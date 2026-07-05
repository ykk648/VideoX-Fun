#!/usr/bin/env python3
"""Preprocess a video dataset for FlashHead V2V training.

Pipeline per video:
  1. Detect faces with SCRFD at regular intervals across the whole video.
  2. Split into contiguous stable-face regions where the face center drift
     stays below --max_center_drift (relative to crop size).
  3. For each stable region (>= --min_seconds long), compute a single fixed
     crop box using median center + P90 face size, matching the target aspect
     ratio.
  4. Crop & resize every frame to an adaptive size from [(512,512),(448,576),(576,448)],
     matched to the face crop aspect ratio, then write as mp4.
  5. Extract 16 kHz mono WAV audio aligned to the cropped region.
  6. Write a training-ready JSON (same schema as VideoSpeechDataset):
     [{"file_path": "...", "audio_path": "...", "text": ""}, ...]

The training DataLoader (VideoSpeechDataset) randomly samples short clips
(e.g. 33 frames) from these longer videos each epoch, so we keep videos as
long as the face remains stable.

Usage:
  python scripts/flashhead/preprocess_face_video.py \
      --input_dir datasets/raw_videos \
      --output_dir datasets/flashhead_v2v \
      --scrfd_model models/face_det/scrfd_500m_bnkps.onnx \
      --face_scale 2.0 \
      --min_seconds 3.0 \
      --fps 25 \
      --max_center_drift 0.3 \
      --lufs -23 \
      --num_workers 8 \
      --gpu_ids 0,1,2,3,4,5,6,7
"""

import argparse
import fcntl
import json
import logging
import multiprocessing as mp
import os
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np
import pyloudnorm as pyln

# Import SCRFD from same directory (standalone mode)
from scrfd import SCRFD

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger(__name__)

VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".flv", ".wmv"}

# Adaptive crop candidates: (width, height) — square, portrait, landscape
CROP_CANDIDATES = [(512, 512), (448, 576), (576, 448)]


def _metadata_relpath(path: str, root: str) -> str:
    """Return a POSIX-style path relative to the metadata data root."""
    return os.path.relpath(path, root).replace(os.sep, "/")


def _progress_key(path: str, root: str) -> str:
    """Return the privacy-preserving key used in progress.json."""
    return _metadata_relpath(path, root)


def _read_video_size(path: str):
    """Return (width, height) for a video file, or (None, None) on failure."""
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        return None, None
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    return width or None, height or None


def _select_candidate_size(crop_w: int, crop_h: int):
    """Pick the (w, h) from CROP_CANDIDATES closest to the crop aspect ratio."""
    orig_ratio = crop_w / max(crop_h, 1)
    return min(CROP_CANDIDATES, key=lambda wh: abs(wh[0] / wh[1] - orig_ratio))


def _temporal_smooth(arr: np.ndarray, win: int = 5) -> np.ndarray:
    """Smooth a 1D sequence with a moving-average kernel."""
    if len(arr) < win:
        return arr
    kernel = np.ones(win) / win
    return np.convolve(arr, kernel, mode="same")


# ---------------------------------------------------------------------------
# Video I/O helpers
# ---------------------------------------------------------------------------

def get_video_info(path: str, target_fps: float):
    """Return (num_target_frames, frame_shape, orig_fps) without loading frames."""
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {path}")
    orig_fps = cap.get(cv2.CAP_PROP_FPS) or target_fps
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total / orig_fps if total > 0 else 0
    n_target = int(duration * target_fps) if duration > 0 else total
    n_target = max(1, n_target)
    # Read one frame to get shape
    ret, frame = cap.read()
    if not ret:
        cap.release()
        raise ValueError(f"Cannot decode first frame: {path}")
    shape = frame.shape
    del frame
    cap.release()
    return n_target, shape, orig_fps


def detect_faces_streaming(detector, video_path, n_target, orig_fps, target_fps,
                           sample_interval: int = 1):
    """Stream sampled frames for face detection; never loads full video.

    Returns list of length n_target with (cx,cy,w,h) or None per frame.
    """
    results = [None] * n_target
    target_indices = np.round(np.arange(n_target) * orig_fps / target_fps).astype(int)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")

    next_sample = 0  # next index in target_indices to detect
    frame_idx = 0
    while next_sample < n_target:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_idx == target_indices[next_sample]:
            # Detect on sampled positions
            if next_sample % sample_interval == 0:
                dets, _ = detector.detect(frame, thresh=0.5, input_size=(640, 640), max_num=1)
                if len(dets) > 0:
                    x1, y1, x2, y2 = dets[0][:4]
                    results[next_sample] = (
                        (x1 + x2) / 2, (y1 + y2) / 2,
                        max(1, x2 - x1), max(1, y2 - y1),
                    )
            del frame  # free immediately
            next_sample += 1
        frame_idx += 1
    cap.release()

    # Fill gaps via nearest detected frame
    valid = [(i, r) for i, r in enumerate(results) if r is not None]
    if valid:
        for i in range(n_target):
            if results[i] is None:
                nearest = min(valid, key=lambda v: abs(v[0] - i))
                if abs(nearest[0] - i) <= sample_interval * 2:
                    results[i] = nearest[1]
    return results


def extract_audio_segment(video_path: str, out_wav: str, start_sec: float, duration_sec: float):
    """Extract audio segment as 16 kHz mono WAV using ffmpeg."""
    cmd = [
        "ffmpeg", "-y", "-loglevel", "error",
        "-i", video_path,
        "-ss", f"{start_sec:.4f}",
        "-t", f"{duration_sec:.4f}",
        "-vn", "-acodec", "pcm_s16le", "-ar", "16000", "-ac", "1",
        out_wav,
    ]
    ret = subprocess.run(cmd, capture_output=True)
    if ret.returncode != 0:
        raise RuntimeError(f"ffmpeg failed: {ret.stderr.decode()}")


def loudness_norm(wav_path: str, target_lufs: float = -23.0):
    """Normalize WAV file to target LUFS (ITU-R BS.1770) in-place."""
    import soundfile as sf
    audio, sr = sf.read(wav_path)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    meter = pyln.Meter(sr)
    loudness = meter.integrated_loudness(audio)
    if abs(loudness) > 100:  # silence or near-silence
        return
    normalized = pyln.normalize.loudness(audio, loudness, target_lufs)
    sf.write(wav_path, normalized, sr)


# ---------------------------------------------------------------------------
# Face detection & crop geometry
# ---------------------------------------------------------------------------


def compute_stable_crop(face_results, frame_shape, scale: float):
    """Compute a fixed crop box from face detections.

    Uses temporal-smoothed sequences with quantile-based robust statistics
    (3%-97% for position, 97% for size) for stability.
    Returns (x1, y1, x2, y2) integer crop box or None.
    """
    valid = [r for r in face_results if r is not None]
    if not valid:
        return None
    cxs = np.array([r[0] for r in valid])
    cys = np.array([r[1] for r in valid])
    fws = np.array([r[2] for r in valid])
    fhs = np.array([r[3] for r in valid])

    # Temporal smoothing to reduce jitter
    cxs = _temporal_smooth(cxs)
    cys = _temporal_smooth(cys)
    fws = _temporal_smooth(fws)
    fhs = _temporal_smooth(fhs)

    # Robust statistics: quantile-based instead of median/P90
    cx = np.percentile(cxs, 50)
    cy = np.percentile(cys, 50)
    face_w = np.percentile(fws, 97)
    face_h = np.percentile(fhs, 97)

    frame_h, frame_w = frame_shape[:2]

    crop_w = face_w * scale
    crop_h = face_h * scale

    crop_w = min(crop_w, frame_w)
    crop_h = min(crop_h, frame_h)

    left = cx - crop_w / 2
    top = cy - crop_h / 2
    left = min(max(0, left), frame_w - crop_w)
    top = min(max(0, top), frame_h - crop_h)

    return (int(round(left)), int(round(top)),
            int(round(left + crop_w)), int(round(top + crop_h)))


def find_stable_segments(face_results, window_size: int, max_drift: float, frame_shape, scale: float):
    """Split frame range into maximal stable-face segments.

    A segment is stable if the face center drift (max - min) within any
    window_size sliding window stays below max_drift * crop_size.

    Returns list of (start, end) frame indices (exclusive end).
    """
    n = len(face_results)
    if n == 0:
        return []

    # Pre-compute crop box for the whole video to get a reference crop size
    # for normalizing drift. Use it as a denominator.
    ref_crop = compute_stable_crop(face_results, frame_shape, scale)
    if ref_crop is None:
        return []
    ref_size = max(ref_crop[2] - ref_crop[0], ref_crop[3] - ref_crop[1], 1)

    # Mark each frame as having a valid face
    has_face = [r is not None for r in face_results]

    # Find contiguous face regions, then check drift within each
    segments = []
    seg_start = None
    for i in range(n):
        if has_face[i]:
            if seg_start is None:
                seg_start = i
        else:
            if seg_start is not None:
                segments.append((seg_start, i))
                seg_start = None
    if seg_start is not None:
        segments.append((seg_start, n))

    # Within each contiguous-face region, split at high-drift points
    stable_segments = []
    for seg_s, seg_e in segments:
        sub_results = face_results[seg_s:seg_e]
        cxs = np.array([r[0] for r in sub_results])
        cys = np.array([r[1] for r in sub_results])

        # Use a sliding window to detect drift jumps
        run_start = 0
        for i in range(1, len(sub_results)):
            window = sub_results[run_start:i + 1]
            w_cxs = np.array([r[0] for r in window])
            w_cys = np.array([r[1] for r in window])
            dx = (w_cxs.max() - w_cxs.min()) / ref_size
            dy = (w_cys.max() - w_cys.min()) / ref_size
            if max(dx, dy) > max_drift:
                # End the current run before this frame
                stable_segments.append((seg_s + run_start, seg_s + i))
                run_start = i
        stable_segments.append((seg_s + run_start, seg_e))

    return stable_segments


# ---------------------------------------------------------------------------
# Main processing
# ---------------------------------------------------------------------------

def _stream_crop_write(video_path: str, target_indices, seg_s: int, seg_e: int,
                       crop_box, out_size, out_path: str, fps: float):
    """Open video, seek to seg_s, crop+resize+pad+write each frame in [seg_s, seg_e).

    out_size: (target_w, target_h) from _select_candidate_size.
    Frames are resized to fit inside out_size while preserving aspect ratio,
    then centered on a black canvas.
    """
    x1, y1, x2, y2 = crop_box
    target_w, target_h = out_size
    crop_w = max(x2 - x1, 1)
    crop_h = max(y2 - y1, 1)
    scale = min(target_w / crop_w, target_h / crop_h)
    new_w = int(round(crop_w * scale))
    new_h = int(round(crop_h * scale))
    x_offset = (target_w - new_w) // 2
    y_offset = (target_h - new_h) // 2

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video for writing: {video_path}")

    cap.set(cv2.CAP_PROP_POS_FRAMES, target_indices[seg_s])
    writer = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"mp4v"),
                             fps, (target_w, target_h))
    try:
        for fi in range(seg_s, seg_e):
            ret, frame = cap.read()
            if not ret:
                log.warning(f"  Frame {fi} read failed, stopping segment early")
                break
            crop = frame[y1:y2, x1:x2]
            resized = cv2.resize(crop, (new_w, new_h),
                                 interpolation=cv2.INTER_LANCZOS4)
            canvas = np.zeros((target_h, target_w, 3), dtype=np.uint8)
            canvas[y_offset:y_offset + new_h, x_offset:x_offset + new_w] = resized
            writer.write(canvas)
            del frame, crop, resized, canvas
    finally:
        writer.release()
        cap.release()


def process_video(
    video_path: str,
    detector: SCRFD,
    output_dir: str,
    face_scale: float,
    min_frames: int,
    fps: float,
    max_center_drift: float,
    detect_interval: int,
    video_idx: int,
    lufs: float = -23.0,
):
    """Process one video into face-cropped stable segments.

    Two-pass streaming approach to avoid OOM:
      Pass 1 — sample frames for face detection, store only coordinates.
      Pass 2 — for each stable segment, seek+crop+write frame by frame.

    Output size is chosen adaptively from CROP_CANDIDATES based on the
    face crop aspect ratio (square / portrait / landscape).
    """
    video_name = Path(video_path).stem
    video_out_dir = os.path.join(output_dir, "videos")
    audio_out_dir = os.path.join(output_dir, "audios")
    os.makedirs(video_out_dir, exist_ok=True)
    os.makedirs(audio_out_dir, exist_ok=True)

    # --- Pass 1: metadata + streaming face detection ---
    try:
        n_frames, frame_shape, orig_fps = get_video_info(video_path, fps)
    except Exception as e:
        log.warning(f"Skip {video_path}: {e}")
        return []

    if n_frames < min_frames:
        log.info(f"Skip {video_path}: only {n_frames} frames < min_frames={min_frames}")
        return []

    face_results = detect_faces_streaming(
        detector, video_path, n_frames, orig_fps, fps,
        sample_interval=detect_interval,
    )
    n_detected = sum(1 for r in face_results if r is not None)
    if n_detected < n_frames * 0.3:
        log.info(f"Skip {video_path}: too few face detections ({n_detected}/{n_frames})")
        return []

    # --- Find stable segments (CPU-light, only face coords in memory) ---
    window_size = int(fps * 2)
    stable_segs = find_stable_segments(
        face_results, window_size, max_center_drift,
        frame_shape, face_scale,
    )

    # Pre-compute target_indices for seeking
    target_indices = np.round(np.arange(n_frames) * orig_fps / fps).astype(int)

    # --- Pass 2: stream-crop-write each stable segment ---
    results = []
    for seg_idx, (seg_s, seg_e) in enumerate(stable_segs):
        seg_len = seg_e - seg_s
        if seg_len < min_frames:
            continue

        seg_face = face_results[seg_s:seg_e]
        crop_box = compute_stable_crop(seg_face, frame_shape, face_scale)
        if crop_box is None:
            continue

        # Adaptive output size based on crop aspect ratio
        x1, y1, x2, y2 = crop_box
        out_size = _select_candidate_size(x2 - x1, y2 - y1)

        out_name = f"{video_name}_{video_idx:04d}_seg{seg_idx:04d}"
        vid_path = os.path.join(video_out_dir, f"{out_name}.mp4")
        try:
            _stream_crop_write(video_path, target_indices, seg_s, seg_e,
                               crop_box, out_size, vid_path, fps)
        except Exception as e:
            log.warning(f"  seg {seg_idx}: video write failed: {e}")
            continue

        start_sec = seg_s / fps
        duration_sec = seg_len / fps
        wav_path = os.path.join(audio_out_dir, f"{out_name}.wav")
        try:
            extract_audio_segment(video_path, wav_path, start_sec, duration_sec)
            loudness_norm(wav_path, target_lufs=lufs)
        except Exception as e:
            log.warning(f"  seg {seg_idx}: audio extraction failed: {e}")
            wav_path = ""

        results.append({
            "file_path": _metadata_relpath(vid_path, output_dir),
            "audio_path": _metadata_relpath(wav_path, output_dir) if wav_path else "",
            "text": "",
            "type": "video",
            "width": out_size[0],
            "height": out_size[1],
        })
        log.info(f"  {video_name} seg {seg_idx}: frames {seg_s}-{seg_e} "
                 f"({seg_len} frames, {duration_sec:.1f}s), "
                 f"crop={crop_box}, size={out_size} -> {out_name}")

    return results


def collect_videos(input_dir: str):
    """Recursively find all video files."""
    videos = []
    for root, dirs, files in os.walk(input_dir):
        for f in sorted(files):
            if Path(f).suffix.lower() in VIDEO_EXTS:
                videos.append(os.path.join(root, f))
    return videos


# ---------------------------------------------------------------------------
# Progress tracking (file-locked JSON for multi-process safety)
# ---------------------------------------------------------------------------

def _load_progress(progress_path: str) -> set:
    """Load the set of already-processed video paths from the progress file."""
    if not os.path.exists(progress_path):
        return set()
    with open(progress_path, "r") as f:
        fcntl.flock(f, fcntl.LOCK_SH)
        try:
            data = json.load(f)
        except json.JSONDecodeError:
            data = []
        finally:
            fcntl.flock(f, fcntl.LOCK_UN)
    return set(data)


def _save_progress(progress_path: str, video_path: str):
    """Append a video path to the progress file (file-locked)."""
    with open(progress_path, "a+") as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        try:
            f.seek(0)
            try:
                data = json.load(f)
            except json.JSONDecodeError:
                data = []
            data.append(video_path)
            f.seek(0)
            f.truncate()
            json.dump(data, f)
        finally:
            fcntl.flock(f, fcntl.LOCK_UN)


# ---------------------------------------------------------------------------
# Multi-GPU worker
# ---------------------------------------------------------------------------

def _worker_fn(worker_args):
    """Process a shard of videos on a single GPU.

    worker_args: (worker_id, gpu_id, video_paths, video_indices, video_keys, scrfd_model,
                  provider, output_dir, face_scale, min_frames, fps,
                  max_center_drift, detect_interval, progress_path, lufs)
    """
    (worker_id, gpu_id, video_paths, video_indices, video_keys, scrfd_model,
     provider, output_dir, face_scale, min_frames, fps,
     max_center_drift, detect_interval, progress_path, lufs) = worker_args

    if gpu_id is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    detector = SCRFD(scrfd_model, provider=provider)
    all_samples = []
    for local_i, (vi, vpath, vkey) in enumerate(zip(video_indices, video_paths, video_keys)):
        log.info(f"[W{worker_id}] [{local_i + 1}/{len(video_paths)}] {vpath}")
        try:
            samples = process_video(
                video_path=vpath,
                detector=detector,
                output_dir=output_dir,
                face_scale=face_scale,
                min_frames=min_frames,
                fps=fps,
                max_center_drift=max_center_drift,
                detect_interval=detect_interval,
                video_idx=vi,
                lufs=lufs,
            )
            all_samples.extend(samples)
            # Mark as done only after full success
            _save_progress(progress_path, vkey)
            log.info(f"[W{worker_id}]   -> {len(samples)} segments")
        except Exception as e:
            log.error(f"[W{worker_id}] Error processing {vpath}: {e}")
    return all_samples


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Preprocess videos for FlashHead V2V training")
    parser.add_argument("--input_dir", type=str, required=True,
                        help="Directory containing source videos (searched recursively)")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Output directory for cropped videos, audio, and metadata JSON")
    parser.add_argument("--scrfd_model", type=str, required=True,
                        help="Path to SCRFD ONNX model (e.g. scrfd_500m_bnkps.onnx)")
    parser.add_argument("--face_scale", type=float, default=2.0,
                        help="Expansion factor from face bbox to crop region")
    parser.add_argument("--min_seconds", type=float, default=3.0,
                        help="Minimum stable segment duration in seconds (shorter segments are dropped)")
    parser.add_argument("--fps", type=float, default=25.0,
                        help="Target FPS for frame sampling")
    parser.add_argument("--max_center_drift", type=float, default=0.3,
                        help="Max face center drift / crop size ratio before splitting segment")
    parser.add_argument("--detect_interval", type=int, default=4,
                        help="Detect face every N frames (others interpolated) for speed")
    parser.add_argument("--provider", type=str, default="gpu", choices=["gpu", "cpu"],
                        help="ONNX Runtime execution provider")
    parser.add_argument("--output_json", type=str, default=None,
                        help="Output JSON path. Default: <output_dir>/metadata.json")
    parser.add_argument("--num_workers", type=int, default=1,
                        help="Number of parallel worker processes (default: 1)")
    parser.add_argument("--lufs", type=float, default=-23.0,
                        help="Target loudness in LUFS for audio normalization (default: -23.0)")
    parser.add_argument("--gpu_ids", type=str, default=None,
                        help="Comma-separated GPU IDs to use (e.g. '0,1,2,3,4,5,6,7'). "
                             "Workers are assigned to GPUs in round-robin. "
                             "Default: single GPU")
    args = parser.parse_args()

    min_frames = max(1, int(args.min_seconds * args.fps))
    os.makedirs(args.output_dir, exist_ok=True)

    # Parse GPU IDs
    gpu_ids = None
    if args.gpu_ids is not None:
        gpu_ids = [int(g.strip()) for g in args.gpu_ids.split(",")]
        log.info(f"Using GPUs: {gpu_ids}")

    # Progress tracking
    progress_path = os.path.join(args.output_dir, "progress.json")
    done_set = _load_progress(progress_path)

    videos = collect_videos(args.input_dir)
    log.info(f"Found {len(videos)} videos in {args.input_dir}")

    # Filter already-processed videos
    remaining = [
        (i, v, _progress_key(v, args.input_dir))
        for i, v in enumerate(videos)
        if _progress_key(v, args.input_dir) not in done_set
    ]
    log.info(f"Already done: {len(done_set)}, remaining: {len(remaining)}")

    if not remaining:
        log.info("All videos already processed. Nothing to do.")
    else:
        if args.num_workers <= 1:
            # --- Single-process mode ---
            if args.provider == "gpu":
                os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_ids[0]) if gpu_ids else "0"
            detector = SCRFD(args.scrfd_model, provider=args.provider)
            for local_i, (vi, vpath, vkey) in enumerate(remaining):
                log.info(f"[{local_i + 1}/{len(remaining)}] Processing {vpath}")
                samples = process_video(
                    video_path=vpath,
                    detector=detector,
                    output_dir=args.output_dir,
                    face_scale=args.face_scale,
                    min_frames=min_frames,
                    fps=args.fps,
                    max_center_drift=args.max_center_drift,
                    detect_interval=args.detect_interval,
                    video_idx=vi,
                    lufs=args.lufs,
                )
                _save_progress(progress_path, vkey)
                log.info(f"  -> {len(samples)} segments extracted")
        else:
            # --- Multi-process mode ---
            n_workers = args.num_workers
            shard_size = (len(remaining) + n_workers - 1) // n_workers
            worker_jobs = []
            for w in range(n_workers):
                start = w * shard_size
                end = min(start + shard_size, len(remaining))
                if start >= len(remaining):
                    break
                shard = remaining[start:end]
                shard_indices = [s[0] for s in shard]
                shard_paths = [s[1] for s in shard]
                shard_keys = [s[2] for s in shard]
                gpu_id = gpu_ids[w % len(gpu_ids)] if gpu_ids else w % 8
                worker_jobs.append((
                    w, gpu_id, shard_paths, shard_indices, shard_keys,
                    args.scrfd_model, args.provider, args.output_dir,
                    args.face_scale, min_frames, args.fps,
                    args.max_center_drift, args.detect_interval,
                    progress_path, args.lufs,
                ))

            log.info(f"Launching {len(worker_jobs)} workers")
            ctx = mp.get_context("spawn")
            with ctx.Pool(processes=len(worker_jobs)) as pool:
                pool_results = pool.map(_worker_fn, worker_jobs)

            total_segments = sum(len(r) for r in pool_results)
            log.info(f"All workers done. Total segments: {total_segments}")

    # --- Merge all partial metadata shards into final JSON ---
    # Re-scan output dir for all video segments that actually exist
    video_out_dir = os.path.join(args.output_dir, "videos")
    audio_out_dir = os.path.join(args.output_dir, "audios")
    all_samples = []
    if os.path.isdir(video_out_dir):
        for vfile in sorted(os.listdir(video_out_dir)):
            if not vfile.endswith(".mp4"):
                continue
            vid_path = os.path.join(video_out_dir, vfile)
            wav_name = vfile.replace(".mp4", ".wav")
            wav_path = os.path.join(audio_out_dir, wav_name)
            width, height = _read_video_size(vid_path)
            sample = {
                "file_path": _metadata_relpath(vid_path, args.output_dir),
                "audio_path": _metadata_relpath(wav_path, args.output_dir) if os.path.exists(wav_path) else "",
                "text": "",
                "type": "video",
            }
            if width is not None and height is not None:
                sample["width"] = width
                sample["height"] = height
            all_samples.append(sample)

    valid_samples = [s for s in all_samples if s["audio_path"]]
    dropped = len(all_samples) - len(valid_samples)
    if dropped:
        log.warning(f"Dropped {dropped} segments without audio")

    out_json = args.output_json or os.path.join(args.output_dir, "metadata.json")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(valid_samples, f, ensure_ascii=False, indent=2)
    log.info(f"Done. {len(valid_samples)} samples written to {out_json}")


if __name__ == "__main__":
    main()
