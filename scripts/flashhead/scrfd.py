"""Minimal SCRFD face detector using ONNX Runtime.

Extracted from insightface/scrfd for local use without external insightface dependency.
Requires: onnxruntime (or onnxruntime-gpu), numpy, opencv-python.
"""

from __future__ import division

import os.path as osp

import cv2
import numpy as np
import onnxruntime


def _distance2bbox(points, distance):
    x1 = points[:, 0] - distance[:, 0]
    y1 = points[:, 1] - distance[:, 1]
    x2 = points[:, 0] + distance[:, 2]
    y2 = points[:, 1] + distance[:, 3]
    return np.stack([x1, y1, x2, y2], axis=-1)


def _distance2kps(points, distance):
    preds = []
    for i in range(0, distance.shape[1], 2):
        px = points[:, i % 2] + distance[:, i]
        py = points[:, i % 2 + 1] + distance[:, i + 1]
        preds.append(px)
        preds.append(py)
    return np.stack(preds, axis=-1)


class SCRFD:
    """SCRFD ONNX face detector with NMS and optional keypoint output."""

    def __init__(self, model_file: str, provider: str = "gpu"):
        assert osp.exists(model_file), f"SCRFD model not found: {model_file}"
        providers = (
            ["CUDAExecutionProvider"] if provider == "gpu"
            else ["CPUExecutionProvider"]
        )
        self.session = onnxruntime.InferenceSession(model_file, providers=providers)
        self.nms_thresh = 0.4
        self.center_cache = {}
        self._init_vars()

    def _init_vars(self):
        input_cfg = self.session.get_inputs()[0]
        input_shape = input_cfg.shape
        self.input_size = (
            None if isinstance(input_shape[2], str)
            else tuple(input_shape[2:4][::-1])
        )
        self.input_name = input_cfg.name
        outputs = self.session.get_outputs()
        self.batched = len(outputs[0].shape) == 3
        self.output_names = [o.name for o in outputs]

        self.use_kps = False
        self._num_anchors = 1
        n_out = len(outputs)
        if n_out == 6:
            self.fmc = 3
            self._feat_stride_fpn = [8, 16, 32]
            self._num_anchors = 2
        elif n_out == 9:
            self.fmc = 3
            self._feat_stride_fpn = [8, 16, 32]
            self._num_anchors = 2
            self.use_kps = True
        elif n_out == 10:
            self.fmc = 5
            self._feat_stride_fpn = [8, 16, 32, 64, 128]
        elif n_out == 15:
            self.fmc = 5
            self._feat_stride_fpn = [8, 16, 32, 64, 128]
            self.use_kps = True

    def _forward(self, img, thresh):
        scores_list, bboxes_list, kpss_list = [], [], []
        input_size = tuple(img.shape[0:2][::-1])
        blob = cv2.dnn.blobFromImage(
            img, 1.0 / 128, input_size, (127.5, 127.5, 127.5), swapRB=True
        )
        net_outs = self.session.run(self.output_names, {self.input_name: blob})
        input_height, input_width = blob.shape[2], blob.shape[3]

        for idx, stride in enumerate(self._feat_stride_fpn):
            if self.batched:
                scores = net_outs[idx][0]
                bbox_preds = net_outs[idx + self.fmc][0] * stride
                kps_preds = net_outs[idx + self.fmc * 2][0] * stride if self.use_kps else None
            else:
                scores = net_outs[idx]
                bbox_preds = net_outs[idx + self.fmc] * stride
                kps_preds = net_outs[idx + self.fmc * 2] * stride if self.use_kps else None

            height = input_height // stride
            width = input_width // stride
            key = (height, width, stride)
            if key in self.center_cache:
                anchor_centers = self.center_cache[key]
            else:
                anchor_centers = np.stack(
                    np.mgrid[:height, :width][::-1], axis=-1
                ).astype(np.float32)
                anchor_centers = (anchor_centers * stride).reshape((-1, 2))
                if self._num_anchors > 1:
                    anchor_centers = np.stack(
                        [anchor_centers] * self._num_anchors, axis=1
                    ).reshape((-1, 2))
                if len(self.center_cache) < 100:
                    self.center_cache[key] = anchor_centers

            pos_inds = np.where(scores >= thresh)[0]
            bboxes = _distance2bbox(anchor_centers, bbox_preds)
            scores_list.append(scores[pos_inds])
            bboxes_list.append(bboxes[pos_inds])
            if self.use_kps:
                kpss = _distance2kps(anchor_centers, kps_preds).reshape((-1, kps_preds.shape[1] // 2, 2))
                kpss_list.append(kpss[pos_inds])
        return scores_list, bboxes_list, kpss_list

    def _nms(self, dets):
        x1, y1, x2, y2, scores = dets[:, 0], dets[:, 1], dets[:, 2], dets[:, 3], dets[:, 4]
        areas = (x2 - x1 + 1) * (y2 - y1 + 1)
        order = scores.argsort()[::-1]
        keep = []
        while order.size > 0:
            i = order[0]
            keep.append(i)
            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])
            inter = np.maximum(0.0, xx2 - xx1 + 1) * np.maximum(0.0, yy2 - yy1 + 1)
            ovr = inter / (areas[i] + areas[order[1:]] - inter)
            order = order[np.where(ovr <= self.nms_thresh)[0] + 1]
        return keep

    def detect(self, img, thresh=0.5, input_size=(640, 640), max_num=0):
        """Detect faces in a BGR image.

        Returns:
            bboxes: (N, 5) array [x1, y1, x2, y2, score]
            kpss:   (N, 5, 2) array or None
        """
        im_ratio = float(img.shape[0]) / img.shape[1]
        model_ratio = float(input_size[1]) / input_size[0]
        if im_ratio > model_ratio:
            new_height = input_size[1]
            new_width = int(new_height / im_ratio)
        else:
            new_width = input_size[0]
            new_height = int(new_width * im_ratio)
        det_scale = float(new_height) / img.shape[0]
        resized_img = cv2.resize(img, (new_width, new_height))
        det_img = np.zeros((input_size[1], input_size[0], 3), dtype=np.uint8)
        det_img[:new_height, :new_width, :] = resized_img

        scores_list, bboxes_list, kpss_list = self._forward(det_img, thresh)
        scores = np.vstack(scores_list)
        order = scores.ravel().argsort()[::-1]
        bboxes = np.vstack(bboxes_list) / det_scale
        kpss = np.vstack(kpss_list) / det_scale if self.use_kps else None

        pre_det = np.hstack((bboxes, scores)).astype(np.float32, copy=False)[order]
        keep = self._nms(pre_det)
        det = pre_det[keep]
        if kpss is not None:
            kpss = kpss[order][keep]

        if max_num > 0 and det.shape[0] > max_num:
            area = (det[:, 2] - det[:, 0]) * (det[:, 3] - det[:, 1])
            det = det[np.argsort(area)[::-1][:max_num]]
            if kpss is not None:
                kpss = kpss[np.argsort(area)[::-1][:max_num]]

        return det, kpss