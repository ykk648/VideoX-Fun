#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"
export NUM_PROCESSES="${NUM_PROCESSES:-2}"

export MODEL_NAME="${MODEL_NAME:-models/Diffusion_Transformer/SoulX-FlashHead-1_3B}"
export AUDIO_MODEL_NAME="${AUDIO_MODEL_NAME:-models/Diffusion_Transformer/wav2vec2-base-960h}"
export DATASET_NAME="${DATASET_NAME:-datasets/X-Fun-Videos-Audios-Demo}"
export DATASET_META_NAME="${DATASET_META_NAME:-datasets/X-Fun-Videos-Audios-Demo/metadata_add_width_height.json}"
export OUTPUT_DIR="${OUTPUT_DIR:-output_dir_flashhead_v2v}"
export NCCL_DEBUG="${NCCL_DEBUG:-INFO}"
export LOG_DIR="${LOG_DIR:-${REPO_ROOT}/logs/flashhead_v2v}"
export VIDEO_SAMPLE_N_FRAMES="${VIDEO_SAMPLE_N_FRAMES:-41}"
export FIX_SAMPLE_HEIGHT="${FIX_SAMPLE_HEIGHT:-512}"
export FIX_SAMPLE_WIDTH="${FIX_SAMPLE_WIDTH:-512}"
export V2V_PROB="${V2V_PROB:-0.8}"
export KEEP_HEADS_MAX="${KEEP_HEADS_MAX:-2}"
export SPATIAL_MARGIN_MAX="${SPATIAL_MARGIN_MAX:-2}"
export AUDIO_DROPOUT_PROB="${AUDIO_DROPOUT_PROB:-0.15}"

mkdir -p "${LOG_DIR}"
LOG_FILE="${LOG_DIR}/train_v2v_$(date +%Y%m%d_%H%M%S).log"

echo "Saving training log to ${LOG_FILE}"

accelerate launch \
  --num_processes="${NUM_PROCESSES}" \
  --num_machines=1 \
  --main_process_port=0 \
  --mixed_precision="bf16" \
  scripts/flashhead/train_s2v_v2v.py \
  --config_path="config/wan2.1/wan_civitai.yaml" \
  --pretrained_model_name_or_path="${MODEL_NAME}" \
  --audio_encoder_path="${AUDIO_MODEL_NAME}" \
  --train_data_dir="${DATASET_NAME}" \
  --train_data_meta="${DATASET_META_NAME}" \
  --video_sample_size=512 \
  --token_sample_size=512 \
  --fix_sample_size "${FIX_SAMPLE_HEIGHT}" "${FIX_SAMPLE_WIDTH}" \
  --video_sample_stride=1 \
  --video_sample_n_frames="${VIDEO_SAMPLE_N_FRAMES}" \
  --train_batch_size=1 \
  --gradient_accumulation_steps=1 \
  --dataloader_num_workers=8 \
  --num_train_epochs=100 \
  --checkpointing_steps=1000 \
  --learning_rate=2e-05 \
  --lr_scheduler="constant_with_warmup" \
  --lr_warmup_steps=100 \
  --seed=42 \
  --output_dir="${OUTPUT_DIR}" \
  --gradient_checkpointing \
  --mixed_precision="bf16" \
  --adam_weight_decay=3e-2 \
  --adam_epsilon=1e-10 \
  --vae_mini_batch=1 \
  --max_grad_norm=0.05 \
  --enable_bucket \
  --uniform_sampling \
  --low_vram \
  --trainable_modules "." \
  --v2v_prob="${V2V_PROB}" \
  --keep_heads_max="${KEEP_HEADS_MAX}" \
  --spatial_margin_max="${SPATIAL_MARGIN_MAX}" \
  --audio_dropout_prob="${AUDIO_DROPOUT_PROB}" \
  2>&1 | tee "${LOG_FILE}"
