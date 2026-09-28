#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="${PBIT_LLM_ROOT:-$(cd "$SCRIPT_DIR/.." && pwd)}"
MODEL="${MODEL_PATH:-$ROOT/models/Qwen2.5-0.5B}"
OUT="$ROOT/results/base_bf16/lm_eval_full"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false

test -f "$MODEL/model.safetensors" || {
  echo "Missing model weights: $MODEL/model.safetensors" >&2
  exit 1
}

lm_eval \
  --model hf \
  --model_args "pretrained=$MODEL,dtype=bfloat16" \
  --tasks hellaswag,arc_easy \
  --num_fewshot 0 \
  --batch_size 32 \
  --device cuda:0 \
  --output_path "$OUT" \
  --seed 0 \
  --verbosity INFO
