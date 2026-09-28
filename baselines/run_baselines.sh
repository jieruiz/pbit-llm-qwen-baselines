#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="${PBIT_LLM_ROOT:-$(cd "$SCRIPT_DIR/.." && pwd)}"
MODEL="${MODEL_PATH:-$ROOT/models/Qwen2.5-0.5B}"
OUT="$ROOT/results/base_bf16"
CORPUS="${WIKITEXT2_PATH:-$ROOT/data/wikitext-2/wiki.test.raw}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export HF_HUB_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
mkdir -p "$OUT"

test -f "$MODEL/model.safetensors" || {
  echo "Missing model weights: $MODEL/model.safetensors" >&2
  exit 1
}
test -f "$CORPUS" || {
  echo "Missing WikiText-2 corpus: $CORPUS" >&2
  exit 1
}

python "$SCRIPT_DIR/verify_model.py" \
  --model "$MODEL" --output "$OUT/model_integrity.json"

python "$SCRIPT_DIR/generate_baseline.py" \
  --model "$MODEL" --output "$OUT/generation.json"

python "$SCRIPT_DIR/benchmark_inference.py" \
  --model "$MODEL" --output "$OUT/inference_performance.json"

python "$SCRIPT_DIR/ffn_probe.py" \
  --model "$MODEL" --layer 0 --output "$OUT/ffn_layer0.json"

python "$SCRIPT_DIR/ffn_probe.py" \
  --model "$MODEL" --layer 12 --output "$OUT/ffn_layer12.json"

python "$SCRIPT_DIR/ffn_probe.py" \
  --model "$MODEL" --layer 23 --output "$OUT/ffn_layer23.json"

python "$SCRIPT_DIR/perplexity.py" \
  --model "$MODEL" --text-file "$CORPUS" \
  --max-length 2048 --stride 1024 \
  --output "$OUT/wikitext2_perplexity.json"

echo "Baseline results written to $OUT"
