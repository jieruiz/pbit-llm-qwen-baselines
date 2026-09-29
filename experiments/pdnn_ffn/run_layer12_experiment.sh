#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="${PBIT_LLM_ROOT:-$(cd "$SCRIPT_DIR/../.." && pwd)}"
MODEL_PATH="${MODEL_PATH:-$ROOT/models/Qwen2.5-0.5B}"
TRAIN_TEXT="${TRAIN_TEXT:-$ROOT/data/wikitext-2/wiki.train.raw}"
TEST_TEXT="${TEST_TEXT:-$ROOT/data/wikitext-2/wiki.test.raw}"
OUTPUT_DIR="${OUTPUT_DIR:-$ROOT/results/pdnn_ffn_layer12_bipolar}"

python "$SCRIPT_DIR/train_layer_distillation.py" \
  --model "$MODEL_PATH" \
  --train-text "$TRAIN_TEXT" \
  --output-dir "$OUTPUT_DIR" \
  --layer 12 \
  --coding bipolar \
  --sequence-length 256 \
  --batch-size 4 \
  --mean-steps 2000 \
  --sample-steps 2000 \
  --train-samples 4 \
  --learning-rate 3e-4 \
  --sample-learning-rate 1e-4 \
  --validate-every 200

for checkpoint in student_mean student_sampled; do
  for samples in 0 1 4 8 16; do
    python "$SCRIPT_DIR/evaluate_perplexity.py" \
      --model "$MODEL_PATH" \
      --checkpoint "$OUTPUT_DIR/${checkpoint}.pt" \
      --text-file "$TEST_TEXT" \
      --output "$OUTPUT_DIR/${checkpoint}_samples${samples}_ppl.json" \
      --sample-count "$samples" \
      --max-length 2048 \
      --stride 1024 \
      --seed 0
  done
done
