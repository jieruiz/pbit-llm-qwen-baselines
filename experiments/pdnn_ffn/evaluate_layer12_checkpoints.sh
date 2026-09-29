#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="${PBIT_LLM_ROOT:-$(cd "$SCRIPT_DIR/../.." && pwd)}"
MODEL_PATH="${MODEL_PATH:-$ROOT/models/Qwen2.5-0.5B}"
TEST_TEXT="${TEST_TEXT:-$ROOT/data/wikitext-2/wiki.test.raw}"
OUTPUT_DIR="${OUTPUT_DIR:-$ROOT/results/pdnn_ffn_layer12_bipolar}"

for checkpoint in student_mean student_sampled; do
  for samples in 0 1 4 8 16; do
    python "$SCRIPT_DIR/evaluate_perplexity.py" \
      --model "$MODEL_PATH" \
      --checkpoint "$OUTPUT_DIR/${checkpoint}.pt" \
      --text-file "$TEST_TEXT" \
      --output "$OUTPUT_DIR/${checkpoint}_samples${samples}_seed0_ppl.json" \
      --sample-count "$samples" \
      --max-length 2048 \
      --stride 1024 \
      --seed 0
  done
done

for seed in 1 2; do
  python "$SCRIPT_DIR/evaluate_perplexity.py" \
    --model "$MODEL_PATH" \
    --checkpoint "$OUTPUT_DIR/student_sampled.pt" \
    --text-file "$TEST_TEXT" \
    --output "$OUTPUT_DIR/student_sampled_samples4_seed${seed}_ppl.json" \
    --sample-count 4 \
    --max-length 2048 \
    --stride 1024 \
    --seed "$seed"
done
