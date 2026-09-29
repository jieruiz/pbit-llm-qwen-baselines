#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="${PBIT_LLM_ROOT:-$(cd "$SCRIPT_DIR/../.." && pwd)}"
MODEL_PATH="${MODEL_PATH:-$ROOT/models/Qwen2.5-0.5B}"
TRAIN_TEXT="${TRAIN_TEXT:-$ROOT/data/wikitext-2/wiki.train.raw}"
TEST_TEXT="${TEST_TEXT:-$ROOT/data/wikitext-2/wiki.test.raw}"
CODING="${CODING:-bipolar}"
OUTPUT_DIR="${OUTPUT_DIR:-$ROOT/results/full_path_pdnn_layer12_${CODING}}"
INPUT_TEMPERATURE="${INPUT_TEMPERATURE:-1.0}"
HIDDEN_TEMPERATURE="${HIDDEN_TEMPERATURE:-1.0}"
MEAN_STEPS="${MEAN_STEPS:-2000}"
SAMPLE_STEPS="${SAMPLE_STEPS:-2000}"
TRAIN_SAMPLES="${TRAIN_SAMPLES:-4}"

python "$SCRIPT_DIR/train_full_path_distillation.py" \
  --model "$MODEL_PATH" \
  --train-text "$TRAIN_TEXT" \
  --output-dir "$OUTPUT_DIR" \
  --layer 12 \
  --hidden-sizes 4864 \
  --coding "$CODING" \
  --input-temperature "$INPUT_TEMPERATURE" \
  --hidden-temperature "$HIDDEN_TEMPERATURE" \
  --sequence-length 256 \
  --batch-size 4 \
  --mean-steps "$MEAN_STEPS" \
  --sample-steps "$SAMPLE_STEPS" \
  --train-samples "$TRAIN_SAMPLES" \
  --learning-rate 3e-4 \
  --sample-learning-rate 1e-4 \
  --validate-every 200

for samples in 0 1 4 8 16; do
  python "$SCRIPT_DIR/evaluate_full_path_perplexity.py" \
    --model "$MODEL_PATH" \
    --checkpoint "$OUTPUT_DIR/student_sampled.pt" \
    --text-file "$TEST_TEXT" \
    --output "$OUTPUT_DIR/student_sampled_samples${samples}_seed0_ppl.json" \
    --sample-count "$samples" \
    --max-length 2048 \
    --stride 1024 \
    --seed 0
done

for seed in 1 2; do
  python "$SCRIPT_DIR/evaluate_full_path_perplexity.py" \
    --model "$MODEL_PATH" \
    --checkpoint "$OUTPUT_DIR/student_sampled.pt" \
    --text-file "$TEST_TEXT" \
    --output "$OUTPUT_DIR/student_sampled_samples4_seed${seed}_ppl.json" \
    --sample-count 4 \
    --max-length 2048 \
    --stride 1024 \
    --seed "$seed"
done
