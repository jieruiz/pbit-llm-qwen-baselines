#!/usr/bin/env bash
# Same training seed/budget and inference seeds for all three conditions.
set -euo pipefail
: "${MODEL_PATH:?Set MODEL_PATH}"
: "${TRAIN_TEXT:?Set TRAIN_TEXT}"
: "${TEST_TEXT:?Set TEST_TEXT}"
: "${COMPARISON_OUTPUT:?Set COMPARISON_OUTPUT to a fresh directory}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
mkdir -p "$COMPARISON_OUTPUT"
if [[ -n "$(ls -A "$COMPARISON_OUTPUT")" ]]; then
  echo 'Refusing to overwrite a nonempty experiment directory' >&2
  exit 1
fi
for condition in bipolar binary_same_temperature binary_half_temperature; do
  coding=binary
  input_temperature=0.25
  hidden_temperature=1.0
  if [[ "$condition" == bipolar ]]; then coding=bipolar; fi
  if [[ "$condition" == binary_half_temperature ]]; then
    input_temperature=0.125
    hidden_temperature=0.5
  fi
  echo "START_CONDITION $condition"
  CODING="$coding" INPUT_TEMPERATURE="$input_temperature" \
    HIDDEN_TEMPERATURE="$hidden_temperature" \
    OUTPUT_DIR="$COMPARISON_OUTPUT/$condition" \
    MEAN_STEPS=2000 SAMPLE_STEPS=2000 TRAIN_SAMPLES=4 \
    bash "$SCRIPT_DIR/run_full_path_layer12_experiment.sh"
done
