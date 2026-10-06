#!/bin/bash
#SBATCH --job-name=cwc
#SBATCH --partition=gpu5090
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=48G
#SBATCH --time=08:00:00
#SBATCH --output=/home/Weican_Chen/projects/pbit-qwen-and-variance-20261005/logs/formal-%j.out
#SBATCH --error=/home/Weican_Chen/projects/pbit-qwen-and-variance-20261005/logs/formal-%j.err
set -euo pipefail
ROOT=/home/Weican_Chen/projects/pbit-qwen-and-variance-20261005
[ "$(id -un)" = Weican_Chen ]
[ "$(realpath ~)" = /home/Weican_Chen ]
[ "$(realpath "$ROOT")" = "$ROOT" ]
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=6 TOKENIZERS_PARALLELISM=false
export TMPDIR="$ROOT/tmp"
PY=/home/Weican_Chen/projects/pbit-qwen-sigmoid-20260929/.venv/bin/python
trap 'result=$?; "$PY" "$ROOT/collect.py" "$result"; exit "$result"' EXIT
cd "$ROOT/formal-code"
"$PY" -u experiments/pdnn_ffn/test_and_variance_loss.py -v
"$PY" -u experiments/pdnn_ffn/run_and_variance_ablation.py --root "$ROOT" --assets /home/Weican_Chen/projects/pbit-qwen-sigmoid-20260929
"$PY" experiments/pdnn_ffn/summarize_and_variance_ablation.py "$ROOT/results"
echo FORMAL_JOB_COMPLETE
