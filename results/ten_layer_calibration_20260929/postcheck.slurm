#!/bin/bash
#SBATCH --job-name=cwc
#SBATCH --partition=gpu5090
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=00:05:00
#SBATCH --output=/home/Weican_Chen/projects/pbit-qwen-calibration-20260929/logs/postcheck-%j.out
#SBATCH --error=/home/Weican_Chen/projects/pbit-qwen-calibration-20260929/logs/postcheck-%j.err
set -euo pipefail
ROOT=/home/Weican_Chen/projects/pbit-qwen-calibration-20260929
[ "$(id -un)" = Weican_Chen ]
[ "$(realpath "$ROOT")" = "$ROOT" ]
export CUDA_VISIBLE_DEVICES="" PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="$ROOT/code/experiments/pdnn_ffn"
export TMPDIR="$ROOT/postchecks/tmp"
mkdir -p "$TMPDIR"
/home/Weican_Chen/projects/pbit-qwen-sigmoid-20260929/.venv/bin/python "$ROOT/postchecks/test_calibrated_pdnn.py" -v
sha256sum "$ROOT/postchecks/test_calibrated_pdnn.py"
echo POSTCHECK_COMPLETE
