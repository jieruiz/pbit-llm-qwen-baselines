#!/bin/bash
#SBATCH --job-name=cwc
#SBATCH --partition=gpu5090
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=00:05:00
#SBATCH --output=/home/Weican_Chen/projects/pbit-qwen-calibration-20260929/logs/compatibility-%j.out
#SBATCH --error=/home/Weican_Chen/projects/pbit-qwen-calibration-20260929/logs/compatibility-%j.err
set -euo pipefail
ROOT=/home/Weican_Chen/projects/pbit-qwen-calibration-20260929
[ "$(id -un)" = Weican_Chen ]
[ "$(realpath ~)" = /home/Weican_Chen ]
[ "$(realpath "$ROOT")" = "$ROOT" ]
export CUDA_VISIBLE_DEVICES="" PYTHONDONTWRITEBYTECODE=1
for version in old upstream; do
  export PYTHONPATH="$ROOT/compatibility/$version"
  /home/Weican_Chen/projects/pbit-qwen-sigmoid-20260929/.venv/bin/python "$PYTHONPATH/test_calibrated_pdnn.py" -v
  sha256sum "$PYTHONPATH/"*.py
done
echo COMPATIBILITY_COMPLETE
