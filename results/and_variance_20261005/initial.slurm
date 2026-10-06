#!/bin/bash
#SBATCH --job-name=cwc
#SBATCH --partition=gpu5090
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=32G
#SBATCH --time=03:00:00
#SBATCH --output=/home/Weican_Chen/projects/pbit-qwen-and-variance-20261005/logs/initial-%j.out
#SBATCH --error=/home/Weican_Chen/projects/pbit-qwen-and-variance-20261005/logs/initial-%j.err
set -euo pipefail
ROOT=/home/Weican_Chen/projects/pbit-qwen-and-variance-20261005
[ "$(id -un)" = Weican_Chen ]
[ "$(realpath ~)" = /home/Weican_Chen ]
[ "$(realpath "$ROOT")" = "$ROOT" ]
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=6 TOKENIZERS_PARALLELISM=false
export TMPDIR="$ROOT/tmp"
cd "$ROOT/initial-code"
/home/Weican_Chen/projects/pbit-qwen-sigmoid-20260929/.venv/bin/python -u experiments/pdnn_ffn/rebuild_and_variance_initial.py --root "$ROOT" --assets /home/Weican_Chen/projects/pbit-qwen-sigmoid-20260929
echo INITIAL_JOB_COMPLETE
