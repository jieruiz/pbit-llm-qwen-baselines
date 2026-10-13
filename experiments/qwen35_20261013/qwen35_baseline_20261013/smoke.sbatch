#!/bin/bash
#SBATCH --job-name=qwen35-smoke
#SBATCH --partition=gpu5090
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --chdir=/home/Hongjie_Zeng/qwen35_baseline_20261013
#SBATCH --output=results/logs/smoke-%j.log
set -euo pipefail
export TOKENIZERS_PARALLELISM=false HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1
exec /opt/slurm/21.08.8/bin/srun venv/bin/python -u baseline.py --task smoke
