#!/bin/bash
#SBATCH --job-name=qwen35-deploy
#SBATCH --partition=gpu5090
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=04:00:00
#SBATCH --chdir=/home/Hongjie_Zeng/qwen35_baseline_20261013
#SBATCH --output=results/logs/deploy-%j.log
set -euo pipefail
export HF_ENDPOINT=https://hf-mirror.com
export HF_HUB_DISABLE_XET=1
export HF_HUB_DOWNLOAD_TIMEOUT=120
exec /opt/slurm/21.08.8/bin/srun venv/bin/python -u setup_model.py
