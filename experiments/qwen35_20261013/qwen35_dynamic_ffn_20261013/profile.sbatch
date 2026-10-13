#!/bin/bash
#SBATCH --job-name=qwen35-delta-profile
#SBATCH --partition=gpu5090
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:20:00
#SBATCH --chdir=/home/Hongjie_Zeng/qwen35_dynamic_ffn_20261013
#SBATCH --output=results/logs/profile-%j.log
set -euo pipefail
export TOKENIZERS_PARALLELISM=false HF_HUB_OFFLINE=1
exec /opt/slurm/21.08.8/bin/srun /home/Hongjie_Zeng/qwen35_baseline_20261013/venv/bin/python -u profile_deltanet.py
