#!/bin/bash
#SBATCH --job-name=qwen35-pbit-prepare
#SBATCH --partition=gpu5090
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --chdir=/home/Hongjie_Zeng/qwen35_pbit_20261013
#SBATCH --output=results/logs/prepare-%j.log
set -euo pipefail
export TOKENIZERS_PARALLELISM=false HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1
python=/home/Hongjie_Zeng/qwen35_baseline_20261013/venv/bin/python
/opt/slurm/21.08.8/bin/srun "$python" -u calibrate.py
exec /opt/slurm/21.08.8/bin/srun "$python" -u verify.py
