#!/bin/bash
#SBATCH --job-name=qwen35-baseline
#SBATCH --partition=gpu5090
#SBATCH --array=0-6%2
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
#SBATCH --time=12:00:00
#SBATCH --chdir=/home/Hongjie_Zeng/qwen35_baseline_20261013
#SBATCH --output=results/logs/baseline-%A_%a.log
set -euo pipefail
export TOKENIZERS_PARALLELISM=false HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1
tasks=(wikitext2 hellaswag piqa winogrande arc_easy arc_challenge generation_performance)
exec /opt/slurm/21.08.8/bin/srun venv/bin/python -u baseline.py --task "${tasks[$SLURM_ARRAY_TASK_ID]}"
