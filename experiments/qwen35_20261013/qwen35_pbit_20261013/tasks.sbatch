#!/bin/bash
#SBATCH --job-name=qwen35-pbit-tasks
#SBATCH --partition=gpu5090
#SBATCH --array=0-2%2
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=08:00:00
#SBATCH --chdir=/home/Hongjie_Zeng/qwen35_pbit_20261013
#SBATCH --output=results/logs/tasks-%A_%a.log
set -euo pipefail
export TOKENIZERS_PARALLELISM=false HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1
variants=(ffn_exact attention_pv joint_pv)
variant="${variants[$SLURM_ARRAY_TASK_ID]}"
python=/home/Hongjie_Zeng/qwen35_baseline_20261013/venv/bin/python
for task in hellaswag piqa winogrande arc_easy arc_challenge; do
    /opt/slurm/21.08.8/bin/srun "$python" -u experiment.py --variant "$variant" --seed 0 --task "$task"
done
