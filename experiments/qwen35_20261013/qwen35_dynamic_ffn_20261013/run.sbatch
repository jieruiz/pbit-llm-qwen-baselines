#!/bin/bash
#SBATCH --job-name=qwen35-dynamic-ppl
#SBATCH --partition=gpu5090
#SBATCH --array=0-2%2
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=04:00:00
#SBATCH --chdir=/home/Hongjie_Zeng/qwen35_dynamic_ffn_20261013
#SBATCH --output=results/logs/slurm-%A_%a.log
set -euo pipefail
export TOKENIZERS_PARALLELISM=false HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1
variants=(int8_dense dynamic_ffn dynamic_joint)
variant="${variants[$SLURM_ARRAY_TASK_ID]}"
python=/home/Hongjie_Zeng/qwen35_baseline_20261013/venv/bin/python
/opt/slurm/21.08.8/bin/srun "$python" -u experiment.py --variant "$variant" --smoke
seeds=(0)
if [[ "$SLURM_ARRAY_TASK_ID" -ge 1 ]]; then seeds=(0 1 2); fi
for seed in "${seeds[@]}"; do
    /opt/slurm/21.08.8/bin/srun "$python" -u experiment.py --variant "$variant" --seed "$seed"
done
