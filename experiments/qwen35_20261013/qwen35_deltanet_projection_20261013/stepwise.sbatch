#!/bin/bash
#SBATCH --job-name=qwen35-delta-stepwise
#SBATCH --partition=gpu5090
#SBATCH --array=0-1%2
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=04:00:00
#SBATCH --chdir=/home/Hongjie_Zeng/qwen35_deltanet_projection_20261013
#SBATCH --output=results/logs/stepwise-%A_%a.log
set -euo pipefail
export TOKENIZERS_PARALLELISM=false HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1
scenarios=(solo combined)
scenario="${scenarios[$SLURM_ARRAY_TASK_ID]}"
python=/home/Hongjie_Zeng/qwen35_baseline_20261013/venv/bin/python
for scope in out out_z out_z_qkv; do
    if [[ "$scenario" == solo ]]; then
        /opt/slurm/21.08.8/bin/srun "$python" -u experiment.py --scope "$scope" --scenario solo --mode weights
    fi
    for seed in 0 1 2; do
        /opt/slurm/21.08.8/bin/srun "$python" -u experiment.py --scope "$scope" --scenario "$scenario" --seed "$seed" --cycles 4096
    done
done
