#!/bin/bash
#SBATCH --job-name=qwen35-delta-8k
#SBATCH --partition=gpu5090
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --chdir=/home/Hongjie_Zeng/qwen35_deltanet_projection_20261013
#SBATCH --output=results/logs/long-context-%j.log
set -euo pipefail
export TOKENIZERS_PARALLELISM=false HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1
python=/home/Hongjie_Zeng/qwen35_baseline_20261013/venv/bin/python
/opt/slurm/21.08.8/bin/srun "$python" -u experiment.py --scope baseline --mode weights --context 8192
for seed in 0 1 2; do
    /opt/slurm/21.08.8/bin/srun "$python" -u experiment.py --scope out_z_qkv --context 8192 --seed "$seed"
done
