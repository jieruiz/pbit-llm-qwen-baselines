#!/bin/bash
#SBATCH --job-name=llama7b-cal
#SBATCH --partition=gpu5090
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --chdir=/home/Hongjie_Zeng/llama7b_joint_pv_20261014
#SBATCH --output=results/logs/calibrate-%j.log

set -euo pipefail
mkdir -p results/logs
export TOKENIZERS_PARALLELISM=false HF_HUB_OFFLINE=1 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
exec /opt/slurm/21.08.8/bin/srun /home/Hongjie_Zeng/.conda/envs/pbit_llm/bin/python -u calibrate_ffn.py \
  --model /home/Hongjie_Zeng/pbit_llm/models/LLaMA-7B \
  --train /home/Hongjie_Zeng/pbit_llm/data/wikitext-2/wiki.train.raw \
  --output calibration_ffn.json --tokens 8192 --context 2048
