#!/bin/bash
#SBATCH --job-name=llama7b-eager
#SBATCH --partition=gpu5090
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=08:00:00
#SBATCH --chdir=/home/Hongjie_Zeng/llama7b_baseline_20261014
#SBATCH --output=results/logs/slurm-%j.log

set -euo pipefail

PYTHON=/home/Hongjie_Zeng/.conda/envs/pbit_llm/bin/python
MODEL_PATH=${MODEL_PATH:-/home/Hongjie_Zeng/pbit_llm/models/LLaMA-7B}
DATA_PATH=/home/Hongjie_Zeng/pbit_llm/data/wikitext-2/wiki.test.raw
OUTPUT_PATH=results/wikitext2_original_eager.json

test -x "$PYTHON"
test -f "$MODEL_PATH/config.json"
test -f "$DATA_PATH"
mkdir -p results/logs

export TOKENIZERS_PARALLELISM=false
export HF_HUB_OFFLINE=1
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4

exec /opt/slurm/21.08.8/bin/srun "$PYTHON" -u baseline.py \
  --model "$MODEL_PATH" \
  --data "$DATA_PATH" \
  --output "$OUTPUT_PATH" \
  --context 2048 \
  --stride 1024 \
  --attn-implementation eager
