#!/bin/bash
#SBATCH --job-name=llama7b-pv
#SBATCH --partition=gpu5090
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=08:00:00
#SBATCH --chdir=/home/Hongjie_Zeng/llama7b_joint_pv_20261014
#SBATCH --output=results/logs/%x-%j.log

set -euo pipefail
: "${VARIANT:?VARIANT is required}"
SEED=${SEED:-${SLURM_ARRAY_TASK_ID:-0}}
SCOPE=${SCOPE:-full}
SAMPLES=${SAMPLES:-512}
mkdir -p "results/$SCOPE" results/logs
SMOKE=()
if [[ "$SCOPE" == smoke ]]; then SMOKE=(--smoke); fi
export TOKENIZERS_PARALLELISM=false HF_HUB_OFFLINE=1 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
exec /opt/slurm/21.08.8/bin/srun /home/Hongjie_Zeng/.conda/envs/pbit_llm/bin/python -u evaluate.py \
  --variant "$VARIANT" --seed "$SEED" --samples "$SAMPLES" \
  --model /home/Hongjie_Zeng/pbit_llm/models/LLaMA-7B \
  --data /home/Hongjie_Zeng/pbit_llm/data/wikitext-2/wiki.test.raw \
  --output "results/$SCOPE/${VARIANT}_seed${SEED}.json" "${SMOKE[@]}"
