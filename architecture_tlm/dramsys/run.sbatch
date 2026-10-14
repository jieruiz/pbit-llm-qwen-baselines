#!/bin/bash
#SBATCH --job-name=pbit-dram-decode
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=02:00:00
#SBATCH --array=0-11%12
#SBATCH --output=decode-%A_%a.log
set -euo pipefail
root=${SLURM_SUBMIT_DIR}
python3 "$root/architecture_tlm/run_dramsys.py" \
 --bridge "$root/build/pbit_dram_bridge" --index "$SLURM_ARRAY_TASK_ID"
