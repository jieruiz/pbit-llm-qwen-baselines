#!/bin/bash
#SBATCH --job-name=pbit-dram-check
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=00:20:00
#SBATCH --output=smoke-%j.log
set -euo pipefail
root=${SLURM_SUBMIT_DIR}
python3 -m unittest discover -s "$root/architecture_tlm/checks" -p 'test_*.py'
python3 "$root/architecture_tlm/dramsys/smoke.py" --bridge "$root/build/pbit_dram_bridge"
