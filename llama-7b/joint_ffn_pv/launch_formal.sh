#!/usr/bin/env bash
set -euo pipefail

cd /home/Hongjie_Zeng/llama7b_joint_pv_20261014
SBATCH=/opt/slurm/21.08.8/bin/sbatch

"$SBATCH" --job-name=l7b-dense --export=ALL,VARIANT=original_dense,SCOPE=formal evaluate.sbatch
"$SBATCH" --job-name=l7b-attnpv --array=0-2 --export=ALL,VARIANT=attention_pv,SCOPE=formal evaluate.sbatch
"$SBATCH" --job-name=l7b-ffn --array=0-2 --export=ALL,VARIANT=ffn_dense,SCOPE=formal evaluate.sbatch
"$SBATCH" --job-name=l7b-joint --array=0-2 --export=ALL,VARIANT=joint_pv,SCOPE=formal evaluate.sbatch
