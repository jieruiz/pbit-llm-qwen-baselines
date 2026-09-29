#!/bin/bash
#SBATCH --job-name=cwc
#SBATCH --partition=gpu5090
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=24G
#SBATCH --time=04:00:00
#SBATCH --output=/home/Weican_Chen/projects/pbit-qwen-calibration-20260929/logs/run-%j.out
#SBATCH --error=/home/Weican_Chen/projects/pbit-qwen-calibration-20260929/logs/run-%j.err
set -euo pipefail
ROOT=/home/Weican_Chen/projects/pbit-qwen-calibration-20260929
ASSETS=/home/Weican_Chen/projects/pbit-qwen-sigmoid-20260929
cd "$ROOT"
finalize() {
  code=$?
  trap - EXIT
  python collect.py "$code"
  exit "$code"
}
trap finalize EXIT
export PATH="$ASSETS/.venv/bin:$PATH"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export OMP_NUM_THREADS=6 MKL_NUM_THREADS=6 PYTHONUNBUFFERED=1
[ "$(id -un)" = Weican_Chen ]
[ "$(realpath "$ROOT")" = "$ROOT" ]
python -c 'import os,torch,transformers; assert os.environ.get("SLURM_JOB_ID"); assert torch.cuda.device_count()==1; assert torch.__version__=="2.7.1+cu128"; assert transformers.__version__=="4.45.2"; print("GPU",torch.cuda.get_device_name(0))'
cd "$ASSETS"
sha256sum --check <<'HASHES'
88c142557820ccad55bb59756bfcfcf891de9cc6202816bd346445188a0ed342  models/Qwen2.5-0.5B/model.safetensors
6707892fa3788b5ab9ed78ab5ff37d9fe825f6011a2ad4fcd6a6d467f0e7da57  data/wikitext-2/wiki.train.raw
173c87a53759e0201f33e0ccf978e510c2042d7f2cb78229d9a50d79b9e7dd08  data/wikitext-2/wiki.test.raw
HASHES
cd "$ROOT"
python code/baselines/perplexity.py --model "$ASSETS/models/Qwen2.5-0.5B" --text-file "$ASSETS/data/wikitext-2/wiki.test.raw" --output "$ROOT/baseline.json"
python code/experiments/pdnn_ffn/run_calibration_ablation.py --model "$ASSETS/models/Qwen2.5-0.5B" --train-text "$ASSETS/data/wikitext-2/wiki.train.raw" --test-text "$ASSETS/data/wikitext-2/wiki.test.raw" --four-layer-checkpoints /home/Weican_Chen/projects/pbit-qwen-joint-ablation-20260929/results/reference_1000 --baseline "$ROOT/baseline.json" --output-dir "$ROOT/results"
