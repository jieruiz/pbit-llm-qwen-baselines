#!/usr/bin/env bash
set -euo pipefail

DEST=/home/Hongjie_Zeng/pbit_llm/models/LLaMA-7B
PYTHON=/home/Hongjie_Zeng/.conda/envs/pbit_llm/bin/python

echo '[first 32 bytes]'
od -An -tx1 -N32 "$DEST/pytorch_model-00001-of-00002.bin"

echo '[index shard names]'
"$PYTHON" - "$DEST" <<'PY'
import json
import pathlib
import sys

root = pathlib.Path(sys.argv[1])
for name in ("model.safetensors.index.json", "pytorch_model.bin.index.json"):
    with (root / name).open(encoding="utf-8") as f:
        data = json.load(f)
    print(name, sorted(set(data["weight_map"].values())))

try:
    import safetensors
    print("safetensors", safetensors.__version__)
except Exception as exc:
    print("safetensors unavailable", repr(exc))
PY
