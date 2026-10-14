#!/usr/bin/env bash
set -euo pipefail

CACHE=/home/Hongjie_Zeng/.cache/huggingface/hub/models--huggyllama--llama-7b
DEST=/home/Hongjie_Zeng/pbit_llm/models/LLaMA-7B

echo '[cache candidates]'
while IFS= read -r -d '' path; do
    stat --printf='%i\t%s\t' "$path"
    printf '%q\n' "$path"
done < <(find "$CACHE" -type f -print0)

echo '[destination files]'
while IFS= read -r -d '' path; do
    stat --printf='%i\t%s\t' "$path"
    printf '%q\n' "$path"
done < <(find "$DEST" -maxdepth 1 -type f -print0)

echo '[large-file hashes]'
while IFS= read -r -d '' path; do
    size=$(stat --printf='%s' "$path")
    if [[ "$size" -ge 1000000000 ]]; then
        sha256sum "$path"
    fi
done < <(find "$DEST" -maxdepth 1 -type f -print0)

echo '[index shard names]'
/home/Hongjie_Zeng/.conda/envs/pbit_llm/bin/python - "$DEST/pytorch_model.bin.index.json" <<'PY'
import json
import sys

with open(sys.argv[1], encoding="utf-8") as f:
    index = json.load(f)
print("\n".join(sorted(set(index["weight_map"].values()))))
PY
