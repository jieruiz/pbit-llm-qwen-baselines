#!/usr/bin/env bash
set -euo pipefail

DEST=/home/Hongjie_Zeng/pbit_llm/models/LLaMA-7B
HASH1=d43476fdd2fca0c44d55ee930039dd5dafb6331764dc0b5e5f89c60b551fcc12
HASH2=94712c4b457815e0f8aaa4f78170aebba2b9ee3339ff2b40fceb983338fbafee
OLD1="$DEST/pytorch_model-00001-of-00002.bin"
OLD2="$DEST/pytorch_model-00002-of-00002.bin"
NEW1="$DEST/model-00001-of-00002.safetensors"
NEW2="$DEST/model-00002-of-00002.safetensors"

printf '%s  %s\n%s  %s\n' "$HASH1" "$OLD1" "$HASH2" "$OLD2" | sha256sum --check --strict
mv -- "$OLD1" "$NEW1"
mv -- "$OLD2" "$NEW2"
printf '%s  %s\n%s  %s\n' "$HASH1" "$NEW1" "$HASH2" "$NEW2" | sha256sum --check --strict

/home/Hongjie_Zeng/.conda/envs/pbit_llm/bin/python - "$NEW1" "$NEW2" <<'PY'
import sys
from safetensors import safe_open

for path in sys.argv[1:]:
    with safe_open(path, framework="pt", device="cpu") as f:
        keys = list(f.keys())
    print(path, "tensors=", len(keys), "first=", keys[0], "last=", keys[-1])
PY
