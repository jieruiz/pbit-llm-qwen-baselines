#!/usr/bin/env bash
set -euo pipefail

DEST=/home/Hongjie_Zeng/pbit_llm/models/LLaMA-7B
PYTHON=/home/Hongjie_Zeng/.conda/envs/pbit_llm/bin/python
HASH1=d43476fdd2fca0c44d55ee930039dd5dafb6331764dc0b5e5f89c60b551fcc12
HASH2=94712c4b457815e0f8aaa4f78170aebba2b9ee3339ff2b40fceb983338fbafee
SIZE1=9976579217
SIZE2=3500297444
TARGET1="$DEST/pytorch_model-00001-of-00002.bin"
TARGET2="$DEST/pytorch_model-00002-of-00002.bin"

declare -a matches1=()
declare -a matches2=()

while IFS= read -r -d '' path; do
    size=$(stat --printf='%s' "$path")
    if [[ "$size" != "$SIZE1" && "$size" != "$SIZE2" ]]; then
        continue
    fi
    hash=$(sha256sum "$path" | awk '{print $1}')
    printf '%s  ' "$hash"
    printf '%q\n' "$path"
    if [[ "$size" == "$SIZE1" && "$hash" == "$HASH1" ]]; then
        matches1+=("$path")
    elif [[ "$size" == "$SIZE2" && "$hash" == "$HASH2" ]]; then
        matches2+=("$path")
    else
        echo "Unexpected hash for weight-sized file: $path" >&2
        exit 1
    fi
done < <(find "$DEST" -maxdepth 1 -type f -print0)

if [[ ${#matches1[@]} -ne 1 ]]; then
    echo "Expected exactly one shard-1 match; found ${#matches1[@]}" >&2
    exit 1
fi
if [[ ${#matches2[@]} -lt 1 ]]; then
    echo 'No shard-2 match found' >&2
    exit 1
fi

if [[ "${matches1[0]}" != "$TARGET1" ]]; then
    mv -- "${matches1[0]}" "$TARGET1"
fi
if [[ "${matches2[0]}" != "$TARGET2" ]]; then
    mv -- "${matches2[0]}" "$TARGET2"
fi

for path in "${matches2[@]:1}"; do
    if [[ "$path" != "$TARGET2" ]]; then
        rm -- "$path"
    fi
done

echo '[final hashes]'
printf '%s  %s\n%s  %s\n' "$HASH1" "$TARGET1" "$HASH2" "$TARGET2" | sha256sum --check --strict

echo '[transformers metadata check]'
"$PYTHON" - "$DEST/pytorch_model.bin.index.json" "$TARGET1" "$TARGET2" <<'PY'
import json
import os
import sys

index_path, shard1, shard2 = sys.argv[1:]
with open(index_path, encoding="utf-8") as f:
    index = json.load(f)
expected = {os.path.basename(shard1), os.path.basename(shard2)}
actual = set(index["weight_map"].values())
if actual != expected:
    raise SystemExit(f"index shards mismatch: expected={expected}, actual={actual}")
print(f"index tensors: {len(index['weight_map'])}")
print(f"index total_size: {index.get('metadata', {}).get('total_size')}")
PY

echo '[final directory]'
while IFS= read -r -d '' path; do
    stat --printf='%s\t' "$path"
    printf '%q\n' "$path"
done < <(find "$DEST" -maxdepth 1 -type f -print0)
