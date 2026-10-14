#!/usr/bin/env bash
set -euo pipefail

DEST=/home/Hongjie_Zeng/pbit_llm/models/LLaMA-7B
PART1="$DEST/.cache/huggingface/download/aoe4E07IMh7reFyUkVoVk040mQk=.d43476fdd2fca0c44d55ee930039dd5dafb6331764dc0b5e5f89c60b551fcc12.incomplete"
PART2="$DEST/.cache/huggingface/download/Dr_lZJDwE1cnGAQMwA77jJEQIk8=.94712c4b457815e0f8aaa4f78170aebba2b9ee3339ff2b40fceb983338fbafee.incomplete"
HASH1=d43476fdd2fca0c44d55ee930039dd5dafb6331764dc0b5e5f89c60b551fcc12
HASH2=94712c4b457815e0f8aaa4f78170aebba2b9ee3339ff2b40fceb983338fbafee
SIZE1=9976579217
SIZE2=3500297444
TARGET1="$DEST/model-00001-of-00002.safetensors"
TARGET2="$DEST/model-00002-of-00002.safetensors"
TMP1="$TARGET1.recovering"
TMP2="$TARGET2.recovering"
BAD_HASH1=8e9a05c322197e42373248bd9ede7948e741afcff82d3cac9070f79daa42a451
BAD_HASH2=f743df358e7ee94ce60bb87fc0c97ed54fce2ace87d89565c3788de6cd4e986a

prefix_hash() {
    local size=$1
    local path=$2
    head -c "$size" "$path" | sha256sum | awk '{print $1}'
}

file_hash() {
    local raw
    raw=$(sha256sum "$1" | awk '{print $1}')
    printf '%s' "${raw#\\}"
}

for path in "$PART1" "$PART2"; do
    [[ -f "$path" ]] || { echo "Missing source: $path" >&2; exit 1; }
done

if [[ -f "$TARGET1" && -f "$TARGET2" ]] \
    && [[ "$(file_hash "$TARGET1")" == "$HASH1" ]] \
    && [[ "$(file_hash "$TARGET2")" == "$HASH2" ]]; then
    echo '[recovered files already valid]'
else
    echo '[source prefix verification]'
    actual1=$(prefix_hash "$SIZE1" "$PART1")
    actual2=$(prefix_hash "$SIZE2" "$PART2")
    echo "$actual1  shard-1-prefix"
    echo "$actual2  shard-2-prefix"
    [[ "$actual1" == "$HASH1" ]] || { echo 'Shard 1 prefix mismatch' >&2; exit 1; }
    [[ "$actual2" == "$HASH2" ]] || { echo 'Shard 2 prefix mismatch' >&2; exit 1; }

    rm -f -- "$TMP1" "$TMP2"
    head -c "$SIZE1" "$PART1" > "$TMP1"
    head -c "$SIZE2" "$PART2" > "$TMP2"

    echo '[recovered-file verification]'
    printf '%s  %s\n%s  %s\n' "$HASH1" "$TMP1" "$HASH2" "$TMP2" | sha256sum --check --strict
    mv -f -- "$TMP1" "$TARGET1"
    mv -f -- "$TMP2" "$TARGET2"
fi

echo '[removing only known-bad generated duplicates]'
while IFS= read -r -d '' path; do
    [[ "$path" == "$TARGET1" || "$path" == "$TARGET2" ]] && continue
    size=$(stat --printf='%s' "$path")
    [[ "$size" == "$SIZE1" || "$size" == "$SIZE2" ]] || continue
    hash=$(file_hash "$path")
    if [[ "$hash" == "$BAD_HASH1" || "$hash" == "$BAD_HASH2" ]]; then
        printf 'remove %q\n' "$path"
        rm -- "$path"
    else
        echo "Refusing to remove unrecognized large file: $path ($hash)" >&2
        exit 1
    fi
done < <(find "$DEST" -maxdepth 1 -type f -print0)

echo '[removing verified redundant extended cache files]'
rm -- "$PART1" "$PART2"

echo '[final verification]'
printf '%s  %s\n%s  %s\n' "$HASH1" "$TARGET1" "$HASH2" "$TARGET2" | sha256sum --check --strict
stat --printf='%n %s bytes\n' "$TARGET1" "$TARGET2"
