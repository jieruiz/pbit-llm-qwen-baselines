#!/usr/bin/env bash
set -euo pipefail

echo '[filesystem]'
df -h /home/Hongjie_Zeng

echo '[all files >= 3 GB under home]'
while IFS= read -r -d '' path; do
    stat --printf='%i\t%s\t' "$path"
    printf '%q\n' "$path"
done < <(find /home/Hongjie_Zeng -xdev -type f -size +3G -print0 2>/dev/null)
