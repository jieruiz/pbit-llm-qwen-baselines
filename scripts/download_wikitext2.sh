#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="${PBIT_LLM_ROOT:-$(cd "$SCRIPT_DIR/.." && pwd)}"
TARGET="${WIKITEXT2_PATH:-$ROOT/data/wikitext-2/wiki.test.raw}"
URL="https://cosmo.zip/pub/datasets/wikitext-2-raw/wiki.test.raw"
EXPECTED_SHA256="173c87a53759e0201f33e0ccf978e510c2042d7f2cb78229d9a50d79b9e7dd08"

mkdir -p "$(dirname "$TARGET")"
python - "$URL" "$TARGET" "$EXPECTED_SHA256" <<'PY'
import hashlib
import sys
import urllib.request
from pathlib import Path

url, target, expected = sys.argv[1:]
path = Path(target)
urllib.request.urlretrieve(url, path)
actual = hashlib.sha256(path.read_bytes()).hexdigest()
if actual != expected:
    path.unlink(missing_ok=True)
    raise SystemExit(f"SHA-256 mismatch: expected {expected}, got {actual}")
print(f"WikiText-2 raw test written to {path} ({actual})")
PY

