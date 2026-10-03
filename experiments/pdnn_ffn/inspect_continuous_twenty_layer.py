"""Compact read-only progress snapshot for the twenty-layer input ablation."""
import json
from pathlib import Path

root = Path("results/continuous_raw_twenty_layer_20261001")
if (root / "status.json").exists():
    print((root / "status.json").read_text().strip())
for mode in ("continuous_raw", "sigmoid"):
    complete = sorted(int(path.parent.name[5:]) for path in (root / "individual" / mode).glob("layer*/summary.json"))
    print(mode, "new_local_complete", complete)
    singles = []
    for path in (root / "evaluation" / "single_layer" / mode).glob("layer*/samples4_seed0.json"):
        data = json.loads(path.read_text())
        if data["layer"] not in (7,8,9,10,11,12,13,14,15,18):
            singles.append((data["layer"], round(data["perplexity"], 6)))
    if singles:
        print("new_single_N4", sorted(singles))
    for path in sorted((root / "individual" / mode).glob("layer*/train_metrics.jsonl")):
        if (path.parent / "summary.json").exists():
            continue
        try:
            record = json.loads(path.read_text().splitlines()[-1])
            print(path.parent.name, record.get("event"), record.get("global_step"), record.get("phase"))
        except (json.JSONDecodeError, IndexError):
            pass
    path = root / "joint" / mode / "train_metrics.jsonl"
    if path.exists():
        records = []
        for line in path.read_text().splitlines():
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                pass
        if records:
            print("joint", mode, {k: records[-1][k] for k in ("event", "step", "elapsed_seconds", "perplexity", "best_step") if k in records[-1]})
            validations = [r for r in records if r.get("event") == "validation"]
            if validations:
                print("latest_validation", validations[-1]["step"], round(validations[-1]["perplexity"], 5))
    for stage in ("independent", "staged", "joint"):
        rows = []
        for path in sorted((root / "evaluation" / stage / mode).glob("*.json")):
            data = json.loads(path.read_text())
            rows.append((data["sample_count"], data["seed"], round(data["perplexity"], 6)))
        if rows:
            print(mode, stage, rows)
