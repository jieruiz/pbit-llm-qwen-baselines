"""Print a compact progress snapshot without loading model weights."""
import json
from pathlib import Path

root = Path("results/continuous_raw_ten_layer_20261001")
if (root / "status.json").exists():
    print((root / "status.json").read_text().strip())
for mode in ("continuous_raw", "sigmoid"):
    complete = sorted(int(path.parent.name[5:]) for path in (root / "individual" / mode).glob("layer*/summary.json"))
    print(mode, "local_complete", complete)
    singles = []
    for path in (root / "evaluation" / "single_layer" / mode).glob("layer*/samples4_seed0.json"):
        data = json.loads(path.read_text())
        singles.append((data["layer"], round(data["perplexity"], 6)))
    if singles:
        print("single_N4", sorted(singles))
    for path in sorted((root / "individual" / mode).glob("layer*/train_metrics.jsonl")):
        if (path.parent / "summary.json").exists():
            continue
        lines = path.read_text().splitlines()
        if lines:
            try:
                record = json.loads(lines[-1])
                print(path.parent.name, record.get("event"), record.get("global_step"), record.get("phase"))
            except json.JSONDecodeError:
                pass
    joint = root / "joint" / mode / "train_metrics.jsonl"
    if joint.exists():
        lines = joint.read_text().splitlines()
        try:
            record = json.loads(lines[-1])
            print("joint", mode, {key: record[key] for key in ("event", "step", "elapsed_seconds", "perplexity", "best_step") if key in record})
        except (json.JSONDecodeError, IndexError):
            pass
    for stage in ("independent", "joint"):
        rows = []
        for path in sorted((root / "evaluation" / stage / mode).glob("*.json")):
            data = json.loads(path.read_text())
            rows.append((data["sample_count"], data["seed"], round(data["perplexity"], 6)))
        if rows:
            print(mode, stage, rows)
