"""Compact progress snapshot; only complete log lines are parsed."""
import json
from pathlib import Path
import re
import sys

root=Path("results/multithreshold_and_twenty_layer_20261002")


def records(path):
    result=[]
    if path.exists():
        for line in path.read_text().splitlines():
            try:
                result.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return result


if __name__=="__main__":
    if "--compact" in sys.argv:
        scores=[]
        pending={}
        for stage in ("independent","joint"):
            for n,seed in ((0,0),(1,0),(2,0),(4,0),(4,1),(4,2),(16,0)):
                result=root/"evaluation"/stage/f"n{n}_seed{seed}.json"
                if result.exists():
                    r=json.loads(result.read_text())
                    scores.append({"stage":stage,"n":n,"seed":seed,"ppl":r["perplexity"]})
                else:
                    log=root/"logs"/f"eval_{stage}_n{n}_seed{seed}.log"
                    text=log.read_text() if log.exists() else ""
                    windows=re.findall(r"windows=(\d+)",text)
                    pending[f"{stage}_n{n}_seed{seed}"]=int(windows[-1]) if windows else 0
        print(json.dumps({"status":json.loads((root/"status.json").read_text())["stage"],
                          "scores":scores,"pending_windows_of_292":pending}))
        sys.exit(0)
    print((root/"status.json").read_text())
    finished=[]
    active=[]
    for directory in sorted((root/"individual").glob("layer*")):
        rows=records(directory/"train_metrics.jsonl")
        if (directory/"summary.json").exists():
            finished.append(int(directory.name[5:]))
        elif rows:
            step=next((r for r in reversed(rows) if r["event"]=="train"),{})
            active.append({"layer":int(directory.name[5:]),"step":step.get("step",0),
                "phase":step.get("phase","calibration"),"elapsed":step.get("elapsed_seconds",0)})
    print(json.dumps({"finished_locals":sorted(finished),"active":active}))
    joint=records(root/"joint"/"train_metrics.jsonl")
    if joint:
        last=next((r for r in reversed(joint) if r["event"]=="train"),{})
        val=[r for r in joint if r["event"]=="validation"]
        print(json.dumps({"joint":last,"last_validation":val[-1] if val else None}))
    scores=[]
    for path in sorted((root/"evaluation").glob("*/*.json")):
        data=json.loads(path.read_text())
        scores.append({"stage":path.parent.name,"n":data["sample_count"],"seed":data["seed"],"ppl":data["perplexity"]})
    print(json.dumps({"evaluations":scores}))
