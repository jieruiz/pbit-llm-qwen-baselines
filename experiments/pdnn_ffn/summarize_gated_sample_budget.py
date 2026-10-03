import argparse
import hashlib
import json
from pathlib import Path
import statistics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("directory", type=Path)
    parser.add_argument("--recover-completed", action="store_true", help="Verify all artifacts and recover completion after an SSH stdout failure")
    args = parser.parse_args()
    root = args.directory
    if not args.recover_completed and json.loads((root / "status.json").read_text())["status"] != "complete":
        raise RuntimeError("experiment has not completed")
    protocol = json.loads((root / "protocol.json").read_text())
    rows = []
    resources = []
    recovery = {}
    for folder in sorted(path for path in root.iterdir() if path.is_dir()):
        with (folder / "train_metrics.jsonl").open() as stream:
            header = json.loads(next(stream))
        architecture = header["args"]["architecture"]
        status = json.loads((folder / "status.json").read_text())
        if status["status"] != "complete":
            if not args.recover_completed or "Broken pipe" not in status.get("error", ""):
                raise RuntimeError(f"unfinished job: {folder}")
            recovery[folder.name] = status
        if header["args"]["initial_checkpoint_sha256"] != protocol["source_sha256"][architecture]:
            raise ValueError("checkpoint source hash mismatch")
        final = json.loads((folder / "summary.json").read_text())
        if final["event"] != "complete" or final["tokens_processed"] != protocol["args"]["steps"] * 4 * 256:
            raise ValueError("training summary is incomplete")
        resources.append({
            "run": folder.name, "training_seconds": final["elapsed_seconds"],
            "peak_allocated_bytes": final["peak_allocated_bytes"],
            "tokens_processed": final["tokens_processed"],
        })
        hashes = json.loads((folder / "checkpoint_hashes.json").read_text())
        for step in sorted({2000, protocol["args"]["steps"]}):
            row = {
                "run": folder.name, "architecture": architecture,
                "train_samples": header["args"]["train_samples"], "sample_training_steps": step,
                "checkpoint_sha256": hashes[f"student_step{step}.pt"],
                "moments": json.loads((folder / f"step{step}_moments.json").read_text())["normalized"],
            }
            row["mean_field_ppl"] = json.loads((folder / f"step{step}_samples0_seed0_ppl.json").read_text())["perplexity"]
            if args.recover_completed:
                checkpoint = folder / f"student_step{step}.pt"
                if hashlib.sha256(checkpoint.read_bytes()).hexdigest() != row["checkpoint_sha256"]:
                    raise ValueError("saved checkpoint checksum mismatch")
            for count in (4, 16):
                evaluations = [json.loads((folder / f"step{step}_samples{count}_seed{seed}_ppl.json").read_text()) for seed in range(3)]
                if any(value["scored_tokens"] != 299077 for value in evaluations):
                    raise ValueError("unexpected full-test scoring count")
                ppls = [value["perplexity"] for value in evaluations]
                row[f"n{count}_ppl_mean"] = statistics.mean(ppls)
                row[f"n{count}_ppl_population_std"] = statistics.pstdev(ppls)
            rows.append(row)
        # Warm diagnostic must also be present and valid before accepting completion.
        warm = json.loads((folder / "warm_moments.json").read_text())
        if warm["architecture"] != architecture or warm["args"]["paths"] != 256:
            raise ValueError("invalid warm-start diagnostic")
    if len(rows) != 8 or len(resources) != 4:
        raise ValueError("expected four arms with two evaluation checkpoints each")
    source_files = [
        "full_path_pdnn_ffn.py", "train_full_path_distillation.py", "evaluate_full_path_perplexity.py",
        "measure_path_moments.py", "test_path_moments.py", "run_gated_sample_budget.py", "summarize_gated_sample_budget.py",
    ]
    source_hashes = {name: hashlib.sha256((Path(__file__).parent / name).read_bytes()).hexdigest() for name in source_files}
    result = {"rows": rows, "training_resources": resources, "source_sha256": source_hashes, "warm_moments": {
        architecture: json.loads((root / f"{architecture}_train4" / "warm_moments.json").read_text())["normalized"]
        for architecture in ("serial", "gated_dual_rail")
    }}
    (root / "comparison.json").write_text(json.dumps(result, indent=2) + "\n")
    if args.recover_completed:
        (root / "completion_recovery.json").write_text(json.dumps({
            "prior_statuses": recovery,
            "verified": "Four completed training summaries; eight checkpoints match stored SHA256; eight mean-field and 48 sampled PPL files; 12 moment files loaded.",
            "reason": "Training and evaluation completed; final stdout writes failed after SSH disconnected. No training or evaluation was rerun.",
        }, indent=2) + "\n")
        for name in recovery:
            (root / name / "status.json").write_text(json.dumps({"status": "complete", "recovered_after_stdout_broken_pipe": True}) + "\n")
        (root / "status.json").write_text(json.dumps({"status": "complete", "completion_recovered": True}) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
