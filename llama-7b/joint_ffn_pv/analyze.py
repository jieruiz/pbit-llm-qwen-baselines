import hashlib
import json
import math
import statistics
from pathlib import Path

import numpy as np


ROOT = Path(__file__).absolute().parent
RESULTS = ROOT / "results" / "formal"
BASELINE = ROOT.parent / "llama7b_baseline_20261014" / "results"
VARIANTS = {
    "original_dense": 1,
    "attention_pv": 3,
    "ffn_dense": 3,
    "joint_pv": 3,
}


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def load_results():
    data = {}
    for variant, seeds in VARIANTS.items():
        rows = []
        for seed in range(seeds):
            path = RESULTS / f"{variant}_seed{seed}.json"
            if not path.is_file():
                raise FileNotFoundError(path)
            rows.append(read(path))
        data[variant] = rows
    return data


def aggregate_ffn(rows):
    result = {}
    for signal in ("x", "g", "u"):
        totals = {key: 0.0 for key in ("count", "out_of_fullscale", "above_top_code", "encoded_zero", "squared_error", "signal_energy")}
        for row in rows:
            for layer in row["metadata"]["ffn_diagnostics"].values():
                source = layer["signals"][signal]
                for key in totals:
                    totals[key] += source[key]
        result[signal] = {
            "count": int(totals["count"]),
            "out_of_fullscale_fraction": totals["out_of_fullscale"] / totals["count"],
            "above_top_code_fraction": totals["above_top_code"] / totals["count"],
            "encoded_zero_fraction": totals["encoded_zero"] / totals["count"],
            "quantization_nmse": totals["squared_error"] / totals["signal_energy"],
        }
    return result


def bootstrap_ratio(left, right, repeats=5000, seed=20261014):
    left_windows = np.mean(
        [[window["nll_sum"] for window in row["result"]["windows"]] for row in left],
        axis=0,
    )
    right_windows = np.mean(
        [[window["nll_sum"] for window in row["result"]["windows"]] for row in right],
        axis=0,
    )
    tokens = np.asarray([window["scored"] for window in left[0]["result"]["windows"]])
    rng = np.random.default_rng(seed)
    ratios = np.empty(repeats)
    for index in range(repeats):
        chosen = rng.integers(0, len(tokens), size=len(tokens))
        denominator = tokens[chosen].sum()
        ratios[index] = math.exp(
            (left_windows[chosen].sum() - right_windows[chosen].sum()) / denominator
        ) - 1.0
    observed = math.exp((left_windows.sum() - right_windows.sum()) / tokens.sum()) - 1.0
    low, high = np.quantile(ratios, [0.025, 0.975])
    return {"relative": observed, "bootstrap95": [float(low), float(high)]}


def bootstrap_interaction(joint, ffn, attention, dense, repeats=5000, seed=20261015):
    def averaged_windows(rows):
        return np.mean(
            [[window["nll_sum"] for window in row["result"]["windows"]] for row in rows],
            axis=0,
        )

    terms = (
        averaged_windows(joint)
        - averaged_windows(ffn)
        - averaged_windows(attention)
        + averaged_windows(dense)
    )
    tokens = np.asarray([window["scored"] for window in dense[0]["result"]["windows"]])
    rng = np.random.default_rng(seed)
    ratios = np.empty(repeats)
    for index in range(repeats):
        chosen = rng.integers(0, len(tokens), size=len(tokens))
        ratios[index] = math.exp(terms[chosen].sum() / tokens[chosen].sum()) - 1.0
    low, high = np.quantile(ratios, [0.025, 0.975])
    return {
        "relative": math.exp(terms.sum() / tokens.sum()) - 1.0,
        "bootstrap95": [float(low), float(high)],
    }


def main():
    data = load_results()
    reference = data["original_dense"][0]
    for variant, rows in data.items():
        for row in rows:
            assert row["metadata"]["status"] == "complete"
            assert row["result"]["scored_tokens"] == 335686
            assert row["result"]["corpus_sha256"] == reference["result"]["corpus_sha256"]
            assert len(row["result"]["windows"]) == 327
            assert row["metadata"]["protected_sha256"] == reference["metadata"]["protected_sha256"]

    summary = {"variants": {}, "comparisons": {}}
    for variant, rows in data.items():
        ppls = [row["result"]["perplexity"] for row in rows]
        seconds = [row["result"]["seconds"] for row in rows]
        summary["variants"][variant] = {
            "seeds": len(rows),
            "ppl": ppls,
            "ppl_mean": statistics.mean(ppls),
            "ppl_stdev": statistics.stdev(ppls) if len(ppls) > 1 else 0.0,
            "seconds_mean": statistics.mean(seconds),
            "peak_cuda_gib_max": max(row["metadata"]["peak_cuda_bytes"] for row in rows) / 2**30,
        }
        if variant in ("ffn_dense", "joint_pv"):
            summary["variants"][variant]["ffn_diagnostics"] = aggregate_ffn(rows)
        if variant in ("attention_pv", "joint_pv"):
            controls = [
                control
                for row in rows
                for control in row["metadata"]["attention_diagnostics"]
            ]
            summary["variants"][variant]["attention_diagnostics"] = {
                "unique_v_fraction_mean": statistics.mean(control["unique_v_fraction"] for control in controls),
                "kv_group_union_v_fraction_mean": statistics.mean(control["kv_group_union_v_fraction"] for control in controls),
            }

    pairs = {
        "attention_pv_vs_dense": ("attention_pv", "original_dense"),
        "ffn_dense_vs_dense": ("ffn_dense", "original_dense"),
        "joint_pv_vs_dense": ("joint_pv", "original_dense"),
        "joint_pv_vs_ffn_dense": ("joint_pv", "ffn_dense"),
    }
    for name, (left, right) in pairs.items():
        summary["comparisons"][name] = bootstrap_ratio(data[left], data[right])
    summary["comparisons"]["multiplicative_interaction"] = bootstrap_interaction(
        data["joint_pv"],
        data["ffn_dense"],
        data["attention_pv"],
        data["original_dense"],
    )

    eager = read(BASELINE / "wikitext2_original_eager.json")
    summary["eager_reference_ppl"] = eager["result"]["perplexity"]
    summary["dense_vs_eager_relative"] = (
        summary["variants"]["original_dense"]["ppl_mean"]
        / summary["eager_reference_ppl"]
        - 1.0
    )
    summary["calibration_sha256"] = hashlib.sha256(
        (ROOT / "calibration_ffn.json").read_bytes()
    ).hexdigest()
    summary["analysis_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()

    output = ROOT / "results" / "summary.json"
    output.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
