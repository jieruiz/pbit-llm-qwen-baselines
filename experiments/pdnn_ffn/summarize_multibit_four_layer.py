"""Summarize the four-layer stochastic K=4 input experiment."""
import json
import math
from pathlib import Path
import statistics


ROOT = Path("results/input_multibit_four_layer_9_12_15_18_20260930")
BASELINE = Path("results/gated_four_layer_9_12_15_18_20260930/comparison.json")
BASE_PPL = 11.652735047455899
LAYERS = (9, 12, 15, 18)


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def evaluation(stage):
    folder = ROOT / "evaluation" / stage
    n4 = [read(folder / f"samples4_seed{seed}_ppl.json")["perplexity"] for seed in (0, 1, 2)]
    return {
        "mean_field": read(folder / "samples0_seed0_ppl.json")["perplexity"],
        "n4_mean": statistics.mean(n4),
        "n4_population_std": statistics.pstdev(n4),
        "n4_values": n4,
        "n16": read(folder / "samples16_seed0_ppl.json")["perplexity"],
    }


def main():
    independent = evaluation("independent")
    joint = evaluation("joint")
    old_rows = read(BASELINE)["rows"]
    old = {row["stage"]: row for row in old_rows if row["architecture"] == "serial"}
    single_layers = {}
    for layer in LAYERS:
        folder = ROOT / "evaluation" / "single_layer" / f"layer{layer}"
        values = [read(folder / f"samples4_seed{seed}_ppl.json")["perplexity"] for seed in (0, 1, 2)]
        single_layers[str(layer)] = {
            "mean_field": read(folder / "samples0_seed0_ppl.json")["perplexity"],
            "n4_mean": statistics.mean(values),
            "n4_population_std": statistics.pstdev(values),
            "n16": read(folder / "samples16_seed0_ppl.json")["perplexity"],
            "path_moments_normalized": read(ROOT / "moments" / f"layer{layer}.json")["normalized"],
        }
    joint_training = read(ROOT / "joint" / "summary.json")
    comparisons = {
        "independent_n4_absolute_gain_over_sigmoid": old["independent"]["n4_ppl_mean"] - independent["n4_mean"],
        "independent_n4_relative_gain_percent": 100 * (old["independent"]["n4_ppl_mean"] - independent["n4_mean"]) / old["independent"]["n4_ppl_mean"],
        "joint_n4_absolute_gain_over_sigmoid": old["joint"]["n4_ppl_mean"] - joint["n4_mean"],
        "joint_n4_relative_gain_percent": 100 * (old["joint"]["n4_ppl_mean"] - joint["n4_mean"]) / old["joint"]["n4_ppl_mean"],
        "joint_n4_gap_above_original_percent": 100 * (joint["n4_mean"] / BASE_PPL - 1),
        "k4_joint_recovery_of_independent_excess_nll_percent": 100 * (math.log(independent["n4_mean"]) - math.log(joint["n4_mean"])) / (math.log(independent["n4_mean"]) - math.log(BASE_PPL)),
    }
    result = {
        "base_qwen_ppl": BASE_PPL,
        "layers": LAYERS,
        "stochastic_k4": {"independent": independent, "joint": joint},
        "sigmoid_input_baseline": {
            stage: {
                "mean_field": row["n0_ppl_mean"], "n4_mean": row["n4_ppl_mean"],
                "n4_population_std": row["n4_ppl_population_std"], "n16": row["n16_ppl_mean"],
            } for stage, row in old.items()
        },
        "single_layers": single_layers,
        "comparisons": comparisons,
        "joint_training": joint_training,
        "notes": [
            "N=4 mean and population standard deviation use inference seeds 0, 1, and 2.",
            "All training uses seed 0; this is not a multi-seed training confidence interval.",
            "Checkpoint selection uses held-out train-tail validation; test PPL is post-training only.",
            "K=4 adjacent stochastic rounding coordinates bitplanes and is an algorithm reference, not an independent-sigmoid p-bit circuit demonstration.",
        ],
    }
    (ROOT / "comparison.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")

    lines = [
        "# 四层入口 4 位 p-bit 编码实验", "",
        "本实验同时替换 Qwen2.5-0.5B Base 的 decoder layer 9、12、15、18。每个替换 FFN",
        "采用宽度 4864 的两矩阵串联结构：入口先做 K=4 相邻档位随机舍入，用四个 0/1",
        "位置位经过共享输入矩阵；隐藏状态再经过 0/1 p-bit，最后连续读出。每条路径只在",
        "最终读出后平均。单层先训练 2000 步 mean-field 和 6000 步 N=4，随后四层联合",
        "蒸馏 1000 步。测试集为完整 WikiText-2 raw test。", "",
        "## 主要结果", "",
        "| 方案 | 阶段 | Mean-field PPL | N=4 PPL（三 seed） | N=16 PPL |",
        "| --- | --- | ---: | ---: | ---: |",
        f"| 原始 Qwen | 无替换 | {BASE_PPL:.6f} | {BASE_PPL:.6f} | {BASE_PPL:.6f} |",
        f"| 原 sigmoid 入口 | 独立组合 | {old['independent']['n0_ppl_mean']:.6f} | {old['independent']['n4_ppl_mean']:.6f} ± {old['independent']['n4_ppl_population_std']:.6f} | {old['independent']['n16_ppl_mean']:.6f} |",
        f"| K=4 入口 | 独立组合 | {independent['mean_field']:.6f} | **{independent['n4_mean']:.6f} ± {independent['n4_population_std']:.6f}** | {independent['n16']:.6f} |",
        f"| 原 sigmoid 入口 | 联合训练 | {old['joint']['n0_ppl_mean']:.6f} | {old['joint']['n4_ppl_mean']:.6f} ± {old['joint']['n4_ppl_population_std']:.6f} | {old['joint']['n16_ppl_mean']:.6f} |",
        f"| K=4 入口 | 联合训练 | {joint['mean_field']:.6f} | **{joint['n4_mean']:.6f} ± {joint['n4_population_std']:.6f}** | {joint['n16']:.6f} |", "",
        f"K=4 使四层独立组合的 N=4 PPL 从 {old['independent']['n4_ppl_mean']:.6f} 降至",
        f"{independent['n4_mean']:.6f}，改善 {comparisons['independent_n4_absolute_gain_over_sigmoid']:.6f}",
        f"（{comparisons['independent_n4_relative_gain_percent']:.2f}%）。联合训练后从",
        f"{old['joint']['n4_ppl_mean']:.6f} 降至 {joint['n4_mean']:.6f}，改善",
        f"{comparisons['joint_n4_absolute_gain_over_sigmoid']:.6f}（{comparisons['joint_n4_relative_gain_percent']:.2f}%）。",
        f"K=4 自身的联合训练又把独立组合从 {independent['n4_mean']:.6f} 降至",
        f"{joint['n4_mean']:.6f}，恢复了相对原模型超额 NLL 的",
        f"{comparisons['k4_joint_recovery_of_independent_excess_nll_percent']:.2f}%。最终仍比原 Qwen",
        f"高 {comparisons['joint_n4_gap_above_original_percent']:.2f}%。", "",
        "## 单层诊断", "",
        "| 层 | Mean-field PPL | N=4 PPL（三 seed） | N=16 PPL | 路径 bias NMSE | 单路径 variance NMSE |",
        "| ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for layer in LAYERS:
        row = single_layers[str(layer)]
        moments = row["path_moments_normalized"]
        lines.append(
            f"| {layer} | {row['mean_field']:.6f} | {row['n4_mean']:.6f} ± {row['n4_population_std']:.6f} | "
            f"{row['n16']:.6f} | {moments['corrected_bias_nmse']:.6f} | {moments['single_path_variance_nmse']:.6f} |"
        )
    lines += [
        "", "四层中 15 和 18 单独替换的损失较大，但 9、12、15、18 都明显优于此前相同层的",
        "sigmoid 单比特入口组合所累积的误差。N=4 到 N=16 仍有约 0.10 PPL 的改善，说明",
        "随机方差还存在；mean-field 与 N=16 的差距较小，剩余主要是映射偏差和多层分布偏移。", "",
        "## 训练与成本", "",
        f"联合训练最佳验证点为第 {joint_training['best_step']} 步，耗时",
        f"{joint_training['elapsed_seconds']:.2f} 秒，峰值已分配显存",
        f"{joint_training['peak_allocated_bytes'] / 2**30:.2f} GiB。四个 K=4 student 共 34,888,196",
        "个参数，比 sigmoid 入口少四个不再使用的输入温度标量。每条路径中，每个替换 FFN",
        "执行四次二值入口投影和一次二值隐藏读出，共五次矩阵运算；原串联结构为两次。",
        "因此矩阵调用数是 2.5 倍，不能把 PPL 改善解释为免费收益。协调随机舍入的编码器",
        "及位权驱动仍需单独做硬件实现和成本验证。", "",
        "## 可复现性", "",
        "训练只使用 seed 0；N=4 仅对推理 seed 0/1/2 报告均值和总体标准差。校准只使用",
        "训练集并排除最后 65536 个验证 token。联合 checkpoint 由该训练尾部验证集选择，",
        "WikiText-2 test 未参与选择。服务器保留全部权重；仓库结果目录保存协议、哈希、",
        "逐项 JSON、校准和路径矩统计。", "",
    ]
    (ROOT / "RESULTS_ZH.md").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
