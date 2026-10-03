"""Audit and summarize the matched floating-input ten-layer experiment."""
import hashlib
import json
import math
from pathlib import Path
import statistics

ROOT = Path("results/continuous_raw_ten_layer_20261001")
LAYERS = (7, 8, 9, 10, 11, 12, 13, 14, 15, 18)
MODES = ("sigmoid", "continuous_raw")


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def main():
    if read(ROOT / "status.json")["stage"] != "complete":
        raise RuntimeError("Experiment incomplete")
    original = read(Path("results/base_bf16/wikitext2_perplexity.json"))
    base = original["perplexity"]
    hashes = read(ROOT / "source_sha256.json")
    for name, expected in hashes.items():
        if hashlib.sha256((Path("experiments/pdnn_ffn") / name).read_bytes()).hexdigest() != expected:
            raise RuntimeError(f"Source changed since execution: {name}")
    results = {}
    rows = []
    for mode in MODES:
        results[mode] = {}
        for stage in ("independent", "joint"):
            folder = ROOT / "evaluation" / stage / mode
            row = {"mode": mode, "stage": stage}
            for count, seeds in ((0, (0,)), (4, (0, 1, 2)), (16, (0,))):
                values = []
                for seed in seeds:
                    data = read(folder / f"samples{count}_seed{seed}.json")
                    if (sorted(data["layers"]) != list(LAYERS) or data["sample_count"] != count
                            or data["seed"] != seed or data["max_length"] != 2048 or data["stride"] != 1024
                            or data["corpus_tokens"] != original["corpus_tokens"]
                            or data["scored_tokens"] != original["scored_tokens"]):
                        raise RuntimeError(f"Unexpected evaluation protocol: {folder}, {count}, {seed}")
                    values.append(data["perplexity"])
                row[f"n{count}_mean"] = statistics.mean(values)
                row[f"n{count}_population_std"] = statistics.pstdev(values)
                row[f"n{count}_values"] = values
            results[mode][stage] = row
            rows.append(row)
    comparisons = {}
    for stage in ("independent", "joint"):
        baseline = results["sigmoid"][stage]["n4_mean"]
        raw = results["continuous_raw"][stage]["n4_mean"]
        comparisons[stage] = {
            "absolute_ppl_gain": baseline - raw,
            "relative_ppl_reduction_percent": 100 * (1 - raw / baseline),
            "raw_ppl_above_original_percent": 100 * (raw / base - 1),
            "excess_nll_reduction_percent": 100 * math.log(baseline / raw) / math.log(baseline / base),
        }
    singles = {}
    training = {}
    for mode in MODES:
        singles[mode] = {}
        for layer in LAYERS:
            folder = ROOT / "evaluation" / "single_layer" / mode / f"layer{layer}"
            singles[mode][str(layer)] = {f"n{count}": read(folder / f"samples{count}_seed0.json")["perplexity"] for count in (0, 4)}
        training[mode] = read(ROOT / "joint" / mode / "summary.json")
    output = {"original_qwen_ppl": base, "rows": rows, "comparisons": comparisons,
              "single_layers": singles, "joint_training": training,
              "protocol": read(ROOT / "protocol.json"), "source_hashes_verified": True}
    (ROOT / "comparison.json").write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    names = {"sigmoid": "单比特 sigmoid 入口", "continuous_raw": "原始浮点入口"}
    stages = {"independent": "独立组合", "joint": "联合训练"}
    lines = [
        "# 十层原始浮点入口与单比特入口对照（2026-10-01）", "",
        "在 Qwen2.5-0.5B Base 同时替换 `{7,8,9,10,11,12,13,14,15,18}` 十个 FFN。",
        "实验组直接使用原始浮点输入，不量化、不裁剪、不先过 sigmoid。每个 FFN 为",
        "`x → W₁x+b₁ → Bernoulli(sigmoid(h/T)) → W₂z+b₂`。隐藏层仍为 0/1，",
        "每条完整路径只在输出之后平均。浮点计算沿用 BF16 激活/混合精度，训练权重为 FP32。", "",
        "## 公平对照与训练协议", "",
        "两组的全部十个局部模型均从头训练：宽度4864，seed0，共同矩阵初始化和训练窗口，",
        "2000步 mean-field + 6000步 N=4，每步1024 token。输入采样使两组随机流不同。",
        "对照组为原 sigmoid 单比特入口，入口温度初值0.125；两组隐藏温度初值0.5。",
        "温度可学习，矩阵偏置保留，无额外 p-bit 阈值。随后各做1000步 N=4 联合蒸馏，",
        "损失为0.8 teacher-logit KL + 0.2 next-token CE，学习率1e-5。原 Qwen 其余参数冻结。",
        "历史十层18.568570使用2000步采样训练及不同编码设置，因此仅作为历史参考，",
        "不能用它与新结果的全部差距归因于入口改动。", "",
        "## 完整 WikiText-2 结果", "",
        f"原始 Qwen PPL：**{base:.6f}**。上下文2048，步长1024，{original['corpus_tokens']}个test token，计分{original['scored_tokens']}个。",
        "N=4为三个推理种子的均值与总体标准差；不是训练种子的置信区间。", "",
        "| 入口 | 阶段 | Mean-field | N=4 | N=16 |",
        "| --- | --- | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append(f"| {names[row['mode']]} | {stages[row['stage']]} | {row['n0_mean']:.6f} | {row['n4_mean']:.6f} ± {row['n4_population_std']:.6f} | {row['n16_mean']:.6f} |")
    for stage, value in comparisons.items():
        lines += ["", f"{stages[stage]}：浮点入口相对匹配单比特对照降低 {value['absolute_ppl_gain']:.6f} PPL",
                  f"（{value['relative_ppl_reduction_percent']:.2f}%），消除了相对原Qwen超额NLL的",
                  f"{value['excess_nll_reduction_percent']:.2f}%；浮点入口PPL仍比原Qwen高",
                  f"{value['raw_ppl_above_original_percent']:.2f}%。"]
    lines += ["", "## 实验判断", "",
              "取消入口二值化带来明确改善，但本轮十层替换没有达到接近原模型的质量。",
              "同预算对照说明入口表示确实是一个瓶颈；同时，浮点入口的mean-field结果",
              "仍明显差于原始Qwen，提示剩余的确定性映射逼近、训练目标和多层分布偏移",
              "需要继续研究。不能据此认定该结构存在不可突破的理论上限。", "",
              "下一步比继续增加入口编码位数更有针对性的对照是：保持浮点入口，比较现有",
              "两矩阵sigmoid隐藏结构与保留SwiGLU幅值/乘积结构的学生；并用中间状态",
              "匹配检验误差累积。增加采样数仍可减少方差，但本轮N=16并未接近原模型。", "",
              "![十层同预算比较](ten_layer_comparison.png)", "",
              "## 单层诊断", "", "每次只替换一层，均为推理seed0。", "",
              "| 层 | sigmoid MF | 浮点 MF | sigmoid N=4 | 浮点 N=4 |",
              "| ---: | ---: | ---: | ---: | ---: |"]
    for layer in LAYERS:
        a, b = singles["sigmoid"][str(layer)], singles["continuous_raw"][str(layer)]
        lines.append(f"| {layer} | {a['n0']:.6f} | {b['n0']:.6f} | {a['n4']:.6f} | {b['n4']:.6f} |")
    lines += ["", "## 成本与解释范围", "", "| 入口 | 最佳联合步数 | 联合时间（秒） | 峰值已分配显存（GiB） |",
              "| --- | ---: | ---: | ---: |"]
    for mode in MODES:
        row = training[mode]
        lines.append(f"| {names[mode]} | {row['best_step']} | {row['elapsed_seconds']:.2f} | {row['peak_allocated_bytes']/2**30:.2f} |")
    lines += ["", "原始浮点入口组的单个 FFN，在固定输入下只含一个随机隐藏层及线性读出，",
              "其条件期望在实数算术下等于 mean-field 输出；BF16 累加会另有舍入误差。",
              "但十个随机 FFN 之间有注意力和其他非线性，整个模型的 mean-field PPL 并非",
              "完整随机模型 N→∞ 的严格期望，也不是无条件的理论性能上界。", "",
              "浮点入口取消了全二值矩阵输入约束，第一矩阵需要处理连续幅值，第二矩阵仍只接收0/1。",
              "训练和测试时间是GPU模拟成本，不能据此推断专用p-bit硬件能耗。两个矩阵的形状",
              "保持896→4864→896，每个浮点入口学生少一个输入温度标量。", "",
              "仅一个训练seed、一个语言建模语料；结果不等于通用任务能力已恢复。模型选择使用",
              "末65536个训练token保留集；test只在固定训练协议后评估。完整权重保留远程，",
              "本目录保存协议、逐项PPL、训练记录、源码哈希和检查点身份。", ""]
    (ROOT / "RESULTS_ZH.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps(comparisons, indent=2))


if __name__ == "__main__":
    main()
