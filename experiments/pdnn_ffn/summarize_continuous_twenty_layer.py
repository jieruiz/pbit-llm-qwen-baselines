"""Validate and summarize staged 10-to-20 floating-input FFN expansion."""
import hashlib
import json
import math
from pathlib import Path
import statistics

ROOT = Path("results/continuous_raw_twenty_layer_20261001")
PREVIOUS = Path("results/continuous_raw_ten_layer_20261001")
LAYERS = [1,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22]
MODES = ("continuous_raw", "sigmoid")
STAGES = ("independent", "staged", "joint")


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def main():
    if read(ROOT / "status.json")["stage"] != "complete":
        raise RuntimeError("Experiment incomplete")
    original = read(Path("results/base_bf16/wikitext2_perplexity.json"))
    base = original["perplexity"]
    for name, expected in read(ROOT / "source_sha256.json").items():
        actual = hashlib.sha256((Path("experiments/pdnn_ffn") / name).read_bytes()).hexdigest()
        if actual != expected:
            raise RuntimeError(f"Source differs from execution: {name}")
    rows = []
    for mode in MODES:
        for stage in STAGES:
            row = {"mode": mode, "stage": stage}
            for count, seeds in ((0, (0,)), (4, (0,1,2)), (16, (0,))):
                values = []
                for seed in seeds:
                    data = read(ROOT / "evaluation" / stage / mode / f"samples{count}_seed{seed}.json")
                    if (data["layers"] != LAYERS or data["sample_count"] != count or data["seed"] != seed
                            or any(data[key] != original[key] for key in ("corpus_tokens", "scored_tokens", "max_length", "stride"))):
                        raise RuntimeError(f"Evaluation mismatch: {mode}, {stage}, {count}, {seed}")
                    for record in data["checkpoints"]:
                        identity = read(ROOT / "checkpoints" / stage / mode / f"layer{record['layer']}.json")
                        if not record["path"].endswith("/" + identity["path"]):
                            raise RuntimeError(f"Checkpoint path mismatch: {record}")
                    values.append(data["perplexity"])
                row[f"n{count}_mean"] = statistics.mean(values)
                row[f"n{count}_population_std"] = statistics.pstdev(values)
                row[f"n{count}_values"] = values
            rows.append(row)
    table = {(row["mode"], row["stage"]): row for row in rows}
    comparisons = {}
    for stage in STAGES:
        a, b = table[("sigmoid", stage)]["n4_mean"], table[("continuous_raw", stage)]["n4_mean"]
        comparisons[stage] = {
            "raw_gain_ppl": a-b, "raw_relative_reduction_percent": 100*(1-b/a),
            "raw_above_original_percent": 100*(b/base-1),
            "raw_recovery_of_sigmoid_excess_nll_percent": 100*math.log(a/b)/math.log(a/base),
        }
    previous = read(PREVIOUS / "comparison.json")
    previous_joint = {r["mode"]: r for r in previous["rows"] if r["stage"] == "joint"}
    expansion = {}
    for mode in MODES:
        initial = table[(mode, "staged")]["n4_mean"]
        final = table[(mode, "joint")]["n4_mean"]
        expansion[mode] = {
            "ten_layer_joint_n4": previous_joint[mode]["n4_mean"],
            "twenty_layer_joint_n4": final,
            "depth_ppl_increase_percent": 100*(final/previous_joint[mode]["n4_mean"]-1),
            "joint_recovery_of_staged_excess_nll_percent": 100*math.log(initial/final)/math.log(initial/base),
        }
    single = {}
    for mode in MODES:
        single[mode] = {}
        for layer in LAYERS:
            single[mode][str(layer)] = {}
            for count in (0,4):
                data = read(ROOT / "evaluation" / "single_layer" / mode / f"layer{layer}" / f"samples{count}_seed0.json")
                if data["layer"] != layer or data["sample_count"] != count:
                    raise RuntimeError("Single-layer identity mismatch")
                if any(data[key] != original[key] for key in ("corpus_tokens", "scored_tokens", "max_length", "stride")):
                    raise RuntimeError("Single-layer protocol mismatch")
                single[mode][str(layer)][f"n{count}"] = data["perplexity"]
    training = {mode: read(ROOT / "joint" / mode / "summary.json") for mode in MODES}
    summary = {"original_qwen_ppl": base, "rows": rows, "comparisons": comparisons,
               "depth_comparison": expansion, "single_layers": single, "joint_training": training,
               "previous_joint": previous_joint, "protocol": read(ROOT / "protocol.json"),
               "source_hashes_verified": True, "new_full_test_evaluations": 70,
               "reused_single_layer_evaluations": 40, "checkpoint_records": 120}
    (ROOT / "comparison.json").write_text(json.dumps(summary, indent=2)+"\n", encoding="utf-8")
    names = {"continuous_raw": "原始浮点入口", "sigmoid": "单比特 sigmoid 入口"}
    stage_names = {"independent": "20层独立组合", "staged": "10层联合＋10层独立", "joint": "20层联合训练后"}
    lines = [
        "# 二十层直接浮点入口扩展实验（2026-10-01）", "",
        "## 结构与训练协议", "",
        "保留第0、2、3、23层原始Qwen FFN；替换层为", "",
        "`{1,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22}`。", "",
        "浮点入口结构：`x → W₁x+b₁ → Bernoulli(sigmoid(h/T)) → W₂z+b₂`，",
        "宽度4864，不裁剪、不量化入口，隐藏层仍为0/1。全部矩阵偏置保留，只有隐藏",
        "温度可学习，无额外p-bit阈值。对照组在入口增加可学习温度的sigmoid二值采样。",
        "两组沿用BF16激活及FP32可训练参数。每FFN只在最终输出处平均N条路径。", "",
        "新增十层各自训练2000步mean-field加6000步N=4；此前十层独立检查点直接复用，",
        "由SHA256核对。先评估20层全独立组合，再换入此前十层联合模型形成分阶段初始化，",
        "最后从该初始化联合训练全部20个FFN共1000步。N=4，每步1024 token，学习率1e-5，",
        "损失0.8 teacher-logit KL + 0.2 next-token CE。其余原模型参数冻结。", "",
        "两种入口使用相同训练日程和验证选择规则；继承的十层最佳联合点分别为浮点900步、",
        "sigmoid 1000步。二十层结果包含此前十层的联合适配，不能当作20层完全从头联合训练。",
        "历史43.383146采用更短局部采样训练及不同阈值/温度设置，只作历史参考。", "",
        "## 完整 WikiText-2 PPL", "",
        f"原始Qwen：**{base:.6f}**。{original['corpus_tokens']}个test token，计分{original['scored_tokens']}个，",
        "上下文2048、步长1024。N=4报告推理seed0/1/2的均值和总体标准差；训练只有seed0。", "",
        "| 阶段 | 入口 | Mean-field | N=4 | N=16 |",
        "| --- | --- | ---: | ---: | ---: |",
    ]
    for stage in STAGES:
        for mode in ("sigmoid", "continuous_raw"):
            r = table[(mode,stage)]
            lines.append(f"| {stage_names[stage]} | {names[mode]} | {r['n0_mean']:.6f} | {r['n4_mean']:.6f} ± {r['n4_population_std']:.6f} | {r['n16_mean']:.6f} |")
    final = comparisons["joint"]
    lines += ["", f"联合后，浮点入口比本轮单比特对照降低{final['raw_gain_ppl']:.6f} PPL",
              f"（{final['raw_relative_reduction_percent']:.2f}%），但仍比原Qwen高{final['raw_above_original_percent']:.2f}%。", "",
              "## 从十层扩展到二十层", "",
              "| 入口 | 十层联合N=4 | 二十层联合N=4 | PPL上升 |",
              "| --- | ---: | ---: | ---: |"]
    for mode in ("sigmoid", "continuous_raw"):
        d = expansion[mode]
        lines.append(f"| {names[mode]} | {d['ten_layer_joint_n4']:.6f} | {d['twenty_layer_joint_n4']:.6f} | {d['depth_ppl_increase_percent']:.2f}% |")
    lines += ["", "![二十层阶段结果](twenty_layer_comparison.png)", "",
              "## 单层诊断", "",
              "每次只替换一个FFN，seed0；此前十层数据复用，新增十层本轮测量。", "",
              "| Layer | sigmoid MF | 浮点 MF | sigmoid N=4 | 浮点 N=4 |",
              "| ---: | ---: | ---: | ---: | ---: |"]
    for layer in LAYERS:
        a, b = single["sigmoid"][str(layer)], single["continuous_raw"][str(layer)]
        lines.append(f"| {layer} | {a['n0']:.6f} | {b['n0']:.6f} | {a['n4']:.6f} | {b['n4']:.6f} |")
    lines += ["", "## 联合训练成本", "",
              "| 入口 | 最佳验证步 | 联合时间（秒） | 峰值显存（GiB） |",
              "| --- | ---: | ---: | ---: |"]
    for mode in ("sigmoid", "continuous_raw"):
        t = training[mode]
        lines.append(f"| {names[mode]} | {t['best_step']} | {t['elapsed_seconds']:.2f} | {t['peak_allocated_bytes']/2**30:.2f} |")
    lines += ["", "![单层结果](single_layer_comparison.png)", "",
              "![联合验证曲线](joint_validation_curve.png)", "",
              "## 结果判断", "",
              "连续入口的收益扩展到20层仍然成立：联合N=4比匹配单比特对照低约30%。",
              "但从10层的15.593升至20层的27.128，说明只解除入口量化仍不足以解决深层组合退化。",
              "N=16为25.703，mean-field为25.354；本轮增加路径平均后仍有很大差距，",
              "不能把退化全部归因于N=4的采样噪声。隐藏映射的逼近误差、训练目标及多层输入",
              "分布变化是后续要分开验证的候选原因，本试验没有识别出唯一因果瓶颈。", "",
              "20个单层对照中浮点N=4均优于sigmoid。后部19–22层单层损失较大，其中21层",
              "浮点单层PPL约12.920；这些层值得做逐层移除诊断，但不能用独立单层排名直接",
              "解释联合模型的全部损失。浮点组最佳验证点在1000步预算终点，不能断言已经收敛。",
              "下一项有针对性的试验是延长同一20层模型的联合训练，并用固定验证集选择checkpoint，",
              "再测保留后部敏感层或改进隐藏表示的增益；本轮尚未执行这些额外试验。", "",
              "## 解释范围与复现", "",
              "浮点入口取消了第一矩阵输入必须为二值的限制。成本是GPU软件测量，不能直接",
              "外推专用p-bit硬件能耗。对于固定输入，单个浮点入口FFN的条件期望等于其",
              "mean-field输出（实数算术下）；整个多层模型还有其他非线性，不能把完整模型",
              "mean-field PPL当作N→∞的严格期望或理论性能下界。", "",
              "新增局部模型、分阶段初始化和最终联合检查点均记录身份与SHA256。全模型共30次",
              "完整test评估，新增单层40次、复用单层40次。test没有参与最佳checkpoint选择；",
              "验证使用训练集末65536个token。只有一个训练seed和一个语料，结论不代表",
              "通用任务性能，也不证明架构的理论上限。", "",
              "入口脚本：`experiments/pdnn_ffn/run_continuous_twenty_layer.py`。",
              "远程权重：`/home/Hongjie_Zeng/pbit_llm/results/continuous_raw_twenty_layer_20261001/`；",
              "原始十层权重仍在同级`continuous_raw_ten_layer_20261001/`。", ""]
    (ROOT / "RESULTS_ZH.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({"comparisons": comparisons, "depth_comparison": expansion, "training": training}, indent=2))


if __name__ == "__main__":
    main()
