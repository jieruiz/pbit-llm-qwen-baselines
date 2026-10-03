"""Audit and report full twenty-layer AND-bank composition and joint adaptation."""
import hashlib
import json
import math
from pathlib import Path
import statistics

ROOT=Path("results/multithreshold_and_twenty_layer_20261002")


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def main():
    if read(ROOT/"status.json")["stage"]!="complete":
        raise RuntimeError("run incomplete")
    protocol=read(ROOT/"protocol.json")
    scheduler=read(ROOT/"completion_scheduler.json")
    if hashlib.sha256(Path("experiments/pdnn_ffn/finish_multithreshold_twenty_layer.py").read_bytes()).hexdigest()!=scheduler["script_sha256"]:
        raise RuntimeError("completion scheduler source changed")
    for name,digest in read(ROOT/"source_sha256.json").items():
        if hashlib.sha256((Path("experiments/pdnn_ffn")/name).read_bytes()).hexdigest()!=digest:
            raise RuntimeError(f"execution source changed: {name}")
    baseline=read(Path("results/base_bf16/wikitext2_perplexity.json"))
    sanity_single=read(ROOT/"sanity"/"single_api_one_layer.json")
    sanity_multi=read(ROOT/"sanity"/"multi_api_one_layer.json")
    for key in ("perplexity","mean_negative_log_likelihood","scored_tokens","sample_count","seed"):
        if sanity_single[key]!=sanity_multi[key]:
            raise RuntimeError(f"single/multi evaluator regression: {key}")
    rows=[]
    for stage in ("independent","joint"):
        row={"stage":stage}
        for count,seeds in ((0,(0,)),(1,(0,)),(2,(0,)),(4,(0,1,2)),(16,(0,))):
            values=[]
            timings=[]
            for seed in seeds:
                data=read(ROOT/"evaluation"/stage/f"n{count}_seed{seed}.json")
                if (data["sample_count"]!=count or data["seed"]!=seed or data["layers"]!=protocol["layers"]
                    or data["replacement_count"]!=20 or data["start_token"]!=0
                    or any(data[k]!=baseline[k] for k in ("corpus_tokens","scored_tokens","max_length","stride"))):
                    raise RuntimeError(f"evaluation identity/protocol mismatch: {stage}, {count}, {seed}")
                if len(data["checkpoints"])!=20:
                    raise RuntimeError("wrong checkpoint count")
                for record in data["checkpoints"]:
                    identity=read(ROOT/"checkpoints"/stage/f"layer{record['layer']}.json")
                    if (record["sha256"]!=identity["sha256"] or record["bits"]!=4
                        or record["step"]!=identity["step"] or record["phase"]!=identity["phase"]
                        or not record["path"].endswith("/"+identity["path"])):
                        raise RuntimeError("checkpoint hash/type/path/step mismatch")
                values.append(data["perplexity"])
                timings.append(data["evaluation_elapsed_seconds"])
            row[f"n{count}_values"]=values
            row[f"n{count}_mean"]=statistics.mean(values)
            row[f"n{count}_std"]=statistics.pstdev(values)
            row[f"n{count}_elapsed_seconds"]=timings
        rows.append(row)
    if len(list((ROOT/"evaluation").rglob("*.json")))!=14:
        raise RuntimeError("wrong full-test result count")
    if len(list((ROOT/"checkpoints").rglob("*.json")))!=40:
        raise RuntimeError("wrong checkpoint identity count")
    joint=read(ROOT/"joint"/"summary.json")
    local={}
    for layer in protocol["layers"]:
        path=(Path("results/multithreshold_and_layer12_20261002/k4") if layer==12 else
              ROOT/"individual"/f"layer{layer}")
        local[str(layer)]=read(path/"summary.json")
    history=read(Path("results/continuous_raw_twenty_layer_20261001/comparison.json"))
    historical={r["mode"]:r for r in history["rows"] if r["stage"]=="joint"}
    initial,final=rows
    comparison={"n4_joint_relative_ppl_reduction_percent":100*(1-final["n4_mean"]/initial["n4_mean"]),
        "n4_joint_recovery_of_excess_nll_percent":100*math.log(initial["n4_mean"]/final["n4_mean"])/math.log(initial["n4_mean"]/baseline["perplexity"]),
        "n4_final_above_original_percent":100*(final["n4_mean"]/baseline["perplexity"]-1),
        "n4_final_relative_reduction_vs_historical_raw_percent":100*(1-final["n4_mean"]/historical["continuous_raw"]["n4_mean"])}
    data={"protocol":protocol,"rows":rows,"original_qwen_ppl":baseline["perplexity"],"joint_training":joint,
        "local_training":local,"historical_joint":historical,"comparison":comparison,
        "completion_scheduler":scheduler,
        "sanity":"single and multi-layer APIs return identical PPL on the same layer12 checkpoint, seed0, 4096-token corpus slice",
        "audit":"14 full-test evaluations, 40 checkpoint identity records and all execution source hashes verified"}
    (ROOT/"comparison.json").write_text(json.dumps(data,indent=2)+"\n",encoding="utf-8")
    lines=["# 二十层四bit双分支AND替换（2026-10-02）","",
        "## 实现与协议","",
        "替换层：`{1,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22}`。",
        "保留原始0、2、3、23层FFN，与上一轮二十层试验相同。每条gate/value分支有4个",
        "线性场驱动的sigmoid p-bit，宽度4864，共享原Qwen初始化的gate/up/down矩阵。",
        "入口保持浮点；输出端展开为8组单bit项与16组AND项，常数折叠为偏置。",
        "输出读出矩阵仅接收0/1，最后平均N条路径；驱动偏置、温度和带符号解码系数可学习。", "",
        "第12层复用上一轮K=4最终checkpoint并核对SHA256，其余19层按完全相同的500步",
        "分支拟合、2000步mean-field、6000步N=4局部蒸馏训练。校准及训练排除训练尾部",
        "65536 token，校准规模32768 token。矩阵偏置保留，未修改原单层实现。", "",
        "先将20个独立训练学生直接组合，然后从该状态联合训练1000步；没有继承此前十层",
        "联合模型。每步1024 token（microbatch1、seq256、累积4），学习率峰值1e-5，",
        "50步warm-up后余弦衰减，损失0.8教师logit KL+0.2 next-token CE。只更新替换FFN，",
        "其余Qwen参数冻结。每100步在保留训练尾部评估，按验证PPL选择最佳checkpoint。", "",
        "训练与验证使用已验证代数等价的因式分解采样形式；全部N>0正式test使用AND展开",
        "的二值输入读出。BF16主模型与入口投影，FP32概率bank、读出及路径累加，最终FFN",
        "输出为BF16。有限精度下两种读出并非逐位相同，不能把训练GPU速度当成硬件速度。", "",
        "完整WikiText-2 test共299078 token，计分299077，上下文2048、stride1024。",
        "N=4用三个推理seed，其他各一个；训练seed只有0，test不参与checkpoint选择。", "",
        "## 完整PPL结果", "", f"原始Qwen：**{baseline['perplexity']:.6f}**。", "",
        "| 阶段 | Mean-field | N=1 | N=2 | N=4（三推理seed） | N=16 |",
        "| --- | ---: | ---: | ---: | ---: | ---: |"]
    for r in rows:
        title="20层独立组合" if r["stage"]=="independent" else "20层联合后"
        lines.append(f"| {title} | {r['n0_mean']:.6f} | {r['n1_mean']:.6f} | {r['n2_mean']:.6f} | {r['n4_mean']:.6f} ± {r['n4_std']:.6f} | {r['n16_mean']:.6f} |")
    lines += ["", "## 结果判断", "",
        f"独立组合N=4为{initial['n4_mean']:.6f}，联合后{final['n4_mean']:.6f}；",
        f"联合训练恢复其相对原模型超额NLL的{comparison['n4_joint_recovery_of_excess_nll_percent']:.2f}%。",
        "单层多bit收益没有在20层组合时完全消失，但仍有显著跨层累积损失。", "",
        f"联合模型N=1/2/4/16分别为{final['n1_mean']:.6f}、{final['n2_mean']:.6f}、",
        f"{final['n4_mean']:.6f}、{final['n16_mean']:.6f}，mean-field为{final['n0_mean']:.6f}。",
        "这些是固定同一checkpoint的推理预算比较，可用来观察增加平均次数的收益；",
        "有限次完整PPL差值不能精确分解成结构偏差与采样方差。", "",
        f"最佳保留验证点在第{joint['best_step']}步，后段验证结果有波动。训练只有一个seed，",
        "不据此认定达到架构上限，也不据一个语料推断所有下游任务表现。", ""]
    lines += ["", "±为推理seed的总体标准差，不是训练置信区间。", "",
        "![二十层PPL与采样次数](twenty_layer_quality.png)", "", "## 与历史试验的关系", "",
        "| 二十层联合模型 | N=4 PPL |",
        "| --- | ---: |",
        f"| 历史sigmoid入口串联 | {historical['sigmoid']['n4_mean']:.6f} |",
        f"| 历史浮点入口串联 | {historical['continuous_raw']['n4_mean']:.6f} |",
        f"| 本轮四bit双分支AND | {final['n4_mean']:.6f} |", "",
        "历史结果仅作实际效果参考。此前浮点串联为两矩阵、20层174,440,980参数，随机初始化、",
        "BF16读出，且部分层继承十层联合适配。本轮为三矩阵、20层264,230,400参数，教师",
        "初始化、分支拟合、FP32读出，联合直接从独立组合开始。替换参数量约增加51.47%，",
        "不能把全部差异单独归因于AND结构。相比原Qwen同20个FFN的261,488,640个参数，",
        "本轮替换部分约增加1.05%。", "",
        f"本轮联合训练使N=4 PPL下降{comparison['n4_joint_relative_ppl_reduction_percent']:.2f}%，",
        f"最终仍比原Qwen高{comparison['n4_final_above_original_percent']:.2f}%。mean-field不是",
        "有限N完整随机模型输出的精确总体期望，也不是PPL性能下界。需要区分有限N期望与",
        "N→∞极限：本结构入口连续、两bank条件独立，每个FFN内部先平均再传给下一层；",
        "在理想实数算术下，增加每层独立路径数会逐层趋向mean-field网络。实际有限精度",
        "累加和有限N不具备精确等式。含额外入口采样等多重内部非线性的旧结构不自动满足此条件。", "",
        "## 训练成本与局部验证", "",
        f"20层联合训练：{joint['elapsed_seconds']:.2f}秒，峰值{joint['peak_allocated_bytes']/2**30:.2f}GiB，",
        f"最佳验证步{joint['best_step']}/1000，最佳验证PPL{joint['best_validation_perplexity']:.6f}。",
        "这些时间只包含联合阶段，局部训练分散在多张卡，每个训练任务单卡运行。", "",
        f"本轮新训练19层的局部训练计时合计{sum(local[str(layer)]['training_seconds'] for layer in protocol['layers'] if layer!=12)/60:.2f} GPU分钟，",
        "另有校准、分支拟合、模型加载和评估成本；第12层复用已有训练结果。", "",
        "![联合验证曲线](joint_validation_curve.png)", "",
        "| Layer | 局部训练秒数 | 保留集均值偏差NMSE | 保留集方差NMSE/4 |",
        "| ---: | ---: | ---: | ---: |"]
    for layer in protocol["layers"]:
        t=local[str(layer)]
        v=t["final_validation"]
        lines.append(f"| {layer} | {t['training_seconds']:.2f} | {v['bias_nmse']:.6f} | {v['single_path_variance_nmse']/4:.6f} |")
    lines += ["", "以上是固定teacher输入分布下的局部验证，不能直接当作联合模型各层的因果贡献。", "",
        "## 计算边界与复现", "",
        "K=4每层每路径24组二值读出，N=4为96组，20层合计1920组；另外还有40次浮点",
        "入口投影。共享参数并不消除连线、权重访问和累加成本，尚未证明专用硬件节能或加速。",
        "驱动、权重与累加器的低精度实现和物理随机相关性也未包含。本轮测试范围仅一个语料。", "",
        "实现沿用`multithreshold_and_ffn.py`；入口`run_multithreshold_twenty_layer.py`；",
        "联合训练接入`train_joint_multithreshold_and.py`；正式评估`evaluate_multi_layer_multithreshold_and.py`。",
        "原有4项AND数学测试与新增多层梯度/checkpoint集成测试在服务器通过。",
        "新旧评估入口对同一单层检查点、相同seed的4096-token片段给出完全相同PPL。",
        "本地集成测试因未安装transformers未运行，未为此安装额外依赖；本地语法检查通过。", "",
        "14份完整test结果、40份checkpoint身份和执行源码哈希已核对。",
        "联合训练完成后用finish_multithreshold_twenty_layer.py重叠独立/联合的长N=16评估，",
        "仅替换等待中的调度进程，没有停止训练或评估worker，没有修改checkpoint或评估设置。",
        "权重保留远程`/home/Hongjie_Zeng/pbit_llm/results/multithreshold_and_twenty_layer_20261002/`。", ""]
    (ROOT/"RESULTS_ZH.md").write_text("\n".join(lines),encoding="utf-8")
    print(json.dumps({"rows":rows,"joint":joint,"comparison":comparison},indent=2))


if __name__=="__main__":
    main()
