# Qwen3.5-0.8B：概率计算替换与完整 PPL 结果

2026-10-13。官方后训练 checkpoint `Qwen/Qwen3.5-0.8B`，revision `2fc06364715b967f1860aea9cf38778875588b17`。只评估文本；24层FFN、18层Gated DeltaNet、6层完整门控注意力。所有正式评估通过Slurm在RTX5090上执行。本次没有微调或重新训练。

## 一眼看结果

下表为完整 WikiText-2 test 滑窗 PPL，越低越好。上下文2048、步长1024、297193个token中的297192个预测目标各计分一次，不额外添加BOS。随机方案为三个推理种子的均值；这不是固定段落后缀PPL。

| 方案 | PPL | 较原始变化 |
|---|---:|---:|
| 原始模型 | 15.661098 | — |
| 仅六层完整注意力PV512采样 | 15.744119 | +0.530% |
| 固定尺度随机FFN | 17.075531 | +9.032% |
| 固定尺度随机FFN＋PV512 | 17.174382 | +9.663% |
| 动态尺度随机FFN | 15.840404 | +1.145% |
| 动态尺度随机FFN＋PV512 | 15.923481 | +1.675% |
| 动态FFN＋PV512＋DeltaNet输出投影随机化 | 15.964670 | +1.938% |
| 动态FFN＋PV512＋DeltaNet输出/z投影随机化 | 16.035416 | +2.389% |
| 动态FFN＋PV512＋DeltaNet输出/z/QKV投影随机化 | **16.124398** | **+2.958%** |

FFN覆盖24层；PV只作用于6层完整注意力，Q/K投影、QK点积、Softmax及输出门控保持原算法。这里没有将Ising Softmax或Qwen2.5的筛选结果混入Qwen3.5实验。

单独替换DeltaNet投影、保持FFN及完整注意力原始计算的对照：

| 全18层DeltaNet替换范围 | 仅INT8权重，无输入采样 | 随机输入＋固定INT8权重 |
|---|---:|---:|
| 输出投影 | 15.697660 | 15.709249 ± 0.003275 |
| 输出＋z | 15.758957 | 15.772502 ± 0.002144 |
| 输出＋z＋QKV | 15.839524 | 15.857678 ± 0.002626 |

±为三个种子间总体标准差，不是语料置信区间。最大范围的采样比量化对照额外增加0.018154 PPL。QKV/z在最大范围共享入口计数流，因此范围增量也包含噪声相关性改变。

8K上下文、步长4096的独立检查：原模型 **14.408686**；仅DeltaNet输出/z/QKV随机化 **14.645752 ± 0.003248**，增加1.645%。未测8K联合方案；每个滑窗重置状态，不能证明超长连续记忆稳定性。

## 计算方法与限制

- 固定尺度FFN：INT8 tile64权重，离线训练文本校准的group64尺度，12位幅度概率编码；入口4096次、产品/down另4096次，分阶段计算。
- 动态FFN：INT8 tile64权重，每token/group64动态maxabs；入口4096次，256次更新一次估计，25%预热，有效产品输出3072次。三流AND产品驱动固定down矩阵。
- DeltaNet投影：固定INT8权重，动态group64概率，每投影4096次采样；所有输出行共享相应输入计数。alpha/beta门、记忆状态更新、卷积、归一化及后续激活保留官方算法。
- PV512：从准确Softmax概率中每query/head带放回采样512次，以V选择累加实现近似PV。
- 软件使用二项分布计数的代数压缩，不展开4096份位流；参考累加为FP32、接口/残差为BF16。这里没有验证有限位宽计数器、物理p-bit相关性或RTL时序。
- 动态max、概率编码、RNG、尺度恢复和读出均有硬件成本。PPL结果不能直接解释为节能或GPU加速。
- 固定与动态FFN同时改变尺度、时间组织与概率舍入，不能把差异全部归因于动态尺度。

## 文件入口

| 子目录 | 内容 |
|---|---|
| [qwen35_baseline_20261013](qwen35_baseline_20261013/README_ZH.md) | 原始PPL、五项任务、生成性能、部署revision与软件环境 |
| [qwen35_pbit_20261013](qwen35_pbit_20261013/RESULTS_ZH.md) | 固定尺度FFN、PV及联合对照，11份正式PPL |
| [qwen35_dynamic_ffn_20261013](qwen35_dynamic_ffn_20261013/README_ZH.md) | 动态FFN对照，7份正式PPL、DeltaNet分析和门控统计 |
| [qwen35_deltanet_projection_20261013](qwen35_deltanet_projection_20261013/RESULTS_ZH.md) | 三步投影替换及联合结果，25份正式PPL（含4份8K） |

所有替换结果统一放在各子目录的 `results/formal/`，包含逐窗口NLL/PPL、种子、计分token数及原始元数据。`PUBLICATION_MANIFEST.json`列出来源与发布文件SHA256；运行代码与PPL原始记录按字节保留。文档更新了完成状态。

原模型五项任务为零样本、全量：HellaSwag acc_norm49.62%，PIQA acc68.06%，WinoGrande acc58.72%，ARC-E acc65.78%，ARC-C acc_norm37.29%。五份JSON保留汇总指标与元数据，移除了逐题`samples`，避免重复发布数据集文本；原始完整文件SHA256记录在清单中。按用户要求，替换方案只评估PPL，五项任务未补测。

## 复现说明

这是已运行实验的源码快照，保留原服务器路径以对应结果内的源码哈希，不是自动适配所有机器的安装包。

1. 四个实验目录必须保持同级；原部署位置为 `/home/Hongjie_Zeng/<子目录名>`。脚本引用相邻baseline及PV实现。
2. baseline模型目录为 `qwen35_baseline_20261013/models/Qwen3.5-0.8B`，使用上述固定revision下载。`setup_model.py`原始脚本会在执行时解析远端main；严格复现应显式下载固定revision，并核验已有`deployment_manifest.json`，不要用未来main覆盖基准。
3. 环境参考baseline的`requirements.lock.txt`和结果元数据：torch2.7.1+cu128、transformers5.19.0、lm_eval0.4.13、fla-core/flash-linear-attention0.5.2。`pip freeze`包含原机器环境信息，不是跨平台的一键安装承诺。
4. WT2文件为 `joint_ffn_attention_20261009/data/wikitext-2/wiki.test.raw`，SHA256 `173c87a53759e0201f33e0ccf978e510c2042d7f2cb78229d9a50d79b9e7dd08`。数据和模型不随本分支发布。若换服务器，调整Slurm工作目录、Python路径及baseline的数据路径，并记录新源码哈希。
5. Slurm脚本：baseline的`baseline.sbatch`；固定尺度的`prepare.sbatch`/`ppl.sbatch`；动态FFN的`run.sbatch`；投影的`prepare.sbatch`/`stepwise.sbatch`/`long_context.sbatch`。`tasks.sbatch`仅保留历史源码，本阶段不提交替换模型的基础任务。
6. GPU数值验证已在远程完成；发布检查仅重新解析Python/JSON、核对43份PPL目标数和结果哈希，没有重复占用GPU。部分入口遇到已有结果会跳过，重新实验应使用独立输出目录，不能把跳过误认为重跑。

Qwen3.5使用后训练权重且分词器不同，不能将其PPL直接与本仓库Qwen2.5-0.5B Base的绝对PPL比较。
