# 全路径 P-DNN：sigmoid 0/1 对照实验

日期：2026-09-29。已在服务器单张 RTX 5090 上完成；Slurm GPU 作业 1056 和 CPU 汇总作业 1057 均为 COMPLETED / ExitCode 0:0。GPU 作业总用时 9 分 39 秒。

结论：在本轮固定训练预算下，直接把 tanh 改为同温度 sigmoid 略差；把输入与隐藏层的 sigmoid 温度均减半后，PPL 和局部拟合误差都略优于 tanh 对照。只训练了每组一个 seed=0 checkpoint；三次重复是推理随机种子，不代表三次独立训练。

## 实现与协议

- 基于 GitHub main `bb76480437b7244bd6ecfb2e090c57512ca6e357`，本地分支 `codex/full-path-sigmoid-comparison`。
- Qwen2.5-0.5B Base，模型 revision `060db6499f32faf8b98477b0a26969ef7d8b9987`；只替换零基索引 12，即第 13 个 decoder block 的 FFN。
- 每条路径：连续输入 → 输入 p-bit → 896×4864 投影 → 隐藏 p-bit → 4864×896 读出。输入与隐藏状态均严格二值；各路径独立执行，最后才平均连续输出。权重和累加仍为浮点数。
- 三组均从相同 seed=0 初始化，2000 步 mean-field 预热＋2000 步四路径 STE 蒸馏，batch=4×256，学习率 3e-4 / 1e-4，AdamW weight_decay=0.01，其他参数不变。每组处理 4,096,000 个训练 token（含重复抽样）。
- 相同 WikiText-2 raw 训练/测试 SHA-256；测试 299,078 token，计分 299,077，window=2048，stride=1024。
- 环境：PyTorch 2.7.1+cu128，Transformers 4.45.2，BF16，SDPA；精确环境见 environment.txt。
- 重新测得原始 Qwen PPL **11.652735047**，与历史基线一致；tanh 对照各 PPL 也复现历史结果。

## 主要结果：4 条完整路径

| 编码 | 输入/隐藏温度 | PPL 均值 ± 样本标准差 | 相对原始 Qwen PPL |
| --- | --- | ---: | ---: |
| tanh / ±1 | 0.25 / 1.0 | 12.119894 ± 0.002591 | +4.009% |
| sigmoid / 0–1，同温度 | 0.25 / 1.0 | 12.157562 ± 0.002160 | +4.332% |
| sigmoid / 0–1，半温度 | 0.125 / 0.5 | 12.093255 ± 0.002523 | +3.780% |

半温度 sigmoid 相对 tanh 的平均 PPL 差为 **-0.026639**（约 -0.220%）。这是小幅改善，尚需独立训练种子验证稳定性。PPL 百分比不是准确率百分比。

## 路径数扫描（推理 seed=0）

| 编码 | Mean-field（N=0） | N=1 | N=4 | N=8 | N=16 |
| --- | ---: | ---: | ---: | ---: | ---: |
| tanh / ±1 | 12.085348 | 12.229104 | 12.117123 | 12.105108 | 12.095442 |
| sigmoid / 0–1，同温度 | 12.110391 | 12.291352 | 12.155268 | 12.138232 | 12.120200 |
| sigmoid / 0–1，半温度 | 12.054641 | 12.203259 | 12.090754 | 12.078650 | 12.066037 |

N=0 是将条件均值直接通过多层网络的 mean-field 代理，不是随机网络的严格无限样本极限。

## 局部验证（4 路径，同一批验证输入）

| 编码 | Normalized MSE | 余弦相似度 |
| --- | ---: | ---: |
| tanh / ±1 | 0.597783 | 0.633552 |
| sigmoid / 0–1，同温度 | 0.636921 | 0.598344 |
| sigmoid / 0–1，半温度 | 0.558282 | 0.660798 |

## 温度与编码的解释

`(1+tanh(z/T))/2 = sigmoid(2z/T)`，所以在相同连续场上，sigmoid 温度取 tanh 的一半才会得到相同的取 1 概率。只把函数换成 sigmoid 而保留温度，也同时改变了随机性和饱和程度。

这两种编码在允许连续权重和偏置时有精确的仿射重编码关系：令 s=2b−1，每一层同时使用 W_binary=2W_bipolar、bias_binary=bias_bipolar−W_bipolar·1，并将输入/隐藏温度减半，则在实数算术和对应随机数下可保持完整路径输出。本轮三组是分别训练，未作这种权重转换；结果反映训练设置下的优化差异，不证明 0/1 编码天然更有表达能力。该重编码关系已通过小型 FP64 测试，尚未在整模型 BF16 下单独评估。

## 核验与限制

- 4 项 CPU 功能测试通过：严格二值矩阵输入和末端平均、sigmoid 采样概率和 STE 梯度、仿射重编码等价性、旧 checkpoint 默认 bipolar。GPU 作业前重复通过，并完成一次短训练和整模型前向检查。
- 32 个下载的结果文件通过逐项 SHA-256 核验；完整 checkpoint 哈希保存在 result-manifest.json，checkpoint 保留在下述服务器目录，三组最终 student_sampled.pt 也已下载到百度网盘本地项目目录并逐项核验哈希。
- 官方 parquet 将空白行清洗为字符串空值；将每个空字符串恢复为一个空格加 LF，再直接拼接，才与本项目 raw 文本哈希完全一致。
- 仅验证一个 FFN、一个训练种子和 WikiText-2 PPL；未测试全部 24 层替换、下游准确率或实际 p-bit 硬件速度/能耗。
- 建议下一步保留 tanh 对照，对半温度 sigmoid 增加独立训练种子，再考察更深随机路径与多层替换。

## 制品位置

- 服务器：`/home/Weican_Chen/projects/pbit-qwen-sigmoid-20260929/`。
- 最终 checkpoint：`results/coding_comparison/{bipolar,binary_same_temperature,binary_half_temperature}/student_sampled.pt`。
- 机器可读汇总：`coding_comparison/comparison.json`；每组目录包含训练记录、局部验证和完整 PPL JSON。
- 结果归档 SHA-256：`1c4f32e4618e4611bb5f2e7edc403791ed94bceb8c57bac04c875765f3499152`。

仓库保存代码、原始指标与哈希清单；模型、数据及 checkpoint 另存于项目存储。

## 发布版本与复现

本实验实际运行基于上述 base commit 加本次实验代码，执行源码哈希保留在 `result-manifest.json`。发布分支合入了后续 main 的多 FFN 评估及语料切分支持；原始数值记录保留不变。文中的服务器绝对路径是执行时的溯源信息，复现时应通过运行参数替换为自己的模型、语料和输出路径。
