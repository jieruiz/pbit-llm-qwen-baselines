# 动态尺度随机 FFN 的 Qwen3.5 对照

2026-10-13，按用户要求先采用之前误差较小的动态尺度方法，仅计算完整WT2 PPL。本目录保持DeltaNet原始计算；后续投影替换已在相邻qwen35_deltanet_projection_20261013完成。基础任务不补测。

| 方案 | 当前完整PPL | 状态 |
|---|---:|---|
| 原始模型 | 15.661098 | 已完成原始基线 |
| INT8权重、不采样 | 15.651730 | 完成 |
| 24层动态尺度随机FFN | 15.840404 ± 0.003995 | 三个种子完成 |
| 动态尺度FFN＋六层完整注意力PV512 | 15.923481 ± 0.002475 | 三个种子完成 |

随机方案为三个种子的最终均值±总体标准差。基线与替换全部使用context2048、stride1024、297192预测token恰好计分一次。

动态FFN复用之前 pbit_streaming_ffn_20261009/streaming_ffn.py 的成功实现：固定INT8权重、64×64权重tile，每token/每64通道动态max，连续概率参考，入口4096采样、有效输出3072采样、每256采样更新估计、25%预热、准确累加、FP32内部/BF16残差。它不是固定尺度分阶段版本的简单尺度开关，也没有新增N12比较器量化，结果包含尺度、时间组织和概率舍入方式的差异。

三流AND后产品用于固定down矩阵；sigmoid仍为理想数值函数。动态max、缩放、连续到概率接口和计数恢复都有硬件成本，不据软件PPL直接宣称节能。所有18层DeltaNet保留官方实现。PV只用于6层完整注意力，并保留其sigmoid输出门控。

远程目录 `/home/Hongjie_Zeng/qwen35_dynamic_ffn_20261013/`；Slurm阵列 **1246**，最多同时两张5090。每个变体先做prefill/decode有限数值检查和保护权重哈希检查，然后计算PPL。INT8对照一个种子，两个随机变体各三个种子。

另有只读门控统计任务 **1254**，在原模型WT2 train前8192token观察alpha/beta分布，已完成，见results/deltanet_gates.json；不训练，不改变模型，不是另一套基础任务评估。

结果在 `results/formal/<方案>_seed<编号>.json`，进度可运行 collect_status.py，当前本地快照为status_snapshot.json。

DeltaNet结构、公式、操作数、存储和随机候选详见 DELTANET_ANALYSIS_ZH.md；可复算数值见 deltanet_operation_counts.json 及 count_deltanet_ops.py。操作数是解码递推模型，非GPU计时或功耗结果，尚不含归一化/激活/Softmax/残差/内存能量。
