# Qwen3.5-0.8B：随机 FFN 与完整注意力 PV 采样

日期：2026-10-13。使用已部署的官方后训练 checkpoint，与原始基线共享同一套冻结环境和离线数据。模型和软件版本见相邻目录 `qwen35_baseline_20261013/deployment_manifest.json`、`requirements.lock.txt`。没有重复下载权重，也没有训练权重。

**用户最新要求：本阶段只使用完整 WT2 PPL。基础任务阵列 1241 已取消；下面的五项任务方案保留为历史计划，暂不继续执行。所有 PPL、校准及验证结果保留。**

## 替换范围

| 方案标识 | FFN | 六层完整门控注意力 | 18 层 Gated DeltaNet |
|---|---|---|---|
| original | 原始 SwiGLU | 原始 SDPA | 原始计算 |
| ffn_mean | INT8 权重、固定幅度编码；不采样 | 原始 SDPA | 原始计算 |
| attention_dense_control | 原始 SwiGLU | 与采样路径一致的 FP32 QK/Softmax，但 PV 准确 | 原始计算 |
| ffn_exact | **24 层随机 FFN，准确计数** | 原始 SDPA | 原始计算 |
| attention_pv | 原始 SwiGLU | **准确 QK、Softmax＋512 次 PV 采样** | 原始计算 |
| joint_pv | **24 层随机 FFN，准确计数** | **准确 QK、Softmax＋512 次 PV 采样** | 原始计算 |

完整门控注意力下标为 3、7、11、15、19、23（从 0 计数，即第 4、8、12、16、20、24 层）。PV 替换覆盖前缀预填充和逐 token 解码。通过注册一个 attention interface 保留原类 forward：Q/K 归一化、RoPE、KV 缓存、sigmoid 输出门控和 o_proj 都仍执行官方代码。没有随机化 Q/K 投影，没有使用 Ising Softmax。

Gated DeltaNet 不是不能改。它以一个递归记忆矩阵表示历史，不执行普通 Softmax→PV，所以不能直接套用已有 PV 采样器。它的固定权重输入/输出投影可以作为下一步随机矩阵候选；记忆更新还需检查噪声对长序列状态的影响。本实验先保留这些算子与权重。FFN 和完整注意力替换会改变其输入，因此其输出可以变化，“保留”不表示复用了准确模型的隐藏状态。

## FFN 的具体随机协议

沿用此前 `pbit_dual_or_ffn_20261012/core.py` 的 **fixed_exact 分阶段协议**，复用时改文件名为 exact_ffn.py。它与此前联合报告中的流式 FFN 是两种时间组织方式，不能混同采样预算：

1. 权重：三个矩阵均按 64×64 tile 对称量化为 INT8，固定权重，不产生随机权重流。
2. 入口 x：12 位无符号幅度概率＋单独符号，用训练集确定的每 64 通道固定二次幂尺度；推理不计算动态最大值。
3. gate/up：输入随机流 **4096 次**，所有乘积在空间和时间上准确求和，恢复连续 g/u。
4. SwiGLU 产品：重新编码 g/u；三条条件独立流分别表示 |g|/Sg、sigmoid(gq)、|u|/Su。AND 输出 0/1，另行保留 g/u 的乘积符号。
5. down：产品流 **另采样 4096 次**，固定权重＋准确求和，恢复尺度并转 BF16，与 BF16 残差相加。

也就是说，这版有两阶段、每阶段 4096 次；不是整条 FFN 总共只占 4096 个时钟。与旧分组 OR 2/4 实验中的 fixed_exact 对照属于同一协议。没有 OR 近似，没有双随机权重近似，也没有流式方案的 25% 预热删样。

计算机模拟利用计数分布和线性代数压缩：固定驱动下，一条独立 Bernoulli 流的 T 次计数直接抽取 Binomial(T,p)，保留同一入口计数在各输出连接及 gate/up 分支间共享的关系，再准确加权求和。这在该算法假设下等价于计数流，不在显存中展开 4096 份位流。累加与尺度恢复用 FP32 数值参考，尚未模拟有限位宽累加器、真实 p-bit 相关性或 RTL 门级延迟。sigmoid(gq) 仍由数值函数计算，真实器件如何生成这一流不属于这次质量测试。

入口与内部 g/u 的尺度根据 **WT2 train 前 8192 token** 校准，2048 上下文，每 64 通道取训练样本最大值向上取二次幂。没有使用 WT2 test、验证集或五项基础任务校准。所有正式 forward 后核对尺度没有更新。calibration.json 保存原始 maxima、token/file SHA256、模型 revision。

## PV 采样的具体协议

完整注意力使用准确 FP32 QK 和 Softmax，每个 query/head 从其概率分布独立抽取 512 个 V 地址，带放回，V 向量求和再除以 512。软件用地址计数加权矩阵计算压缩这一求和；实际目标硬件是地址选择＋累加。因此这不是宣称 GPU 上获得了加速。

处理 causal mask、padding 和整行无有效 key 的情况，未来位置不会被抽中。模型原有的输出 sigmoid 门控在采样平均之后继续执行，随后 o_proj 和残差不变。固定 Q/K/V/门控条件下，PV 估计无偏；整个多层网络仍可因后续非线性及误差传播产生性能变化。

attention_dense_control 用同一 FP32 分块注意力路径，但不采样，用来分离 SDPA/BF16 运算次序与 PV 抽样的差异。ffn_mean 则分离 INT8 权重、12 位编码、固定尺度及运算次序与抽样噪声的差异。

## 验证和正式评估

已通过：小规模 signed AND 位平面计数与整数点积严格相等；512 次 PV 估计的 Monte Carlo 均值检查；因果/padding mask 检查；五个变体的 prefill 和 cached decode 有限数值检查；非 FFN 参数逐字节 SHA256 不变；DeltaNet 模块身份不变。verification.json 保存验证结果。

原始模型准确基线：WT2 PPL 15.6610978；HellaSwag acc_norm 49.62%；PIQA acc 68.06%；WinoGrande acc 58.72%；ARC-E acc 65.78%；ARC-C acc_norm 37.29%。均为全量零样本、同一环境。其生成样例仅用于观察正常运行，不能代替任务质量评估。

| 评估 | 协议 |
|---|---|
| 完整 WT2 test | 上下文 2048、步长 1024、297192 个预测 token 恰好计分一次 |
| 随机变体 WT2 | ffn_exact、attention_pv、joint_pv 各 3 个种子（0、1、2） |
| 确定性对照 WT2 | ffn_mean、attention_dense_control 各 1 次 |
| 五项基础任务 | 三个主要随机变体 seed0；HellaSwag 10042、PIQA 1838、WinoGrande 1267、ARC-E 2376、ARC-C 1172 题 |

最初安排 PPL 阵列成功后自动开始基础任务阵列，后按用户要求取消了后者；当前只比较完整 PPL。重复提交时跳过已有完成结果。没有把短文本段后缀 PPL 或 smoke 小样本作为正式成绩。

## 远程任务与位置

目录：`/home/Hongjie_Zeng/qwen35_pbit_20261013/`。

- 1235：初次准备；尺度校准成功，模块身份审计引用了不存在的 config 字段，已修复。
- 1237：准备与全部验证成功，复用已完成的训练集校准。
- **1236**：完整 PPL，五个变体阵列，已全部完成；三种主要随机方案各三个种子。
- **1241**：三种主要方案的全量基础任务，曾在 1236 成功后启动，现已按用户要求取消。

两组正式阵列按依赖顺序运行，不会同时各占两张卡。GPU 执行均要求 Slurm。每个任务一张卡、4 CPU、16 GB 主存。没有启动权重训练。

```bash
cd /home/Hongjie_Zeng/qwen35_pbit_20261013
/home/Hongjie_Zeng/qwen35_baseline_20261013/venv/bin/python collect_status.py
/opt/slurm/21.08.8/bin/squeue -u Hongjie_Zeng
```

结果：`results/formal/<方案>/<任务>_seed<种子>.json`；日志：`results/logs/`。结果元数据包含模型 revision、程序哈希、校准哈希、替换层列表、诊断统计、计分范围和保护参数哈希。完整 WT2：ffn_exact 17.075531 ± 0.028020；attention_pv 15.744119 ± 0.001543；joint_pv 17.174382 ± 0.022787，± 为三种子的总体标准差。ffn_mean 对照 15.655289，attention_dense_control 对照 15.662724。基础任务已取消。结果汇总见 RESULTS_ZH.md，实际完成快照见 status_snapshot.json。
