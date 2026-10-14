# Python TLM 与 attention 模块一致性核对

核对日期：2026-10-14。TLM 分支：`pbit-tlm-macro-decode-20261014`；核对提交：`3445921193b7495f69ff3728de95e37d664fc2a9`。attention 源码采用上一步已核验的 `c021f5bd9d051e306c2c41630b8ef25ca5dc7e93` 快照；其中 Qwen2.5 attention 实现沿自 `pbit-joint-fixed-ffn-attention-20261009`。本次未修改这些实现，未运行 LLM 数值推理、训练、RTL 仿真或综合。

结论：**完整 decode 的外层骨架与我们的 Qwen2.5 attention 设计相容；当前 TLM 实现的是稠密数字 attention 基线，尚未实现 QK 筛选、Ising softmax 或采样 PV。现有结果不能作为这些新模块的性能验证。** README 和中文说明已经明确声明这些边界，不是模块接错了，而是尚未集成。

## 对照范围与源代码

- [TLM 概述](https://github.com/jieruiz/pbit-llm-qwen-baselines/blob/3445921193b7495f69ff3728de95e37d664fc2a9/architecture_tlm/README.md)
- [系统模型及 attention()](https://github.com/jieruiz/pbit-llm-qwen-baselines/blob/3445921193b7495f69ff3728de95e37d664fc2a9/architecture_tlm/system_decode.py#L259)
- [成本来源与限制](https://github.com/jieruiz/pbit-llm-qwen-baselines/blob/3445921193b7495f69ff3728de95e37d664fc2a9/architecture_tlm/configs/cost_provenance.json)
- [QK 筛选及压缩 attention](https://github.com/jieruiz/pbit-llm-qwen-baselines/blob/c021f5bd9d051e306c2c41630b8ef25ca5dc7e93/experiments/joint_router_ffn_20261009/attention.py#L53)
- [K 摘要与 max4 评分](https://github.com/jieruiz/pbit-llm-qwen-baselines/blob/c021f5bd9d051e306c2c41630b8ef25ca5dc7e93/experiments/joint_router_ffn_20261009/metrics.py)
- [Ising 强互斥 CTMC](https://github.com/jieruiz/pbit-llm-qwen-baselines/blob/c021f5bd9d051e306c2c41630b8ef25ca5dc7e93/experiments/joint_ffn_attention_20261009/sampling.py#L4)

TLM 是标准库 Python 的宏级、松散计时模型，不是 SystemC/TLM-2.0；它通过尺寸、动作数量和资源时间估算成本，不实际计算 Q/K/V 或 logits。数值 PPL 是代码中关联的历史实验参考值。测试通过不表示数值 attention、随机相关性或物理器件行为通过验证。

## 模块对应关系

| 部分 | 当前 TLM | 已有 attention 代码／之前的硬件设计 | 判断与接入动作 |
|---|---|---|---|
| 模型尺寸 | 24 层、hidden 896、FFN 4864、14 Q heads、2 KV heads、head dim 64 | Qwen2.5-0.5B 相同尺寸 | 一致；不能套到 Qwen3.5 的混合 attention 架构 |
| Q/K/V/O、RoPE、RMSNorm、残差 | 数字矩阵、向量和非线性成本代理 | 筛选方案保留原投影及 BF16 接口；Ising 对照允许准确 Q/K | 外层位置一致；具体吞吐、精度和操作单价仍须标定；随机 Q/K 投影另作独立开关 |
| KV cache | 每步写新 K/V；按 128-token tile 扫描全上下文，K 和 V 合并读取 | 筛选先读摘要，候选确定后取 K；采样得到地址后取 V | 需拆开 K/V 事务，支持候选地址与采样地址；保留完整 KV 数据 |
| GQA | 将全量 KV 按 2 KV heads 读一次，QK/PV 计算按 14 Q heads 计数 | 每 7 个 Q heads 共享一组 KV，但其筛选和采样地址不同 | 维度和全量计数一致；稀疏方案需按组做地址 union、保留 per-head mask；还须计广播代价 |
| QK 筛选 | 无 | 每 64-token block 分 4 个子块，post-RoPE K 的 FP32 sum/mean；四个 q·mean 取最大值；free blocks 上标准化、校准 sigmoid；首块及最后两块强制保留 | 增加摘要 SRAM、增量更新、评分、标准化、p-bit 偏置、mask 和候选队列 |
| QK 点积 | 计算全上下文 QK | 筛选代码 gather 候选 K 后确定性点积；Ising 实验本身仍全量点积 | 增加候选输入；不能因添加 Ising 就同时减少 QK 计算 |
| softmax | 数字 online softmax 的宏级动作代理 | 筛选分支仍用准确 softmax；Ising 分支通过 reference/δ 与互斥随机过程采样 | 需单独实现 Ising 成本与时序模块；不能把 exp 的单价换成 p-bit 单价即视为接入 |
| PV | 全量 weighted PV MAC | 筛选分支可做准确 PV 或 512 次 IID V-gather/平均；Ising GPU 模拟为 dense counts@V，硬件提议为地址选择、累加和平均 | 增加事件 FIFO、去重/重放策略、V 事务、加法器、计数与平均；真实去重和缓存容量决定访存 |
| 随机源 | 集中式 16-bit 均匀随机字源，FFN 输入 12-bit 比较，双随机权重需独立随机字；输入流跨输出复用 | 本次 attention 假设外部可编程 tanh p-bit 阵列 | 两种接口不同，应并存为不同契约，不能共享未核验的位率、能量或时序假设 |
| 数值质量 | BF16、INT8 FFN、双随机 Sz 的历史完整 WT2 参考；fixed_sc_exact 无独立 PPL | 联合 attention 是另一个 FFN 版本、后缀 PPL 协议；筛选仅 decode，Ising 实验也覆盖 prefill | 不继承 PPL；为每个集成配置单独绑定模型、FFN/attention 版本、精度、种子、计分协议 |

目前筛选分支与 Ising 分支是两个独立实现。`筛选 → 候选 QK → Ising → V 累加` 是可研究的组合方案，并非已有联合代码或结果已经证实的配置。

## p-bit 接口必须明确的区别

用户指定“接口尚未确定，先按可编程阵列设计”。沿用约定：`m ∈ {-1,+1}`，`E[m|u]=tanh(u)`，`b=(m+1)/2`，则 `P(b=1|u)=sigmoid(2u)`。因此 sigmoid(a) 对应驱动 **u=a/2**。驱动范围、每通道 gain/offset、配置带宽、settling、读出、反馈延迟、相关性和能量都应是参数；真实器件若采用其他 tanh 定义，需要重新映射。

筛选的驱动是：

`u = ((score - mean_free)/max(std_free,0.01) + bias_cal)/(2*temperature_cal)`。

Ising 的有限耦合原型采用：

`delta_i = score_i - score_ref; C = sum(b); u_i = (delta_i - lambda*(C-b_i))/2`。

reference 由 all-zero 状态表示，single-active 状态表示其他候选。`C>1` 是非法状态，不能简单 priority encode 后当作正确样本。常数抑制耦合可用 C 反馈，无须保存完整 J 矩阵；反馈传播、计数更新与自项排除仍有硬件成本。

已发表在仓库里的 LLM Ising 仿真是理想强互斥 CTMC。`burn=16`、`spacing=4`、512 次观测的时间单位是其更新过程的单位时间，**不能直接解释为 100MHz 下的 16/4 个周期**。静态 tanh 响应不保证相同转移率。受控逐位 Gibbs 原型与理想 CTMC 的动态不同，须重测混合时间、ESS、有限样本质量和非法状态率。

TLM 顶层 `samples=4096` 当前服务 FFN；接入时须单独增加 `attention_samples=512`、`burn`、`spacing` 和时间单位/更新调度配置。FFN 的 8 个 64×64 engines 也不自动意味着有足够的 Ising 通道或并行 attention heads。

## 已复现检查与发现

在本地精简源码快照中执行，不需要模型权重或 GPU：

- `import_evidence.py`：证据包及功耗 CSV 校验通过。
- `checks/test_model.py`：8 项通过。
- `checks/test_system_decode.py`：11 项通过。
- 重新运行 2K/8K、4 种变体共 8 个基础场景，所有 metrics 与发布 JSON 一致。
- 源文件字节、SHA256 和测试日志已保存在同目录；详见 `source_manifest.json`、`verification.json`、`checks/`。

这些结果支持现有动作账本和基线结果可复现；没有重新执行完整 96 项扫描，也没有把其发布审计误报成本次新执行的验证。

**需要修正：数字 attention 的 scratch 带宽没有约束时延。** `attention()` 计算 scratch K/V buffer 的读写能量与字节数，但没有向 scratch 资源预约读写时间；`vectors()` 和 `digital_matrix()` 的局部接口流量也主要只计能量。只有随机 FFN replay 等路径使用了 scratch 带宽，而且没有形成统一的争用资源。

隔离一个 8K 单层 `attention(0,8192,0)`：

| scratch 带宽 | 已记 scratch K/V 流量 | attention 完成时间 |
|---|---:|---:|
| 64 GB/s | 8,388,608 B | 468,048 ns |
| 0.064 GB/s | 8,388,608 B | 468,048 ns |

在当前“这部分流量经过配置的 scratch 端口”的账本假设下，64 GB/s 下仅服务这些字节需要约 0.131 ms，0.064 GB/s 下需要约 131.072 ms，而模型仍输出 0.468 ms。若实际体系采用独立旁路或专用双口 K/V buffer，应显式建立其端口、容量和带宽，不能以通用 scratch 计费后省略服务时间。结论只针对所声明的资源模型，不是实测硬件性能。

**随机供给也只达到宏级服务时间。** `system_decode.py:rng_service()` 通过 bits/rate 预约均匀源服务，不包含微周期 FIFO 水位、分发仲裁和 CDC。旧 `model.py` 工作台的有限 RNG FIFO 没有直接成为系统 decode 的 FIFO 模型。总位数不超名义供给是必要检查，不能证明每个消费周期都有足量且满足相关性要求的随机位。

**摘要内存须另加。** 8K、block64、4 个 FP32 K sums 的容量为 256 KiB/层，24 层合计 6 MiB，外加计数、mask、score、索引、采样状态等。现有 1 MiB scratch 未包含这 6 MiB。可用专用 SRAM 或外存/缓存，但容量、访问、漏电、面积必须进入模型；摘要可在 prefill 流式建立，但当前 decode-only 模型尚未计其初始化。

## 现有参数模型中，attention 改动的整体上限

下表为本次复现的参数模型结果，不是芯片测量。attention 核心指 `QK_softmax_PV` 阶段，包括该阶段的 KV 读取等待；保持投影、FFN、lm_head 等阶段不变。假设这个阶段的时间降为零，得到该配置下的理想整体加速上限；实际新模块会增加摘要、采样和接口开销。

| 配置 | context | 全 token ms | attention 核心 ms | 占比 | attention 核心零成本时整体加速 |
|---|---:|---:|---:|---:|---:|
| 数字 BF16 | 2K | 66.637 | 2.847 | 4.27% | 1.045× |
| 数字 BF16 | 8K | 75.023 | 11.234 | 14.97% | 1.176× |
| 数字 INT8 FFN | 8K | 54.854 | 11.234 | 20.48% | 1.258× |
| 固定权重 SC | 8K | 494.738 | 11.234 | 2.27% | 1.023× |
| 双随机 Sz | 8K | 45,062.518 | 11.234 | 0.025% | 1.00025× |

这些上限会随资源参数、缓存方案和修正后的调度改变。它们说明当前默认随机 FFN/RNG 会掩盖 attention 的收益，不能用该默认整机配置直接判断 Ising 模块是否有价值。建议同时报告 attention 子系统、数字 FFN＋概率 attention 整机，以及随机 FFN＋概率 attention 整机。

## 建议接入顺序与接口契约

1. **先补资源时间。** 为 scratch、K/V streaming buffer、摘要 SRAM、状态 SRAM 建立明确端口与带宽；将数字向量、矩阵接口和 attention 读写都纳入预约。保留 overlap，但不能忽略资源冲突；增加可观察的 read/write/stall 计数。为 context 明确“本次 attention 看到的含当前 token 长度”还是“原有 KV 长度”，统一新增 K/V 与尾块索引。
2. **先对接两个独立算法。** 保留 dense 基线；分别添加 screen+accurate softmax/PV、screen+IID512、full QK+Ising512 三个模式。QK/KV/FFN 精度固定。screen+Ising 作为后续新组合，独立验证。
3. **把可编程 p-bit 当作模块。** 建立 configure/bias、settle、sample、state feedback、valid/readout 事务；参数包含通道数、可并行行数、更新顺序/动力学、反馈带宽和时间、偏置范围、相关性、动作能量。均匀字源与原生概率输出分开配置，按实际接口计费。
4. **用真实地址 trace 验证收益。** trace 至少包含 token/layer/head/KV-group、K block mask、global 候选索引、512 个带重复的 V 地址及其时间。按组 union K、按实际缓存策略处理 V 去重，并逐次计 burst。不能把约 30% 的单 head 保留率直接乘到整机 KV 字节数上。
5. **绑定数值与时序配置。** 先检查 63/64/65、127/128/129 等尾块、首尾必选、GQA mask、参考/空态读出和有限耦合非法态。随后使用相同权重、固定精度和配对种子验证 attention 均值/误差、PPL、ESS，再标定吞吐和能量。不同 FFN 编码、warmup、间隔及采样数应各有质量证据。

建议新增接口的事务拆分如下；这里只规定对接关系，没有实现或声称完成新硬件：

`K_append → summary_update → score_max4 → free_mean_std → configure/sample_screen → candidate_address_queue → K_read_union → QK → [digital_softmax+PV / digital_softmax+IID / Ising_feedback] → sample_index_FIFO → V_read → accumulate/normalize → O_projection`。

对于 full QK+Ising 模式，绕过 summary/screen；对于 dense baseline，沿用现有数字流水。这样能够逐项核验算法一致性，也能够从时间账本中辨认筛选、Ising 和 V 读出各自的贡献。
