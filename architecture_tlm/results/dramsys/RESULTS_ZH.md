# DRAMSys 接入与 Qwen2.5-0.5B 对照结果

**已把外部内存换成真实 DRAMSys/SystemC 时序仿真，并让完成时间反馈给计算调度。**计算阵列、片上 SRAM、随机源仍采用原来的宏级参数。这里只改变硬件时间与能耗模型，不重新运行模型数值推理，不产生新的 PPL。

## 一眼看懂的汇总

下面采用 DRAMSys 的 NoPowerDown 配置；全部是 batch=1、完整24层和 lm_head、连续生成2个 token 的平均值。时间单位 ms/token，能量单位 mJ/token。数字方案也运行在同一套假想 ASIC 上，**不是 GPU 实测结果**。

| 上下文 | 方案 | 原简化模型时间 | DRAMSys 时间 | 原简化模型总能量 | DRAMSys 总能量 | 其中 DRAM 能量 |
|---|---|---:|---:|---:|---:|---:|
| 2K | 数字 BF16 | 66.64 | 70.39 | 77.56 | 156.66 | 147.10 |
| 2K | 数字 INT8 权重 FFN | 46.47 | 49.04 | 53.95 | 108.62 | 101.63 |
| 2K | 固定权重＋随机输入＋精确计数 | 486.35 | 488.00 | 175.99 | 545.29 | 438.28 |
| 2K | 双随机＋Sz＋精确计数 | 45054.13 | 45056.12 | 6477.12 | 38471.91 | 34292.14 |
| 8K | 数字 BF16 | 75.02 | 78.76 | 84.14 | 171.56 | 160.66 |
| 8K | 数字 INT8 权重 FFN | 54.85 | 57.39 | 60.53 | 123.50 | 115.18 |
| 8K | 固定权重＋随机输入＋精确计数 | 494.74 | 496.38 | 182.65 | 560.32 | 451.90 |
| 8K | 双随机＋Sz＋精确计数 | 45062.52 | 45064.47 | 6483.78 | 38486.37 | 34305.19 |

**主要判断：外存细化没有改变随机FFN的时间瓶颈。**2K固定权重随机FFN时间变化 0.34%，双随机变化 0.004%。前者主要受采样/阵列服务时间限制，后者仍主要受现有全局均匀随机字源供给限制；不能把后者作为原生概率输出p-bit的性能结论。

**能量估算变高，主要是原来的外存背景功耗假设偏低。**DRAM工作/等待/刷新按实际命令与状态计费。长时间计算期间也要保持外存；减少计算时间、采用合适的内存低功耗策略、减少权重反复外存搬运，都可能降低这项成本。这里没有因为接入DRAMSys而改变FFN的采样数、并行度或计算成本。

## 把内存规格差异与详细时序差异分开

旧模型假设外存 32 GB/s；本次双通道 DDR4-1866 根据 tCK=1.072ns 的理论峰值为 29.850746 GB/s。因此另算一组相同峰值带宽的简化模型。下面最后一列才更接近“增加 bank、刷新和控制器时序后”的影响；仍包含旧固定80ns延迟与实际延迟的差异。

| 上下文 | 方案 | 同带宽简化时间 ms/token | DRAMSys 时间 ms/token | DRAMSys / 同带宽简化 |
|---|---|---:|---:|---:|
| 2K | 数字 BF16 | 68.86 | 70.39 | 1.022 |
| 2K | 数字 INT8 权重 FFN | 47.99 | 49.04 | 1.022 |
| 2K | 固定权重＋随机输入＋精确计数 | 487.33 | 488.00 | 1.001 |
| 2K | 双随机＋Sz＋精确计数 | 45055.11 | 45056.12 | 1.000 |
| 8K | 数字 BF16 | 77.25 | 78.76 | 1.020 |
| 8K | 数字 INT8 权重 FFN | 56.37 | 57.39 | 1.018 |
| 8K | 固定权重＋随机输入＋精确计数 | 495.71 | 496.38 | 1.001 |
| 8K | 双随机＋Sz＋精确计数 | 45063.49 | 45064.47 | 1.000 |

## 地址布局检查

最初诊断配置把row bit 11放在较低位置，导致连续读取也频繁切换行：2K BF16平均每次ACT仅约1次读写，估算358.83mJ/token。该批任务已停止/归档；它不属于上面的最终12项结果。最终映射把完整列地址放在行地址之前，具体为column=[3,4,5,11..17]、row=[18..32]，并新增对应检查。

| 2K 方案 | 每次 ACT 平均读写次数（越大表示行复用越充分） |
|---|---:|
| 数字 BF16 | 23.71 |
| 数字 INT8 权重 FFN | 24.86 |
| 固定权重＋随机输入＋精确计数 | 17.98 |
| 双随机＋Sz＋精确计数 | 18.03 |

## 等待期间的外存功耗不能忽略

原简化模型把外存背景功耗固定为0.05W。DRAMPower 使用具体芯片数、工作状态和刷新命令计算；长时间随机计算等待时，DRAM仍会消耗能量。下面对比低功耗策略，不能把某一策略的结果直接推广到所有真实DDR4器件。

| 2K 方案 | NoPowerDown 时间 ms/token | Staggered 时间 ms/token | NoPowerDown DRAM mJ/token | Staggered DRAM mJ/token |
|---|---:|---:|---:|---:|
| 数字 BF16 | 70.39 | 70.42 | 147.10 | 162.43 |
| 数字 INT8 权重 FFN | 49.04 | 49.03 | 101.63 | 112.29 |
| 固定权重＋随机输入＋精确计数 | 488.00 | 488.94 | 438.28 | 363.14 |
| 双随机＋Sz＋精确计数 | 45056.12 | 45056.72 | 34292.14 | 22492.44 |

本配置中，Staggered降低了长空闲随机FFN情景的能耗，但数字路径没有节能。差异可能来自状态切换、行关闭/重开、唤醒和电流参数；示例中的VPP power-down电流系数尚未核验，不能仅凭本表判断目标器件的低功耗能力。

## 这次具体接入了什么

- 持续运行的 C++/SystemC 进程；Python 每次提出真实地址、读写方向和长度，C++返回实际响应时间。
- 非阻塞四阶段 TLM（BEGIN_REQ/END_REQ/BEGIN_RESP/END_RESP），未使用 DRAMSys 的固定延迟 blocking 接口。
- 每个 DMA 拆成64字节请求；最多32笔请求同时未完成，DMA提交前端1GHz。这个频率不同于100MHz随机计算阵列频率。
- 单个宏DMA内部可以并行，但当前DMA全部返回后才返回给Python；宏DMA之间保持原双缓冲调度。尚未模拟任意多个DMA引擎跨任务并发。
- 为 tied embedding/lm_head、每层 QKV/Wo/FFN、每层KV建立不重叠地址。KV采用 token-major K/V 交错布局；这是设计选择，不是声称与GPU物理缓存布局相同。
- 每层注意力仍为原有完整准确 QK、softmax、PV；随机FFN路径不变。
- 地址采用64B通道交错和128B bank交错。权重外部线性布局在SC路径对应原64×64块顺序。
- DRAMPower在整个decode结束时刻计算核心和接口能量，包含计算尾部的等待时间；原来每字节能耗与0.05W背景项被移除，避免双计。

## 结果可信度和当前边界

DRAMSys固定提交9f15aaf1fe6f05ff16a2e521a20a96f3bfe29d52；SystemC3.0.2、DRAMPower6.3.1、DRAMUtils1.16.0。公开版本保留配置、源码哈希、实际请求计数和汇总结果；完整日志、内部路径、作业标识和逐笔DMA记录仅保留在本地。公开模型只使用Qwen公开形状和合成地址，不包含权重、文本输入或KV实际内容。

**时序来自真实DRAM命令模型；能耗仍是参数估计。**采用官方 MICRON_4Gb_DDR4-1866_8bit_A 示例，2个通道，每通道8颗x8器件，合计8GiB。配置中的vddq=0，接口能量因此不能用作可信结论；阻抗参数也属于示例。DRAM核心能量使用该文件的电流系数，尚未用目标器件数据表核验，包括power-down电流。

新总能量=原来的片上计算/存储/随机源估算（按新时长计费）＋DRAMPower外存核心能量＋示例接口能量。不是板卡实测，不包含内存控制器/PHY完整ASIC面积功耗、主机、CTS、封装；未做prefill/TTFT。

PPL不随外存时序模型改变。固定权重随机FFN的这套精确硬件路径仍缺少独立匹配的数值PPL；双随机Sz引用之前的数值实验，并不证明本次硬件随机相关性已验证。

## 复现与文件

- `dramsys/CMakeLists.txt` 和 `bridge.cpp`：真实库接入。CMake自动固定拉取上述DRAMSys提交；也可设置DRAMSYS_SOURCE使用已下载源码。
- `dramsys/ddr4_2ch_*.json`：两种低功耗配置；`provenance.json`：配置来源与带宽。
- `dramsys_backend.py`：地址分配、Python/C++协议、完成时间反馈、能量替换。
- `run_dramsys.py`：12项对照。`dramsys/build.sbatch`、`smoke.sbatch`、`run.sbatch`：CPU Slurm作业。
- `results/dramsys/*.json`：去除内部运行元数据后的结果；运行代码会重新生成本地DMA请求记录和DRAMSys日志，公开分支不附原始记录。
- `summary.csv` / `summary.json`：总括数据；`verification.json`：12项逐项审计；`integration_smoke.json`：实际SystemC集成测试。

普通联网Linux编译：
```bash
cmake -S architecture_tlm/dramsys -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j 4
python3 -m unittest discover -s architecture_tlm/checks -p 'test_*.py'
python3 architecture_tlm/dramsys/smoke.py --bridge build/pbit_dram_bridge
python3 architecture_tlm/run_dramsys.py --bridge build/pbit_dram_bridge --index 0
python3 architecture_tlm/report_dramsys.py
```

最后一条汇总要求index0..11全部完成。Slurm脚本的partition按实际集群修改；不申请GPU。原简化模型保留作对照；新的主实验入口是run_dramsys.py。

参考：[DRAMSys](https://github.com/tukl-msd/DRAMSys)、[DRAMPower](https://github.com/tukl-msd/DRAMPower)。配置取自DRAMSys BSD-3-Clause示例，版权归RPTU及原作者，见dramsys/UPSTREAM_LICENSE.txt。
