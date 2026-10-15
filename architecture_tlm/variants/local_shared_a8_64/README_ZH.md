FFN 本地 PRNG、8位随机字、64路共享：独立 TLM v2
日期：2026-10-15

入口与结果
  新调度：system_decode_local_shared_v2.py
  新随机源宏模型：rng_local_shared_v2.py
  位流参考：local_shared_reference_v2.py
  参数：config/local_shared_v2.json
  运行：run_local_shared_v2.py
  检查：tests/test_local_shared_v2.py、audit_local_shared_v2.py
  报告：results/RESULTS_ZH.md

旧 system_decode.py / rng_hierarchy.py / rng_reference.py 默认行为保留。
新版本通过继承复用普通LLM、权重宏、外存及权重随机事件生成算法，不覆盖历史结果。

运行（标准库，无GPU）：
  python architecture_tlm/variants/local_shared_a8_64/tests/test_local_shared_v2.py
  python architecture_tlm/variants/local_shared_a8_64/run_local_shared_v2.py
  python architecture_tlm/variants/local_shared_a8_64/audit_local_shared_v2.py

默认18项：两种权重方案，各自旧A12、旧A8，以及新版7种输入/播种配置。
全部使用相同形状、T4096、100MHz、2K上下文、8阵列、2token、简化Link外存。
小规模调试请使用不同目录：
  python architecture_tlm/variants/local_shared_a8_64/run_local_shared_v2.py --small --output architecture_tlm/variants/local_shared_a8_64/results/small

也可使用已编译的DRAMSys桥：
  python architecture_tlm/variants/local_shared_a8_64/run_local_shared_v2.py --bridge /path/to/pbit_dram_bridge --output /path/to/new_results
本轮新版本未重跑DRAMSys，报告不把简化Link结果写成DRAMPower或实际硬件测量。

一、8位的准确含义
输入阈值q=0..255，随机字R=0..255，bit=(R<q)，概率q/256。
64个输入比较同一个R；它们的q不同，不是只保留一个概率值。
原v1输入使用12位随机字，新增old_A8将位宽影响与共享/局部化影响分开。
权重仍为原来的8位候选字，经<254、除2转换成0..126，实现|code|/127。
没有把输入和权重的随机字混用；没有OR近似，也没有改变注意力、SiLU、残差等算法。

二、硬件资源与“没有额外概率寄存器”
一个FFN同步阵列簇前端放一套本地32位LFSR（8位输出），生成同字并广播到64个比较器。
比较后输入bit行广播给本波多个输出阵列。保留一份已有输入阈值/符号银行，不再为
本地PRNG增设概率银行，也不再按每个输出阵列重复复制前端。
逻辑阈值+符号共64*(8+1)=576bit；功耗保守沿用一份原13位概率+符号+比较前端，
未因位宽下降或时钟门控额外索取收益。面积中前端删减信用默认0，待工艺标定再抵扣。

PRNG有32bit工作状态和32bit检查点，包含在本地模块面积参数中。
计算时随机字随用随生；不缓存T拍原始随机字。每波恢复同一个检查点，再生相同
输入序列。恢复不会重新获取物理熵，也不会把全部权重源复位。
输入阈值/符号在该输入tile的所有输出波之间保持不变。
源码位流参考使用既有rng32x64默认端口0的8个抽头，每次消费后执行16步反馈。
它与既有RTL参考向量核对过；不是完整LLM PPL验证或新的LFSR工艺综合结果。

三、三种不同的种子操作
1) 根刷新：默认每1秒按需获取256bit调理种子，扩展并分发。
   冷启动、熵收集/调理、扩展、总线传输、权重核预热、等待与能量全部收费。
   只在当前输入tile完成后刷新，避免破坏重放；空闲过期不补做所有历史刷新。
   间隔0表示只在启动时获取根种子。
2) 子种子：默认每个新的64输入tile，从现有根状态数字派生一个32bit输入种子。
   身份包含实际任务顺序/层/阶段/输入tile；派生和写入分别收费，不消耗新的物理熵。
   continuous只推进状态；periodic_child每1ms数字重键；均可配置。
3) 重放：跨输出波恢复同一输入tile的起始状态；只收费恢复动作。
   不能每一波换一个新输入种子，否则不再是本轮PPL测试的跨输出共享语义。

fresh_entropy情景故意在每个64输入tile都请求32bit新的调理熵，以展示其成本。
这里“每次请求”粒度是64输入tile，不是每个FFN、每个token或每个采样拍。
两token全24层共有4320个这种输入tile。此模式额外消耗138240bit调理输出；
默认数字派生模式只消耗根刷新所需的256bit（本次运行小于1秒，仅启动一次）。

四、权重侧保持的内容
原HierarchicalRng.weight_wave代码直接继承。
每阵列64列，每列4个Bivium，160bit输出；8阵列共2048个核。
核数、输出位宽、拒绝采样、独立连接事件策略、更新次数与动态单价不变。
默认同样8位输入的前后对比，权重更新和比较选择能量核对相等。
root10ms只作为根刷新频率敏感性，会增加所有源的播种/预热次数，报告单独标明。

五、统一能量口径
旧源码单列的是原始随机字生成/输送，并未明确单列最后bit-mask到每乘法单元的线成本。
为避免只向新版添加该公共成本造成不公平，本轮新旧两边都按同一系数单列它。
因此报告中的“旧版能量”比历史原始数字多这一公共项；JSON的historical_raw_metrics
保存未加公共项的旧结果，并核对与历史基线完全一致。新增公共线成本不产生虚构节能。
source->64比较器的随机字扇出则按不同源模式分别计费。

默认输入模块1pJ/更新、0.001mm²、1µW等系数未标定；这些不是新LFSR的PT结果。
共享前端也尚无实际布局布线校准。scratch仍保留1MiB，因为词表logits的峰值没有消失。
双随机SC核仍沿用原固定权重核代理；面积、能量与速度结论用于方案排序。

六、质量与验收
共14个新测试、23个旧回归检查通过；完整18项运行通过操作数、内存、能量、时间、
源代码/配置hash、播种/熵预算与权重侧保持检查。验证文件verification.json。
引用此前IID A8共享输入PPL（固定权重12.4535、双随机14.1483），明确不把它当成
实际LFSR或硬件时序验证；TLM不加载模型或计算PPL。
