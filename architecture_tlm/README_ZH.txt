Qwen随机计算系统模型：第一版事务级工作台
2026-10-14

最新进展：已新增完整24层decode宏模型，权重存储按宏参数计费，其他部分使用普通准确LLM算法。
入口请先读SYSTEM_MODEL_ZH.txt，完整总表见results/system_decode/完整decode宏模型与瓶颈分析.txt。
运行python run_system_sweep.py与python report_system.py。下面保留早期单核心工作台说明，不代表新模型的全部范围。

一、可以开始建模，但要区分已经完成的内容

已有Qwen形状、数值实验、16组旧标定、G1/G4新版功耗分项和SRAM/门控DFF功能验证，足以建立首版时延、动作与成本覆盖模型。
早期Python事务级工作台用于先核对权重驻留、SRAM事务、随机源供给和账本；新system_decode.py已扩展24层attention、外存与词表输出的参数化模型。SystemC/TLM-2.0与prefill尚未实现。
原需求中的SystemC/AT、真实注意力索引trace和完整系统调度仍是后续阶段；不把这次小工作台写成完成了全系统TLM。

二、怎样运行

在本目录运行：
  python run_workbench.py
  python checks/test_model.py

Python只使用标准库，无需GPU。import_evidence.py核对分支内附带的CSV、ZIP与冻结成本契约，不需要原工作区目录。
run_workbench.py执行12组配置，保存逐组事务/时间记录、summary.json/CSV、24组标定能量折算检查、容量核算。
Config位于model.py，可选择列数、输入帧数、SRAM bank/延迟/容量、outstanding数、随机源数和12位理想打包/16位整字策略。每份结果保存完整配置。
所有默认SRAM时序与容量都是明确假设，不是SRAM工艺宏的实测数据。

三、实际建模边界

1. 固定权重64输入点积，OR4、B1、T4096、100MHz；概率阈值和输入符号寄存器保持原结构。
2. 同一份权重处理两帧不同输入；权重加载一次，帧间不重载。
3. 对比未门控DFF、加载时开启的门控DFF，以及普通分bank SRAM读取到显式本地保持阵列。
4. SRAM每请求带地址、bank、issue与response周期；有限outstanding，数据返回后才能计算。
5. 随机源有限供给与FIFO；原始位不足时sample_valid停顿。默认源一直运行，记录生成、使用、FIFO留存和丢弃数量。
6. 一个输入随机流广播到所有输出列。默认64列情景是复制64个单输出核心，包含各自概率/符号存储；没有假设免费共享前端。
7. 保留已标定的活跃阶段成本，SRAM/ICG/本地保持、加载和stall成本缺项输出null与energy_complete=false。

本工作台的配置源直接提供常驻tile，没有把这些写请求解释为HBM访问；全模型权重搬运待接入外存/DMA。结果中的elapsed_us是工作台时间，不是TPOT。
12情景中64列功耗按单输出模块复制，是外推；真实64列共享、广播负载和布局需要另标定。
暂停时已有宏的活动会变化，但现有case0是零概率、时钟运行，不是sample_valid=0标定。本模型不把case0作为stall功耗，也不把stall能耗填零。

四、SRAM包检查结论

已经核对zip内容与本地6个关键源文件的SHA256一致。
sc_ws_weight_store.sv：权重保存9位符号/幅度，MODE2用valid位屏蔽未写地址。
sc_ws_sram_parallel.sv：一个写端口，全部64个9位词同时可见；定义SYNTHESIS时要求真实sc_ws_tech_sram_parallel绑定。
因此该行为模型表达了需要满足的并行可见接口，并不是已有普通64x9单端口SRAM宏的物理实现。
包中VALIDATION与波形审计证明门控版4097个主时钟期间权重时钟0次上升沿、权重数据0次变化；并没有DC/PT或真实SRAM的PPA结果。

建议现实的第一步采用：普通SRAM保留模型权重tile，读取一次到本地关闭写入/时钟的权重保持阵列，再运行4096次采样。
SRAM减少较大权重库的存储成本，本地保持避免每拍读取同一权重。576位局部保持仍存在，但计算期间不会像原未门控DFF一样持续时钟翻转。
若真正希望取消本地保持，应采用分布式本地存储或定制并行可见CIM宏，再标定连续输出、读选择、漏电和外围；不能把普通单端口SRAM直接接上64路权重并行输入。

五、与最新Sz实验的关系

这次电路是固定多位权重×随机输入、OR4归约。最新Sz实验是权重和输入双随机、准确空间计数，语义不同。
不能把13.95等Sz PPL配到OR4功耗上，也不能用本包SRAM替换结果证明双随机的能效。新Sz路径应另加权重概率驱动/比较器、随机流及准确归约的成本。
当前工作台没有用OR4质量结果去代表G1精确结构，也没有继承联合attention实验的质量。

六、继续建设完整系统模型需要什么

已有即可加入：模型shape/tile展开、24层权重总容量、已标定核心运行成本、随机源参数扫描。
需要新标定：同接口baseline/gated DFF、真实SRAM与本地保持宏、加载/保持/stall、非线性和编码、共享64列前端。
需要从算法导出：候选K、PV采样索引及重复次数、GQA关系与次序；否则不能声称有真实稀疏bank冲突结果。
需要设定或扫描：SRAM容量/端口、外存带宽与burst、引擎数量、数字吞吐；每个假设保留在配置中。
随后先完成数字单层decode，再加入准确softmax+PV采样、随机FFN、筛选器，最后扩展24层与prefill/lm_head。
原4096输入样本/3072产品样本的流水FFN，与新Sz完整gate/up后再down的分阶段FFN必须分别建立DAG。

SystemC/TLM的接口与并发定义参考Accellera：
https://systemc.org/overview/systemc-tlm/
外存详细模型候选Ramulator（当前在线仓库已更新，应固定版本/commit）：
https://github.com/CMU-SAFARI/ramulator2
当前未安装或集成这两项依赖，不把Python工作台描述为它们的实现。

七、主要文件

import_evidence.py：校验zip与功耗来源，生成成本契约。
configs/evidence_contract.json：G1/G4成本、旧标定、SRAM空缺参数及来源hash。
model.py：SRAM有限请求、RNG FIFO、权重保持、帧调度与部分能量账本。
run_workbench.py：12情景扫描、24组能量折算和容量核算。
checks/test_model.py：8项独立行为检查。
results/首版模型结论.txt：便于阅读的总表与下一步仿真需求。
results/summary.json、summary.csv：工作台情景结果。
results/*cols*.json：地址事务、各阶段时间、随机源守恒与遗漏成本。
