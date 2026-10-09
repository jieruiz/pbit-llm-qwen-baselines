# 全24层FFN＋Attention联合替换（2026-10-09）

## 本轮结论

全24层联合替换可以完成推理和续写，但相对原模型的精度损失仍明显。
当前FFN加准确attention为12.3427；加入随机Q/K与Ising后为13.2158，
较该FFN基线增加7.07%。验证集通过的5%新增误差预算，在本轮test没有保持。
仅把Ising换回准确Softmax、保留随机Q/K和PV采样，得到13.2377，未见明显改善。
恢复准确Q/K与Softmax、只在PV采样则为12.4901，较FFN基线增加约1.19%，是本轮更稳妥的组合。

随机Q/K但Softmax/PV都准确的对照为13.0390，已增加5.64%，
提示当前组合新增损失主要来自随机Q/K投影这一组，而非Ising与IID采样方式的差别。
这包括投影量化、采样及其与FFN的相互作用，不是单独测出的纯采样误差。
Ising与IID的成对差异区间跨0，不能认定其中一个精度更好；仅PV采样相对准确attention的区间也接近并跨0。

建议暂以“当前固定尺度FFN＋准确Q/K/Softmax＋PV采样”为集成基线，Ising保留为研究分支。
FFN+Ising而Q/K准确的单seed为12.4183，值得后续复测，但尚不能替代三seed结论。
若继续扩展随机Q/K，先在独立验证集扩大流长或减少替换投影，再做新测试；本轮不按test调参。
主要组合仍比原模型高约20%—27% PPL，不能称为无损替换。

## 当前版本与范围

FFN沿用最新固定尺度版：全部24层，W8，12位幅度+符号，group_max离线固定尺度，
输入4096次、输出3072次采样，每256次更新，25%预热；保留理想sigmoid和FP32中间估计。
这不是旧训练AND版本，也不是需要在线最大值的动态尺度高精度版本。
残差、RMSNorm与BF16接口保持原模型路径，没有训练。

随机Q/K指固定q_proj/k_proj投影：W10、9位幅度+符号、固定二次幂尺度，选择T=256。
QK点积仍确定性，V/O投影不变。Ising为理想强互斥连续时间热浴，
预热16、读取间隔4、512次观测；准确Softmax回退也采样512个类别位置。
Ising直接从score差的sigmoid速率生成驻留轨迹，并按固定时间观测，没有用softmax概率生成其输出。
这不是有限惩罚、有限延迟硬件。时间单位也不能与Q/K采样周期直接相加。

**预填充与解码都实际启用替换，覆盖24层。** KV缓存来自替换后的前缀。
这比以往只在decode替换attention、保持精确prefill的协议更严格。

## 匹配测试结果

测试为WT2 test的32段相同文本，每段context2048、128步teacher forcing，
合计4096计分token。不是完整语料滑窗PPL，不能与11.652735或13.873066直接横比。
本轮原模型对应PPL为10.381261。

| 方案 | seed数 | PPL均值±总体标准差 | 相对原模型 | 相对当前FFN＋准确dense |
|---|---:|---:|---:|---:|
| 原模型SDPA | 1 | 10.381261 ± 0.000000 | +0.00% | -15.89% |
| 原模型手写dense | 1 | 10.368166 ± 0.000000 | -0.13% | -16.00% |
| 仅Ising＋V采样，原FFN/投影 | 1 | 10.441393 ± 0.000000 | +0.58% | -15.40% |
| 当前FFN＋原attention | 3 | 12.274026 ± 0.067247 | +18.23% | -0.56% |
| 当前FFN＋手写准确attention | 3 | 12.342713 ± 0.030084 | +18.89% | +0.00% |
| 当前FFN＋随机Q/K＋准确Softmax/PV | 3 | 13.038964 ± 0.096486 | +25.60% | +5.64% |
| 当前FFN＋Ising，无随机Q/K | 1 | 12.418348 ± 0.000000 | +19.62% | +0.61% |
| A：当前FFN＋随机Q/K＋Ising | 3 | 13.215760 ± 0.081924 | +27.30% | +7.07% |
| B：当前FFN＋随机Q/K＋准确Softmax＋PV采样 | 3 | 13.237724 ± 0.026454 | +27.52% | +7.25% |
| C：当前FFN＋准确Q/K/Softmax＋PV采样 | 3 | 12.490136 ± 0.050821 | +20.31% | +1.19% |

A/B/C三种主组合和FFN基线均复测三个seed。原FFN的Ising及固定FFN无随机Q/K的Ising是单seed诊断。
固定FFN+SDPA与固定FFN+dense均保留三seed，避免将原attention软件路径差别与FFN随机波动混淆。

验证集在train尾部65536token上8段，context512/decode64，共512计分。
FFN+dense验证PPL=19.175124，联合Ising T256/T1024为19.301743/19.401433。
预设规则以FFN+dense新增误差5%为探索预算，优先较短T；本轮选择256。
回退B/C都按预定对照执行，没有用test选择流长。5%不是一般能力可接受保证，
总退化需同时看原模型基线；小验证集的轻微反向波动不构成架构更优证据。

## 成对文本差异

以下先对每段的三个seed平均NLL，再对32段做5000次成对bootstrap。
区间只描述本次文本抽样的不确定性，不包含所有模型能力/器件误差，不能当作全面显著性证明。
正数表示左侧PPL更高；区间跨0时，不应声称左/右方案确定更好。

| 对比 | 几何PPL相对变化 | 文本bootstrap95%区间 |
|---|---:|---:|
| A：当前FFN＋随机Q/K＋Ising 对 当前FFN＋手写准确attention | +7.07% | [+5.43%, +8.73%] |
| B：当前FFN＋随机Q/K＋准确Softmax＋PV采样 对 当前FFN＋手写准确attention | +7.25% | [+5.76%, +8.79%] |
| C：当前FFN＋准确Q/K/Softmax＋PV采样 对 当前FFN＋手写准确attention | +1.19% | [-0.04%, +2.35%] |
| A：当前FFN＋随机Q/K＋Ising 对 B：当前FFN＋随机Q/K＋准确Softmax＋PV采样 | -0.17% | [-1.63%, +1.22%] |
| A：当前FFN＋随机Q/K＋Ising 对 C：当前FFN＋准确Q/K/Softmax＋PV采样 | +5.81% | [+4.29%, +7.25%] |
| 当前FFN＋随机Q/K＋准确Softmax/PV 对 当前FFN＋手写准确attention | +5.64% | [+4.08%, +7.17%] |

## 逻辑访存和采样统计

每层/seed等权统计，分母仅包括因果可见位置。数值是非零采样计数对应的地址，
不是实测DRAM事务或物理KV缓存容量节省。GQA每7个Q头共享1个KV头，地址并集更大。
软件依旧生成完整QK，并使用dense counts@V模拟选择累加，未实现真实稀疏加速。

| 方案 | 单头V地址比例 | GQA组并集比例 | Ising相邻样本重复率 |
|---|---:|---:|---:|
| 仅Ising＋V采样，原FFN/投影 | 7.28% | 26.45% | 32.45% |
| 当前FFN＋Ising，无随机Q/K | 7.37% | 26.80% | 31.63% |
| A：当前FFN＋随机Q/K＋Ising | 7.32% | 26.74% | 31.14% |
| B：当前FFN＋随机Q/K＋准确Softmax＋PV采样 | 7.39% | 26.96% | — |
| C：当前FFN＋准确Q/K/Softmax＋PV采样 | 7.46% | 27.04% | — |

## GPU参考实现耗时

各seed均值，单位秒；每次处理相同32段前缀和4096个解码计分token。包含统计和诊断开销，不能换算成p-bit电路速度。

| 方案 | prefill | decode |
|---|---:|---:|
| 原模型SDPA | 0.75 | 12.03 |
| 当前FFN＋原attention | 46.47 | 198.94 |
| 当前FFN＋手写准确attention | 52.68 | 214.70 |
| 当前FFN＋随机Q/K＋准确Softmax/PV | 53.92 | 241.89 |
| A：当前FFN＋随机Q/K＋Ising | 62.96 | 259.28 |
| B：当前FFN＋随机Q/K＋准确Softmax＋PV采样 | 56.07 | 249.93 |
| C：当前FFN＋准确Q/K/Softmax＋PV采样 | 54.53 | 223.55 |

## 精度解释与局限

先比较原模型与当前FFN，再比较FFN+准确attention与A/B/C。
如果联合退化主要已经存在于FFN-only中，换回准确Softmax不能消除FFN的有限流长噪声。
Q/K随机化、概率采样和FFN会相互影响，PPL增量不应按独立误差简单相加。
即使某采样方案单seed低于准确attention，也可能只是不同随机轨迹，不代表采样优于准确算子。

固定尺度来自旧原模型校准，未针对本组合重新训练或重新校准。
FFN越界与局部量化NMSE汇总保存在summary.json；局部NMSE不是相对原模型的端到端误差。
宽累加、sigmoid、Ising速率和驻留时间均为软件参考；未测有限整数溢出、
tanh驱动/偏置电路、随机相关性、耦合与广播延迟、面积或能耗。
GPU耗时属于仿真，不是候选硬件延迟；4096/256/512也不是可直接相加的同类时钟。

## 验证和审计


12项验证、22项测试、15条续写完成。3项采样单元测试和281条源码/协议/结果审计通过。
所有24层的prefill/decode调用、FFN与投影数量、固定尺度、其他参数哈希均核对。
手写dense对原SDPA整段logit平均/最大差为0.039365/0.343750；
同一cached decode路径为0.049595/0.265625。
因果前缀不受未来token变更影响的检查通过。BF16整段与逐token路径不是逐位相等：
原模型平均差0.033144，手写图0.075775。
首次过紧的跨路径检查失败后，加入原模型同路径对照；正式评测前已重新通过所有检查。

源代码与校准SHA256、作业参数、每段NLL和token哈希均保留。
发布目录：experiments/joint_ffn_attention_20261009
此发布包包含三个联合方案、匹配基线和完整审计记录。

## 全部固定提示续写

Base模型，greedy，最多48新token；仅作定性检查，不是通用任务评测。

### original_q0_sdpa_s0 / The capital of France is

```text
 Paris. It is the largest city in Europe and the second largest in the world. It is also the capital of France, the second largest country in Europe, and the third largest country in the world. It is the seat of the French
```

### original_q0_sdpa_s0 / 人工智能可以帮助人类

```text
解决很多问题，比如在医疗领域，人工智能可以辅助医生进行诊断和治疗；在教育领域，人工智能可以为学生提供个性化的学习方案；在交通领域，人工智能可以优化交通流量，提高道路通行效率。这些
```

### original_q0_sdpa_s0 / To calculate the area of a rectangle,

```text
 multiply the length by the width. The area of a rectangle is equal to the length times the width. The area of a rectangle is equal to the length times the width. The area of a rectangle is equal to the length times the width
```

### fixed_q0_sdpa_s0 / The capital of France is

```text
 Paris. The city of Paris is the capital of France. Paris is the capital of France. The capital of France is Paris. The capital of France is Paris. The capital of France is Paris. The capital of France is Paris. The
```

### fixed_q0_sdpa_s0 / 人工智能可以帮助人类

```text
进行以下哪些活动？
A. 以上所有
B. 通过学习，我们能够更好地理解世界
C. 通过学习，我们能够更好地理解自己
D. 通过学习，我们能够更好地理解他人
答案
```

### fixed_q0_sdpa_s0 / To calculate the area of a rectangle,

```text
 one must first know the length and width. The area of a rectangle is calculated by multiplying the length by the width. For example, if the length of a rectangle is 5 and the width is 3, the area is 1
```

### fixed_q256_ising_s0 / The capital of France is

```text
 Paris, the capital of Germany is Berlin, the capital of Japan is Tokyo, and the capital of the United States is Washington DC. What is the capital of the United States?
The capital of the United States is Washington, D.C.
```

### fixed_q256_ising_s0 / 人工智能可以帮助人类

```text
的哪些方面？
人工智能（AI）已经渗透到我们生活的方方面面，从智能手机到智能家居，再到医疗保健，几乎无处不在。以下是一些人工智能在不同领域的应用及其带来的好处：

1. 工业领域：
```

### fixed_q256_ising_s0 / To calculate the area of a rectangle,

```text
 we can multiply the length by the width. If the length of a rectangle is increased by 10% and the width is decreased by 10%, then the area of the rectangle will be:
Answer Choices Why?
10%
```

### fixed_q256_iid_s0 / The capital of France is

```text
 located in the city of Paris, which is situated in the center of the country. The city of Paris is the largest city in Europe and is the capital of France. It is the second largest city in the world and the 13
```

### fixed_q256_iid_s0 / 人工智能可以帮助人类

```text
进行决策，但是否真的可以代替人类进行决策呢？
是的，人工智能可以帮助人类进行决策。人工智能可以收集和分析大量数据，通过学习和推理，可以做出比人类更准确的决策。例如，在医疗领域
```

### fixed_q256_iid_s0 / To calculate the area of a rectangle,

```text
 you multiply the length by the width. The area of a rectangle is given by the formula: Area = length × width. The length of the rectangle is 10 inches and the width is 6 inches. What is the area of
```

### fixed_q0_iid_s0 / The capital of France is

```text
 Paris. It is the seat of the government of France. Paris is the capital of France. The capital of France is Paris. It is the seat of the government of France. The capital of France is Paris. The capital of France is
```

### fixed_q0_iid_s0 / 人工智能可以帮助人类

```text
解决很多问题，比如在医疗领域，人工智能可以帮助医生进行诊断和治疗方案的制定。那么，人工智能在教育领域有什么应用呢？ 人工智能在教育领域应用广泛，比如在智能教育系统中，人工智能可以为
```

### fixed_q0_iid_s0 / To calculate the area of a rectangle,

```text
 which of the following formulas should be used?
A. Length × Width
B. Length + Width
C. Length × Width
D. Length × Width ÷ 2
Answer:
A

Which of the following statements about the
```
