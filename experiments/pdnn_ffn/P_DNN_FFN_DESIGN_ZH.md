# Qwen2.5-0.5B 的 P-DNN FFN：逐层原理与持续设计记录

本文档是本项目 P-DNN FFN 的**唯一持续设计文档**。它说明当前代码究竟做了什么、每个计算阶段如何工作、哪些量已经二值化、训练和测试结果如何，以及后续改进应怎样记录。以后修改网络结构、训练方法、采样策略或硬件映射时，都在本文档中追加新版本，保留旧版本和旧结果，不覆盖历史结论。

当前实现版本为 **v0**，对应实现与结果提交 `40d043b`，日期为 2026-09-29。当前结论只适用于 Qwen2.5-0.5B Base 的第 12 个 decoder block 中单个 FFN 的替换实验。

第 13 节新增“入口和隐藏层都采用 p-bit”的候选方案。它处于分析阶段，尚未实现、训练或获得 PPL 结果，当前实现仍为 v0。

## 1. 实验对象和替换边界

Qwen2.5-0.5B Base 有 24 个 decoder block。每个 block 可以简化为：

```text
输入 hidden state
    │
    ├─ RMSNorm → Self-Attention → 残差相加
    │
    └─ RMSNorm → SwiGLU FFN       → 残差相加
                                      │
                                      ▼
                              下一个 decoder block
```

原始 Qwen FFN 的输入维数为 896，中间维数为 4864，输出维数回到 896。忽略 token 和 batch 维后，其计算为：

$$
g=W_{gate}x,\qquad u=W_{up}x
$$

$$
F_{Qwen}(x)=W_{down}\left(\operatorname{SiLU}(g)\odot u\right)
$$

其中 `x` 是 `post_attention_layernorm` 的连续输出，`SiLU(a)=a·sigmoid(a)`，`⊙` 表示逐元素相乘。FFN 的输出随后与 attention 后的主残差流相加。

v0 只把 **decoder layer 12 的整个 SwiGLU FFN** 换成 P-DNN student。Attention、RMSNorm、embedding、其余 23 个 FFN、LM head 和残差连接都保持原样。替换后的完整路径为：

```text
连续 x (896)
  → 连续输入投影和场 z (4864)
  → p-bit 条件均值 μ (4864)
  → N 次双极性随机采样 s^(n) ∈ {-1,+1}
  → 样本平均 s_bar (4864)
  → 连续线性读出 y (896)
  → 与主残差流相加
```

它受到 P-DNN“中间传播二值状态、最后进行连续读出/平均”思想的启发，但目前是在一个 FFN 边界内进行的功能验证，并不是把整套 LLM 变成论文中的物理 P-DNN。

## 2. v0 每个计算阶段的工作原理

下面的“层”指 P-DNN FFN 内的计算阶段。代码入口是 [`pdnn_ffn.py`](pdnn_ffn.py)。

### 阶段 0：接收 Qwen 的归一化连续输入

输入张量形状为：

```text
[batch, sequence_length, 896]
```

该张量已经经过 Qwen decoder block 的 `post_attention_layernorm`。它仍是 BF16/浮点连续值，并未二值化。这一接口很重要：当前 student 学的是“原始 FFN 的输入到原始 FFN 输出”的映射，不负责模拟前面的 attention 和 RMSNorm。

### 阶段 1：输入投影产生 p-bit 局部场

`input_proj` 是带偏置的全连接层：

$$
z=W_{in}x+b_{in},
$$

其中：

- \(x\in\mathbb{R}^{896}\)；
- \(W_{in}\in\mathbb{R}^{4864\times896}\)；
- \(b_{in}\in\mathbb{R}^{4864}\)；
- \(z\in\mathbb{R}^{4864}\)。

`z` 可以理解为 4864 个 p-bit 的连续输入场。当前连接是稠密的：每个 p-bit 都接收全部 896 个输入维度的加权和。它还不满足固定扇入、稀疏耦合或局部互连等硬件约束。

### 阶段 2：把连续场映射为条件均值

当前使用双极性编码 `coding="bipolar"`。每个 p-bit 的条件均值为：

$$
\mu_i=\tanh\left(\frac{z_i}{T}\right),\qquad \mu_i\in[-1,1].
$$

v0 的温度 \(T=1\)。条件概率由均值决定：

$$
P(s_i=+1\mid z_i)=\frac{1+\mu_i}{2},\qquad
P(s_i=-1\mid z_i)=\frac{1-\mu_i}{2}.
$$

因此 \(E[s_i\mid z_i]=\mu_i\)。这里的 `tanh` 不是额外保留的 ReLU 或 SiLU；它是双极性随机 p-bit 的概率参数化。代码也支持 0/1 编码，其条件均值改为 `sigmoid(z/T)`，但本轮训练和评估只使用了 -1/+1 编码。

当 `sample_count=0` 时，代码直接传播 \(\mu\)。这称为 **conditional-mean 模式**，是无采样噪声的确定性上限测试，并不代表真实 p-bit 在一次时钟更新中的二值状态。

### 阶段 3：生成双极性 p-bit 状态

当样本数 \(N>0\) 时，对每个 token、每个隐藏单元和每次采样分别生成均匀随机数，并按上述概率得到：

$$
s_i^{(n)}\in\{-1,+1\},\qquad n=1,\ldots,N.
$$

单次采样时，真正传给下一步的是严格的 -1/+1 状态。当前软件实现使用 PyTorch 伪随机数，各 p-bit 在给定 `z` 后独立采样。它没有模拟物理器件的翻转时间、时钟、空间相关性、随机源偏差、有限精度场或器件失配。

### 阶段 4：对 N 条随机路径求平均

代码累加 N 次二值状态，再计算：

$$
\bar{s}=\frac{1}{N}\sum_{n=1}^{N}s^{(n)}.
$$

因此：

- `N=1` 时，`s_bar` 的每个元素仍严格为 -1 或 +1；
- `N>1` 时，单条路径内部仍是二值传播，但路径平均值是多级离散数；
- `N→∞` 时，按大数定律 `s_bar` 接近条件均值 `μ`。

P-DNN 原文强调多条二值随机路径在末端平均。当前 FFN 只有一个随机隐藏层，后面紧跟线性读出，所以有严格等价关系：

$$
W_{out}\left(\frac{1}{N}\sum_ns^{(n)}\right)+b_{out}
=\frac{1}{N}\sum_n\left(W_{out}s^{(n)}+b_{out}\right).
$$

因此代码先平均隐藏状态、再做一次输出矩阵乘法，与 N 条路径各自做输出读出后再平均在数学上相同，可以减少 GPU 计算量。若以后加入多个随机层和层间非线性，就不能在中间提前平均，否则会改变网络函数；那时必须明确保留每条随机路径，直到规定的读出位置再平均。

### 阶段 5：连续线性读出

`output_proj` 把 4864 维随机平均状态投影回 Qwen 的 896 维残差空间：

$$
y=W_{out}\bar{s}+b_{out},
$$

其中 \(W_{out}\in\mathbb{R}^{896\times4864}\)。这一层保持连续权重和连续输出。它承担两项作用：把 p-bit population 解码为有符号连续 FFN 更新量，并恢复 Qwen block 所需的 896 维接口。

所以 v0 的准确描述是“**连续输入投影 + 一个双极性随机隐藏层 + 连续线性读出**”。它并不是权重二值网络，也不是所有层间信号都为 1 bit。训练时参数有浮点 master copy，推理时矩阵计算按模型运行精度完成。

### 阶段 6：回接 Qwen 残差流

P-DNN student 返回 `y` 后，Qwen decoder layer 执行原有残差相加：

$$
h_{out}=h_{attention}+y.
$$

这里不再添加激活函数。下一个 decoder block 会按原结构对 `h_out` 做 RMSNorm 和 attention。残差流保持连续值，这也是目前能够只替换一个 FFN 而无需改动整个 Qwen 执行图的关键接口。

## 3. 前向与反向传播并不完全相同

采样不可微。v0 在 sample-aware 训练阶段使用 straight-through estimator（STE）：

```python
hidden = mean + (hard_average - mean).detach()
```

前向计算的数值严格等于 `hard_average`，所以训练时确实看到随机采样误差；反向计算则把采样节点近似为条件均值函数，通过 `tanh(z/T)` 的导数传梯度：

$$
\frac{\partial\mu}{\partial z}=\frac{1}{T}\left(1-\tanh^2(z/T)\right).
$$

这是有偏的替代梯度，不是离散随机变量期望损失的精确梯度。场绝对值过大时 `tanh` 饱和，梯度也会变小。后续若调整温度、加入场归一化或采用其他随机梯度估计器，应在新版本中同时报告训练稳定性和最终 PPL。

## 4. 参数量与连接度

| 部分 | 形状 | 参数量 | 连接特点 |
| --- | --- | ---: | --- |
| 输入投影权重 | 4864 × 896 | 4,358,144 | 稠密，每个 p-bit 连接 896 个输入 |
| 输入投影偏置 | 4864 | 4,864 | 每个 p-bit 一个偏置场 |
| 输出投影权重 | 896 × 4864 | 4,358,144 | 稠密，每个输出连接全部 4864 个 p-bit |
| 输出投影偏置 | 896 | 896 | 每个输出维一个偏置 |
| **v0 student 合计** |  | **8,722,048** | 两个连续仿射层，一个随机隐藏层 |

原始 Qwen SwiGLU FFN 有三个无偏置的矩阵 `gate_proj`、`up_proj`、`down_proj`，共 13,074,432 个参数。v0 减少 4,352,384 个参数，即单个 FFN 减少约 33.29%；相对 494,032,768 参数的完整模型约减少 0.88%。这个参数下降不等同于硬件能耗或延迟下降，因为当前两个投影仍是稠密浮点矩阵乘法，且多次采样还会增加随机状态生成成本。

## 5. 训练方法

训练脚本是 [`train_layer_distillation.py`](train_layer_distillation.py)。Teacher 为冻结的 Qwen2.5-0.5B Base。脚本在 decoder layer 12 的原始 MLP 上注册 forward hook，记录：

- teacher MLP 的输入 `x`：即该 block 归一化后的 FFN 输入；
- teacher MLP 的输出 `y_teacher`：即原始 SwiGLU FFN 的连续输出。

Student 只学习局部映射 `x → y_teacher`。目标损失为归一化 MSE 与余弦方向损失之和：

$$
L=\frac{\operatorname{MSE}(y_{student},y_{teacher})}
{\operatorname{mean}(y_{teacher}^2)+\epsilon}
+0.05\left(1-\cos(y_{student},y_{teacher})\right).
$$

训练分两阶段：

| 阶段 | 前向模式 | 步数 | 学习率 | 目的 |
| --- | --- | ---: | ---: | --- |
| Phase 1 | conditional mean，`N=0` | 2,000 | 3e-4 | 先学习无采样噪声的函数近似 |
| Phase 2 | 随机采样，`N=4`，STE | 2,000 | 1e-4 | 让参数适应有限样本噪声 |

共同设置为 batch 4、sequence length 256、AdamW、weight decay 0.01。训练语料为 WikiText-2 raw train，共 2,518,423 tokenizer tokens；验证保留最后 65,536 tokens。累计处理 4,096,000 token。

这属于**单层局部蒸馏**。它没有用下一 token cross-entropy、teacher logits KL 或下游任务损失端到端优化整个 LLM，因此它能回答“一个简单 P-DNN 能否近似一个 FFN”，不能直接回答“24 个 FFN 全部替换后能否保持语言能力”。

## 6. v0 完整模型结果

评估脚本 [`evaluate_perplexity.py`](evaluate_perplexity.py) 将 student 实际插回第 12 层，然后在相同的 WikiText-2 raw test 流程中计算完整模型 PPL。原模型 PPL 为 11.652735。

| Checkpoint | 推理样本数 | Seed | PPL | 相对原模型变化 |
| --- | ---: | ---: | ---: | ---: |
| 原始 Qwen | 确定性 | 0 | 11.652735 | 0.00% |
| Mean-distilled | conditional mean | 0 | 12.048572 | +3.40% |
| Mean-distilled | 1 | 0 | 13.463740 | +15.54% |
| Mean-distilled | 4 | 0 | 12.356535 | +6.04% |
| Mean-distilled | 8 | 0 | 12.205344 | +4.74% |
| Mean-distilled | 16 | 0 | 12.123609 | +4.04% |
| Sample-aware | conditional mean | 0 | 12.037538 | +3.30% |
| Sample-aware | 1 | 0 | 12.175210 | +4.48% |
| Sample-aware | 4 | 0 | 12.070811 | +3.59% |
| Sample-aware | 4 | 1 | 12.074098 | +3.62% |
| Sample-aware | 4 | 2 | 12.073038 | +3.61% |
| Sample-aware | 8 | 0 | 12.055022 | +3.45% |
| Sample-aware | 16 | 0 | 12.046654 | +3.38% |

Sample-aware、4 samples 的三次运行平均 PPL 为 12.072649，样本标准差为 0.001678。其相对原模型退化约 3.60%。同一 checkpoint 的 conditional-mean PPL 为 12.037538，所以 4 samples 相对其确定性极限只多约 0.29%；剩余误差主要来自 student 结构和局部蒸馏，而不是 4 次采样本身。

本轮使用一张 RTX 5090 32 GB。记录的训练循环耗时为 66.20 秒，峰值 allocated GPU memory 为 1,244,770,304 bytes（约 1.16 GiB）。这些数字是当前软件实验的测量值，不代表物理 p-bit 硬件性能。

完整结果和 checkpoint 哈希见 [`results/pdnn_ffn_layer12_bipolar/RESULTS.md`](../../results/pdnn_ffn_layer12_bipolar/RESULTS.md)。两个约 100 MB 的 checkpoint 未放入 Git 仓库：

| 文件 | SHA-256 |
| --- | --- |
| `student_mean.pt` | `61e5eecadf9d254ae3fc395eefcaf345ead42600859e38cbf487b74bd35215de` |
| `student_sampled.pt` | `0557b82fcccb1e7f5659eb1fd54e599fbfbe3e490db34525257768f3439933bd` |

## 7. 目前已经证明和尚未证明的内容

v0 已证明：在只替换第 12 层 FFN 时，单随机隐藏层 student 能让完整语言模型正常运行；使用 sample-aware 训练后，4 samples 已很接近该 student 的 conditional-mean 极限；PPL 退化主要来自结构近似误差。

v0 尚未证明：

- 24 个 FFN 全部替换时误差是否会逐层累积；
- 0/1 编码与 -1/+1 编码在等价校准后谁更好；
- 权重、场累加和残差流能否降低位宽；
- 固定连接度或稀疏硬件图能否保持当前精度；
- 物理 p-bit 的相关噪声、器件失配和更新时序对结果的影响；
- 下游任务准确率、生成质量、吞吐和能耗是否可接受；
- 当前稠密 GPU 仿真是否会转化为硬件速度或能耗优势。

## 8. 下一步改进顺序

1. **先降低 deterministic structure error。** conditional-mean 模式已经比原模型差 3.30%，说明简单的单层 `tanh` student 表达力不足。优先测试保留门控信息、增加小型连续旁路、分组 p-bit population，或增大/分解随机隐藏层。
2. **渐进式替换多个 FFN。** 从 1 层扩到 2、4、8、24 层，每一步都重新评估 PPL，定位误差累积最敏感的层。
3. **加入端到端蒸馏。** 在局部输出损失之外加入 teacher logits KL 和下一 token loss，让上游与下游层共同适应 P-DNN 误差。
4. **系统比较编码和温度。** 对 0/1 与 -1/+1 使用等价的中心化、缩放和参数预算，扫描固定温度、可学习全局温度和逐通道温度。
5. **逐步引入硬件约束。** 先限制 fan-in/fan-out，再加入权重与场量化、采样相关性、偏置和更新异步性；每引入一种约束都与前一版本做回归比较。
6. **扩大评估。** 除 WikiText-2 PPL 外，继续运行 ARC-Easy、HellaSwag、固定 prompts 生成一致性、不同 seeds 方差、吞吐、显存和最终硬件能耗估算。

## 9. 代码与复现实验入口

| 文件 | 作用 |
| --- | --- |
| [`pdnn_ffn.py`](pdnn_ffn.py) | P-DNN 配置、条件均值、随机采样、STE 和 checkpoint 加载 |
| [`train_layer_distillation.py`](train_layer_distillation.py) | teacher hook、两阶段局部蒸馏、验证与 checkpoint 保存 |
| [`evaluate_perplexity.py`](evaluate_perplexity.py) | 把 student 插回完整 Qwen 并计算 PPL |
| [`run_layer12_experiment.sh`](run_layer12_experiment.sh) | layer 12 训练与主评估入口 |
| [`evaluate_layer12_checkpoints.sh`](evaluate_layer12_checkpoints.sh) | 不同 checkpoint、样本数与 seed 的批量评估 |
| [`upstream/modeling_qwen2.py`](../../upstream/modeling_qwen2.py) | 固定为 Transformers v4.45.2 的 Qwen2 参考实现 |
| [`config/qwen2.5-0.5b-config.json`](../../config/qwen2.5-0.5b-config.json) | 本实验所用 Qwen2.5-0.5B 配置 |

## 10. 本文档的维护规则

以后每次改进都在本文档末尾新增一个版本小节，并更新最上方的“当前实现版本”。必须保留旧版本，不能只改公式或表格使旧结果失去上下文。每个新版本至少记录：

- 日期、Git commit、实验目的和相对上一版的唯一变量；
- 完整前向公式和哪些节点是真正的 0/1、-1/+1 或连续值；
- 替换的 decoder layer 集合、参数量、连接度和精度；
- 训练数据及哈希、训练步数、损失、训练 sample count 和 STE/梯度方法；
- 推理 sample count、全部随机 seeds 和 checkpoint SHA-256；
- 局部误差、完整 PPL、下游任务、吞吐、显存与硬件条件；
- 相对上一版的收益、代价、失败现象和下一步结论。

推荐命名：`v1`、`v2`……；若只修复实现错误，使用新版本并明确写出受影响的旧结果，不能静默覆盖。

### 后续版本追加模板

```markdown
## vN：版本名称（YYYY-MM-DD）

- Git commit：
- 相对上一版的变化：
- 实验假设：
- 替换层：
- 编码 / 温度 / 训练 samples / 推理 samples：
- 参数量与连接度：
- 数据与 checkpoint 哈希：
- 训练配置：
- 局部验证结果：
- 完整模型 PPL 与多 seed 方差：
- 下游任务和性能结果：
- 结论、局限与下一步：
```

## 11. v0 版本记录（2026-09-29）

- Git 实现与结果提交：`40d043b`
- 结构：`896 continuous → dense field → 4864 bipolar p-bits → continuous linear readout → 896`
- 替换位置：decoder layer 12 的完整 SwiGLU FFN
- 训练：2,000 mean steps + 2,000 sampled steps，sample-aware 阶段 `N=4`
- 最佳当前工作点：sample-aware checkpoint，推理 `N=4`
- 完整模型结果：PPL `12.072649 ± 0.001678`（3 seeds），原模型 `11.652735`
- 主要结论：4-sample 随机误差已经较小，下一版应优先降低 conditional-mean student 的结构误差。

## 12. 参考资料

- [P-DNN 主论文（Nature，文章编号 s44335-026-00063-7）](https://www.nature.com/articles/s44335-026-00063-7)；本地补充材料文件为 `44335_2026_63_MOESM1_ESM.pdf`。
- [相关 arXiv 版本 2605.01910](https://arxiv.org/abs/2605.01910)。
- [Extropic Z1T 的 encoding 说明](https://extropic.ai/writing/z1t/#sec-encoding)，用于区分概率编码、瞬时 bit 状态和多次采样平均。
- [Qwen2.5-0.5B 官方模型页](https://huggingface.co/Qwen/Qwen2.5-0.5B)。本仓库的模型配置和 Transformers v4.45.2 参考源码已固定在 [`sources.json`](../../sources.json) 中。

## 13. 候选方案：入口与隐藏层均采用 p-bit（2026-09-29，待验证）

### 13.1 目的与必须保留的中间步骤

目标是让 `W_in` 和 `W_out` 的输入操作数都为 -1/+1，使两个矩阵都能使用二值输入专用的带符号累加电路。这里首先二值化的是激活，不要求矩阵权重同时变成 1 bit。

只采用 `x → 入口 p-bit → W_in → W_out → 平均` 不能实现这个目标：`W_in` 的加权和输出已经是连续或多位数，所以 `W_out` 仍接收多位输入。若两个仿射层之间没有非线性，它们还可以合并成一个仿射层，失去原来 4864 维隐藏非线性的作用。

因此候选结构应保留当前隐藏 p-bit，再增加一个入口 p-bit 编码层：

```text
归一化后的 Qwen 输入 x（896 维连续值）
 → 逐通道范围/概率编码
 → 入口 p-bit b^(n)（896 个 -1/+1）
 → W_in 的带符号累加 + 偏置（4864 维多位场）
 → 隐藏 p-bit s^(n)（4864 个 -1/+1）
 → W_out 的带符号累加 + 偏置（896 维连续读出）
 → N 条完整路径的输出平均
 → 原有连续残差相加
```

每条路径都必须先采样入口，再根据该路径的场采样隐藏层。不能在 `W_in` 后、隐藏 p-bit 前跨路径平均，否则会改变模型。

### 13.2 连续输入如何编码为 -1/+1

Qwen RMSNorm 不保证每个坐标落在 [-1,1]。不能直接设置 `P(+1)=(1+x)/2`。一种便于解释和校准的入口编码是固定或可学习的正尺度 a_i：

$$
u_i=\operatorname{clip}(x_i/a_i,-1,1),\qquad
P(b_i^{(n)}=+1\mid x)=(1+u_i)/2.
$$

于是 `E[b_i|x]=u_i`，没有裁剪时 `E[a_i b_i|x]=x_i`。令 A=diag(a)，可将固定的逐通道尺度吸收进权重 `W'_in=W_in A`，无需逐样本显式计算 `A b`：

$$
z^{(n)}=W'_{in}b^{(n)}+b_{in},\qquad
P(s_j^{(n)}=+1\mid b^{(n)},x)=\frac{1+\tanh(z_j^{(n)}/T)}{2},
$$

$$
y_N=\frac{1}{N}\sum_{n=1}^{N}\left(W_{out}s^{(n)}+b_{out}\right).
$$

尺度需要折衷：过小会裁剪，过大则多数输入的概率接近 1/2，重建噪声变大。在无裁剪、独立采样条件下，单通道重建平均的方差为 `(a_i^2-x_i^2)/N`。均值可重建输入，不代表非线性 FFN 输出无偏。

另一种入口是 `E[b_i|x]=tanh(x_i/T_in)`，更直接匹配理想 tanh p-bit 的响应，但它会改变输入特征，需要重新训练，不能声称其平均值等于原始 x。若物理器件本身服从 tanh 响应而希望实现上面的线性概率编码，所需驱动场为 `atanh(u)`；端点需有限范围处理，编码电路的成本和误差也要计入。

### 13.3 两个矩阵都可免通用乘法，但不是全二值权重网络

对于任意多位权重 w 和激活 b∈{-1,+1}，`w*b` 只需选择 `+w` 或 `-w`。因此两个矩阵都可以采用带符号权重累加，但权重存储、累加器、中间场和最终读出仍是多位数。

如进一步把权重也约束为 -1/+1，内积才可以实现为 XNOR + popcount，再做符号转换与尺度恢复。这属于更强的约束，应作为独立实验；不能用当前的浮点权重 checkpoint 直接声称效果不变。权重和激活同时二值化的经典参考是 [Courbariaux 等，Binarized Neural Networks](https://arxiv.org/abs/1602.02830)，其结果不构成本项目在 Qwen 上的精度证据。

### 13.4 计算量与 v0 的区别

令 K=896×4864=4,358,144，忽略偏置、采样和归约等额外工作，逐 token 的主要运算为：

| 方案 | 主要矩阵工作 |
| --- | --- |
| 原始 SwiGLU | 3K 次连续乘加 |
| v0，先平均隐藏样本再读出 | 2K 次连续乘加 |
| v0，N 条二值读出路径 | K 次连续乘加 + NK 次带符号累加 |
| 候选方案，N 条完整二值路径 | 2NK 次带符号累加 |

候选方案在 N=4 时约有 34,865,152 次带符号累加。每个入口样本不同，`W_in` 的计算也不同，不能像 v0 那样只算一次输入场再反复采样隐藏层。忽略其他成本，若每次累加成本小于原连续 MAC 的 3/8，该方案才在此简单模型下低于原始 SwiGLU；这不是硬件收益测量。

权重应尽量驻留并跨样本复用，避免 N 倍读取主存。入口概率编码、多位累加器、隐藏 p-bit 驱动、最终平均与残差成本仍需计入。若软件仍调用普通浮点 GEMM，二值输入不会自动使矩阵乘法变快，需要相应内核或专用硬件。

### 13.5 不能直接沿用“4 samples 已经足够”的结论

v0 只有输出前的隐藏采样噪声，候选方案增加了入口采样，而且入口噪声还会经过 `W_in` 和非线性。因此 v0 的 4-sample PPL 不能外推到该方案。

即使 N 趋于无穷，候选方案也一般不回到 v0 的确定性函数。其输出期望包含：

$$
E[y\mid x]=W_{out}E_{b\mid x}\left[\tanh((W'_{in}b+b_{in})/T)\right]+b_{out},
$$

一般不等于把入口平均值直接代入 tanh。应重新训练，让网络学习期望输出所定义的新函数。把两层 p-bit 都换成条件均值的前向计算，只能作为确定性对照，不能标成该两层随机网络的无限样本极限；后者应使用足够大的 Monte Carlo 样本数估计。

独立完整路径的平均可以降低输出方差。若多次隐藏采样共用同一次入口样本，则入口噪声会在这些路径之间相关，不能按同样的总次数宣称相同的降噪效果。输入每个通道有限次平均只有少量可能值，但后续有非线性，不能把整个随机网络简单视为“先做一个低位宽输入量化器”。

### 13.6 建议的验证顺序

1. 保持 layer 12、896→4864→896 和浮点权重，先只新增入口概率编码及采样，隔离这一项改动。
2. 校准入口逐通道尺度，记录裁剪比例、概率饱和比例、场分布和噪声方差；比较线性概率编码与 tanh 编码。
3. 使用完整随机路径做 sample-aware 蒸馏，必要时以连续松弛预热，但最终训练必须看到两层采样噪声。
4. 测试 N=1、4、8、16、32 及至少 3 个 seeds，报告完整模型 PPL，同时保留 v0、原始 Qwen 和较大 N 的随机参考。
5. 精度可接受后再对权重做低位宽量化；1-bit 权重与 XNOR/popcount 作为单独方案比较。

本节是设计分析；没有启动新的训练，没有新 checkpoint 或性能结果。正式实现后应新增版本条目并补齐数据、配置、哈希和结果。
