# FFN 实验补充：多阈值 p-bit 与 AND（2026-10-03）

本次补充发布最新的双分支多阈值 p-bit FFN，实现、训练/评估脚本、原理说明和结果
一并保留。另补齐此前连续输入、多位输入、门控及宽度对照的代码和实验记录，便于
追溯设计变化、满足脚本依赖并复核历史比较。本次不包含随机注意力实验。

## 最新结构

保留两个浮点入口投影，以原 Qwen gate/up/down 矩阵初始化。每条分支、每个隐藏
通道用 K 个不同阈值/温度的 0/1 p-bit 表达函数；gate 分支先拟合 SiLU，value 分支
先拟合线性函数，再作局部蒸馏和多层联合蒸馏。两个分支的乘积展开为单 bit 项和
AND 项，带符号系数合并进共享读出权重，常数折叠为偏置。每个 FFN 平均 N 条路径
后输出连续值。K 是分支编码位数，N 是重复采样数，两者不同。

训练使用代数等价的因式分解采样形式；正式采样评估使用展开二值读出。入口投影
为 BF16，概率 bank、读出及路径累加为 FP32。这些是软件参考，不是二值硬件加速实现。

## 完整 WikiText-2 test 结果

原始 Qwen2.5-0.5B Base PPL 为 **11.652735**。全部结果使用 299078 个输入 token，
计分 299077 个 token，上下文 2048、stride 1024。N=4 的 ± 是三个推理 seed 的总体标准差。

| 配置 | N=4 PPL | N=16 PPL | Mean-field PPL |
| --- | ---: | ---: | ---: |
| 第12层，K=4 | 11.720476 ± 0.000883 | 11.700873 | 11.694797 |
| 20层独立组合，K=4 | 14.849078 ± 0.013730 | 13.782845 | 13.460670 |
| 20层联合后，K=4 | 13.672837 ± 0.002889 | 12.741809 | 12.469567 |

20层保留原始0、2、3、23层，其余FFN替换；联合训练1000步，验证集选择第800步。
联合阶段耗时534.06秒、峰值allocated memory约12.58 GiB，不含此前局部训练和评估。

输入仍然连续；K=4每路径有24组动态二值读出项。当前没有证明硬件加速或节能。
历史两矩阵模型在参数量、初始化、读出精度和联合训练起点上不同，不能把全部精度
改善归因于AND结构。训练只有一个seed、测试仅一个语料，不能据此认定架构极限。

## 实现与记录

- [原理和演变记录](P_DNN_FFN_DESIGN_ZH.md)：第33、34节描述最新单层和20层实验。
- [核心实现](multithreshold_and_ffn.py)：p-bit概率、编码、AND展开及最后平均。
- [单层报告](../../results/multithreshold_and_layer12_20261002/RESULTS_ZH.md)：45份评估、9份checkpoint身份。
- [20层报告](../../results/multithreshold_and_twenty_layer_20261002/RESULTS_ZH.md)：14份评估、40份checkpoint身份。

权重和语料不纳入Git；结果保留checkpoint SHA256和执行源码哈希。既有权重保留在
训练服务器。训练脚本默认使用仓库根目录下的`models/Qwen2.5-0.5B`和`data/wikitext-2`。

提交前在远程现有环境运行19项测试，覆盖AND二值边界、解析矩、STE梯度等价、
多层梯度/checkpoint集成、门控、多位输入、连续输入和路径矩；全部通过。
单层和20层汇总程序同时核对评估协议、源码哈希和checkpoint身份。

```bash
cd experiments/pdnn_ffn
python -m unittest test_multithreshold_and test_joint_multithreshold_and \
  test_gated_pdnn_ffn test_multibit_input test_continuous_input_ffn test_path_moments
cd ../..
python experiments/pdnn_ffn/summarize_multithreshold_and.py
python experiments/pdnn_ffn/summarize_multithreshold_twenty_layer.py
```
