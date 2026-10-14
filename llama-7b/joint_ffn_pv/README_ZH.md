# LLaMA-1 7B随机FFN与P–V采样实验

本目录把此前Qwen2.5-0.5B实验中表现最稳妥的组合迁移到原始LLaMA-1 7B：FFN继续使用固定尺度随机流参考计算；attention保持Q/K投影、QK点积和softmax准确，仅把`P @ V`替换为从精确softmax分布进行的IID采样计数。

## 迁移后的固定配置

- 覆盖全部32层，模型hidden size 4096、FFN intermediate size 11008。
- FFN权重按64×64 tile做W8对称量化。
- 激活为12位幅度加符号；输入4096周期，256周期更新一次；25%预热，因此输出累计3072周期。
- sigmoid保持理想FP32；随机计数使用精确Binomial压缩，不是逐bit RTL或有限位累加器仿真。
- attention使用精确Q/K、RoPE、缩放`1/sqrt(128)`和softmax。
- P–V使用512次IID类别采样；计数除以512后与V相乘。软件仍构造dense计数张量，不代表真实稀疏加速。
- LLaMA-1 7B是32Q/32KV的MHA；不使用Qwen的14Q/2KV GQA共享逻辑。

## 标定

固定尺度只使用WikiText-2 train的前8192个token，不更新权重。每层分别统计x、gate和up信号的64通道组最大绝对值，并向上取二次幂。测试集不参与尺度选择。

```bash
/opt/slurm/21.08.8/bin/sbatch calibrate.sbatch
```

生成`calibration_ffn.json`后，先运行四种128-token烟雾测试，再运行完整WT2：

| 方案 | FFN | Attention |
|---|---|---|
| `original_dense` | 原始SwiGLU | 手写精确dense |
| `attention_pv` | 原始SwiGLU | 精确Q/K/softmax，512次P–V采样 |
| `ffn_dense` | 随机FFN | 手写精确dense |
| `joint_pv` | 随机FFN | 精确Q/K/softmax，512次P–V采样 |

完整评测沿用基线协议：WikiText-2 `wiki.test.raw`，context 2048、stride 1024、batch 1、不加特殊token，共335,686个计分token。`original_dense`是确定性单次；其余方案固定运行seed 0/1/2。最终统计由`analyze.py`生成。

完整结果已经生成，见`results/RESULTS_ZH.md`和`results/summary.json`。联合方案PPL为5.561454 ± 0.004251，相对同路径dense增加7.294%。

## 远程目录

```text
/home/Hongjie_Zeng/llama7b_joint_pv_20261014
```

模型权重仍读取：

```text
/home/Hongjie_Zeng/pbit_llm/models/LLaMA-7B
```
