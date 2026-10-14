# LLaMA-1 7B 随机计算实验

本目录记录将 Qwen2.5-0.5B 上的随机计算方案迁移到原版 LLaMA-1 7B 的基线、实现与完整 WikiText-2 评测结果。模型使用 Hugging Face 转换镜像 `huggyllama/llama-7b`，权重不存入仓库。

## 目录

- `baseline/`：原模型 BF16 基线，包含 SDPA 与 eager attention 的完整 WikiText-2 结果。
- `joint_ffn_pv/`：全部 32 层随机 FFN，以及保持 Q/K/softmax 精确、仅对 `P @ V` 做 512 次 IID 采样的实验。

两个实验使用相同评测协议：WikiText-2 `wiki.test.raw`、context 2048、stride 1024、batch 1，共计分 335,686 个 token。

## 主要结果

| 方案 | PPL | 相对同路径 dense |
|---|---:|---:|
| 原版 SDPA 基线 | 5.182965 | — |
| 手写 dense 对照 | 5.183385 | — |
| 仅 Attention P–V 采样 | 5.289522 ± 0.002792 | +2.048% |
| 仅随机 FFN | 5.413979 ± 0.003129 | +4.449% |
| 随机 FFN + P–V 采样 | 5.561454 ± 0.004251 | +7.294% |

详细配置、运行命令和逐 seed 结果分别见：

- [`baseline/README_ZH.md`](baseline/README_ZH.md)
- [`baseline/results/RESULTS_ZH.md`](baseline/results/RESULTS_ZH.md)
- [`joint_ffn_pv/README_ZH.md`](joint_ffn_pv/README_ZH.md)
- [`joint_ffn_pv/results/RESULTS_ZH.md`](joint_ffn_pv/results/RESULTS_ZH.md)
