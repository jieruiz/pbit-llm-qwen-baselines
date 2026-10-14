# LLaMA-1 7B WikiText-2原模型基线

## 结论

原模型基线已经在远程单张RTX 5090上完成。完整WikiText-2 `wiki.test.raw`结果为：

| Attention实现 | PPL | 计分token | 峰值显存 | 完整窗口时间 |
|---|---:|---:|---:|---:|
| SDPA | 5.1829651235 | 335,686 | 12.783 GiB | 55.85 s |
| eager | 5.1829770184 | 335,686 | 13.943 GiB | 86.11 s |

两者PPL相对差异仅+0.0002295%。因此：

- 原模型总基线采用SDPA的5.1829651235。
- 后续P-V采样的直接算法基线采用eager的5.1829770184，或进一步实现的同路径手写dense attention。
- LLaMA与Qwen使用不同tokenizer，绝对PPL和token数不能直接横向比较；只比较各自相对本模型基线的变化。

## 固定协议

- 模型：公开转换镜像`huggyllama/llama-7b`，原始LLaMA-1 7B结构。
- 架构：32层，hidden size 4096，FFN intermediate size 11008，32个Q/K/V头，head dim 128，context 2048，vocab 32000。
- 参数量：6,738,415,616。
- 语料：`wiki.test.raw`，SHA-256为`173c87a53759e0201f33e0ccf978e510c2042d7f2cb78229d9a50d79b9e7dd08`。
- 不添加特殊token；context 2048，stride 1024，batch 1；除语料首token外全部计分。
- 模型计算BF16，LM head输出转FP32后计算交叉熵；关闭TF32；PPL路径不使用KV cache。
- 共327个滑动窗口，语料335,687个token，计分335,686个token。

## 可审计信息

- SDPA Slurm作业：1413；结果文件`wikitext2_original_sdpa.json`。
- eager Slurm作业：1414；结果文件`wikitext2_original_eager.json`。
- 权重分片SHA-256：
  - `model-00001-of-00002.safetensors`: `d43476fdd2fca0c44d55ee930039dd5dafb6331764dc0b5e5f89c60b551fcc12`
  - `model-00002-of-00002.safetensors`: `94712c4b457815e0f8aaa4f78170aebba2b9ee3339ff2b40fceb983338fbafee`
- SDPA NLL总和：552330.1277413368；NLL/token：1.6453773101688387。
- eager NLL总和：552330.8981361389；NLL/token：1.6453796051552312。

缓存解码探针用于检查prefill/decode路径是否数值一致。BF16下SDPA的mean/max absolute logit difference为0.01839/0.125；eager为0.04800/0.15625。这些是不同矩阵形状和归约顺序带来的BF16差异，不参与PPL计算；后续随机attention需要使用同一探针继续做回归检查。
