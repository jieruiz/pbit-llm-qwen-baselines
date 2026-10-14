# 原版 LLaMA-1 7B 基线

目标是先建立可审计的原模型基线，再引入随机FFN和Attention P-V采样。

## 固定范围

- 模型严格校验为原版LLaMA-1 7B：4096隐藏维、11008 FFN、32层、32个Q/K/V头、2048上下文、32000词表。
- 运行时BF16权重和激活，分别建立SDPA与eager attention基线，关闭TF32。
- WikiText-2 `wiki.test.raw`，不添加特殊token。
- context=2048，stride=1024，batch=1；除第一个token外全部计分。
- LM head和交叉熵使用FP32。
- 保存配置、小文件哈希、语料哈希、软件版本、参数量、峰值显存、逐窗口NLL和缓存解码探针。

协议与此前Qwen完整WT2实验一致，但由于tokenizer不同，两个模型的绝对PPL和计分token数不能直接横向比较。后续所有LLaMA随机方案必须使用本目录基线作为配对对照。

## 模型目录

默认要求转换为Hugging Face格式并放在：

```text
/home/Hongjie_Zeng/pbit_llm/models/LLaMA-7B
```

使用公开转换镜像 `huggyllama/llama-7b`，无需把访问token写进脚本、日志或仓库。两个safetensors分片已经按镜像LFS对象哈希验证：

```text
model-00001-of-00002.safetensors  d43476fdd2fca0c44d55ee930039dd5dafb6331764dc0b5e5f89c60b551fcc12
model-00002-of-00002.safetensors  94712c4b457815e0f8aaa4f78170aebba2b9ee3339ff2b40fceb983338fbafee
```

只读配置检查：

```bash
/home/Hongjie_Zeng/.conda/envs/pbit_llm/bin/python baseline.py \
  --model /home/Hongjie_Zeng/pbit_llm/models/LLaMA-7B \
  --data /home/Hongjie_Zeng/pbit_llm/data/wikitext-2/wiki.test.raw \
  --output results/unused.json \
  --check-only
```

检查通过后提交：

```bash
/opt/slurm/21.08.8/bin/sbatch baseline.sbatch
/opt/slurm/21.08.8/bin/sbatch baseline_eager.sbatch
```

结果写入：

```text
results/wikitext2_original_sdpa.json
results/wikitext2_original_eager.json
```

## 已完成结果

2026-10-14在单张NVIDIA GeForce RTX 5090上完成。两条结果均计分335,686个token（语料token总数335,687），参数量为6,738,415,616。

| Attention实现 | Slurm作业 | PPL | NLL/token | 峰值显存 | 评测时间 |
|---|---:|---:|---:|---:|---:|
| SDPA | 1413 | 5.1829651235 | 1.6453773102 | 12.783 GiB | 55.85 s |
| eager | 1414 | 5.1829770184 | 1.6453796052 | 13.943 GiB | 86.11 s |

eager相对SDPA的PPL差异仅+0.0002295%。SDPA作为原模型性能基线；后续P-V采样应优先与eager或同一份手写dense实现比较，避免把attention内核差异计入随机计算误差。

## 后续门槛

基线完成后再依次执行：

1. 手写dense attention与SDPA/eager同路径对照。
2. cached decode数值和因果掩码测试。
3. 仅P-V采样的128/256/512采样数扫描。
4. 随机FFN重新校准后做单层、分层和全32层评估。
