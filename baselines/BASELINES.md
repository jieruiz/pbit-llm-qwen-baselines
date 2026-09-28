# Qwen2.5-0.5B baseline protocol

The baseline model is the unmodified `Qwen/Qwen2.5-0.5B` Base checkpoint. It is
loaded from local files only and evaluated in BF16 with PyTorch SDPA. The
published measurements used one NVIDIA GeForce RTX 5090 (32 GB).

The baseline suite records:

1. Model/configuration integrity and finite logits.
2. Deterministic greedy continuations for fixed Chinese and English prompts.
3. Prefill latency, generation throughput, and peak allocated CUDA memory.
4. Sliding-window WikiText-2 raw test perplexity with a 2048-token window and
   1024-token stride.
5. FFN activation distributions at layers 0, 12, and 23, including the gate
   projection, SiLU output, up projection, gated product, and down projection.
6. Zero-shot ARC-Easy and HellaSwag accuracy through lm-evaluation-harness.

Run everything:

```bash
CUDA_VISIBLE_DEVICES=0 bash baselines/run_baselines.sh
```

Results are JSON files under:

```text
results/base_bf16
```

Run the downstream task baseline after its datasets have been cached:

```bash
CUDA_VISIBLE_DEVICES=0 bash baselines/run_lm_eval.sh
```

The established Base BF16 reference results are:

| Metric | Value |
| --- | ---: |
| WikiText-2 raw test perplexity | 11.652735 |
| ARC-Easy 0-shot accuracy | 0.6460 |
| ARC-Easy 0-shot normalized accuracy | 0.5867 |
| HellaSwag 0-shot accuracy | 0.4059 |
| HellaSwag 0-shot normalized accuracy | 0.5208 |
| Greedy generation throughput, 64 new tokens | 88.239 tokens/s |

For every p-bit experiment, copy the suite to a new result directory and keep
the model, corpus, tokenizer, prompt set, precision, window/stride, and random
seed unchanged. Stochastic models should additionally report the sample count
and at least three independent seeds. Compare perplexity, task accuracy,
throughput, peak memory, and output variance against this baseline.
