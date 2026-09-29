# p-bit LLM: Qwen2.5-0.5B baselines

Reproducible BF16 reference measurements and inspection utilities for studying
p-bit replacements inside the feed-forward network of Qwen2.5-0.5B Base.

This repository contains the baseline code and measured results used before any
p-bit modification. Model weights and benchmark datasets are intentionally not
committed.

## Reference results

The measurements below used one NVIDIA GeForce RTX 5090 (32 GB), PyTorch
2.7.1+cu128, Transformers 4.45.2, BF16 weights, PyTorch SDPA, and seed 0.

| Metric | Result |
| --- | ---: |
| Model parameters | 494,032,768 |
| Non-embedding parameters | 357,898,112 |
| WikiText-2 raw test perplexity | **11.652735** |
| ARC-Easy 0-shot accuracy | **0.6460** |
| ARC-Easy 0-shot normalized accuracy | **0.5867** |
| HellaSwag 0-shot accuracy | **0.4059** |
| HellaSwag 0-shot normalized accuracy | **0.5208** |
| Greedy generation, 64 new tokens | **88.239 tokens/s** |
| 2048-token prefill peak allocated memory | **2.70 GiB** |

FFN probes are included for layers 0, 12, and 23. The gated-product RMS changes
from 0.1049 to 0.1183 to 1.4467 across those layers, which motivates per-layer
calibration for later stochastic or binary activations.

## First P-DNN FFN replacement

`experiments/pdnn_ffn/` contains the first layer-distillation experiment. It
replaces decoder layer 12's SwiGLU FFN with a bipolar stochastic hidden layer
and a continuous linear readout. After conditional-mean distillation and
sample-aware training, four-sample inference reached WikiText-2 perplexity
12.0726 +/- 0.0017 across three random seeds, compared with 11.6527 for the
unmodified model. See `results/pdnn_ffn_layer12_bipolar/RESULTS.md` for the
protocol, full table, and limitations. The Chinese living design document
[`experiments/pdnn_ffn/P_DNN_FFN_DESIGN_ZH.md`](experiments/pdnn_ffn/P_DNN_FFN_DESIGN_ZH.md)
describes every layer and is the canonical place for all future P-DNN FFN
changes and version records.

## Repository contents

- `baselines/`: integrity, generation, performance, perplexity, FFN probe, and
  lm-evaluation-harness entry points.
- `results/base_bf16/`: raw JSON results plus environment and artifact hashes.
- `results/pdnn_ffn_layer12_bipolar/`: first single-layer P-DNN training and
  perplexity results; large training checkpoints are retained off-repository.
- `experiments/pdnn_ffn/`: P-DNN module, layer distillation, and evaluation.
- `config/`: the Qwen2.5-0.5B configuration used for architecture accounting.
- `upstream/`: pinned Transformers v4.45.2 Qwen2 implementation for source
  comparison. Its upstream Apache-2.0 license is retained.
- `sources.json`: pinned source URLs, revisions, sizes, and hashes.

## Setup

Install a PyTorch build suitable for the target GPU first. The reference machine
used PyTorch 2.7.1 with CUDA 12.8. Then install the remaining packages:

```bash
python -m pip install -r requirements.txt
python scripts/download_model.py
bash scripts/download_wikitext2.sh
```

The default layout is:

```text
models/Qwen2.5-0.5B/
data/wikitext-2/wiki.test.raw
```

Paths can be overridden with `MODEL_PATH` and `PBIT_LLM_ROOT`.

## Run

```bash
CUDA_VISIBLE_DEVICES=0 bash baselines/run_baselines.sh
CUDA_VISIBLE_DEVICES=0 bash baselines/run_lm_eval.sh
```

See [baselines/BASELINES.md](baselines/BASELINES.md) for the measurement
definitions and comparison protocol.

## Notes for p-bit comparisons

Keep the tokenizer, corpus, prompt set, precision, window/stride, and random seed
fixed. For stochastic variants, report the sample count and at least three
independent seeds. Compare task accuracy and perplexity alongside peak memory,
throughput, output variance, and layer-local error.

## License

Apache License 2.0. Qwen model weights and WikiText-2 are obtained separately
and remain subject to their own licenses and terms.
