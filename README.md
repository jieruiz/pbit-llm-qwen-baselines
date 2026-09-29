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

## Full-path bipolar P-DNN replacement

The second experiment also converts the FFN input into bipolar p-bit samples:

```text
continuous input -> input p-bits -> W_in -> hidden p-bits -> W_out -> average
```

Every learned matrix therefore receives -1/+1 activations. With four complete
paths, decoder layer 12 reached WikiText-2 PPL **12.1199 +/- 0.0026** across
three seeds. This is 4.01% above the original Qwen result and only 0.39% above
the first P-DNN replacement, whose first matrix still received continuous
inputs. See
[`results/full_path_pdnn_layer12_bipolar_t0p25/RESULTS.md`](results/full_path_pdnn_layer12_bipolar_t0p25/RESULTS.md)
for the temperature pilot, protocol, full table, and limitations.

Progressive composition was then tested with independently distilled students.
Layers 6 and 12 reached four-path PPL 13.2025, while layers 6, 12, and 18
reached 14.3115. Adding layer 0 caused PPL to exceed 46 even though that layer
had low local MSE. Replacing it with screened middle layers `{9,12,15,18}`
reduced the four-layer result to **15.3493 +/- 0.0051**, confirming that layer
0 caused the catastrophic failure while also showing that independent
replacement errors still accumulate. Late layers 21 and 23 were also more
sensitive than their local normalized MSE suggested, so candidate layers must
be screened with full-model perplexity. See
[`results/full_path_pdnn_progressive_2_4_layers/RESULTS.md`](results/full_path_pdnn_progressive_2_4_layers/RESULTS.md).

All 24 layers were then trained and ranked by single-layer sensitivity. The
safest layers are concentrated around decoder layers 9–14, while layers 0 and
2 are catastrophic even alone. With safest-first cumulative replacement,
held-out-half PPL rises from 12.1563 for original Qwen to 14.4869 at four
layers, 16.9659 at six, 20.4379 at eight, and 24.1865 at nine. Thus eight
independently distilled FFNs is a lenient boundary and nine to ten is already
unacceptable. See
[`results/full_path_pdnn_expansion_5_24_layers/RESULTS.md`](results/full_path_pdnn_expansion_5_24_layers/RESULTS.md).

## Repository contents

- `baselines/`: integrity, generation, performance, perplexity, FFN probe, and
  lm-evaluation-harness entry points.
- `results/base_bf16/`: raw JSON results plus environment and artifact hashes.
- `results/pdnn_ffn_layer12_bipolar/`: first single-layer P-DNN training and
  perplexity results; large training checkpoints are retained off-repository.
- `results/full_path_pdnn_layer12_bipolar_t0p25/`: full-path bipolar P-DNN
  training and perplexity results.
- `results/full_path_pdnn_progressive_2_4_layers/`: two-, three-, and four-FFN
  composition results, timing estimates, and per-layer sensitivity data.
- `results/full_path_pdnn_expansion_5_24_layers/`: all-layer sensitivity,
  cumulative 4–24-FFN curves, held-out split checks, and sampling controls.
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

## Sigmoid coding and internal P-DNN depth (2026-09-29)

Both studies replace only decoder FFN index 12 of Qwen2.5-0.5B Base;
original BF16 WikiText-2 PPL is 11.652735. Every matrix input is sampled,
and only the final continuous path readouts are averaged.

| Study | Four-path PPL | Replication and conclusion |
| --- | --- | --- |
| [Coding comparison](results/full_path_coding_comparison_20260929/RESULTS_ZH.md) | tanh: 12.119894; same-temperature sigmoid: 12.157562; half-temperature sigmoid: 12.093255 | One trained checkpoint per condition, three inference seeds; half-temperature sigmoid slightly improves this fixed recipe. |
| [Internal depth comparison](results/full_path_depth_comparison_20260929/RESULTS_ZH.md) | 2 matrices: 12.096229; 3: 12.128913; 4: 12.136852 | Three independent training seeds per depth, inference seed 0; deeper and narrower students do not improve PPL at matched parameters and training steps. |

Here 2/3/4 layers means matrices **inside one student FFN**, not the number
of decoder FFNs replaced. See the [experiment instructions](experiments/pdnn_ffn/README.md)
for runners and tests. Raw metrics, environments and SHA-256 manifests are
included; weights, datasets and checkpoints remain outside Git. These are
software feasibility measurements, not p-bit hardware speed or energy results.

## License

Apache License 2.0. Qwen model weights and WikiText-2 are obtained separately
and remain subject to their own licenses and terms.
