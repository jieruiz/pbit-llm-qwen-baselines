# Full-path bipolar P-DNN FFN: layer 12

This experiment tests a stricter P-DNN replacement in which every learned
matrix receives bipolar activations. One complete stochastic path is:

```text
continuous Qwen FFN input (896)
  -> input p-bits (-1/+1, 896)
  -> W_in and bias (4864-dimensional field)
  -> hidden p-bits (-1/+1, 4864)
  -> W_out and bias (continuous 896-dimensional readout)
```

Four independent complete paths are used during sample-aware training. Their
continuous readouts are averaged only after `W_out`. Unlike the first v0
experiment, the input of `W_in` is also bipolar, so both matrices can in
principle use sign-controlled weight accumulation. Weights, fields,
accumulators, and the final readout remain multi-bit.

## Model and training

- Teacher: Qwen2.5-0.5B Base, BF16, frozen
- Replaced module: decoder layer 12 FFN
- Shape: `896 input p-bits -> 4864 hidden p-bits -> 896 readout`
- Parameters: 8,722,048, the same as v0
- Input p-bit mean: `tanh(x / 0.25)`
- Hidden p-bit mean: `tanh(field / 1.0)`
- Data: WikiText-2 raw train, 2,518,423 tokenizer tokens
- Phase 1: 2,000 mean-field warm-up steps
- Phase 2: 2,000 full-path sample-aware steps, 4 paths, STE
- Batch: 4 sequences x 256 tokens
- Optimizer: AdamW, weight decay 0.01
- Learning rates: 3e-4 warm-up, 1e-4 sample-aware
- Total examples processed: 4,096,000 tokens
- GPU: one NVIDIA GeForce RTX 5090 32 GB
- Training time: 70.9996 seconds
- Effective training rate: 57,690.4 tokens/s
- Peak allocated training memory: 1,270,759,936 bytes (1.18 GiB)

Mean-field mode propagates conditional means through the matrices. It is a
training surrogate and diagnostic, not the exact infinite-sample expectation
of a multilayer stochastic network.

## Input-temperature pilot

Each pilot used the same seed, data split, architecture, 400 mean-field steps,
400 four-path sample-aware steps, and eight validation batches.

| Input temperature | Four-path validation normalized MSE |
| ---: | ---: |
| 0.125 | 0.629495 |
| **0.25** | **0.624446** |
| 0.35 | 0.626466 |
| 0.5 | 0.635921 |
| 1.0 | 0.689295 |
| 2.0 | 0.780949 |

Temperature 0.25 was selected before the full run. The local optimum between
0.125 and 0.35 suggests a real tradeoff: a low temperature reduces input-bit
uncertainty, while an excessively hard sign encoding discards input magnitude.

## Layer-local validation after full training

| Complete paths | Normalized MSE | Cosine similarity |
| ---: | ---: | ---: |
| Mean-field surrogate | 0.537018 | 0.680980 |
| 1 | 0.817803 | 0.519954 |
| 4 | 0.597783 | 0.633552 |
| 8 | 0.550871 | 0.668031 |
| 16 | 0.563133 | 0.660777 |

The local validation batches differ across sample counts, so the complete
language-model perplexity below is the primary cross-sample comparison.

## Full WikiText-2 test perplexity

All runs use the same 299,078-token corpus, 2,048-token windows, and stride
1,024 as the original BF16 and v0 measurements.

| Model | Complete paths | Seed | PPL | Change from 11.652735 |
| --- | ---: | ---: | ---: | ---: |
| Original Qwen | deterministic | 0 | 11.652735 | 0.00% |
| v1 full-path P-DNN | mean-field | 0 | 12.085348 | +3.71% |
| v1 full-path P-DNN | 1 | 0 | 12.229104 | +4.95% |
| v1 full-path P-DNN | 4 | 0 | 12.117123 | +3.99% |
| v1 full-path P-DNN | 4 | 1 | 12.122254 | +4.03% |
| v1 full-path P-DNN | 4 | 2 | 12.120305 | +4.01% |
| v1 full-path P-DNN | 8 | 0 | 12.105108 | +3.88% |
| v1 full-path P-DNN | 16 | 0 | 12.095442 | +3.80% |
| v1 full-path P-DNN | 32 | 0 | 12.091615 | +3.77% |

The three four-path runs have mean PPL **12.119894** and sample standard
deviation **0.002591**, a **4.01%** increase over the original model. The v0
four-sample mean was 12.072649, so making the input of `W_in` bipolar adds only
0.047245 PPL, or 0.391% relative to v0.

## Interpretation

The experiment supports the proposed full-path construction. Both learned
matrices receive strict -1/+1 values, yet replacing one Qwen FFN keeps the
complete model close to both the original Qwen and the less restrictive v0.
Four paths remain a useful operating point: increasing from four to sixteen
improves PPL by about 0.0245, while requiring four times as many complete
matrix paths. Increasing to 32 paths reaches 12.091615, only 0.052% above the
mean-field surrogate, with eight times the path computation of the four-path
setting.

For four paths and `K = 896 x 4864`, the conceptual hardware workload is `8K`
sign-controlled weight accumulations per token. This repository uses ordinary
BF16 PyTorch matrix multiplication, so the measured GPU time is a functional
training measurement and does not demonstrate a hardware speed or energy
gain.

This result covers one of 24 FFNs. It does not establish the behavior of a
fully converted Qwen model, binary weights, fixed-degree sparse connectivity,
or physical p-bit correlations. The next structural test should add another
matrix and p-bit layer at a controlled parameter budget, then compare it with
this two-matrix full-path baseline before converting multiple decoder layers.

## Checkpoint identities

Large checkpoints remain on the experiment server and are excluded from Git.

| Checkpoint | Bytes | SHA-256 |
| --- | ---: | --- |
| `student_mean.pt` | 104,671,457 | `0a78abf007a928c8073b7226758f4620fa092bf0a168f5bf9a93ee413b04ed1c` |
| `student_sampled.pt` | 104,671,587 | `24cdcf43c8e643ea27ccb1427218b1b97b53f80a2750b09e1d3b62f18e046912` |
