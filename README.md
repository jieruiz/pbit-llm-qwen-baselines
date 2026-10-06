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

Joint end-to-end adaptation was then tested on the four safest layers
`{10,11,12,13}`. Only the 34.9M P-DNN parameters were updated using original
Qwen logits and next-token loss. A 1,000-step run on one RTX 5090 took 225
seconds and reduced four-path PPL from 13.8953 to **13.1535 +/- 0.0014**,
recovering 31.2% of the excess NLL caused by independent replacement. See
[`results/joint_full_path_pdnn_layers10_13_v1/RESULTS.md`](results/joint_full_path_pdnn_layers10_13_v1/RESULTS.md).

The same method was extended to ten layers
`{7,8,9,10,11,12,13,14,15,18}`. A staged initialization used the joint
four-layer checkpoints for layers 10–13 and independent checkpoints elsewhere.
The 1,000-step run took 316 seconds on one RTX 5090 and reduced full-test N=4
PPL from the ten-independent result of 27.3051 to **19.2261 +/- 0.0069**.
This recovers 41.2% of the excess NLL, but the remaining 65.0% PPL increase
over original Qwen shows that the present structure still needs improvement.
See
[`results/joint_full_path_pdnn_layers7_15_18_ten_v1/RESULTS.md`](results/joint_full_path_pdnn_layers7_15_18_ten_v1/RESULTS.md).

A twenty-layer staged run then kept only layers 0, 2, 3, and 23 as original
Qwen FFNs. Joint adaptation reduced N=4 PPL from 101.3749 at the staged
initialization to **48.9108 +/- 0.0963**. This is a large recovery but still
319.7% above original Qwen; mean-field PPL is already 42.8818. The experiment
therefore confirms that the current P-DNN structure is not acceptable at
twenty layers even after joint training. See
[`results/joint_full_path_pdnn_layers1_22_twenty_v1/RESULTS.md`](results/joint_full_path_pdnn_layers1_22_twenty_v1/RESULTS.md).

Layer-specific 0/1 encoding was then tested by learning a scalar threshold and
positive temperature at each p-bit boundary. On representative layers 10, 12,
and 19, four-path PPL improved in all three cases. When the three students were
composed, PPL fell from 14.0926 +/- 0.0032 with fixed encoding to
**13.8903 +/- 0.0016**. Learned input temperatures reduced input probability
saturation substantially, while hidden encodings became sharper. See
[`results/layerwise_binary_encoding_20260929/RESULTS_ZH.md`](results/layerwise_binary_encoding_20260929/RESULTS_ZH.md).

The learned 0/1 encoding was then extended to the same ten-layer set used in
the earlier joint experiment. End-to-end adaptation reduced N=4 PPL from
23.7393 to **18.5686 +/- 0.0009**. This is 0.6575 better than the previous
joint ten-layer result, but still 59.35% above original Qwen. Mean-field PPL
of 17.4551 confirms that most remaining error is structural. See
[`results/joint_binary_learnable_encoding_layers7_15_18_ten_v1/RESULTS_ZH.md`](results/joint_binary_learnable_encoding_layers7_15_18_ten_v1/RESULTS_ZH.md).

At twenty replaced FFNs, learned 0/1 encoding reduced N=4 PPL from the earlier
48.9108 to **43.3831 +/- 0.0100**. The staged initialization also improved from
101.3749 to 75.8560, but the final model remains 272.3% above original Qwen.
Late layers 17, 20, 21, and 22 account for a large part of the accumulated
error. See
[`results/joint_binary_learnable_encoding_layers1_22_twenty_v1/RESULTS_ZH.md`](results/joint_binary_learnable_encoding_layers1_22_twenty_v1/RESULTS_ZH.md).

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
- `results/joint_full_path_pdnn_layers10_13_v1/`: joint four-layer training
  curve, cost measurement, and full-model perplexity recovery.
- `results/joint_full_path_pdnn_layers7_15_18_ten_v1/`: joint ten-layer
  training curve, robust full-test evaluation, and checkpoint identities.
- `results/joint_full_path_pdnn_layers1_22_twenty_v1/`: joint twenty-layer
  training, staged-initialization comparison, and failure-boundary result.
- `results/layerwise_binary_encoding_20260929/`: fixed-versus-learned 0/1
  encoding, saturation measurements, and three-layer composition results.
- `results/layerwise_binary_encoding_ten_20260929/`: independent training logs
  for the additional layers used by the learned-encoding ten-layer run.
- `results/joint_binary_learnable_encoding_layers7_15_18_ten_v1/`: joint
  ten-layer 0/1 learned-encoding curve, evaluations, and saturation data.
- `results/layerwise_binary_encoding_twenty_20260929/`: independent training
  logs for the additional learned-encoding twenty-layer initializers.
- `results/joint_binary_learnable_encoding_layers1_22_twenty_v1/`: joint
  twenty-layer learned-encoding results, saturation, and layer ablations.
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

## Binary gated dual-rail pilot

A new 0/1 gate/value FFN retains all matrix biases upstream of p-bits and learns
temperatures without separate p-bit thresholds. Positive and negative binary
rails share a readout matrix; complete paths are averaged at the output.
Matched-parameter single-layer tests at layers 10/12/19 did **not** improve
PPL over serial FFNs: N=4 PPL was 12.0720/12.1016/12.9578 versus
12.0298/12.0666/12.8741. See the
[implementation and full pilot results](results/gated_dual_rail_20260929/RESULTS_ZH.md).

Continuing layer 12 to 6,000 sample-training updates narrows the gated/serial
N=4 PPL gap to 12.0560 versus 12.0465. Training with N=16 instead improves
N=16 inference, but worsens N=4 inference; repeated-path diagnostics measure
the resulting bias/variance tradeoff. See the
[controlled continuation results](results/gated_sample_budget_20260929/RESULTS_ZH.md).

The four-layer `{9,12,15,18}` comparison confirms that the gated design does
not improve composition in this setup. After matched joint training, N=4 PPL
is 13.6933 for gated and 13.5591 for serial. See the
[four-layer comparison](results/gated_four_layer_9_12_15_18_20260930/RESULTS_ZH.md).

A corrected layer-12 width sweep finds modest N=4 PPL improvements as parameters
grow from 8.72M to 13.08M: serial 12.04655 to 12.03119, gated 12.05600 to
12.04255. An earlier evaluation selected the wrong global-step checkpoint for
new widths; those results are superseded. Width helps but leaves most of the
gap to original Qwen. Single-seed local distillation does not establish an
architectural limit. See the
[width-sweep results](results/pdnn_width_sweep_layer12_20260930/RESULTS_ZH.md).

A fixed-width input-encoding experiment lowers N=4 PPL from 12.04655 to
11.95572 using four tied position bits per input feature. This is close to the
continuous-input control (11.95222), while deterministic 4-bit input gives
11.95921. It improves amplitude representation with unchanged matrix parameter
count, at greater bitplane computation cost; the coordinated encoder remains
an algorithmic reference. See
[input multibit results](results/input_multibit_layer12_20260930/RESULTS_ZH.md).

Extending the stochastic K=4 input encoder to layers `{9,12,15,18}` lowers
four-layer N=4 PPL from 14.45444 to 13.68340 before joint adaptation, and from
13.55906 to 12.90971 after 1,000 joint updates. The latter remains 10.79% above
original Qwen and uses 2.5x as many matrix calls per replacement path as the
two-matrix sigmoid-input student. See the
[four-layer multibit results](results/input_multibit_four_layer_9_12_15_18_20260930/RESULTS_ZH.md).

A matched ten-layer experiment removes input encoding entirely and feeds raw,
unclipped floating activations into the first matrix, retaining binary hidden
p-bits. After 1,000 joint updates, N=4 PPL is 15.59254 versus 17.83837 for a
fresh sigmoid-input control trained for the same duration. This is a 12.59%
improvement, but remains 33.81% above original Qwen. See the
[ten-layer floating-input results](results/continuous_raw_ten_layer_20261001/RESULTS_ZH.md).

A staged twenty-layer extension retains original FFNs 0, 2, 3, and 23 and
jointly adapts the other 20 replacements for 1,000 additional updates. Raw-input
N=4 PPL is **27.12776 +/- 0.03691**, versus **38.76411 +/- 0.02749** for the
matched sigmoid-input control. The 30.02% benefit persists, but the raw-input
result is 73.98% worse than its ten-layer predecessor. N=16 reaches 25.70319;
additional averaging does not close the observed gap. See the
[twenty-layer results](results/continuous_raw_twenty_layer_20261001/RESULTS_ZH.md).

A teacher-initialized layer-12 experiment uses sigmoid p-bit banks on both
SwiGLU branches and binary AND features with tied readout weights. With 1, 2,
or 4 bits per branch, final N=4 PPL is **11.90592, 11.78858, and 11.72048**.
The four-bit result is 0.58% above original Qwen; at N=1 it reaches 11.80462.
All sampled evaluations use actual binary readout inputs. Input projections
remain floating point, and readout terms grow as K²+2K, so these quality gains
do not establish hardware efficiency or multi-layer performance. See the
[AND-bank experiment](results/multithreshold_and_layer12_20261002/RESULTS_ZH.md).

The [FFN supplement](experiments/pdnn_ffn/FFN_SUPPLEMENT_20261003_ZH.md)
collects the latest implementation, reproducibility checks, and result links.
The K=4 AND-bank model was extended to 20 FFNs, retaining original layers
0, 2, 3, and 23. Independent composition gives N=4 PPL **14.84908 +/- 0.01373**;
1,000 joint updates reduce it to **13.67284 +/- 0.00289**, still 17.34% above
original Qwen. N=16 reaches **12.74181**, with mean-field at **12.46957**.
This improves substantially over the historical two-matrix raw
input result, but uses more replacement parameters, teacher initialization,
and different readout precision. See the
[twenty-layer AND-bank experiment](results/multithreshold_and_twenty_layer_20261002/RESULTS_ZH.md).

A separate [p-bit attention experiment](experiments/pbit_attention/README.md)
retains all original FFNs and changes only decode-time AV aggregation in all
24 attention layers. At S=512, categorical-tree / independent-Bernoulli PPL is
10.42392 / 10.44439 versus 10.37356 original SDPA on 2k-context sampled suffixes,
and 9.49162 / 9.49449 versus 9.41898 at 8k. Three inference seeds are reported.
These are cached-decode subset metrics, not the historical full-test PPL.
See the [report](results/pbit_attention_20261003/RESULTS_ZH.md) for all 34 runs,
sampling-law tests, and the distinction between logical sparsity and hardware speed.

## License

Apache License 2.0. Qwen model weights and WikiText-2 are obtained separately
and remain subject to their own licenses and terms.
