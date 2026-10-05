# P-DNN replacement for one Qwen2.5 FFN

This experiment replaces one Qwen2.5-0.5B SwiGLU FFN with a single
stochastic binary hidden layer and a continuous linear readout. It first
distills the conditional-mean network, then continues with sampled p-bit
activations and a straight-through gradient estimator.

The initial experiment targets decoder layer 12 and uses bipolar activations.
The final evaluation inserts the student into the complete language model and
reports WikiText-2 perplexity for deterministic conditional means and for
1, 4, 8, and 16 samples.

The continuously maintained Chinese design document
[`P_DNN_FFN_DESIGN_ZH.md`](P_DNN_FFN_DESIGN_ZH.md) explains every computation
stage, the exact binary/continuous boundaries, training method, current
results, limitations, and the required format for recording future versions.

The v1 full-path experiment adds an input p-bit layer, keeps the p-bit layer
between the two matrices, and averages only after the continuous readout. Both
matrices therefore receive bipolar inputs. Its implementation is in
`full_path_pdnn_ffn.py`, with separate training and perplexity scripts. Results
are in
[`results/full_path_pdnn_layer12_bipolar_t0p25/RESULTS.md`](../../results/full_path_pdnn_layer12_bipolar_t0p25/RESULTS.md).

`evaluate_multi_layer_full_path_perplexity.py` installs any number of trained
students into one Qwen model. The first progressive test covers layers 6 and
12; layers 6, 12, and 18; and a four-layer set including layer 0. Its results
show that layer 0 dominates the catastrophic four-layer degradation. A second
screen of layers 9, 15, 21, and 23 selected `{9,12,15,18}`, which reached
four-path PPL 15.3493 +/- 0.0051 without layer 0. The remaining accumulated
error shows that independent local distillation still needs joint end-to-end
adaptation before all 24 FFNs can be converted. See
[`results/full_path_pdnn_progressive_2_4_layers/RESULTS.md`](../../results/full_path_pdnn_progressive_2_4_layers/RESULTS.md).

The expansion experiment trains students for all 24 layers, ranks them by
single-layer full-model PPL, and evaluates cumulative replacement from four to
24 FFNs. A fixed corpus split confirms that eight replacements is the lenient
boundary and that nine to ten independent replacements is already
unacceptable. `--start-token` can be combined with `--max-tokens` in all PPL
evaluators to keep selection and final evaluation segments disjoint. See
[`results/full_path_pdnn_expansion_5_24_layers/RESULTS.md`](../../results/full_path_pdnn_expansion_5_24_layers/RESULTS.md).

`train_joint_full_path_distillation.py` inserts multiple existing students
into one model and jointly adapts them against frozen original-Qwen logits and
next-token loss. On layers `{10,11,12,13}`, 1,000 updates reduced N=4 PPL from
13.8953 to 13.1535 in 225 seconds on one RTX 5090. See
[`results/joint_full_path_pdnn_layers10_13_v1/RESULTS.md`](../../results/joint_full_path_pdnn_layers10_13_v1/RESULTS.md).

The ten-layer extension uses layers `{7,8,9,10,11,12,13,14,15,18}` and a
staged initialization. In 316 seconds on one RTX 5090, 1,000 updates reduced
the N=4 full-test result from 27.3051 for ten independent students to
19.2261 +/- 0.0069. Mean-field PPL remains 18.2533, so the residual error is
mainly structural rather than sampling variance. See
[`results/joint_full_path_pdnn_layers7_15_18_ten_v1/RESULTS.md`](../../results/joint_full_path_pdnn_layers7_15_18_ten_v1/RESULTS.md).

The twenty-layer staged run leaves only layers 0, 2, 3, and 23 unchanged.
One thousand updates reduce N=4 PPL from 101.3749 to
48.9108 +/- 0.0963 in 439 seconds, with 6.02 GiB peak allocated memory.
Mean-field PPL remains 42.8818, confirming an unacceptable deterministic
structural gap at this depth. See
[`results/joint_full_path_pdnn_layers1_22_twenty_v1/RESULTS.md`](../../results/joint_full_path_pdnn_layers1_22_twenty_v1/RESULTS.md).

The layer-specific encoding experiment adds binary 0/1 operation and optional
learned scalar thresholds and positive temperatures to `full_path_pdnn_ffn.py`.
Across layers 10, 12, and 19, learned encoding consistently lowers local error
and full-model perplexity. Composing all three improves N=4 PPL from
14.0926 +/- 0.0032 to 13.8903 +/- 0.0016. The accompanying saturation tool
measures how much probability mass lies below 0.01 or above 0.99. See
[`results/layerwise_binary_encoding_20260929/RESULTS_ZH.md`](../../results/layerwise_binary_encoding_20260929/RESULTS_ZH.md).

Extending learned 0/1 encoding to the ten-layer set and jointly adapting all
students reduces full-test N=4 PPL from 23.7393 to 18.5686 +/- 0.0009. This
beats the earlier bipolar ten-layer result by 0.6575 PPL, while the remaining
mean-field PPL of 17.4551 still identifies structural approximation as the
main limit. See
[`results/joint_binary_learnable_encoding_layers7_15_18_ten_v1/RESULTS_ZH.md`](../../results/joint_binary_learnable_encoding_layers7_15_18_ten_v1/RESULTS_ZH.md).

The twenty-layer learned-encoding extension reduces N=4 PPL from the earlier
48.9108 to 43.3831 +/- 0.0100, while mean-field remains 38.0083. Restoring late
layers 17, 20, 21, and 22 to their original Qwen FFNs lowers an indicative
seed-0 ablation to 31.6983, locating a major source of accumulated error. See
[`results/joint_binary_learnable_encoding_layers1_22_twenty_v1/RESULTS_ZH.md`](../../results/joint_binary_learnable_encoding_layers1_22_twenty_v1/RESULTS_ZH.md).

Run with:

```bash
CUDA_VISIBLE_DEVICES=7 bash experiments/pdnn_ffn/run_layer12_experiment.sh
```

This is a layer-distillation feasibility test. It does not establish hardware
speed or energy savings, and replacing only one of 24 FFNs does not represent
the quality of a model whose complete FFN stack has been converted.

## Binary gated dual-rail FFN

`GatedPBitFFN` adds two independently sampled 0/1 branches, `gate` and `value`.
The rails `gate*value` and `gate*(1-value)` feed the same readout weight; their
continuous outputs are subtracted and the readout bias is added once. All
matrix biases are retained upstream of p-bits; only temperatures are learned,
with no separate p-bit threshold. Full paths are averaged only at the output.

Use `train_full_path_distillation.py --architecture gated_dual_rail --coding
binary --temperature-only --hidden-sizes 3242` together with its required model,
training text and output directory arguments. Existing PPL and joint-training
scripts load the architecture from its checkpoint. The default serial model
and legacy checkpoint behavior are preserved.

See section 23 of [the evolving design document](P_DNN_FFN_DESIGN_ZH.md) for the
equations, hardware interface, matched-parameter protocol and compute cost.
The completed three-layer pilot did not beat the matched serial controls; see
[full results](../../results/gated_dual_rail_20260929/RESULTS_ZH.md).

The layer-12 continuation forks common warm-start checkpoints into N=4/N=16
training arms for 6,000 updates. Longer training narrows the gated/serial N=4
PPL gap to 0.00946, but N=16 training worsens N=4 inference while improving
N=16 inference. Direct repeated-path measurements quantify this bias/variance
tradeoff. See [continuation results](../../results/gated_sample_budget_20260929/RESULTS_ZH.md).

A controlled four-layer test on `{9,12,15,18}` finds no multi-layer benefit
from the gated design. After identical 1,000-update joint adaptation, N=4 PPL
is 13.6933 for gated versus 13.5591 for serial. Mean-field and N=16 also favor
serial, while gated joint training is 15.7% slower in the current dense PyTorch
implementation. See [four-layer results](../../results/gated_four_layer_9_12_15_18_20260930/RESULTS_ZH.md).

The layer-12 width sweep compares three near-equal parameter budgets for serial
and gated models. Increasing the serial model from 8.72M to 13.08M trainable
parameters changes N=4 PPL from 12.0465 to 12.0312; gated improves from
12.0560 to 12.0425. These corrected results evaluate 6000 sample-training updates
for every model; the initial new-width evaluations mistakenly used checkpoints
with only 4000 sample updates. Width helps modestly; the remaining error is not
proven to be a fundamental architectural limit. See the
[width-sweep results](../../results/pdnn_width_sweep_layer12_20260930/RESULTS_ZH.md).

Input position-bit encoding improves quality without widening the student:
4-bit stochastic input reaches N=4 PPL 11.95572 versus 12.04655 for the original
sigmoid input, close to the clipped continuous-input control at 11.95222.
Deterministic 4-bit input reaches 11.95921; its hidden layer still samples.
Bitplanes share matrix weights and receive only 0/1 inputs; the encoding uses
coordinated adjacent rounding, not independent physical sigmoid p-bits. N=4
matrix terms increase 2.5x for K=4. See
[input-encoding results](../../results/input_multibit_layer12_20260930/RESULTS_ZH.md).

The same stochastic K=4 input encoder was then installed at layers
`{9,12,15,18}`. N=4 PPL is 13.68340 for independently trained replacements and
12.90971 after 1,000 joint updates, compared with 14.45444 and 13.55906 for the
matched sigmoid-input serial baseline. Mean-field/N=16 and three N=4 inference
seeds agree on the improvement. The coordinated encoder and 2.5x matrix-call
cost remain material limitations. See
[four-layer multibit results](../../results/input_multibit_four_layer_9_12_15_18_20260930/RESULTS_ZH.md).

Use `--input-encoding continuous_raw --coding binary --temperature-only` for
unclipped floating inputs with sampled binary hidden nodes. This is distinct
from `continuous`, which retains calibrated input clipping. No calibration file
is needed for `continuous_raw`. A matched ten-layer run gives joint N=4 PPL
15.59254 versus 17.83837 for the sigmoid-input control; see
[results and limitations](../../results/continuous_raw_ten_layer_20261001/RESULTS_ZH.md).

`run_continuous_twenty_layer.py` extends this run to 20 layers, retaining
original FFNs 0, 2, 3, and 23. It trains the ten new local students per arm,
audits reused ten-layer checkpoints, and evaluates independent, staged, and
joint models. After 1,000 joint updates, raw-input N=4 PPL is 27.12776 versus
38.76411 for the matched sigmoid-input control. The raw-input best validation
point is the final update, so convergence is not established. See the
[twenty-layer report](../../results/continuous_raw_twenty_layer_20261001/RESULTS_ZH.md).

`multithreshold_and_ffn.py` implements teacher-initialized gate/value projections,
per-channel sigmoid banks, and tied binary single-bit/AND readout terms.
`run_multithreshold_and.py` compares K=L=1/2/4 at layer 12 with common training
budgets; `run_multithreshold_sample_budget.py` evaluates N=1/2 on the same fixed
final checkpoints. N=4 PPL is 11.90592/11.78858/11.72048. Four mathematical
tests cover the binary boundary, exact moments including shared-AND covariance,
factorized/expanded gradients, final averaging, and checkpoint loading.
Training uses the equivalent factorized form; sampled PPL uses expanded 0/1
inputs with FP32 readout and accumulation. The input projections remain
floating point. See the [full report](../../results/multithreshold_and_layer12_20261002/RESULTS_ZH.md)
and section 33 of the living design document for cost and comparison limits.

`run_multithreshold_twenty_layer.py` extends K=4 to 20 FFNs, reusing the
verified layer-12 checkpoint and training 19 additional local students. The
joint wrapper reuses the established KL/CE protocol with the AND-bank loader;
all sampled full-test evaluations use binary expanded readouts. Independent
N=4 PPL is 14.84908, improving to 13.67284 after 1,000 joint updates, with
validation selecting step 800. N=16 reaches 12.74181; mean-field is 12.46957.
Historical architecture comparisons are not
parameter/initialization/precision matched. See the
[twenty-layer results](../../results/multithreshold_and_twenty_layer_20261002/RESULTS_ZH.md)
and section 34 of the design document.

## Fixed-N=4 analytical output-variance ablation (completed)

Three paired continuation seeds, each trained for 2,000 steps from one
reconstructed initializer, held K=4 and inference N=4 fixed. A used KL/CE;
B added normalized analytical conditional variance, with lambda=0.1 selected
only on held-out local validation. All 26 full-test evaluations completed.
N=4 PPL was 13.386293 +/- 0.024100 for A
and 13.390170 +/- 0.029083 for B
(sample standard deviations across continuation-seed means).
The prespecified 2% gain was not met: B changed PPL by
+0.029% and won only 1/3 pairs.
Shared-input local variance fell by 0.057%,
while recorded training time increased by 31.6%.
A layer-12 actual binary-AND sampling check agreed with analytical variance
within 0.03%; this does not establish whole-model variance or hardware speed.

See the [full report](../../results/and_variance_20261005/RESULTS_ZH.md),
[interpretation and future goals](../../results/and_variance_20261005/ANALYSIS_ZH.md),
[completion audit](../../results/and_variance_20261005/completion-audit.json),
and [prespecified protocol](AND_VARIANCE_PROTOCOL_ZH.md).
Next experiments should first screen lambda in short joint validation with
separate component-gradient measurements. A conditional-mean anchor, if used,
must be identical in both A and B; inference remains K4/N4. These follow-ups
have not been executed.
