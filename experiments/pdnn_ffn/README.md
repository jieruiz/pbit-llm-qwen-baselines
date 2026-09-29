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

Run with:

```bash
CUDA_VISIBLE_DEVICES=7 bash experiments/pdnn_ffn/run_layer12_experiment.sh
```

This is a layer-distillation feasibility test. It does not establish hardware
speed or energy savings, and replacing only one of 24 FFNs does not represent
the quality of a model whose complete FFN stack has been converted.
