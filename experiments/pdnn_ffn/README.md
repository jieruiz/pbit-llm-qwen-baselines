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
show that independent local distillation needs joint end-to-end adaptation
before all 24 FFNs can be converted. See
[`results/full_path_pdnn_progressive_2_4_layers/RESULTS.md`](../../results/full_path_pdnn_progressive_2_4_layers/RESULTS.md).

Run with:

```bash
CUDA_VISIBLE_DEVICES=7 bash experiments/pdnn_ffn/run_layer12_experiment.sh
```

This is a layer-distillation feasibility test. It does not establish hardware
speed or energy savings, and replacing only one of 24 FFNs does not represent
the quality of a model whose complete FFN stack has been converted.
