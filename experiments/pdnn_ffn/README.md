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

Run with:

```bash
CUDA_VISIBLE_DEVICES=7 bash experiments/pdnn_ffn/run_layer12_experiment.sh
```

This is a layer-distillation feasibility test. It does not establish hardware
speed or energy savings, and replacing only one of 24 FFNs does not represent
the quality of a model whose complete FFN stack has been converted.

## Full-path sigmoid coding comparison

`FullPathPBitFFNConfig.coding` selects `bipolar` (the backward-compatible
default, tanh and -1/+1 samples) or `binary` (sigmoid and 0/1 samples).
Both options sample the input and every hidden layer on each complete path,
then average only the continuous readouts. A checkpoint records its coding;
old checkpoints without that field load as bipolar.

`run_coding_comparison.sh` runs three conditions with identical data, layer,
width, seed, optimizer and 2000+2000-step budget: bipolar temperatures
0.25/1.0, binary temperatures 0.25/1.0, and binary temperatures 0.125/0.5.
Set `MODEL_PATH`, `TRAIN_TEXT`, `TEST_TEXT`, and a fresh `COMPARISON_OUTPUT`
directory before running it inside an allocated GPU job. Results include
mean-field and 1/4/8/16-path PPL, plus three inference seeds at four paths.
Use `summarize_coding_comparison.py` to validate and aggregate these results.

Halving temperature aligns sigmoid with the corresponding tanh probability
link at the same field; it does not alone make the networks identical.
Exact recoding additionally requires every affine layer to use `W'=2W` and
`bias'=bias-W*1`, since `s=2b-1`. The CPU tests in
`test_full_path_coding.py` check that identity, strict bit-valued matrix
inputs, readout-only averaging, STE gradients and old checkpoint loading.

## Matched-parameter depth comparison

`run_depth_comparison.py` compares 2, 3, and 4 matrices inside the same decoder
FFN (zero-based layer 12). The shapes are `896-4864-896`,
`896-2192-2192-896`, and `896-1688-1688-1688-896`; parameter counts differ by
less than 0.2%. This is a depth/width tradeoff at a fixed parameter budget,
not a fixed-width capacity expansion or a replacement of more decoder FFNs.
Input and every intermediate activation are sampled 0/1 p-bits, and paths
are averaged only after the final affine readout.

The script keeps sigmoid temperatures at 0.125 (input) and 0.5 (hidden),
uses 2000 mean-field plus 2000 four-path steps, and independently trains each
depth with seeds 0, 1, and 2. Primary PPL compares those three checkpoints at
four paths and inference seed 0. The training-seed-0 checkpoints additionally
get 0/1/8/16-path scans and inference seeds 1 and 2 at four paths. Training-seed
and inference-seed variation are reported separately by
`summarize_depth_comparison.py`.

Run the depth script with `--model`, `--train-text`, `--test-text`, and a fresh
`--output-dir` inside a GPU allocation. It first performs short training and
whole-model evaluation for every full-size shape. `test_depth_configuration.py`
checks the parameter budget, binary boundaries, readout averaging and gradient
connectivity for all depths. The inherited learning rates, temperature and
step budget are controlled settings, not independently optimized for each depth.
