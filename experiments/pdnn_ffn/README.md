# P-DNN replacement for one Qwen2.5 FFN

This experiment replaces one Qwen2.5-0.5B SwiGLU FFN with a single
stochastic binary hidden layer and a continuous linear readout. It first
distills the conditional-mean network, then continues with sampled p-bit
activations and a straight-through gradient estimator.

The initial experiment targets decoder layer 12 and uses bipolar activations.
The final evaluation inserts the student into the complete language model and
reports WikiText-2 perplexity for deterministic conditional means and for
1, 4, 8, and 16 samples.

Run with:

```bash
CUDA_VISIBLE_DEVICES=7 bash experiments/pdnn_ffn/run_layer12_experiment.sh
```

This is a layer-distillation feasibility test. It does not establish hardware
speed or energy savings, and replacing only one of 24 FFNs does not represent
the quality of a model whose complete FFN stack has been converted.
