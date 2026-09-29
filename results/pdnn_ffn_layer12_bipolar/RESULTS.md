# Layer 12 P-DNN FFN experiment

This experiment replaces only decoder layer 12's Qwen SwiGLU FFN:

```text
896 continuous inputs -> 4864 bipolar p-bits -> 896 continuous outputs
```

The student has 8,722,048 parameters, compared with 13,074,432 parameters in
the original FFN. It uses a dense input projection, stochastic bipolar hidden
activations, and a continuous linear readout. This is a functional software
experiment and does not model sparse physical connectivity or demonstrate
hardware speed or energy savings.

## Training

- Teacher: Qwen2.5-0.5B Base, BF16, frozen
- Data: WikiText-2 raw train, 2,518,423 tokenizer tokens
- Distilled layer: decoder layer 12
- Phase 1: 2,000 conditional-mean steps, batch 4 x sequence length 256
- Phase 2: 2,000 sample-aware steps with 4 p-bit samples and an STE
- Total examples processed: 4,096,000 tokens
- GPU: one NVIDIA GeForce RTX 5090 32 GB
- Training time: 66.20 seconds
- Peak allocated training memory: 1,244,770,304 bytes (1.16 GiB)
- Train corpus SHA-256: `6707892fa3788b5ab9ed78ab5ff37d9fe825f6011a2ad4fcd6a6d467f0e7da57`

The conditional-mean checkpoint reached validation normalized MSE 0.4357 and
cosine similarity 0.7527. The final sample-aware checkpoint at 4 samples
reached normalized MSE 0.4997 and cosine similarity 0.7076.

The large checkpoints remain on the experiment server and are excluded from
Git. Their artifact identities are:

| Checkpoint | Bytes | SHA-256 |
| --- | ---: | --- |
| `student_mean.pt` | 104,671,393 | `61e5eecadf9d254ae3fc395eefcaf345ead42600859e38cbf487b74bd35215de` |
| `student_sampled.pt` | 104,671,459 | `0557b82fcccb1e7f5659eb1fd54e599fbfbe3e490db34525257768f3439933bd` |

## Full WikiText-2 test perplexity

All runs use the same 299,078-token corpus, 2,048-token windows, and stride
1,024 as the BF16 reference.

| Checkpoint | Samples | Seed | PPL | Change from 11.652735 |
| --- | ---: | ---: | ---: | ---: |
| Original Qwen | deterministic | 0 | 11.652735 | 0.00% |
| Mean-distilled | conditional mean | 0 | 12.048572 | +3.40% |
| Mean-distilled | 1 | 0 | 13.463740 | +15.54% |
| Mean-distilled | 4 | 0 | 12.356535 | +6.04% |
| Mean-distilled | 8 | 0 | 12.205344 | +4.74% |
| Mean-distilled | 16 | 0 | 12.123609 | +4.04% |
| Sample-aware | conditional mean | 0 | 12.037538 | +3.30% |
| Sample-aware | 1 | 0 | 12.175210 | +4.48% |
| Sample-aware | 4 | 0 | 12.070811 | +3.59% |
| Sample-aware | 4 | 1 | 12.074098 | +3.62% |
| Sample-aware | 4 | 2 | 12.073038 | +3.61% |
| Sample-aware | 8 | 0 | 12.055022 | +3.45% |
| Sample-aware | 16 | 0 | 12.046654 | +3.38% |

The three sample-aware 4-sample runs have mean PPL 12.072649 and sample
standard deviation 0.001678. Sample-aware training substantially improves the
one-sample result and makes four samples approach the student's deterministic
conditional-mean limit.

## Interpretation

The experiment supports the feasibility of a P-DNN replacement at one FFN:
the complete language model remains usable, and four stochastic samples incur
about 3.6% perplexity degradation. Most of the remaining gap is structural:
the conditional-mean student is already about 3.3% worse than the original
model. Additional samples cannot remove that error.

This result must not be extrapolated directly to replacing all 24 FFNs. Errors
can compound across layers, and the current dense projections do not satisfy a
fixed-degree p-bit hardware graph. The next useful experiment is progressive
multi-layer replacement with end-to-end language-model distillation, while
keeping this single-layer result as a regression baseline.
