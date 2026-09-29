# Progressive multi-FFN full-path P-DNN experiment

This experiment applies the two-matrix full-path bipolar P-DNN to multiple
Qwen2.5-0.5B decoder FFNs. It tests whether independently distilled layer
replacements can be composed before attempting all 24 FFNs.

Each replacement keeps the v1 structure and training protocol:

```text
continuous FFN input
  -> input p-bits (-1/+1)
  -> W_in
  -> hidden p-bits (-1/+1)
  -> W_out
  -> average complete-path readouts
```

All layers use input temperature 0.25, hidden temperature 1.0, 2,000
mean-field warm-up steps, and 2,000 four-path sample-aware steps. Each layer is
distilled independently from the original frozen Qwen model.

## Training time and 24-layer estimate

| Decoder layer | Training seconds | N=4 local normalized MSE |
| ---: | ---: | ---: |
| 0 | 72.3658 | 0.326524 |
| 6 | 75.1327 | 0.614678 |
| 12 | 70.9996 | 0.597783 |
| 18 | 71.7043 | 0.609486 |
| **Mean** | **72.5506** | — |

The measured total for 24 independent students is about 29.0 GPU-minutes.
With seven free RTX 5090 GPUs, four parallel waves would take roughly 6–8
minutes including repeated model loading and checkpoint writes. Sequential
training on one GPU would take roughly 30–35 minutes.

Each layer writes three approximately 104.7 MB checkpoints (`student_mean`,
`student_latest`, and `student_sampled`). Keeping all three for 24 layers would
use about 7.5 GB; retaining only mean and final sampled checkpoints would use
about 5.0 GB.

The training time is practical. The accuracy results below show that the
current independent local-distillation method is not ready for all 24 layers.

## Single-layer sensitivity

| Replaced layer | Paths | PPL | Change from 11.652735 |
| ---: | ---: | ---: | ---: |
| 0 | mean-field | 30.180632 | +159.01% |
| 0 | 4 | 40.512179 | +247.66% |
| 6 | 4 | 12.627425 | +8.36% |
| 12 | 4, 3-seed mean | 12.119894 | +4.01% |
| 18 | 4 | 12.497141 | +7.25% |

Layer 0 is a decisive counterexample to selecting layers by local MSE. It has
the best local normalized MSE of the four measured layers, yet replacing it
alone severely damages full-model perplexity. Errors in the first decoder
block propagate through all subsequent blocks, and the locally normalized
output loss does not capture that downstream sensitivity.

## Progressive combinations

| Replaced layers | Paths | Seed | PPL |
| --- | ---: | ---: | ---: |
| 12, 18 | 4 | 0 | 13.038061 |
| 6, 12 | mean-field | 0 | 13.067955 |
| 6, 12 | 4 | 0 | 13.204659 |
| 6, 12 | 4 | 1 | 13.198722 |
| 6, 12 | 4 | 2 | 13.204057 |
| 6, 12 | 16 | 0 | 13.098192 |
| 6, 12, 18 | mean-field | 0 | 14.089006 |
| 6, 12, 18 | 4 | 0 | 14.302053 |
| 6, 12, 18 | 4 | 1 | 14.311866 |
| 6, 12, 18 | 4 | 2 | 14.320606 |
| 6, 12, 18 | 16 | 0 | 14.141447 |
| 0, 6, 12, 18 | mean-field | 0 | 35.013256 |
| 0, 6, 12, 18 | 4 | 0 | 46.348907 |
| 0, 6, 12, 18 | 4 | 1 | 47.299433 |
| 0, 6, 12, 18 | 4 | 2 | 46.653581 |
| 0, 6, 12, 18 | 16 | 0 | 40.450719 |

The `{6,12}` four-path mean is **13.202479 ± 0.003268**. The `{6,12,18}`
four-path mean is **14.311508 ± 0.009282**. The four-layer set containing layer
0 has mean **46.767307 ± 0.485361**.

For the stable middle and late layers, negative-log-likelihood degradation is
approximately additive but includes a growing interaction term. The measured
single-layer NLL increases for layers 6, 12, and 18 sum to 0.18960; the
three-layer increase is 0.20552. This is consistent with independently trained
students receiving shifted input distributions after upstream replacements.

Increasing the number of paths reduces only the stochastic part. For layers
`{6,12,18}`, mean-field PPL is 14.0890, 16-path PPL is 14.1414, and four-path
PPL is 14.3115. More samples cannot remove the large deterministic structural
and distribution-shift error.

## Evaluation-time scaling

On one RTX 5090, the complete 299,078-token WikiText-2 PPL loop at four paths
took approximately:

| Replacements | Evaluation seconds | Total process seconds |
| ---: | ---: | ---: |
| 2 | 7.5–7.8 | 13.7–14.4 |
| 3 | 7.9–8.1 | 14.1–14.6 |
| 4 | 8.3–8.4 | 15.2–15.6 |

The vocabulary projection and unchanged transformer work dominate this small
model's PPL runtime. A linear extrapolation suggests a 24-layer four-path PPL
loop around 15–20 seconds and a total process time around 25–40 seconds on the
same GPU. This is an engineering estimate; quality failure currently prevents
a meaningful 24-layer accuracy run.

## Decision

Do not train all 24 independent replacements with the current objective. The
hardware and wall-time budget is sufficient, but the four-layer test already
shows unacceptable quality loss, especially at layer 0.

The next useful experiment is joint end-to-end adaptation of the already
trained students for layers `{6,12,18}`. Keep all three replacements active,
unfreeze their parameters, and train against teacher logits and next-token
loss so each student sees the shifted upstream distribution. Early layers such
as layer 0 should remain original until this joint method is shown to recover
the three-layer PPL. After recovery, expand in the order 3, 6, 12, and 24
layers, with a full-model PPL gate at every stage.

## Checkpoint identities

Large checkpoints remain on the experiment server.

| Layer | Sample-aware checkpoint SHA-256 |
| ---: | --- |
| 0 | `4cc3c864477f56d63e59f8ec392fabd986201a32477de3259b9cf04f4da44dd6` |
| 6 | `bdea3a5fe756e9e786d5597757ad5b5076a1ec950e255ae3a805a438b0208208` |
| 12 | `24cdcf43c8e643ea27ccb1427218b1b97b53f80a2750b09e1d3b62f18e046912` |
| 18 | `2f7d5cc7a0e40927e9a93cfd51e6e6e7c6f4dd9ff1f07269e3ad825d25cf54a8` |

