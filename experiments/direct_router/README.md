# Direct Bernoulli p-bit block router, 2026-10-07

This is an original Qwen adaptation, not a reproduction of BoltzFormer. All Qwen
FFNs and backbone weights remain original and frozen. Only cached, single-query,
unpadded decode attention in all 24 layers is changed. Prefill remains SDPA.

## What runs at inference

1. Partition RoPE-transformed Keys into 64-token blocks. Cache one mean per block
   (`r1`) or four contiguous 16-token means (`r4`). Build the sums once from the
   prefix and then add only the newly appended Key. Retain the full KV cache.
2. For each query/head compute dot products against these summaries, not all Keys.
3. Produce block logits with a head-specific affine map (`r1`, 2,688 parameters
   across 24x14 heads), or a 16-hidden-unit ReLU MLP (`r4`, 64,848 parameters).
   ReLU here is in the small digital selector, not a changed Qwen FFN.
4. A software Bernoulli sample of `sigmoid(logit + head_bias)` emulates a p-bit.
   Force the first and most recent two blocks to be kept. This is an independent
   mask; it does not require Ising coupling, equilibration, a top-k sort or a
   block-level numerical softmax. There is one sampled mask per query/head.
5. `sparse.py` branches on that mask BEFORE loading each K/V tile. Only selected
   tiles execute continuous QK, streaming softmax and PV. No dense teacher QK is
   computed in this path. The exact candidate softmax remains; this experiment
   does not replace every softmax with hardware p-bits.

Features are the summary dot products minus their query-wide maximum, that maximum,
query RMS, relative block age, log block count, log block-size fraction and log
requested budget. Normalization uses training data only. The budget is an input
to the predictor, not a guaranteed retained fraction. A scalar head bias is folded
into its final affine output before driving the p-bit.

## Training and calibration

`collect.py` extracts teacher attention on 32 disjoint, spread-out 8k windows of
WT2 **train**, reserving its final 65,536 tokens as eight validation windows.
Queries at 1k,2k,...,8k yield 256 training and 64 validation queries per head.
WT2 test is not used for gradients, normalization, checkpoint choice or bias fitting.

Let `m_b` be full teacher attention mass in block b and `M=ceil(ratio*C)`.
Soft-label BCE targets `1-(1-m_b)^M`, across ratios 0.125,0.25,0.5. Mandatory blocks
are excluded from this loss. Train for 1,800 AdamW steps and choose the minimum
validation-BCE checkpoint. This is block-inclusion distillation, **not** attention
output matching or end-to-end language-model training.

The initial test screen was weak. A transparently separate second stage
(`calibrate.py`) then adds nonnegative per-layer/head biases so that **expected**
teacher mass retained, averaged over held-out validation queries, is at least 95%
at ratio 0.5. No test labels fit these offsets. This is not a 95% guarantee for
each query. This stage was motivated by the first screen, so it is exploratory;
an independent dataset is needed for a confirmatory conclusion.

## Evaluation and cost scope

- WT2 test, teacher-forced cached decode, original FFNs.
- 2k: 32 windows x 128 scored tokens, batch 16; 8k: 16 x 128, batch 4.
- Same evenly spaced starts as prior experiments. Three inference seeds per
  stochastic configuration. This is sampled suffix PPL, not full-corpus WT2 PPL.
- Initial screen: 54 runs (including SDPA, all-kept sparse/dense controls, old
  mean-router controls, two learned variants at three budgets).
- Second stage: 12 calibrated runs.
- Kernel tests cover GQA, multiple batches, partial blocks, sparse and all-kept
  masks; incremental summary updates are compared against full recomputation.
- All-kept FP32 sparse/dense attention can differ after BF16 rounding and error
  propagation. At 2k PPL is 10.379405 / 10.376637, at 8k 9.425201 / 9.423508;
  these small differences are far below the failed initial router's degradation.

MAC estimates count selected QK and PV rows plus selector projection/network MACs.
They omit reductions, nonlinearities, summary maintenance, memory, launches,
Q/K/V/output projections and FFNs. Summary projection/network overhead is roughly
0.87% (r1) or 5.3% (r4) of dense QK+PV. Candidate count is measured, not inferred
from requested budget. These are not whole-model FLOP or energy savings.

Full KV storage remains allocated. The extra FP32 summary-sum cache is about
1.56% (r1) or 6.25% (r4) of BF16 K+V capacity, excluding temporary tensors.
GQA union is a logical candidate-union statistic, not measured DRAM traffic.

`benchmark.py` separately times real layer-12 Q/K/V snapshots on one idle RTX 5090.
It includes a small benchmark-only subtraction to reset the last summary update.
The dense comparator is Qwen's repeat_kv + SDPA, not the fastest imaginable native
GQA kernel. The selector currently has many tiny PyTorch launches and the sparse
kernel loops over blocks sequentially per head. **The present implementation is
slower overall than SDPA despite computing fewer QK/PV entries.** No hardware p-bit
energy, settling time, or hardware throughput is measured.

## Reproduce on the remote experiment directory

Runtime: `/home/Hongjie_Zeng/pbit_direct_router_20261007`; Python:
`/home/Hongjie_Zeng/.conda/envs/pbit_llm/bin/python`; PyTorch 2.7.1+cu128,
Transformers 4.45.2, Triton 3.3.1, RTX 5090. `models/` and `data/` point to the
existing local Qwen2.5-0.5B and WT2 files.

```bash
python experiments/direct_router/verify.py
python experiments/direct_router/collect.py
python experiments/direct_router/train.py --reps 1
python experiments/direct_router/train.py --reps 4
python experiments/direct_router/validate.py
python experiments/direct_router/run.py
python experiments/direct_router/calibrate.py
python experiments/direct_router/run_calibrated.py
python experiments/direct_router/benchmark.py
ROUTER_CALIBRATED=1 python experiments/direct_router/benchmark.py
```

Use separate idle GPUs for independent runs; the launchers discover GPUs below
500 MiB allocation and keep one process per selected GPU. They resume by skipping
existing result files. Use a new result directory for a new protocol. The small
trained weights and offsets are retained remotely and locally under
`artifacts/direct_router_20261007/`; `.pt` files are intentionally Git-ignored.
Raw teacher calibration tensors remain remote (447 MiB).

Results and the Chinese analysis are in `results/direct_router_20261007/` in the
local worktree. Root `results/` is used when executing the remote isolated copy.
