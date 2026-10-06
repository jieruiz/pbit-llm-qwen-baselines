# Query-side p-bit encoding for Qwen attention

This experiment approximates only decode-time QK in all 24 Qwen2.5-0.5B Base
layers. Original FFNs, projections, RoPE, dense prefill, and continuous PV are
retained. There is no training and no value-side sampling.

For each post-RoPE query head, set `a = max(abs(q))`, sample
`b ~ Bernoulli(abs(q)/a)`, retain `sign(q)`, and average B signed bit vectors.
The reconstructed query is `sign(q) * a * count / B`. Zero queries remain zero.
Keys remain continuous. The mean-group-query correction from the paper is not
implemented in this experiment.

- `iid` uses Binomial counts, equivalent in distribution to B independent bits.
- `stratified` uses `floor(B*p) + Bernoulli(frac(B*p))`, exactly the count law
  obtained by comparing p with one independent uniform draw in each of B strata.
- These count shortcuts reproduce the arithmetic distribution, not physical
  p-bit runtime. The reference still uses dense FP32 matmuls and additionally
  computes exact scores for same-state diagnostics. No sparse GPU speedup is claimed.

## Results

70 evaluations: two contexts, three inference seeds per sampled configuration,
IID B=2/4/8/16 and stratified B=2/4/8/16/32/64/128, plus dense baselines.
The larger stratified budgets were added after seeing severe small-budget errors.

| Method | B | 2k PPL | 8k PPL |
| --- | ---: | ---: | ---: |
| Original SDPA | - | 10.37356 | 9.41898 |
| IID | 16 | 27.20501 | 31.80593 |
| Stratified | 16 | 11.57242 | 10.72265 |
| Stratified | 32 | 10.63490 | 9.75520 |
| Stratified | 64 | 10.43481 | 9.50974 |
| Stratified | 128 | 10.38378 | 9.44642 |

At B=64 the PPL increase is below 1% in both contexts, but logical key-feature
access across each seven-query GQA group is above 99.99%. Literal bit-round
accumulation would require about 7.2 times as many signed additions as the
original dense dot product has MAC terms; this is not a latency or energy ratio.
The current per-head encoding has not demonstrated a useful key-bandwidth saving
at high fidelity. Other encodings and shared sampling remain untested.

See the [Chinese report](../../results/pbit_qk_20261006/RESULTS_ZH.md) for the
full tables, diagnostics, scope, and limitations. These are matched sampled
WT2 cached-decode suffixes (4096/2048 scored tokens), not full-test PPL.
The [paper comparison](../../results/pbit_qk_20261006/PAPER_COMPARISON_ZH.md)
separates measured attention distortion from model/protocol differences and
unverified explanations for the gap to the paper's BitNet example.

## Reproduction

Use CUDA PyTorch and Transformers 4.45.2, the original model at
`models/Qwen2.5-0.5B`, and WT2 test text at `data/wikitext-2/wiki.test.raw`.
Run from the repository root:

```bash
python experiments/pbit_qk/test_sampling.py
python experiments/pbit_qk/test_integration.py
python experiments/pbit_qk/evaluate.py --mode stratified --samples 64 --seed 0 --context 2048 --examples 32 --decode 128 --batch 16 --output outputs/pbit_qk_reproduction/ctx2048_stratified_b64_seed0.json
```

For 8k, use `--context 8192 --examples 16 --batch 4`. Repeat seeds 0/1/2,
and compare `iid`, `dense`, and `sdpa` with the same target positions. Each
output path must be new. `--samples` denotes query sampling rounds B, not the
value-sampling S used in `experiments/pbit_attention`.

`run_experiment.py` launches the original 52 configurations across idle GPUs;
`extend_budget.py` adds 18 configurations after they finish. Both refuse to
overwrite existing protocols. The runners archive the original Python executable
path; substitute your environment's Python when replaying individual commands.

`summarize.py` audits all 70 results, source/FFN/corpus hashes, target positions,
and the earlier PV experiment's dense baselines; then regenerates the report and
plot. It requires matplotlib but no GPU. LF source line endings preserve the
recorded byte-level hashes on Windows.
