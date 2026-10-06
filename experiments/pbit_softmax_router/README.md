# Ising SoftMax and BoltzFormer-inspired Qwen routing

All original Qwen2.5-0.5B weights and FFNs remain fixed. Only all 24 decode
attention modules change; prefill is exact. Same WikiText-2 suffix targets as
the previous 2k/8k attention experiments (4096/2048 scored tokens).

## Ising scope

For L logits there are L-1 bits and degree L-2. The 0/1 energy is
`E=-sum_i (z_i-z_ref)b_i + lambda*sum_(i<j)b_i*b_j`.
All-zero encodes the reference category; single-active states encode the rest.
The reference is the largest logit, making all relative fields nonpositive.
Within valid states the exact conditional distribution is softmax.

The LLM experiment uses the **hard-penalty limit**, with exact continuous-time
heat-bath dynamics, not a finite-conductance hardware model. From all-zero,
birth rates are sigmoid(z_i-z_ref); from active i, death rate is 1-birth_i.
We generate the alternating exponential holding times in bulk and observe
states at fixed clock intervals. This preserves residence weighting and
sample correlations. Counting transition events instead would be biased.
Time is in unit per-bit update clocks, NOT microseconds or GPU runtime.
Burn-in is at target temperature (beta=1); there is no optimization annealing.

diagnostics.py separately checks finite penalties by exact enumeration and
literal asynchronous Gibbs updates, and compares compressed hard-limit
transients with a matrix-exponential solution of the generator. No claim that
the paper's finite circuitry, delays, noise or energy consumption is reproduced.

## Router scope

This is an **untrained language-model adaptation**, not a reproduction of the
trained image BoltzFormer predictor. Each 64-token block has cached key mean
and coordinate variance. The cache is built once from the prefix and updated
incrementally. full-rank and 16-coordinate low-rank proxies are compared.

- block_mean: query dot mean key + log block size.
- block_moment: Gaussian moment approximation, mean score + half variance
  (diagonal covariance) + log block size.
- block_paper: sigmoid confidence with tau=1/(layer+1), inspired by paper.
- oracle: true full-QK block probability mass; diagnostic, cannot save QK.
- oracle_topk: deterministic full-QK relevance control, not a mathematical
  upper bound on end-to-end language-model quality.

Bernoulli inclusion is `1-(1-p_block)^K`. The first and last two blocks are
always retained and included in reported sparsity. Selected blocks get exact
softmax attention. No importance correction: this changes the target function.

The software computes dense QK and masked dense PV for correctness and teacher
diagnostics. Logical selected fractions are NOT measured sparse-kernel speedups.
FFN hashes, source hashes, text hashes, sample positions and wall times are saved.

## Published results and reproduction

- [Chinese publication notes, comparison, and KV-cache limits](PUBLICATION_NOTES_ZH.md)
- [Full results](../../results/pbit_softmax_router_20261006/RESULTS_ZH.md)
- [All 140 evaluations](../../results/pbit_softmax_router_20261006/evaluation/)
- [Audit](../../results/pbit_softmax_router_20261006/audit.json)

Run from repository root using Python 3.11+ and the existing Qwen environment (transformers 4.45.2,
CUDA-enabled PyTorch), plus numpy, scipy and matplotlib. Obtain the Base model
and WikiText-2 using the repository setup instructions. No weights or corpus are
included. The launchers use all GPUs reporting less than 500 MiB allocated;
review that policy before launching on a shared server.

For fresh results use a separate checkout and move the published result directory
to an archive location first: batch launchers skip existing evaluation JSON files.
Preserve analysis_scripts/ before moving the directory and restore it afterward.
The diagnostics step recreates the result directory.

```bash
python experiments/pbit_softmax_router/diagnostics.py
python experiments/pbit_softmax_router/run_experiment.py
python experiments/pbit_softmax_router/repeat_selected.py
python results/pbit_softmax_router_20261006/analysis_scripts/real_logits.py
python results/pbit_softmax_router_20261006/analysis_scripts/confirm_burn.py
python experiments/pbit_softmax_router/summarize.py
```

The first three evaluation stages produce 70 + 60 + 10 runs; real_logits.py
performs the separate transient analysis. The summarizer audits all 140 saved
evaluations and regenerates tables and plots. Published Python sources preserve
the bytes used for recorded evaluation hashes. Supplementary interpretation is
kept in PUBLICATION_NOTES_ZH.md so report regeneration does not overwrite it.
Transient launcher logs, command dumps and progress files are omitted from the
publication; evaluation JSON, protocols, diagnostics and summary data are retained.
