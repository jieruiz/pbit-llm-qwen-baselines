# P-bit attention with original Qwen FFNs

This experiment changes only the decode-time attention value aggregation in
all 24 layers of Qwen2.5-0.5B Base. All FFNs, Q/K/V/output projections, RoPE,
and dense prefill are retained. No training is performed.

- `sampling.py`: categorical selection through a binary Bernoulli tree, and
  independent per-position Bernoulli sampling compressed as Binomial counts.
- `attention.py`: decode-only patch for Transformers 4.45.2.
- `evaluate.py`: true cached, teacher-forced decode with identical held-out text
  positions across methods. Prefixes are exact; subsequent caches include the
  effects of stochastic attention.
- `test_sampling.py`: Monte Carlo means/variances, tree padding, boundary
  probabilities, fixed versus variable selection counts, and literal-bit law.
- `test_integration.py`: unchanged FFN objects, bit-identical prefill and SDPA
  delegation, and manual-dense versus SDPA numerical comparison.
- `run_experiment.py`: original 25 evaluations on eight idle GPUs.
- `repeat_long_context.py`: nine additional runs to resolve single-seed 8k
  uncertainty, giving three inference seeds and both dense baselines per context.
- `summarize.py`: audit all 34 runs and source/FFN identities; generate report.

The sampled suffix PPLs at S=512 are 10.42392 (tree) / 10.44439 (independent)
versus 10.37356 original SDPA at 2k, and 9.49162 / 9.49449 versus 9.41898 at 8k.
These use 4096/2048 scored suffix tokens respectively, not full WT2 sliding-window
PPL. They must not be compared directly against the historical 11.652735 result.

This is a probability-law/accuracy reference, not a sparse CUDA implementation.
Both modes use dense FP32 count/S @ V. Binomial counts have the same law as S
independent 0/1 p-bit rounds but do not simulate their physical runtime. The tree
implements IID SANTA, not variance-reduced S²ANTA. Logical address sparsity is
not measured bandwidth savings. Actual devices need probability/drive mapping
and address/control circuitry.

See the [Chinese report](../../results/pbit_attention_20261003/RESULTS_ZH.md).

## Reproduce on one CUDA GPU

Use the repository's CUDA PyTorch setup and `requirements.txt`, including
Transformers 4.45.2. Put the original model at `models/Qwen2.5-0.5B` and the
WikiText-2 test text at `data/wikitext-2/wiki.test.raw` as described in the root
README. Run from the repository root; model loading is local-only.

```bash
python experiments/pbit_attention/test_sampling.py
python experiments/pbit_attention/test_integration.py
python experiments/pbit_attention/evaluate.py --mode tree --samples 128 --seed 0 --context 2048 --examples 32 --decode 128 --batch 16 --output outputs/pbit_attention_reproduction/ctx2048_tree_s128_seed0.json
```

Repeat the evaluation with `--mode independent`, `dense`, and `sdpa`, preserving
the other settings and choosing a new output filename each time. For the 8k
protocol use `--context 8192 --examples 16 --decode 128 --batch 4`; repeat sampled
configurations with seeds 0, 1, and 2. Outputs must not already exist. Reducing
batch size can help with memory but changes the original reproduction settings.

The archived launchers use eight idle GPUs and refuse to overwrite an existing
experiment protocol. They are historical orchestration scripts, not required
for single-GPU reproduction. Archived command records include the original
machine's Python path; use your own activated environment's `python` instead.
The saved GPU test reports document the original run; they are distinct from
the source/result integrity audit performed when publishing this branch.

`summarize.py` verifies all 34 archived results and their source hashes before
regenerating the report/plot (requires matplotlib). Source files are kept with
LF line endings so SHA256 checks also work after a Windows checkout.

## Additional attention methods (2026-10-06)

The original AV sampling baseline above is retained. The independent
[Ising SoftMax and BoltzFormer-inspired supplement](../pbit_softmax_router/README.md)
adds two decode-only alternatives with original FFNs and 140 matched evaluations.
See the [scope and KV-cache notes](../pbit_softmax_router/PUBLICATION_NOTES_ZH.md).
