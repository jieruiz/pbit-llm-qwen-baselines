# Joint stochastic FFN and post-softmax PV attention

Branch: `pbit-attention-ffn-joint`. This combines the user-confirmed twenty-layer
K=4 AND FFN joint checkpoint with the existing exact-softmax categorical-tree
PV sampler. It does not use the Ising SoftMax or BoltzFormer router.

## Frozen configuration

- FFN: historical validation-best joint checkpoint at step 800 from
  `results/multithreshold_and_twenty_layer_20261002/joint/best_layer*.pt`.
  Replace layers 1 and 4 through 22; retain original FFNs 0, 2, 3 and 23.
- Primary combination: 16 FFN paths averaged, 512 categorical PV samples in
  all 24 decode attention layers. Four-path FFNs are the lower-budget control.
- FFNs are active during both prefill and decode. Attention prefill stays exact.
  The prefix KV cache therefore already includes the selected FFN's effects.
- All sampled FFN readouts use expanded 0/1 single-bit and AND features, FP32
  signed readout weights and accumulation, with BF16 entrance projections.
  Token chunks of 256 bound temporary memory without changing the sampling law.
- Attention uses full continuous QK, numerical softmax, Bernoulli-tree category
  sampling, and FP32 `counts/S @ V`. This is logically sparse sampling with a
  dense arithmetic reference, not a sparse CUDA gather kernel or a speed claim.
- The FFN global CUDA RNG and attention private generator have separate state;
  both are initialized from the inference seed. Attention sampling does not
  advance FFN state. Paired comparisons reuse FFN uniform draws across modes.
- No new training, checkpoint selection, or test-driven hyperparameter tuning.

The Oct 5 variance-regularization branch uses separately trained checkpoints on
another server. It is not the checkpoint used here. The user confirmed the
previously successful twenty-layer AND joint model after this distinction was
explained; this study does not claim to rank all historical training seeds.

## Factorial controls and interpretation

The fixed plan has 48 evaluations over 2k and 8k contexts: original, attention
only, FFN only, and both; N=4/N=16 FFNs; manual dense attention references; and
N=0 conditional-mean FFN diagnostics. Main stochastic conditions have three
inference seeds. An N=0 network is not the exact mean of the whole random model.

Use the same WikiText-2 suffix positions as prior attention experiments:
32 examples × 128 scored tokens at 2k and 16 × 128 at 8k, batch 16/4 respectively.
These are cached teacher-forced decode subset PPLs, not full-test PPL. Compare
within a context and matched conditions. Three inference seeds do not quantify
training variation or broad task generalization.

The report includes `NLL(both) - NLL(FFN) - NLL(attention) + NLL(original)` to
describe interaction on this subset. Small positive/negative values are not
proof of universal amplification/cancellation. Per-seed and per-example paired
differences are retained. Dense numerical references distinguish FP32 manual
attention from fused BF16 SDPA.

## Files and reproduction

- `evaluate.py`: checkpoint identity verification, chunked binary FFN evaluation,
  cached decode and independent RNG-state plumbing.
- `test_integration.py`: real checkpoint hashes, original delegate equivalence,
  unchanged FFN RNG stream and matched prefix, finite combined outputs.
- `run_experiment.py`: frozen 48-evaluation schedule. It refuses duplicate launch
  and uses only GPUs reporting under 500 MiB allocated and under 5% utilization.
  Review resource availability before using it on a shared machine.
- `summarize.py`: validate all results and exact executed source snapshots,
  aggregate PPL and paired NLL changes, regenerate the report and figure.
- [Results](../../results/pbit_joint_20261006/RESULTS_ZH.md),
  [protocol](../../results/pbit_joint_20261006/protocol.json),
  [integration checks](../../results/pbit_joint_20261006/integration.json).

Use Python 3.11+, CUDA-enabled PyTorch (executed with 2.7.1+cu128), Transformers
4.45.2, and matplotlib for the figure. Run from repository root. Obtain the Base
Qwen model and WikiText-2 through the existing repository instructions. The
twenty checkpoints stay on the training server; weights and corpus are not in Git.
Checkpoint names and SHA256 identities are in the earlier joint N16 evaluation
manifest and each combined evaluation JSON. Do not substitute another model
without recording a new protocol and expected checkpoint manifest.

Example single matched 2k evaluation:

```bash
python experiments/pbit_joint/evaluate.py \
  --ffn and --ffn-samples 16 --mode tree --samples 512 --seed 0 \
  --context 2048 --examples 32 --decode 128 --batch 16 --ffn-chunk 256 \
  --checkpoint-dir results/multithreshold_and_twenty_layer_20261002/joint \
  --checkpoint-manifest results/multithreshold_and_twenty_layer_20261002/evaluation/joint/n16_seed0.json \
  --output outputs/joint_reproduction/ctx2048_and_n16_tree_seed0.json
```

For a fresh full run use a separate checkout, archive the published
`results/pbit_joint_20261006/` directory first, run `test_integration.py`, then
`run_experiment.py`. Record the exact runtime source bytes under
`results/pbit_joint_20261006/executed_sources/<repository-relative source path>`
(paths appear in evaluation JSON) before running `summarize.py`. Published
snapshots preserve legacy CRLF where applicable; the audit also checks their
equivalence to repository sources after line-ending normalization.

Launcher logs/commands/progress are transient; the published evidence consists
of evaluation JSON, protocol, integration and smoke checks, source snapshots,
summary, plot, and audit. No physical p-bit device, throughput, or energy result
is claimed, and full KV storage is still required.
