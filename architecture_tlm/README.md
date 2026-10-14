# Qwen2.5 macro-level transaction model

This branch captures the current **global uniform random-word supply** model. It has not yet modeled distributed local probability-output p-bit arrays. The current RNG timing results apply to that specific mapping.

The implementation is a Python macro-level loosely timed model, **not a SystemC/TLM-2.0 implementation**. It models batch-one decode through all 24 Qwen2.5-0.5B layers, dense accurate attention, FFN, residuals, normalization and the vocabulary output. Prefill and TTFT are not modeled.

Weights are exposed through a configurable storage macro: capacity, load bandwidth/latency, read/write energy, holding power and area. SRAM costs are scenario parameters. Stochastic fixed-weight core costs are derived from the supplied mapped-power evidence; dual-stream core power is an adjustable proxy.

## Run

Python 3.10 or newer; standard library only. No GPU, model weights, tokenized corpus or original local workspace is required.

```sh
cd architecture_tlm
python import_evidence.py
python checks/test_model.py
python checks/test_system_decode.py
python run_workbench.py
python run_system_sweep.py
python report_system.py
python audit_system_results.py
python checks/check_publication_metrics.py
```

The full sweep includes 96 scenarios. The single-core workbench remains available for storage/RNG checks.

## Configuration and results

- [Default system parameters](configs/system_decode_default.json)
- [Cost provenance and limitations](configs/cost_provenance.json)
- [Bundled Qwen configuration](configs/model_config.json)
- [Chinese implementation guide](SYSTEM_MODEL_ZH.txt)
- [Full Chinese bottleneck report](results/system_decode/完整decode宏模型与瓶颈分析.txt)
- [Scenario table](results/system_decode/summary.csv)
- [Sensitivity results](results/system_decode/sensitivity.json)
- [Consistency audit](results/system_decode/verification.json)

Scenario energies and areas are not measured chip values. The digital and stochastic resource configurations are not strictly area-matched. Numerical PPL references do not validate the hardware precision or random-stream correlations. In particular, do not transfer the global uniform-source bottleneck results to a native probability-output p-bit module without replacing its supply and cost contract.

Power CSVs and the RTL reference package are included in `evidence/`; their original bytes are preserved and checked by SHA256. No technology library, credentials, SSH configuration or model weights are included.

## Snapshot

The publication changes only input paths and evidence verification so that this snapshot runs outside the original research directory. All 96 published scenario metrics are checked against the pre-publication results.
