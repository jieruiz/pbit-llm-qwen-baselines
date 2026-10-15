# Local A8 share64 FFN RNG TLM

This self-contained TLM variant evaluates one FFN-local 32-bit PRNG that emits
an 8-bit uniform word and broadcasts it to 64 input probability comparators.
It regenerates an input stream by restoring a 32-bit checkpoint instead of
storing 4096 random words. The weight-side stochastic generator, Qwen2.5-0.5B
decode work, dense attention, weight SRAM model and external-memory interface
are inherited from the shared TLM.

Directory layout:

- `config/`: A8 local-source and inherited weight-RNG macro parameters.
- `tests/`: unit tests and the compact RTL port-0 reference vector.
- `results/`: the checked 18-scenario summary, comparison and audit status.
- `rng_local_shared_v2.py`: seed, checkpoint, replay and broadcast accounting.
- `system_decode_local_shared_v2.py`: the variant scheduler.
- `system_decode_base.py`: pinned shared decode base with hierarchical-RNG hooks.
- `run_local_shared_v2.py`: full experiment driver.
- `audit_local_shared_v2.py`: saved-run consistency checks.
- `README_ZH.md`: detailed Chinese design and accounting notes.

From the repository root:

```sh
python architecture_tlm/variants/local_shared_a8_64/tests/test_local_shared_v2.py
python architecture_tlm/variants/local_shared_a8_64/run_local_shared_v2.py
python architecture_tlm/variants/local_shared_a8_64/audit_local_shared_v2.py
```

The default run writes detailed JSON files to `results/generated/`. The checked
summary in `results/` uses the simplified Link memory backend. It is a macro
estimate; local-LFSR PPA and actual LFSR-driven full-model PPL remain unmeasured.
