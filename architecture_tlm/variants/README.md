# TLM variants

This directory keeps architecture experiments separate from the shared decode
model and the DRAMSys backend:

- [`local_shared_a8_64/`](local_shared_a8_64/): 8-bit input probability words,
  one FFN-local PRNG word shared by 64 comparators, state checkpoint/replay,
  and the original weight-side stochastic generator.

Each variant owns its configuration, tests, audit, saved summary and design
notes. Shared dense-attention, weight-SRAM and external-memory code remains in
the parent `architecture_tlm/` directory.
