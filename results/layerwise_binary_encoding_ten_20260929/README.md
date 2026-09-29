# Independent initializers for the ten-layer learned-encoding run

This directory contains the training summaries and JSONL curves for decoder
layers 7, 8, 9, 11, 13, 14, 15, and 18. Each student uses binary 0/1 states,
a width of 4,864, and learned scalar threshold/temperature parameters at the
input and hidden p-bit boundaries. Training used 2,000 mean-field updates
followed by 2,000 four-path sampled updates.

Layers 10 and 12 reuse the matching checkpoints and logs from
[`../layerwise_binary_encoding_20260929/`](../layerwise_binary_encoding_20260929/).
The ten students are jointly adapted and evaluated in
[`../joint_binary_learnable_encoding_layers7_15_18_ten_v1/`](../joint_binary_learnable_encoding_layers7_15_18_ten_v1/).

Large checkpoint files remain on the remote training server and are not stored
in Git.
