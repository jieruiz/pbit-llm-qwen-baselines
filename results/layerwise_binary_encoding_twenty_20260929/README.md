# Additional independent initializers for the twenty-layer run

This directory contains training summaries and JSONL curves for decoder layers
1, 4, 5, 6, 16, 17, 20, 21, and 22. Each student uses binary 0/1 matrix inputs,
a width of 4,864, and learned scalar threshold/temperature parameters. Training
used 2,000 mean-field updates followed by 2,000 four-path sampled updates.

Layer 19 reuses the learned checkpoint in
[`../layerwise_binary_encoding_20260929/`](../layerwise_binary_encoding_20260929/).
The remaining ten layers use the jointly adapted checkpoints from
[`../joint_binary_learnable_encoding_layers7_15_18_ten_v1/`](../joint_binary_learnable_encoding_layers7_15_18_ten_v1/).
The final twenty-layer run is recorded in
[`../joint_binary_learnable_encoding_layers1_22_twenty_v1/`](../joint_binary_learnable_encoding_layers1_22_twenty_v1/).

Large checkpoint files remain on the remote training server and are not stored
in Git.
