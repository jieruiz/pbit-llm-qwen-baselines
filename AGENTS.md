# Experiment reproducibility

- Before comparing a new coding or architecture against recorded results, verify model revision and train/test text SHA-256, then rerun the unchanged BF16 reference in the experiment environment.
- Salesforce/WikiText parquet represents whitespace-only raw lines as empty strings. To reproduce this repository's `wiki.train.raw` and `wiki.test.raw`, restore each empty row as `" \n"` (one space followed by LF), concatenate rows without an additional separator, and require the recorded corpus hashes. Do not assume ordinary newline joining recreates the raw corpus.
- Keep `bipolar` as the default for older full-path checkpoints without a `coding` field. Both encodings must keep every matrix input binary on sampled paths and average only after the final readout. Run `experiments/pdnn_ffn/test_full_path_coding.py` after changing these boundaries.

- Preserve LF line endings for result artifacts covered by SHA-256 manifests. Verify committed blob bytes against the manifest before publishing; Windows CRLF conversion changes hashes even when JSON values are unchanged.
