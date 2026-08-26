# Camera-ready manuscript checks

The tracked Overleaf source is
`_overleaf_/ARR_May_Revision_Overleaf/emnlp_2026.tex`.

## Local PDF build

Run from the repository root:

```bash
bash scripts/manuscript/compile_emnlp_local.sh
```

The script requires `pdflatex`, `bibtex`, and the standard LaTeX packages
`multirow` and `placeins`. It performs one LaTeX pass, BibTeX, and two final
LaTeX passes, copies the PDF to
`archive/local_exports/emnlp_2026_camera_ready.pdf`, and rasterizes every page
for visual inspection. Rasterization requires either `pdftoppm` or
Ghostscript. The temporary TeX format cache is isolated under `/tmp`, avoiding
failures when the account's default TeX cache is not writable.

The 2026-08-24 receipt used pdfTeX 1.40.19 (TeX Live 2018), CTAN `multirow`
2.9 (2024-11-12), CTAN `placeins` 2.2 (2005-04-18), and Ghostscript
rendering. It produced a 20-page PDF with Conclusion ending on content page 9,
followed by the excluded Limitations and Ethical Considerations sections,
references, and the appendix.

## Citation audit

Run the reproducible bibliography check with:

```bash
python3 scripts/manuscript/audit_citations.py \
  --paper-root _overleaf_/ARR_May_Revision_Overleaf \
  --output /absolute/path/to/citation-audit.json
```

It parses multiline citation groups and rejects missing or duplicate keys,
duplicate normalized titles, and cited entries whose author list uses
`and others`. A cited entry without a URL, DOI, or arXiv eprint is reported as
a warning for manual review.

## Long-context evidence gates

First validate each formal arm:

```bash
python3 scripts/manuscript/audit_long_context_results.py \
  --root /absolute/path/to/formal-results \
  --length 65536 \
  --arm baseline \
  --output /absolute/path/to/baseline-audit.json
```

Use `--arm intervention` for exclude-self Flash-LSE. Once both manifests are
complete, compare them with:

```bash
python3 scripts/manuscript/compare_long_context_results.py \
  --baseline /absolute/path/to/baseline-audit.json \
  --intervention /absolute/path/to/intervention-audit.json \
  --position-diagnostic-receipt /absolute/path/to/position-diagnostic-receipt.json \
  --output /absolute/path/to/matched-comparison.json
```

The comparison refuses partial task sets or mismatched per-sample input
signatures. It reports both the direct mean over 13 tasks and the five-family
macro used by different paper tables. It also refuses an intervention manifest
when any task lacks the required position-bin diagnostics; warning-only audit
rows therefore cannot be silently promoted into a paper aggregate. A separate
reviewed diagnostic receipt must first gain an explicit, schema-validated gate
before it can satisfy this requirement.

That receipt must bind, rather than merely name, all of the following: the
diagnostic result and sample-file SHA-256 values; exact model snapshot,
Transformers version, source commit, context length, task, and sample identity;
exclude-self `flash_lse`; layers `[9,28)`; parallel/perpendicular scales `0/1`;
the positive `XSA_STATS_POSITION_BINS` value; finite aggregate and per-layer
statistics; and, for every enabled bin, `self_probability`,
`nonself_over_total`, `self_over_total`, and `para_nonself_over_total` under
both the overall branch and every active layer. The diagnostic sample identity
must match a row in the corresponding formal intervention shard. A receipt
that only hashes an arbitrary JSON file is insufficient.

Generate and validate it with:

```bash
python3 scripts/manuscript/audit_position_diagnostic.py \
  --result /absolute/path/to/diagnostic-result.json \
  --samples /absolute/path/to/diagnostic-samples.jsonl \
  --formal-audit /absolute/path/to/intervention-audit.json \
  --allowed-git-hash e0cbc87b21c72dc88bfda8b46752eab22bbeff0c \
  --output /absolute/path/to/position-diagnostic-receipt.json
```

The validator fails closed if the diagnostic lacks any bin/layer/metric, uses
the wrong runtime contract, contains a nonfinite value, or logs an input not
present in the audited formal shard.

The audit also requires every sample row to contain a unique `doc_id` and
valid SHA-256 `prompt_hash` and `target_hash`; absent hashes can therefore not
produce a false matched-input signature. Intervention manifests record whether
position-bin diagnostics were enabled. The current formal launcher used
`XSA_STATS_POSITION_BINS=0`, so those task artifacts carry an explicit warning
and require a separate reviewed position-diagnostic receipt before paper
integration.

By default the audit accepts only `git_hash=e0cbc87b`. A full 40-character
commit may be supplied with `--allowed-git-hash`; the validator safely resolves
the 7-or-more-character abbreviation emitted by lm-eval only when it uniquely
matches an allowed commit. The four reviewed legacy Beacon 64k intervention
shards report `git_hash=efcde87`; after their raw artifacts are recovered, an
explicitly reviewed composite audit may declare both hashes by repeating
`--allowed-git-hash`. Never broaden this list merely to make an unexplained
mismatch pass.

Capture the exact formal environment and model contents once per runtime with
`capture_long_context_environment.py`. The receipt contains no credentials; it
hashes the complete local snapshot inventory, key implementation files, and
launcher, and records Python/PyTorch/CUDA/driver/package/FlashAttention
provenance. Store the generated JSON under the corresponding
`results/jetski_runs/**/evidence/` tree, not in the model cache or paper source.
