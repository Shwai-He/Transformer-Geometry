# EMNLP 2026 Findings camera-ready checklist

Baseline inspected: `main@3626beabfe60407b9fd493d84edbdb6399180b84`.

## Verified

- The paper compiles locally to a 20-page PDF with the final ACL style. The
  2026-08-24 export is
  `archive/local_exports/emnlp_2026_camera_ready.pdf` (SHA-256
  `9f7e7dae67ecbc76e073894b3de2155fe3fad54d7287c360751b1e4edaa0f48c`,
  772,498 bytes).
- The official EMNLP 2026 rule allows nine content pages for a final long
  paper. The fresh local PDF ends Conclusion on page 9; the excluded
  Limitations and Ethical Considerations sections follow before References.
  See
  <https://2026.emnlp.org/calls/main_conference_papers/>.
- ARR permits appendices after references without a numeric page limit and
  requires the official double-column format, which the current appendix uses:
  <https://aclrollingreview.org/cfp>.
- Three LaTeX passes plus BibTeX produce no undefined citations or references
  and no overfull boxes.
- All 31 cited keys exist in `references.bib`; labels and normalized titles
  are unique, cited author lists are not truncated with `and others`, every
  cited entry has a URL/DOI/eprint locator, and all references resolve. The
  check is reproducible with `scripts/manuscript/audit_citations.py`.
- Citation metadata for Llama 3, Qwen3, and the Transformer Circuits article
  has been checked against their official paper pages. The prior Llama entry
  incorrectly omitted first author Aaron Grattafiori; it now uses the
  established corporate `Llama Team` form, and Qwen3 uses `Qwen Team`.
- Primary publication locators added or corrected during the citation audit
  (CVF, NeurIPS, PMLR, OpenReview, ACL Anthology, and arXiv) all returned HTTP
  200 on 2026-08-24.
- A 20-page raster contact sheet has been visually inspected; no clipped,
  blank, or grossly malformed page was found. The latest hash-bound export was
  rechecked at full-page scale on pages 1, 9, 10, and 20: the author block and
  internship footnote render, Conclusion remains on page 9, Limitations and
  Ethical Considerations transition cleanly into References, and the final
  appendix figures/captions are not clipped.
- Reproducible figure status is recorded in `CAMERA_READY_FIGURE_AUDIT.md`.
- Recomputed claims and frozen long-context cells are recorded in
  `CAMERA_READY_NUMERIC_AUDIT.md`.
- Formal long-context artifacts are gated by
  `scripts/manuscript/audit_long_context_results.py`.
- The formal Transformers-4.52.4 runtime now has a non-secret environment and
  full model-inventory receipt under
  `results/jetski_runs/formal_runtime_tf4524_evidence_20260824/`. It binds the
  15-file, 12.86-GB snapshot inventory, clean `v2@e0cbc87b`, core-source and
  launcher hashes, package/CUDA/driver/glibc versions, the FlashAttention CUDA
  extension and resolved ABI dependencies, and a real causal
  `return_attn_probs=True` CUDA smoke with finite output and LSE.
- Matched long-context aggregation is gated by
  `scripts/manuscript/compare_long_context_results.py`, which requires all 13
  tasks and matched per-sample input signatures before reporting either the
  13-task mean or five-family macro. A separate position diagnostic is accepted
  only through a passing `audit_position_diagnostic.py` receipt whose source
  result/sample hashes and formal-audit hash are rechecked by the comparison
  gate.

## Frozen pending matched evidence

The existing 4k--128k table in `sections/appendix.tex` predates the active
exact-snapshot runs. Do not replace, combine with, or cite new 64k/128k cells
until all of the following hold for each reported arm:

1. exact local Llama-3.2-3B snapshot
   `13afe5124825b4f3751f836b40dafda64c1ed062`;
2. Transformers 4.52.4 and source commit
   `e0cbc87b21c72dc88bfda8b46752eab22bbeff0c`; the four owner-reviewed
   legacy intervention shards (`niah_single_1`, `niah_single_2`,
   `niah_single_3`, and `niah_multikey_1`) may instead use `efcde87` only
   after their raw result, sample, log, and hash evidence is recovered and
   passes the same audit;
3. all 13 tasks at 500 examples with no sample limit;
4. baseline uses FlashAttention-2;
5. intervention uses exclude-self Flash-LSE, layers `[9,28)`,
   `s_parallel=0`, and `s_perp=1`;
6. no nonfinite values or error termination;
7. result and sample hashes are recorded; and
8. baseline and intervention have matched input document IDs, prompt hashes,
   and target hashes. The harness task hash is not required to match because it
   includes arm-specific model arguments; full sample-file hashes also differ
   because logged generations and metrics are arm-dependent.
9. a reviewed exclude-self run records the required position-bin diagnostics.
   The active formal launcher left `XSA_STATS_POSITION_BINS=0`; score shards
   therefore remain usable as task evidence but cannot be integrated until a
   separate protocol-matched diagnostic receipt closes this instrumentation
   gap. The receipt contract in `scripts/manuscript/README.md` requires exact
   runtime binding, finite overall/per-layer values for every position bin,
   artifact hashes, and a diagnostic sample that matches the corresponding
   formal shard; a standalone unvalidated JSON hash is not sufficient.

## Human confirmation required

- Confirm the final author list and order. The manuscript currently says
  `Shwai He, Haichao Zhang, Shen Yan`, while Shwai He's public publication
  page listed the accepted EMNLP 2026 paper as `Shwai He, Ang Li` when checked
  on 2026-08-24: <https://shwai-he.github.io/>. This conflict must be resolved
  by the authors; do not infer the camera-ready list from either source alone.
- Confirm all three affiliations and whether the ByteDance internship footnote
  should remain. The current custom author block also emits the harmless but
  unresolved Hyperref warning `Hfootnote.1 ... does not exist`; resolve the
  author metadata first, then repair or remove that footnote link without
  changing its approved text.
- Supply or explicitly omit Shen Yan's email; the current author block lists
  only Shwai He's and Haichao Zhang's addresses.
- Confirm the final acknowledgements and any required funding disclosures.
- Confirm that the appendix length is acceptable for the final submission;
  ARR does not impose a numeric appendix limit, but authors should confirm that
  all included material is appropriate for the final submission.
- Provide the exact Responsible NLP Research Checklist responses from the
  accepted OpenReview submission. ARR states that the checklist is published
  as an appendix for accepted EMNLP papers and subsequent *ACL conferences:
  <https://aclrollingreview.org/responsible-nlp-checklist-appendices>. Neither
  the repository/full Git history nor a 2026-08-24 exact-title public
  OpenReview search exposed a unique forum ID or checklist response; do not
  reconstruct answers from the paper text.

## Remaining technical work

- Recover the missing exact sources for the all-layer head-localization and
  composed overview figures. The exact depth-profile and component-scaling
  entry points are now verified.
- Finish and validate the active 64k/128k matched runs.
- Integrate only completed matched aggregates, then rerun the numeric,
  citation, page-count, and visual PDF audits.
