# Camera-ready numeric audit

Audit baseline: `main@3626beabfe60407b9fd493d84edbdb6399180b84` on
2026-08-24. This ledger separates values that can be recomputed from committed
evidence from long-context values that remain frozen pending matched manifests.

## Recomputed from committed evidence

### Component-scaling claims

Source:
`_overleaf_/ARR_May_Revision_Overleaf/figs/para_ablation_corrected/ppl_component_scaling_corrected_sources.csv`.
The manifest contains all 54 plotted points, their retained scales, PPL,
matched-no-op deltas, series labels, and original result paths.

- Within value-space parallel retained scales `[0, 1]`, the maximum absolute
  delta is `1.3160003669613367`, reported as `1.32`.
- Full value-space parallel removal has delta `0.460112094774523`, reported as
  `0.46`.
- At retained scales `-0.5`, `-1`, and `3`, the value-space deltas are
  `18.132347754074953`, `188.7744593491261`, and `67.43790154846876`, reported
  as `18.13`, `188.77`, and `67.44`.
- Residual(Attn) and Residual(MLP) parallel scale `0` deltas are
  `21.01062778319317` and `22.278521874268566`, supporting “more than 20.”
- Every series contains its complete expected scale grid and has a zero delta
  at retained scale `1`; the replot validator rejects missing, duplicate, or
  non-no-op scale-1 rows.

### Post-training summary arithmetic

Source: the six task columns in Appendix Table
`tab:appendix-pretraining-downstream`.

- 1.4B displayed task means recompute to `58.47`, `58.78`, and `59.20`, which
  round to the main-table values `58.5`, `58.8`, and `59.2`.
- 2.7B displayed task means recompute to `60.17`, `60.93`, and `61.72`, which
  round to `60.2`, `60.9`, and `61.7`.
- Main-table deltas are reported from the underlying unrounded scores; the
  task-wise one-decimal cells are therefore a display check rather than an
  exact reconstruction of every delta.

### Short-context RULER display arithmetic

Main Table `tab:longcontext-ruler` explicitly states that deltas are computed
before displayed scores are rounded. Each displayed parenthetical delta is
within `0.01` of intervention minus baseline using the displayed two-decimal
scores. The only visible one-hundredth discrepancy (`74.76 - 79.30 = -4.54`
versus reported `-4.53`) is consistent with that declared rounding policy.

## Frozen pending matched evidence

Do not use the active experiment outputs to revise the paper until the audit
and comparison gates pass. In particular:

- 64k baseline currently has only a partial 13-task manifest.
- The local 64k exclude-self run has nine tasks; the four owner-reviewed
  legacy shards still require their raw result, sample, log, and hash evidence.
- 128k baseline and exclude-self runs are incomplete.
- The formal exclude-self launcher has aggregate and per-layer mechanism
  statistics but `xsa_stats_position_bins=0`. A separate reviewed diagnostic
  with position bins enabled is required before final integration; no score is
  invalidated or silently accepted solely because of this missing diagnostic.
- A task metric is not an aggregate. Both the direct 13-task mean and the
  five-family macro may be reported only from complete, input-matched
  manifests, with their aggregation definitions named explicitly.

As of 2026-08-24 22:41 UTC, five 64k task pairs pass both arm audits and have
identical per-sample input signatures. They are retained as partial evidence,
not averaged or inserted into the paper:

| Task | Baseline | Excl.-self, $s_\parallel=0$ | Delta |
| --- | ---: | ---: | ---: |
| `niah_multikey_3` | 0.3720 | 0.0000 | -0.3720 |
| `niah_multivalue` | 0.9105 | 0.0930 | -0.8175 |
| `ruler_cwe` | 0.0066 | 0.0004 | -0.0062 |
| `ruler_fwe` | 0.7827 | 0.3607 | -0.4220 |
| `ruler_qa_squad` | 0.4258 | 0.1602 | -0.2657 |

For `ruler_qa_squad`, the baseline result and sample SHA-256 values are
`bfac5fc6310afbcc09295543ad9fbbe43b2eb75c88136a23d1d99a08608c1835`
and `edd57f2a859089edfc728c502dce097d3f0a06df1f7f9f516194a9ef56ea9ecb`;
both arms share input signature
`708b2bc0552ad163b1076994ac10f5bca5fa9c5a1a30edb100a870f43ec864bf`.

The authoritative gates are:

```bash
python scripts/manuscript/audit_long_context_results.py ...
python scripts/manuscript/compare_long_context_results.py ...
```

The comparison gate rejects any intervention manifest whose task rows do not
record position-bin diagnostics. Audit warnings cannot therefore be silently
converted into a paper aggregate; support for a separate diagnostic receipt
must be explicit and schema-validated before that receipt can close the gate.

The shared formal-runtime receipt is
`results/jetski_runs/formal_runtime_tf4524_evidence_20260824/environment_receipt.json`
(SHA-256
`f08c7192cc90c4846801c56b3d6b229ab0817a521d17faac3bbb6a7ea2e06c3e`).
Its model-inventory SHA-256 is
`f18ec2a6bfd25df302d28f179d1e4123e26bafad1755a0f6f9a23304f89735d1`.

## Remaining numeric checks

- Recompute the final 64k and 128k baseline/intervention rows after both arms
  have complete protocol-matched evidence.
- Replace or retain the pre-existing long-context cells only after recording
  their exact provenance and aggregation definition.
- Re-run this ledger, the citation audit, and the PDF page/visual audit after
  any long-context table edit.
