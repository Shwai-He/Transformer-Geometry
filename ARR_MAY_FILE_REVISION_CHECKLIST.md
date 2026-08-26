# ARR May / EMNLP Transformer Geometry — File-by-File Revision Checklist

This checklist governs the next manuscript-editing pass. Work through it in
order because later claims depend on definitions and evidence fixed earlier.

Current manuscript reviewed:
`_overleaf_/ARR_May_Revision_Overleaf/emnlp_2026.tex`.

## 0. Freeze the editing baseline

- [ ] Confirm whether the canonical editing source is the Beacon
  `_ARR_May/manuscript_working_copy/` or the current dirty
  `_overleaf_/ARR_May_Revision_Overleaf/` mirror.
- [ ] Preserve all existing uncommitted manuscript, bibliography, figure, and
  audit-script changes; do not overwrite or normalize them mechanically.
- [ ] Diff the canonical working copy against the Overleaf mirror before the
  first content edit.
- [ ] Record the chosen baseline commit and the initial dirty-file list.
- [ ] Keep new 64k/128k no-early evidence frozen until the ordered 500-sample
  identity gate passes. Do not promote `returned-unverified` artifacts.

## 1. `sections/method.tex` — establish the authoritative method

- [ ] Resolve the implementation-defining question first: is value-space
  decomposition performed independently per query head, or jointly after
  expanding/concatenating heads before `W_O`?
- [ ] Make the answer consistent across the value-space definition, unified
  scaling equations, diagonal-form derivation, captions, and appendix.
- [ ] Verify the raw intervention definition against the actual implementation.
- [ ] Verify the exclude-self definition: keep `A_tt v_t` fixed, decompose only
  the non-self aggregate, and perform no attention-row renormalization.
- [ ] Clarify that "self-value-aligned" denotes alignment with the current
  token's value direction; it is not synonymous with the direct self message.
- [ ] Check whether the claimed effective-diagonal form is exact for the chosen
  per-head or joint implementation and state its scope precisely.
- [ ] Check notation for GQA/MQA head expansion and the domain/dimension of
  `v_t`, `o_t`, `c_t`, and `W_O`.
- [ ] Ensure scale conventions are uniform: retained parallel scale 0 means
  removal, scale 1 means a validated no-op, and perpendicular scale remains 1.
- [ ] Completion gate: every mathematical statement maps unambiguously to one
  tested code path.

## 2. `sections/appendix.tex` — align derivations and implementation details

- [ ] Update the attention-diagonal derivation after `method.tex` is frozen.
- [ ] Remove the current per-head versus concatenated-pre-`W_O` contradiction.
- [ ] Distinguish raw value-space, exclude-self value-space, residual-attention,
  and hard diagonal removal in both equations and prose.
- [ ] Verify that the forward-hook description matches eager and Flash-LSE
  implementations, including GQA expansion and layer windows.
- [ ] Audit every RULER table cell against its evidence status and aggregation
  type: 13-task mean versus five-family macro.
- [ ] Keep existing 64k/128k cells frozen unless their current provenance is
  already independently valid; do not substitute the blocked no-early runs.
- [ ] Label incomplete, diagnostic, legacy, simulated, or single-run evidence
  explicitly and keep it out of validated aggregate claims.
- [ ] Recheck head-localization, MLP-internal, compression-null, overhead, and
  multi-seed sections against their source artifacts.
- [ ] Completion gate: all appendix claims are reproducible or explicitly
  scoped, and none silently upgrades pending evidence.

## 3. `sections/experiments.tex` — rebuild the evidence hierarchy

- [ ] Make the experimental setup inherit the exact frozen method definition.
- [ ] Check all intervention locations, edited layer ranges, models, tasks,
  sample counts, scale conventions, and baseline/no-op definitions.
- [ ] Keep the central inference claim comparative: value-space half retention
  is less disruptive than matched alternatives; do not imply that full removal
  is generally harmless.
- [ ] Preserve the negative boundary that exclude-self full removal is damaging.
- [ ] Check every value in the Qwen benchmark and RULER main tables against the
  numeric audit and source artifacts.
- [ ] Make the 4k--12k main-table metric and the 4k--128k appendix metric
  visibly distinguishable where readers compare them.
- [ ] Ensure diagonal-map interpretation does not assign a layer-level
  post-`W_O` residual coefficient to an individual head.
- [ ] Keep compression as component-resolved diagnosis, not a new predictive
  metric independent of total error.
- [ ] Present the 12/12 matched validation-loss result as the replicated
  training evidence; label 1.4B/2.7B downstream gains as single-run results.
- [ ] Completion gate: every main-text claim has a validated table, figure, or
  appendix pointer and uses the narrowest defensible wording.

## 4. `sections/introduction.tex` — state the corrected story

- [ ] Rewrite only after Method and Experiments are stable.
- [ ] Define residual-space and value-space interventions without conflating
  direct self routing with non-self projection along `v_t`.
- [ ] Replace broad redundancy language with sensitivity/relative-robustness
  language wherever full-removal evidence contradicts redundancy.
- [ ] State the length boundary: robustness weakens as context grows.
- [ ] Keep editing as the central causal contribution; compression and training
  remain supporting applications.
- [ ] Match the three contribution bullets to results that survive the evidence
  audit.
- [ ] Completion gate: a reader can predict the paper's actual positive and
  negative findings before seeing the tables.

## 5. `sections/abstract.tex` — compress the final claims

- [ ] Update only after Introduction and Experiments are frozen.
- [ ] Clarify "self-value-aligned component" using the corrected exclude-self
  meaning.
- [ ] Avoid implying that parallel components are globally redundant.
- [ ] Mention comparative robustness and its scope rather than universal
  near-losslessness.
- [ ] Preserve the distinction between replicated fixed-validation improvement
  and single-run downstream observations.
- [ ] Keep compression wording proportional to its small/non-universal gains.
- [ ] Completion gate: every abstract sentence is directly supported in the
  main paper without relying on pending 64k/128k evidence.

## 6. `sections/background.tex` — notation and motivation

- [ ] Verify the two reported parallel-fraction ratios are named distinctly and
  are not treated as numerically interchangeable.
- [ ] Make "direction-preserving" explicitly local to the chosen reference
  vector and intervention site.
- [ ] Check that the prevalence plot motivates causal testing without implying
  functional redundancy from magnitude alone.
- [ ] Ensure notation matches `method.tex` after the method revision.

## 7. `sections/related_work.tex` — positioning

- [ ] Check XSA terminology and distinguish this intervention from deleting or
  masking self-attention.
- [ ] Position against residual scaling, attention sinks/gating, activation
  steering, compression diagnostics, and geometric representation analysis.
- [ ] Remove any novelty language broader than the demonstrated decomposition
  and matched interventions.
- [ ] Verify all citations against `references.bib` and primary sources.

## 8. `sections/discussion.tex` — interpretation boundaries

- [ ] Add the long-context boundary explicitly.
- [ ] Explain why high-dimensional concentration limits the independent value
  of perpendicular compression error.
- [ ] Separate observed intervention sensitivity from an optimization-mechanism
  hypothesis.
- [ ] State the operational implication of head/depth heterogeneity without
  claiming additive head effects.
- [ ] Avoid deployment or efficiency claims beyond measured overhead.

## 9. `sections/conclusion.tex` — final scope, limitations, and ethics

- [ ] Match the conclusion to the corrected abstract and contribution list.
- [ ] Retain limitations on models, modalities, tasks, context distributions,
  training budgets, and reference/site dependence.
- [ ] State that the replicated training result is early fixed-validation loss,
  not stable downstream or convergence improvement.
- [ ] Ensure Limitations and Ethical Considerations remain outside the nine-page
  main-content boundary under the final venue rules.
- [ ] Confirm acknowledgements/funding and Responsible NLP checklist handling.

## 10. `emnlp_2026.tex` and `macros.tex` — document-level consistency

- [ ] Confirm final author order, affiliations, internship footnote, emails,
  acknowledgements, and funding disclosures with the authors.
- [ ] Resolve the author-footnote Hyperref warning after metadata is approved.
- [ ] Check title against the final claim scope.
- [ ] Remove unused or ambiguous macros only if doing so cannot disturb current
  uncommitted work.
- [ ] Verify review/final mode, anonymity, page-limit treatment, and appendix
  placement for the intended submission stage.

## 11. `references.bib` — citation audit

- [ ] Preserve existing manual bibliography changes.
- [ ] Run the citation audit and resolve missing/unused/duplicate keys.
- [ ] Verify new or changed citations against primary publication pages.
- [ ] Rebuild BibTeX output only after source text and bibliography are frozen.

## 12. Figures, captions, and tables

- [ ] Verify each figure is generated by the corrected implementation and that
  its sources are recorded in `CAMERA_READY_FIGURE_AUDIT.md`.
- [ ] Regenerate figures only from validated source data; never retitle a legacy
  figure into a corrected result.
- [ ] Check scale-1 identity, raw/exclude-self labels, model names, layer ranges,
  aggregation metrics, and baseline pairing in every caption.
- [ ] Confirm the overview diagram matches the final per-head/joint method.
- [ ] Audit table arithmetic before applying display rounding.

## 13. Final cross-file verification

- [ ] Search globally for ambiguous legacy terms: `self-value`, `redundant`,
  `lossless`, `per-head`, `concatenated`, `all layers`, `no early`, `64k`, and
  `128k`.
- [ ] Confirm raw/exclude-self/residual/diagonal terminology is identical across
  abstract, introduction, method, experiments, captions, and appendix.
- [ ] Compile with the repository script and run citation, numeric,
  long-context, and figure audits.
- [ ] Check for undefined references/citations, overfull boxes, broken links,
  missing assets, and page-boundary regressions.
- [ ] Visually inspect all PDF pages, with special attention to pages 1, 9, 10,
  tables, multi-panel figures, and the final appendix page.
- [ ] Diff the canonical source and Overleaf mirror; synchronize only after the
  canonical version is approved.
- [ ] Record the final PDF hash, source commit, dirty status, and unresolved
  human decisions.

## Recommended editing order

1. `sections/method.tex`
2. method-related parts of `sections/appendix.tex`
3. `sections/experiments.tex`
4. result-related parts of `sections/appendix.tex`
5. `sections/introduction.tex`
6. `sections/abstract.tex`
7. `sections/background.tex`
8. `sections/related_work.tex`
9. `sections/discussion.tex`
10. `sections/conclusion.tex`
11. `emnlp_2026.tex`, `macros.tex`, and `references.bib`
12. figures/tables, compilation, audits, and final PDF inspection
