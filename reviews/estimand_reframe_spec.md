# Spec: Remove the summed-DiD "upper bound" estimand from all tables and pipeline outputs

**Date:** 2026-07-14
**Status of manuscript prose:** ALREADY DONE. `manuscript.tex` (abstract), `body.tex`, and `appendix.tex` have been rewritten and no longer contain any "summed", "upper bound", or `exp(β₂+β₄)` language. **Do not edit the prose files except where this spec explicitly says so.** Your job is to make the generated tables, the bootstrap, and the pipeline code consistent with the new framing.

---

## 1. Rationale (read before touching anything)

The Cox model (Model A) has two nested time-varying treated indicators:

- `treated_post_question` (**β₂**): switches on at question posting, stays on through the post-answer phase. During the waiting period the treated−control gap equals β₂.
- `is_treated_active` (**β₄**): additionally switches on at answer arrival. Post-answer the gap equals β₂+β₄.

**New official framing (decided 2026-07-14):**

- **β₄ is THE treatment effect** — the difference-in-differences increment at answer arrival. Referred to in tables as "Answer arrival (β₄)" / "Arrival HR".
- **β₂ is a parallel-trends diagnostic**, reported separately, never added to β₄. It measures pre-answer divergence (anticipation/selection); it has no sign guarantee.
- **The summed contrast exp(β₂+β₄) is REMOVED from every LaTeX table, caption, and footnote.** The old "upper bound" label was statistically wrong: β₂ can be negative (e.g. treated users helping less while monitoring their own unanswered question), in which case the "upper bound" sits *below* a legitimate estimate of the effect. It bounded nothing.
- Keep the summed computation available in **CSV outputs only** (fields `did_coef/did_se/did_p/did_hr/did_ci_lo/did_ci_hi`) as internal diagnostics. Do not delete the computation functions; just stop emitting the values into any `.tex` output.

Terminology for table headers/footnotes: β₄ → "Arrival HR (β₄)" or "Answer arrival (×Post-Answer Received)"; β₂ → "Waiting-period β₂ (parallel-trends diagnostic)". The standard footnote sentence replacing the old summed sentence is:

> `$\beta_4$ (the answer-arrival increment) is the difference-in-differences treatment effect; $\beta_2$ (waiting period) is reported as a parallel-trends diagnostic and is not added to the effect.`

---

## 2. Code changes

### 2.1 `analysis/create_figures.py` (writes ALL `.tex` tables in `analysis/output_tables/`)

All table writers live here (see ~lines 1559–1670). For each, remove summed emission and add/keep the β₂ diagnostic:

1. **`regression_all.tex`** (`tab:pooled_experienced`, writer near line 329/1559):
   - DELETE the whole block `\textit{Total elevation (summed DiD, upper bound)}` — its coef row ("Received Answer (net post-answer effect)"), SE row, model-based HR/CI row, and the "Hazard Ratio [bootstrap 95% CI]" row.
   - KEEP the β₄ block, retitle its section header to `\textit{Treatment effect (difference-in-differences): increment at answer arrival ($\beta_4$)}`.
   - MOVE the bootstrap row into the β₄ block: `\hspace{1em} Hazard Ratio [bootstrap 95\% CI]` under the model-based CI row of β₄, populated from the re-run β₄ bootstrap (§2.2). Show `—` in the Main+Speed column (bootstrap is Model A only).
   - PROMOTE the β₂ row from the nested sub-item into its own block titled `\textit{Parallel-trends diagnostic: waiting period ($\beta_2$)}`, and ADD its standard error row and significance stars (currently coef only). β₂'s SE is in the fit summary (`treated_post_question`, `se(coef)`).
   - Response-time rows: DELETE `Net: Treatment × log(RT), summed` (and its SE row). KEEP `Component: Post-Answer × log(RT)` renamed to `Answer arrival × log(RT) ($\gamma$)`, and ADD a `Waiting period × log(RT) ($\delta$)` diagnostic row with SE.
   - Footnote: replace the summed cross-reference sentence with the standard footnote sentence (§1) plus `Matched-pair bootstrap 95\% CI for the pooled arrival increment is from Table~\ref{tab:pair_bootstrap}.`

2. **`main_results.tex`** (`tab:main_results`, writer near line 1573):
   - DELETE the 3-row summed block (`\textit{Total elevation (summed DiD, upper bound)}`, coef, SE, HR/CI rows).
   - KEEP the β₄ block; retitle header as in 2.1(1).
   - β₂ row: retitle `Waiting period ($\beta_2$, parallel-trends diagnostic)` and ADD an SE row beneath it (SEs exist in the per-bucket fit summaries).
   - Footnote: standard sentence.

3. **`speed_results.tex`** (`tab:speed_results`, writer near line 1581):
   - DELETE the block `\textit{Base treatment effect (summed DiD, upper bound)}` (coef + SE rows).
   - DELETE the rows `Net: Treatment × log(RT), summed ($\gamma+\delta$, upper bound)` (coef + SE).
   - KEEP: β₄ base block (retitle `\textit{Treatment effect: arrival increment ($\beta_4$, at mean response time)}`), the `Answer arrival × log(RT) ($\gamma$)` rows, and the `Waiting period × log(RT) ($\delta$)` row — give δ an SE row too and label the γ/δ section `\textit{Response-time moderation ($\gamma$ = arrival; $\delta$ = waiting-period diagnostic)}`.
   - Footnote: standard sentence.

4. **`composite_outcome.tex`** (`tab:outcome_decomposition`, writer near line 1650):
   - Target columns: `Outcome | Arrival HR [95% CI] (β₄) | Waiting-period β₂ (diagnostic) | Events`. DELETE the `Summed HR (upper bnd)` column.
   - IMPORTANT: the checked-in `.tex` was hand-edited on 2026-07-14 to drop the degenerate `Accepts only` and `All types` rows. The **generator must reproduce that**: emit only the rows `Answers only / Comments only / Edits only / Answers + comments / Answers + comments + edits`. Drop the accepts-inclusive outcome variants from the emitted table (keep them in the CSV if produced) and drop the now-orphaned footnote `$^{a}$ Accept events are treated-only by construction…`.
   - Footnote: replace the `Arrival HR is… Summed HR is the DiD… (upper bound)…` sentence with the standard sentence. Keep the Events-definition and model-based-SE sentences.

5. **Spec/model robustness tables** (generic writer, list at lines 1625–1628: `observable_controls.tex`, `answer_quality_robustness.tex`, `newcomer_robustness.tex`, `cohort_robustness.tex`):
   - Columns become: `Specification | Arrival HR | 95% CI | Waiting β₂ | N | Events`. DELETE `Summed HR (upper bnd)`; ADD `Waiting β₂` populated from `gap_coef` (with stars from `gap_p`).
   - Footnote: replace the `Primary HR is… shown as an upper bound` sentence with the standard sentence.

6. **`selection_bounds.tex`** (writer near line 1610): DELETE the `Summed HR (upper bnd)` column and the footnote line defining it. Keep `Base HR (arrival)` and rest unchanged.

7. **`absolute_effects.tex`** (writer near line 1667; CSV from `analysis/effect_sizes.py`): DELETE the trailing `Summed HR` column and the footnote clause ``` ``Summed HR'' is the summed DiD…(upper bound) ```. ARD/NNT are already computed from the arrival HR — verify that and do not change the computation.

8. **`response_time_bins.tex`** and **`response_time_bins_quality.tex`** (writers near lines 1591/1602): no summed columns. Only DELETE the footnote sentence `…the summed DiD contrast is reported in the pooled regression tables.` Keep everything else.

9. **`pair_bootstrap.tex`** — see §2.2.

### 2.2 `analysis/pair_bootstrap_se.py` — re-target the bootstrap onto β₄ (COMPUTE RERUN REQUIRED)

- Currently (docstring lines 9–11) the bootstrapped statistic is the SUMMED contrast `treated_post_question + is_treated_active`. Change the extracted per-replicate statistic to the **`is_treated_active` coefficient (β₄) alone**; optionally also record β₂ per replicate as a second column so a β₂ percentile CI can be quoted as a diagnostic.
- Checkpoint compatibility: old checkpoints (`pair_bootstrap_checkpoint_*`) hold summed coefs — they are NOT reusable for β₄. Bump the checkpoint naming/compat key (e.g. add `_b4` suffix or a `stat` column) so stale checkpoints cannot silently merge.
- Re-run 100 replicates, Model A, pooled scope, same pair-resampling protocol (resample whole `match_id` pairs with replacement).
- `pair_bootstrap.tex` output: caption `Matched-Pair Bootstrap Uncertainty for the Answer-Arrival DiD Increment ($\beta_4$; Model A)`; footnotes: `Replicates resample whole matched pairs with replacement.` and `HR is the answer-arrival increment $\exp(\beta_4)$, the difference-in-differences treatment effect.` No summed language.
- Feed the resulting CI into the bootstrap row of `regression_all.tex` (§2.1.1).
- Expected result: point HR ≈ 1.00 (base fit: β₄ = 0.0024, model CI [0.99, 1.02]); the bootstrap CI should straddle 1. **If the β₄ bootstrap CI excludes 1.0, STOP and report back before regenerating prose-adjacent tables** — the prose asserts the pooled arrival increment is n.s.

### 2.3 `analysis/cox_config.py`

- `HEADLINE_ESTIMAND` stays `"arrival"`. Update the comment block (lines ~18–23): delete the instruction "Never delete the summed computation" and the description of summed as a reported secondary estimand; state instead that `did_*` (summed) fields are computed for CSV diagnostics only and must not be emitted into `.tex` outputs, per the 2026-07-14 estimand decision (β₂+β₄ is not a bound of any kind).

### 2.4 `analysis/cox_fit.py`, `analysis/cohort_robustness.py`, `analysis/revision_robustness.py`, `analysis/effect_sizes.py`, `analysis/selection_sensitivity.py`

- KEEP `_did_fields`/`did_*` computation flowing into CSVs (diagnostic). Grep each file for the strings `upper bound`, `upper bnd`, `summed` in any **string literal destined for LaTeX** and remove/replace per §1. Column data changes are all in `create_figures.py` (§2.1); these files should only need comment/label hygiene unless they hard-code table text.

### 2.5 Figures

- Verify `strength_rec.pdf`, `interaction_effect.pdf`, and any figure caption or axis label built in `create_figures.py` plots the **arrival** HR (they should already, given `HEADLINE_ESTIMAND="arrival"`). If any figure or its caption mentions summed/upper bound, switch to arrival and regenerate.

---

## 3. Rerun scope

- **Cox refits: NONE needed.** All arrival (`arrival_*`, `treat_*`) and β₂ (`gap_*`) fields already exist in the results CSVs; tables are re-rendered from cached results.
- **Bootstrap: FULL RERUN** (§2.2), 100 replicates of Model A pooled.
- After table regeneration: `pdflatex manuscript && bibtex manuscript && pdflatex manuscript && pdflatex manuscript` must exit clean.

## 4. Verification checklist (run all)

1. `grep -rn "summed\|Summed\|upper bound\|upper bnd" analysis/output_tables/*.tex body.tex appendix.tex manuscript.tex` → **zero hits**. ("lower bound" occurrences in prose are intentional; do not touch.)
2. `composite_outcome.tex` has exactly 5 outcome rows and no `$^{a}$` footnote.
3. Every robustness table (`observable_controls`, `answer_quality_robustness`, `newcomer_robustness`, `cohort_robustness`) has a `Waiting β₂` column and no summed column.
4. `pair_bootstrap.tex` caption/footnote reference β₄ only; `regression_all.tex` bootstrap row sits in the β₄ block and matches `pair_bootstrap.tex`.
5. Cross-check prose numbers still match regenerated tables — the prose asserts: pooled arrival 1.00 [0.99, 1.02] n.s., pooled β₂ ≈ +0.006 n.s.; newcomer arrival 1.06 [1.04, 1.08], β₂ ≈ +0.045; answers-only 1.03 [1.02, 1.05]; comments-only 1.00 [0.98, 1.01]; edits-only 1.14, β₂ = +0.07; composite+edits 1.02; quality-controlled pooled 0.98, newcomer 0.94; observables newcomer 1.10; pre-2020 pooled 1.014 vs 1.002, newcomer 1.056 vs 1.059; 30–60 min bin 1.22 → 1.06 with quality controls; NNT 143.5 newcomer / ~1995 pooled; ARD 0.7 pp newcomer.
6. Full manuscript compiles with no undefined references.

## 5. Follow-up NOT in this task (needs the β₄ bootstrap number first)

`response_to_reviewers.tex` still describes the old framing (bootstrapping the summed contrast as "more conservative", "upper bound" labels in R1.3, R1.4, R2 responses, and the preamble item (v)). After the β₄ bootstrap lands, the letter must be rewritten to say: the matched-pair bootstrap now targets the headline arrival increment (β₄) directly; β₂ is reported as an explicit parallel-trends diagnostic; the summed quantity was dropped because a signed pre-trend cannot bound the effect. This is a *stronger* answer to R1's inference comment, not a retreat. Do not edit the letter in this task.
