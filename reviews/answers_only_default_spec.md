# Spec: Answers-only default outcome — full table/figure regeneration and placeholder fill

**Date:** 2026-07-15
**Decision (user-approved):** The paper's default (headline) outcome reverts to **answers-only** — the outcome of the first submission — with the answers+comments composite demoted to the reviewer-requested re-estimation reported in the decomposition table. Rationale: R2 point 4 offered "re-estimate with a composite **or** justify answers-only"; we do both, but the *headline* stays continuous with the submitted manuscript. The title has also been reverted to the v1 title ("Help Converts Newcomers, Not Veterans: …").

**Prose status: ALREADY EDITED.** `manuscript.tex`, `main.tex`, `title_page.tex`, `body.tex`, `appendix.tex`, and `Highlights.txt` have been rewritten for the answers-only default. Every number that requires an answers-only refit is marked with a greppable placeholder token `[[AO:field]]` (sometimes with an inline choice like `[[AO:x --- attenuated/reversed]]` — pick the empirically correct wording and delete the token). **Do not restructure the prose; only fill placeholders and resolve choice-markers.**

---

## 0. Sequencing and coordination (READ FIRST)

1. **`reviews/estimand_reframe_spec.md` must land first.** Its table-writer refactor in `analysis/create_figures.py` (delete all summed/upper-bound rows and columns, promote β₂ as a labeled parallel-trends diagnostic, standard footnote sentence) applies verbatim here. This spec assumes the post-refactor table structures.
2. **One exception to that spec:** its §2.2 says to re-run the 100-replicate matched-pair bootstrap on the *composite* β₄ and expects HR ≈ 1.00 straddling 1. **Do not run the composite bootstrap.** Run the bootstrap ONCE, on the **answers-only** β₄ (see §3.9). Its §4.5 verification numbers (pooled 1.00 n.s. etc.) are superseded by §5 below.
3. **No rematch.** Matching is outcome-independent; the exact-numTags matched file (24,349,760 rows / 12,174,880 pairs / 3,732,706 users) and Tables `tab:covariates`, `tab:balance` stay exactly as they are.

## 1. Pipeline switch

- `analysis/cox_config.py`: set `PRIMARY_HELP_TYPES = ["answer"]` and update its comment (primary outcome = answers to other users' questions; self-accepts excluded; composite variants live in the decomposition only).
- Bump `DATA_VERSION` (e.g. `"rev2ao"`) so interval/model caches miss rather than silently mixing outcome generations. Same for any checkpoint-compat key the bootstrap uses (suffix `_ao_b4`).
- Everything downstream re-runs from the existing matched event histories with the answer-only event filter; no Stage-1–4 work.

**Known anchor numbers (already computed in the decomposition row "Answers only," post-rematch): pooled Model A arrival HR = 1.03 [1.02, 1.05]; β₂ ≈ −0.00; Events = 3,435,973.** The refit pooled Model A must reproduce this row (it is the same model). If it does not, STOP — cache contamination.

## 2. Final table inventory (main text order, with required structure)

Model naming: **Model A** = eq. (1), main DiD; **Model B** = eq. (2), Model A + γ, δ response-time interactions. Tenure buckets in `BUCKET_ORDER` (7 buckets). All Cox tables: model-based SEs; stars `*** p<.001, ** p<.01, * p<.05, † p<.1`; footnote sentence per estimand spec §1 (β₄ = DiD treatment effect; β₂ = parallel-trends diagnostic, never summed).

| # | Label | File | Refit needed | Structure |
|---|-------|------|-------------|-----------|
| T1 | `tab:variables` | inline in body.tex | none | static (already edited: helpEvent = answers-only) |
| T2 | `tab:covariates` | inline | none | static |
| T3 | `tab:balance` | inline | none | static (matching unchanged) |
| T4 | `tab:desc_stats` | `desc_stats.tex` | **recompute events row** | Row "Help events per window" retitled **"Help events per window (answers to others)"** with answers-only mean/SD/median. All sample-size and response-time rows unchanged (verify they reproduce). |
| T5 | `tab:pooled_experienced` | `regression_all.tex` | **REFIT pooled Model A + Model B** | Columns: `Main` (Model A), `Main + Speed` (Model B). Blocks (post-estimand-refactor): (i) *Treatment effect (DiD): increment at answer arrival (β₄)* — coef, SE, HR [95% CI], and `Hazard Ratio [bootstrap 95% CI]` row (Model A column only, from T-A3); (ii) *Parallel-trends diagnostic: waiting period (β₂)* — coef, SE, stars; (iii) *Response-time moderation* — `Answer arrival × log(RT) (γ)` coef+SE and `Waiting period × log(RT) (δ)` coef+SE (Model B column only). N, Events rows (Events = answers-only). |
| T6 | `tab:outcome_decomposition` | `composite_outcome.tex` | none (fits exist) | Rows exactly: `Answers only / Comments only / Edits only / Answers + comments / Answers + comments + edits`. Columns: `Outcome | Arrival HR [95% CI] (β₄) | Waiting β₂ (diagnostic) | Events`. Caption: **"Outcome Decomposition by Help Type"**. Footnote adds: "The headline outcome throughout the paper is Answers only." No accepts rows, no summed column. |
| T7 | `tab:main_results` | `main_results.tex` | **REFIT Model A × 7 buckets** | 7 bucket columns. Blocks: β₄ coef/SE/HR-CI; β₂ coef/SE with stars; N; Events. Landscape. |
| T8 | `tab:speed_results` | `speed_results.tex` | **REFIT Model B × 7 buckets** | 7 bucket columns. Blocks: *Arrival increment at mean response time (β₄)* coef/SE; *Response-time moderation*: γ coef/SE, δ coef/SE; N; Events. Landscape. |
| T9 | `tab:response_time_bins` | `response_time_bins.tex` | **REFIT Model A × 10 bins** | Bins: 0–15 min, 15–30 min, 30–60 min, 1–2 hr, 2–4 hr, 4–8 hr, 8–12 hr, 12–24 hr, 1–3 days, >3 days. Columns: `Response time | HR | 95% CI | N | Events` (HR = arrival β₄ within bin). |
| T10 | `tab:answer_quality_robustness` | `answer_quality_robustness.tex` | **REFIT 4 pooled specs** | Rows (human-readable labels, not slugs): `Baseline (no quality controls)`, `+ answer length only`, `+ answer score only`, `+ length, score, accepted (full)`. Columns: `Specification | Arrival HR | 95% CI | Waiting β₂ | N | Events`. |

Appendix tables:

| # | Label | File | Refit | Structure |
|---|-------|------|-------|-----------|
| A1 | `tab:selection_bounds` | `selection_bounds.tex` | **recompute** on answers-only fits/event shares | Columns: `Scope | Assumed frac. | Base HR (arrival) | Control zero share | Adjusted RR proxy` (summed column deleted per estimand spec). Scopes: All + 7 buckets × assumed fractions {0, .10, .25, .50}. |
| A2 | `tab:absolute_effects` | `absolute_effects.tex` | **recompute** | Rows: 7 buckets + `Pooled (all tenures)` (rename `AllData_Main`). Columns: `Base risk | HR | ARD | NNT | NNT 95% CI` (no Summed HR). Base risk = answers-only event risk. |
| A3 | `tab:pair_bootstrap` | `pair_bootstrap.tex` | **RERUN 100 replicates** (§3.9) | Caption: "Matched-Pair Bootstrap Uncertainty for the Answer-Arrival DiD Increment (β₄; Model A)". Columns: `Scope | HR | Bootstrap 95% CI | Replicates | N`. Footnotes per estimand spec §2.2. |
| A4 | `tab:cohort_robustness` | `cohort_robustness.tex` | **REFIT pre-2020 pooled + pre-2020 newcomer** | Rows: `Full sample (pooled)`, `Pre-2020 (pooled)`, `Full sample (< 1 week)`, `Pre-2020 (< 1 week)`. Columns: `Specification | Arrival HR | 95% CI | Waiting β₂ | N | Events`. |
| A5 | `tab:observable_controls` | `observable_controls.tex` | **REFIT pooled + observables** | Rows: `Baseline (Model A)`, `+ observable controls (hour, weekday, lengths, reputation)`. Same columns as A4. No viewCount (collider — keep exclusion note in footnote). |
| A6 | `tab:newcomer_robustness` | `newcomer_robustness.tex` | **REFIT newcomer specs** | Rows: `Baseline (< 1 week)`, `+ observable controls`, `+ answer-quality controls`. Same columns as A4. |
| A7 | `tab:response_time_bins_quality` | `response_time_bins_quality.tex` | **REFIT 10 bins + quality controls** | Same bins as T9; columns `Response time | HR (quality-controlled) | 95% CI | N | Events`; footnote: quality controls = first-answer acceptance, score, length; post-treatment caveat sentence. |

Deleted (stays deleted): legacy "Analysis of Life-Time Contributions" appendix (`tab:quartile_statistics`). Disclose in the response letter (§6).

## 3. Figure inventory

All help-rate figures are event-based and must be **regenerated with answers-only events** (they currently plot the composite). Keep axis labels/caption text as in the checked-in files; captions in body.tex are already correct.

1. `fig:study_design` — TikZ, static. No change.
2. `fig:help_rate_pooled` (`help_rate_pooled.pdf`) — regenerate.
3. `fig:strength_rec` (`strength_rec.pdf`) — regenerate from T7 bucket fits; plots **arrival HR (β₄)** per bucket with 95% CIs.
4. `fig:interaction_effect` (`interaction_effect.pdf`) — regenerate from T9 bin fits (arrival HR per bin).
5. `fig:help_rate_adoption_pooled` — regenerate.
6. `fig:help_rate_adoption_by_tenure` — regenerate (now a main-text figure).
7. Appendix: `common_support.pdf`, `love_plot.pdf` — static (matching unchanged). `help_rate_by_tenure.pdf` — regenerate.

### 3.9 Bootstrap (feeds T-A3 and T5)

100 replicates, Model A pooled, answers-only events, resample whole `match_id` pairs with replacement, statistic = per-replicate `is_treated_active` coefficient (β₄); optionally record β₂ per replicate. New checkpoint key (`_ao_b4`) — composite and summed checkpoints are NOT reusable. **Expectation: point HR ≈ 1.03; the bootstrap CI should exclude 1.0.** If it straddles 1.0, STOP and report back before filling prose — the prose asserts a significant pooled effect and would need re-hedging.

## 4. Placeholder fill map (`grep -rn "\[\[AO:" *.tex` must end at zero)

Files: `body.tex` (18), `manuscript.tex` (1), `main.tex` (1), `appendix.tex` (8). **In the .tex files the tokens are kebab-case (hyphens, e.g. `[[AO:nc-arrival-hr]]`) so drafts compile; the map below lists them with underscores — treat them as identical.** Tokens:

| Token | Source |
|-------|--------|
| `nc_arrival_hr` (×7: abstract ×2, intro ×2/theory, H2, discussion, conclusion, appendix) | T7 `< 1 Week` arrival HR |
| `nc_arrival_pct`, `nc_arrival_ci` | T7 newcomer (pct = (HR−1)×100) |
| `mid_bucket_hr_range` | T7 buckets 2–5 HR range |
| `vet_pattern`, `b6_arrival_hr_ci_sig`, `b7_arrival_hr_ci_sig`, `b7_arrival_hr`, `b7_reversal` | T7 buckets 6–7 (choose wording per sign/significance; drop the reversal insert if >6-yr HR not significantly < 1) |
| `nc_beta2_beta4_pattern` + conditional pre-trend sentence | T7 newcomer β₂/β₄ (keep the caution sentence only if β₂ > 0 and significant) |
| `pooled_beta2` | T5 Model A β₂ |
| `boot_consistency` | A3 vs T5 CI comparison |
| `pooled_quality_direction` (intro) / `quality_full_direction` / `quality_full_limitation` / `quality_conclusion` / `quality_channel_discussion` / `quality_full_direction_discussion` | T10 full-spec row: pick "attenuated" vs "reversed/below one" per the actual estimate |
| `quality_baseline_hr`, `quality_length_hr`, `quality_score_hr`, `quality_full_hr`, `quality_length_effect`, `quality_length_limitation` | T10 rows |
| `nc_quality_hr`, `nc_quality_pattern`, `nc_observables_pattern` | A6 rows |
| `gamma_range`, `delta_sign`, `delta_offset_pattern` | T8 |
| `bin_3060_hr_ci`, `bin_late_range`, `bin_3060_quality_verdict`, `bin_3060_quality_discussion`, `late_bump`, `late_bump_discussion` | T9 + A7 (drop the late-bump sentences if no multi-day elevation) |
| `pre2020_pooled_hr`, `pre2020_nc_hr`, `pre2020_nc_pattern` | A4 |
| `observables_pooled_hr`, `observables_pooled_pattern` | A5 |
| `nc_ard_pp`, `nc_nnt`, `pooled_nnt` | A2 |
| `desc_mean_events`, `desc_sd_events` | T4 |

## 5. Qualitative assertions the prose now makes — verify each; STOP and report if violated

1. Pooled answers-only arrival HR = 1.03 [1.02, 1.05], p<.001 (anchor; must reproduce).
2. Newcomer (<1 wk) arrival HR > pooled HR, positive, p<.001, and the largest bucket estimate.
3. Arrival HR declines to ~null in the top tenure bucket(s) ("fades to zero among the longest-tenured users").
4. γ < 0 in every bucket ("negative and statistically significant in every tenure bucket").
5. RT bins: <30-min bins ≈ null; 30–60-min bin is the early-window maximum and significant; survives quality controls as attenuated early maximum.
6. Quality controls attenuate the pooled effect, with length-only ≈ baseline and full spec the strongest attenuation ("graded story"; the bad-control ordering length < score < full).
7. Pre-2020 newcomer HR ≈ full-sample newcomer HR ("essentially unchanged").
8. Observable controls leave pooled and newcomer estimates essentially unchanged/positive.
9. Bootstrap CI for pooled β₄ excludes 1.
10. Selection bounds: pooled effect "remains modest" under optimistic bounds; newcomer contrast more sensitive (limitations paragraph) — reword if the new A1 pattern differs.

## 6. Response letter (`response_to_reviewers.tex`) — STRUCTURAL EDITS DONE 2026-07-15; only number fill remains

All number-independent edits are already applied: verbatim reviewer quotes in every left cell, retitle claim deleted (title reverted in letter header too), R2 P1/P4 row reframed (answers-only justified + composite re-estimated), bootstrap row rewritten to target β₄ directly with the summed-drop rationale, lifetime-appendix disclosure added to the preamble, Highlights wording fixed, letter compiles clean.

**Remaining: fill the 12 `[[AO:...]]` tokens in the letter** from the regenerated tables. Beyond the shared tokens in §4, the letter adds: `boot-hr` (A3 point HR), `nc-observables-hr` (A6 observables row), `quality-length-effect-letter` / `quality-full-direction-letter` (T10 wording choices, letter phrasing), `bin-3060-quality-verdict-letter`, `late-bump-letter` (A7), `nc-per-1000` (= newcomer ARD × 1000, from A2), `nc-nnt-ci` (A2). Then recompile the letter AFTER the manuscript (it reads `manuscript.aux` via `\externaldocument`) and verify `grep -c "\[\[AO:" response_to_reviewers.tex` → 0.

## 7. Final verification checklist

1. `grep -rn "\[\[AO:" *.tex` → zero hits.
2. `grep -rn "summed\|Summed\|upper bound\|upper bnd" analysis/output_tables/*.tex body.tex appendix.tex manuscript.tex main.tex` → zero hits ("lower bound" prose occurrences OK).
3. `grep -rn "composite" body.tex manuscript.tex main.tex` → only in decomposition contexts (Table 6 discussion, limitations, outcome paragraphs).
4. Events in T5 Model A = 3,435,973 = sum of T7 bucket Events.
5. Every number quoted in prose matches its table (esp. abstract, intro, H2, design implications NNT, conclusion).
6. `pdflatex manuscript && bibtex manuscript && pdflatex manuscript && pdflatex manuscript` exits clean, no undefined refs; same for `main.tex` build.
7. Letter compiles; letter numbers match tables.
