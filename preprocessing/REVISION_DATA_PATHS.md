# Revision data: pipeline file-path mapping

Maps `data/input/*_revision.parquet` to pipeline stages and canonical filenames.

**Recommended approach before re-run:** promote revision files to canonical names (backup old files first), so existing hard-coded paths keep working. Alternative: symlink or edit paths in the table below.

## Stage 0 — Promote revision inputs (one-time)

```bash
cd data/input
# backup originals
for f in posts_questions posts_answers Users Votes; do
  mv "${f}.parquet" "${f}_pre_revision.parquet"
done
# promote revision
mv posts_questions_revision.parquet posts_questions.parquet
mv posts_answers_revision.parquet posts_answers.parquet
mv Users_revision.parquet Users.parquet
mv Votes_revision.parquet Votes.parquet
# Comments is new — no canonical predecessor
mv Comments_revision.parquet Comments.parquet
```

`Posts_revision.parquet` is optional if splits are promoted; keep `Posts.parquet` (37GB, raw `Body`/`Title`) for text fallbacks.

---

## Stage 1 — Dump parse / ingest

| Output (plan name) | Revision source | Script | Notes |
|--------------------|-----------------|--------|-------|
| `posts_questions.parquet` | `posts_questions_revision.parquet` | `processing_data_dump.py` → `process_posts` | +`ViewCount`, text metrics |
| `posts_answers.parquet` | `posts_answers_revision.parquet` | same | +answer `BodyLenChars`, etc. |
| `Users.parquet` | `Users_revision.parquet` | `parse_generic_row_users` (extend) | +`Reputation`, `UpVotes`, `DownVotes` |
| `Comments.parquet` | `Comments_revision.parquet` | **new** `parse_generic_row_comments` | `Id, PostId, UserId, CreationDate, Score` |
| `Votes.parquet` | `Votes_revision.parquet` | `parse_generic_row_votes` | Unchanged schema; type-2 rows still lack `UserId` |
| `Badges.parquet` | *(unchanged)* | existing | Rep reconstruction / badges |
| `PostHistory.parquet` | `PostHistory_revision.parquet` | **new** `parse_generic_row_posthistory` | Edit rows only: types 4/5/6 + `UserId`; no `Text` |

### Column aliases (apply at ingest or in SQL views)

| Revision (PascalCase) | Plan / pipeline (snake_case) |
|-------------------------|------------------------------|
| `BodyLenChars` | `body_len_chars` (or `body_len` in Stage 2/4) |
| `BodyLenWords` | `body_len_words` |
| `TitleLenChars` | `title_len_chars` (or `title_len`) |
| `TitleLenWords` | `title_len_words` |
| `NCodeBlocks` | `n_code_blocks` |
| *(derived at Stage 3)* | `hour`, `dayofweek` → alias to `postHour`, `postDayOfWeek` at Stage 4 |

---

## Stage 2 — Raw reciprocity dataset

**Script:** `preprocessing/creating_raw_reciprocity_dataset.py`

| Reads | Path today | After promotion |
|-------|------------|-----------------|
| Questions | `data/input/posts_questions.parquet` | promoted revision |
| Answers | `data/input/posts_answers.parquet` | promoted revision |
| Votes | `data/input/Votes.parquet` | promoted revision |
| Users | `data/input/Users.parquet` | promoted revision |
| Badges | `data/input/Badges.parquet` | unchanged |

**New columns to thread through SELECTs (not yet wired):**

- `ViewCount`, `body_len_*`, `title_len_*`, complexity metrics (from questions)
- `first_answer_body_len` etc. (join first answer → answers revision)
- `Comments.parquet` (ISS-04 composite — new view)
- `Users.Reputation` or reconstructed rep (ISS-02)

**Already computed, needs export to event histories:**

- `first_answer_score`, `has_accepted_answer` (in `eligible_questions` when `include_all_questions=True`)

---

## Stage 3 — Processed reciprocity

**Script:** `preprocessing/processing_reciprocity_dataset.py`

| Input | Path |
|-------|------|
| Raw all-questions | `data/input/question_centered_model_7d_all_questions.parquet` |

**Derive here:**

- `hour`, `dayofweek` from `timestamp` / `question_timestamp`
- `numTags` from `tag_ids`
- Pass through `ViewCount`, length metrics, `first_answer_score`

**Output:** `data/study_datasets/question_centered_model_7d_processed.parquet`

---

## Stage 4 — Matching

**Script:** `matching/create_matched_dataset.py`

| Input | Path |
|-------|------|
| Processed | `data/study_datasets/question_centered_model_7d_processed.parquet` |

**Add to `CONTINUOUS_COVS` / exact-match keys:**

- `postHour`, `postDayOfWeek`, `numTags`
- `body_len_chars`, `title_len_chars` (or binned)
- `ViewCount`, reputation (optional)
- complexity metrics (optional robustness)

**Output:** `data/input/matched_questions.parquet` (or `data/study_datasets/question_centered_model_7d_matched.parquet`)

---

## Stage 5 — Event histories

| Script | Reads | Writes |
|--------|-------|--------|
| `calculate_matched_questions_helping.py` | `matched_questions.parquet`, `posts_questions.parquet`, `posts_answers.parquet` | helping metrics |
| `create_matched_event_histories.py` | `matched_questions`, `posts_questions`, `posts_answers`, `Users` | `data/event_history/study_timelines.parquet`, `study_events.parquet` |

**New for revision:**

| Column | Source | Issue |
|--------|--------|-------|
| `question_year` | `EXTRACT(year FROM question_ts)` | ISS-10 |
| `hasAcceptedAnswer`, `first_answer_score` | matched / raw | ISS-06 |
| `ViewCount` | questions | ISS-16 placebo |
| `help_type` | UNION answers + comments + edits + accept | ISS-04 |
| `Comments.parquet` | join asker comments on others' posts | ISS-04 |
| `PostHistory.parquet` | join asker edits (types 4/5/6) on others' posts | ISS-04 |

---

## Stage 6 — Analysis

**Scripts:** `analysis/fit_cox_models.py`, `cox_data.py`, etc.

| Input | Path |
|-------|------|
| Timelines | `data/event_history/study_timelines.parquet` |
| Events | `data/event_history/study_events.parquet` |

No direct reads of `*_revision` files — everything must be propagated through Stages 2–5.

---

## Issue → data source quick reference

| Issue | New raw data needed? | Primary sources |
|-------|---------------------|-----------------|
| ISS-02 (#8) observable controls | Promoted revision + `Badges` | questions revision, Users revision, Tags, Votes (rep reconstruction) |
| ISS-04 (#9) composite outcome | `Comments` + `PostHistory` + plumbing | Comments revision, edit events (types 4/5/6); accept from questions/Votes; **not** upvotes-given |
| ISS-06 (#11) answer quality | Promoted answers revision | `first_answer_score` (pipeline), `BodyLenChars` on first answer |
| ISS-10 (#15) cohort | None (plumbing) | `year` / `question_year` from timestamps |
| ISS-16 (#16) ViewCount placebo | Promoted questions revision | `ViewCount` |
| ISS-01, ISS-03, ISS-05 | None | existing event histories |

---

## Files not replaced by revision

| File | Role |
|------|------|
| `Posts.parquet` | Full `Body`/`Title` fallback (37GB); optional for custom text metrics |
| `Badges.parquet` | Reputation reconstruction, badge events |
| `bounty_timeline.parquet` | Bounty side analyses only |
| `question_centered_model_7d_all_questions.parquet` | Regenerated by Stage 2 |

## Run the full pipeline

```bash
bash preprocessing/run_revision_pipeline.sh
```

Or step-by-step (see stages above). After event histories are rebuilt:

```bash
cd analysis
python revision_robustness.py --input ../data/event_history --no-cache
```

Outputs:
- `analysis/model_cache/results_observable_controls.csv` (ISS-02)
- `analysis/model_cache/results_answer_quality.csv` (ISS-06)
- `analysis/model_cache/results_composite_outcome.csv` (ISS-04)
- `analysis/model_cache/results_viewcount_placebo.csv` (ISS-16)
- `analysis/model_cache/results_newcomer_robustness.csv`
- `analysis/model_cache/results_cohort_robustness.csv` (ISS-10)

Blocked / not in dump:
- Upvote **voter** identity on `VoteTypeId=2` — not in SO dump (ISS-04)
