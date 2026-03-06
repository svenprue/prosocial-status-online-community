# Preprocessing ETL Pipeline

This folder contains the ETL (Extract, Transform, Load) pipeline for building study datasets from the Stack Overflow data dump. The pipeline supports two research streams: **reciprocity** (help-seeking and helping behavior) and **bounty** (reputation incentives).

---

## Pipeline Overview

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                         RAW DATA (Stack Overflow Dump)                            │
│  Posts.xml, Votes.xml, Users.xml, Badges.xml, bounty_timeline.parquet            │
└─────────────────────────────────────────────────────────────────────────────────┘
                                          │
                                          ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│  1. processing_data_dump.py                                                       │
│     • Convert/split Posts → posts_questions.parquet, posts_answers.parquet        │
│     • (Optional) Convert Votes, Users, Badges to Parquet                          │
│     • Fix column types (IDs, timestamps)                                          │
└─────────────────────────────────────────────────────────────────────────────────┘
                                          │
                    ┌─────────────────────┴─────────────────────┐
                    ▼                                           ▼
┌───────────────────────────────────────┐   ┌───────────────────────────────────────┐
│  2a. creating_raw_reciprocity_dataset │   │  2b. creating_raw_bounty_dataset       │
│      • Build question-centered events │   │     (archive/preprocessing_bounty/)     │
│      • Phase 1/2 windows, first answer│   │     • Join answers with bounty timeline│
│      • Output: question_centered_*    │   │     • Output: user_answers_bounty_*    │
└───────────────────────────────────────┘   └───────────────────────────────────────┘
                    │                                           │
                    ▼                                           ▼
┌───────────────────────────────────────┐   ┌───────────────────────────────────────┐
│  3a. processing_reciprocity_dataset   │   │  3b. processing_bounty_dataset         │
│      • Compute pre-treatment metrics  │   │     (archive/preprocessing_bounty/)     │
│      • Rolling windows (7d, 30d, AT)  │   │     • Similar metric computation       │
│      • Output: *_processed.parquet    │   │     • Output: *_processed.parquet      │
└───────────────────────────────────────┘   └───────────────────────────────────────┘
                    │
                    ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│  4. matching/create_matched_dataset.py (propensity score matching)                │
│     • PSM on reciprocity data → question_centered_model_7d_matched.parquet        │
│     • Output: matched_questions.parquet (used downstream)                         │
└─────────────────────────────────────────────────────────────────────────────────┘
                    │
                    ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│  5. calculate_matched_questions_helping.py                                        │
│     • Add helps_given_in_window for each matched question                         │
│     • Output: matched_questions_helping_metrics.parquet                           │
└─────────────────────────────────────────────────────────────────────────────────┘
                    │
                    ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│  6. create_matched_event_histories.py                                             │
│     • Build event-history dataset for survival/Cox analysis                       │
│     • Output: study_timelines.parquet, study_events.parquet                       │
└─────────────────────────────────────────────────────────────────────────────────┘
```

---

## Execution Order

Run scripts in this order for a full reciprocity pipeline:

| Step | Script | Purpose |
|------|--------|---------|
| 0 | (External) | Obtain Stack Overflow XML dump + bounty timeline; place in `data/input/` |
| 1 | `processing_data_dump.py` | Prepare base Parquet tables (Posts split, types fixed) |
| 2 | `creating_raw_reciprocity_dataset.py` | Create raw question-centered reciprocity dataset |
| 3 | `processing_reciprocity_dataset.py` | Add metrics, rolling windows, reciprocity activation |
| 4 | `matching/create_matched_dataset.py` | Propensity score matching (PSM) on reciprocity data |
| 5 | `calculate_matched_questions_helping.py` | Compute helping behavior within matched windows |
| 6 | `create_matched_event_histories.py` | Build event-history dataset for Cox/survival analysis |

For bounty: run `creating_raw_bounty_dataset` and `processing_bounty_dataset` (in `archive/preprocessing_bounty/`) between steps 1 and 3. The main entry point `main.py` orchestrates steps 2–4 for both reciprocity and bounty, but requires bounty scripts to be present in the preprocessing folder or on the import path.

---

## File Descriptions

### Core pipeline

| File | Function |
|------|----------|
| **main.py** | Entry point that runs the full reciprocity + bounty pipeline. Calls creating_raw_bounty_dataset, creating_raw_reciprocity_dataset, processing_bounty_dataset, and processing_reciprocity_dataset in sequence. |
| **processing_data_dump.py** | Converts Stack Overflow XML to Parquet. Splits `Posts.parquet` into `posts_questions.parquet` and `posts_answers.parquet` (by PostTypeId 1/2), drops heavy `Body`/`Title` columns, and fixes column types (IDs → integer, dates → timestamp). Can also process Votes, Users, and Badges. |
| **creating_raw_reciprocity_dataset.py** | Builds the raw question-centered reciprocity dataset. Joins questions, answers, votes, and users; identifies first non-self, non-downvoted answers; creates Phase_One_Start, Phase_Two_Start/End, and Window_Answer events. Outputs `question_centered_model_{N}d_all_questions.parquet` and a tag dictionary. |
| **processing_reciprocity_dataset.py** | Processes the raw reciprocity dataset. Adds pre-treatment metrics (all-time, 30d, 7d rolling windows), reciprocity activation flags, experience categories, and fixed effects. Uses Numba JIT for metric computation. Outputs `*_processed.parquet`. |

### Post-matching / analysis support

| File | Function |
|------|----------|
| **calculate_matched_questions_helping.py** | Takes `matched_questions.parquet` (from PSM) and raw questions/answers. For each matched question, computes `helps_given_in_window` (answers posted to others between question and response-time window). Outputs `matched_questions_helping_metrics.parquet` in `data/study_datasets/`. |
| **create_matched_event_histories.py** | Builds event-history data for survival analysis. Uses matched questions, aligns timelines per match_id, and finds all “help” events (answers to others) within study windows. Outputs `study_timelines.parquet` and `study_events.parquet` in `data/event_history/`. |

### Quality and testing

| File | Function |
|------|----------|
| **sanity_checking_reciprocity_dataset.py** | Validates the processed reciprocity dataset: checks for nulls, logical consistency, metric ranges, and prints example rows for inspection. |

### Bounty pipeline (archived)

The bounty pipeline lives in `archive/preprocessing_bounty/`:

- **creating_raw_bounty_dataset.py** — Builds user-answers dataset with bounty timeline (bounty_start, bounty_end).
- **processing_bounty_dataset.py** — Processes bounty dataset with similar metric logic.

---

## Data Dependencies

| Input | Source | Used by |
|-------|--------|---------|
| `Posts.parquet` | `processing_data_dump` (or pre-converted) | Split into questions/answers |
| `posts_questions.parquet` | From Posts split | Reciprocity, helping, event histories |
| `posts_answers.parquet` | From Posts split | Reciprocity, helping, event histories |
| `Votes.parquet` | Optional | Reciprocity (accepted answer timing) |
| `Users.parquet` | Optional | Reciprocity (registration, tenure) |
| `Badges.parquet` | Optional | Bounty, badge-based features |
| `bounty_timeline.parquet` | External | Bounty pipeline |

---

## Output Locations

| Output | Location |
|--------|----------|
| Base Parquet tables | `data/input/` |
| Raw reciprocity dataset | `data/input/question_centered_model_*d_*.parquet` |
| Processed reciprocity | `data/study_datasets/*_processed.parquet` |
| Matched questions (PSM) | `data/study_datasets/question_centered_model_*d_matched.parquet` |
| Helping metrics | `data/study_datasets/matched_questions_helping_metrics.parquet` |
| Event histories | `data/event_history/study_timelines.parquet`, `study_events.parquet` |

---

## Where questions are lost (low observation count)

If the pipeline produces fewer observations than expected, losses occur at these stages:

| Stage | Where | What is dropped |
|-------|--------|-------------------|
| **1. Raw creation** | `creating_raw_reciprocity_dataset.py` | Questions with `OwnerUserId` IS NULL (anonymous/deleted users). |
| **2. Processing** | `processing_reciprocity_dataset.py` | **Cutoff filter**: questions with `phase_two_end > cutoff_date` (default `2025-04-01`) are removed so the full 7-day window is observed. This drops all questions whose 7-day window extends past the cutoff (e.g. questions asked after ~2025-03-25). **To retain more questions**, increase `cutoff_date` in `main.py` and in `processing_reciprocity_dataset.py` (e.g. to the latest date in your dump). |
| **3. Matching** | `matching/create_matched_dataset.py` | (a) Rows with `hasSelfAnswer == 1`; (b) `hasAnswer` recoded to 0 when `responseTimeHours > 7*24`; (c) rows with missing covariates (dropped before PSM); (d) year–tag strata with &lt; 300 questions (`MIN_QUESTIONS_PER_STRATUM`); (e) questions that do not get a propensity-score match within caliper (many controls/treated remain unmatched). |

**To investigate your data:** run the diagnostic script (requires pandas and input/study parquet files):

```bash
python preprocessing/investigate_question_loss.py
```

It reports counts at each stage (source → raw → processed → matched) and how many questions are lost to the cutoff in processing.

---

## Notes

1. **Reciprocity vs. bounty**: The reciprocity pipeline is self-contained in this folder. The bounty pipeline is in `archive/preprocessing_bounty/`; copy those scripts into `preprocessing/` if you need `main.py` to run both.
2. **`process_question_data`**: Implemented as an alias for `process_accepted_answer_data` in `creating_raw_reciprocity_dataset.py` so `main.py` can call it.
3. **Resource usage**: Reciprocity scripts use DuckDB with large memory limits and temp directories; ensure sufficient disk and RAM for the full Stack Overflow dump.
