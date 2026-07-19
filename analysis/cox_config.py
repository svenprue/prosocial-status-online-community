"""Shared config for Cox pipeline: paths, bucket order, model covariates."""
import os

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CACHE_DIR = os.path.join(_SCRIPT_DIR, "model_cache")
DATA_CACHE_DIR = os.path.join(_SCRIPT_DIR, "data_cache")

BUCKET_ORDER = [
    "< 1 Week", "1 Week - 1 Month", "1 - 6 Months",
    "6 - 12 Months", "1 - 3 Years", "3 - 6 Years", "> 6 Years",
]

ROUND_TO_HOURS = 1
# Cap on rows passed to a single lifelines fit. Above this, fit_cox_cached draws a
# matched-pair subsample. Overridable via COX_MAX_FIT_ROWS so the point-estimate fits
# can run on the full data (seed-independent MLE) while the matched-pair bootstrap keeps
# the default 8M cap per replicate for tractability. 2026-07-16: the default-8M pooled
# fit proved subsample-fragile — its beta_4 varies ~0.006-0.01 across subsample seeds
# (comparable to the ~0.04 effect itself), so the reported point estimate must come from
# a full-data (or near-full) fit, not a single 8M subsample.
MAX_FIT_ROWS = int(os.environ.get("COX_MAX_FIT_ROWS", "8000000"))
SUBSAMPLE_SEED = 42

# ISS-24 re-headline: which estimand LEADS the reported tables/figures. "arrival" =
# the answer-arrival increment (beta_4 = is_treated_active), the sole DiD treatment
# effect. beta_2 (treated_post_question) is reported separately as a parallel-trends
# diagnostic and is never added to beta_4. Per the 2026-07-14 estimand decision, the
# summed quantity beta_2+beta_4 bounds nothing (beta_2 has no sign guarantee), so the
# `did_*` (summed) fields are computed only as CSV diagnostics — cox_fit.py and the
# robustness scripts keep populating them for internal use — but must NEVER be
# emitted into any .tex table, caption, or footnote.
HEADLINE_ESTIMAND = "arrival"  # "arrival" | "summed"

# fix(cache): bump when the data/covariate construction changes so stale model/interval
# caches miss instead of silently mixing generations. Backward-safe (old caches just miss).
# rev2ao (2026-07-15): default outcome flipped to answers-only per
# reviews/answers_only_default_spec.md — composite-era caches must not merge.
DATA_VERSION = "rev2ao"
# Cap concurrent lifelines fits (each can hold up to MAX_FIT_ROWS). Override via
# --n-jobs / COX_MAX_FIT_WORKERS; default keeps Stage-6 RT-bin and bootstrap pools
# from OOM'ing when cpu_count is large.
MAX_FIT_WORKERS = int(os.environ.get("COX_MAX_FIT_WORKERS", "4"))
VARIANCE_ESTIMATOR = "robust_sandwich"
CLUSTER_COL = "match_id"

# Primary Cox outcome (2026-07-15 decision, reviews/answers_only_default_spec.md):
# answers the focal user posts to other users' questions — the paper's headline
# outcome, matching the first submission. Self-directed accepts are excluded.
# Comment-/edit-inclusive composites are estimated only for the outcome
# decomposition (tab:outcome_decomposition), not as the default.
PRIMARY_HELP_TYPES = ["answer"]

CONTINUOUS_COVARIATES = [
    "hasAnswer_response_time_interaction",
    "treated_response_time_interaction",
    "treated_post_question_response_time_interaction",
]

# fix(scale): the three response-time interaction terms are now built in
# cox_data._build_covariates from a SINGLE standardized log-RT (standardized once on
# the treated rows), so they already share a common scale. fit_cox_cached must therefore
# SKIP its per-column winsorize+z-score for these (which would re-scale each by its own
# nonzero-row SD and make gamma+delta a sum of differently-scaled coefficients). They
# remain counted for has_continuous / penalizer / step_size.
RT_INTERACTION_TERMS = {
    "hasAnswer_response_time_interaction",
    "treated_response_time_interaction",
    "treated_post_question_response_time_interaction",
}

COVARIATES_MAIN = [
    "hasAnswer",
    "phase_post_question",
    "treated_post_question",
    "phase_post",
    "is_treated_active",
]

COVARIATES_SPEED = COVARIATES_MAIN + [
    "hasAnswer_response_time_interaction",
    "treated_response_time_interaction",
    "treated_post_question_response_time_interaction",
]

# ISS-02: observable selection controls (robustness spec)
# viewCount is intentionally excluded: matching treats it as post-treatment /
# collider (answered questions attract more views). Conditioning on it in the
# Cox robustness specs would reopen that channel.
OBSERVABLE_CONTROL_COVARIATES = [
    "postHour", "postDayOfWeek", "numTags",
    "bodyLenChars", "titleLenChars", "ownerReputation",
]
COVARIATES_MAIN_OBSERVABLE = COVARIATES_MAIN + OBSERVABLE_CONTROL_COVARIATES

# ISS-06 (revision): alternative codings of firstAnswerScore for the score-coding
# sensitivity check (revision_robustness.run_score_coding_sensitivity). The tabled
# control is the LINEAR firstAnswerScore, winsorized (p5–p95) and z-scored via
# CONTINUOUS_COVARIATES_EXTENDED. These two alternatives probe whether the arrival HR
# depends on the functional form of the score control.
#
# Concave "signed-log" coding: asinh(score) = ln(score + sqrt(score^2 + 1)), computed in
# revision_robustness._add_score_codings. NOT log1p — log1p is undefined for the ~2.08M
# NEGATIVE (downvoted) scores present in the data, whereas asinh is defined on all of R
# and behaves like ln(2·score) for large positive score. asinh(x) != log(1+x) for x > 0,
# so do not re-implement this as log1p or describe it as such in prose/tables. The column
# STRING stays "firstAnswerScoreLog" (a legacy id) only so the existing full-data fit
# caches keep hitting — its VALUES are asinh. Added to CONTINUOUS_COVARIATES_EXTENDED below
# so it flows through the IDENTICAL winsorize(p5–p95)+z-score pipeline as the linear
# version; the only thing that differs between the linear and asinh specs is the
# functional form, not the standardization.
FIRST_ANSWER_SCORE_LOG = "firstAnswerScoreLog"

# Ordinal bin dummies: 1–2, 3–9, 10+ votes. Reference is the omitted cell — intended as
# "score 0", but empirically "score <= 0": ~2.08M treated rows carry NEGATIVE scores
# (downvoted answers), which fail all three >= tests and fall into the reference (the
# score >= 0 extraction assumption was wrong; see revision_robustness._add_score_codings).
# These deliberately BYPASS the continuous winsorize+z-score pipeline — they are 0/1
# indicators that need no standardization — so they are intentionally NOT added to
# CONTINUOUS_COVARIATES_EXTENDED.
FIRST_ANSWER_SCORE_BINS = [
    "firstAnswerScoreBin_1to2",
    "firstAnswerScoreBin_3to9",
    "firstAnswerScoreBin_10plus",
]

# ISS-06: answer-quality robustness (speed spec extension)
QUALITY_COVARIATES = ["hasAcceptedAnswer", "firstAnswerScore", "firstAnswerBodyLenChars"]
COVARIATES_SPEED_QUALITY = COVARIATES_SPEED + QUALITY_COVARIATES
# Discrete RT bins already condition on response-time window, so quality controls
# attach to Model A (no continuous log-RT interactions within bin).
COVARIATES_MAIN_QUALITY = COVARIATES_MAIN + QUALITY_COVARIATES


CONTINUOUS_COVARIATES_EXTENDED = CONTINUOUS_COVARIATES + OBSERVABLE_CONTROL_COVARIATES + [
    "firstAnswerScore", "firstAnswerBodyLenChars",
    # asinh (signed-log) score coding: flows through the SAME winsorize(p5–p95)+z-score
    # pipeline as the linear firstAnswerScore. The bin dummies (FIRST_ANSWER_SCORE_BINS)
    # are deliberately absent here — they bypass continuous processing.
    FIRST_ANSWER_SCORE_LOG,
]

RT_BIN_EDGES_HOURS = [0, 0.25, 0.5, 1, 2, 4, 8, 12, 24, 72, float("inf")]
RT_BIN_LABELS = [
    "0-15 min", "15-30 min", "30-60 min", "1-2 hr", "2-4 hr", "4-8 hr", "8-12 hr",
    "12-24 hr", "1-3 days", ">3 days",
]
