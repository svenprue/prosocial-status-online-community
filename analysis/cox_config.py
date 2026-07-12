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
MAX_FIT_ROWS = 8_000_000
SUBSAMPLE_SEED = 42
# Cap concurrent lifelines fits (each can hold up to MAX_FIT_ROWS). Override via
# --n-jobs / COX_MAX_FIT_WORKERS; default keeps Stage-6 RT-bin and bootstrap pools
# from OOM'ing when cpu_count is large.
MAX_FIT_WORKERS = int(os.environ.get("COX_MAX_FIT_WORKERS", "4"))
VARIANCE_ESTIMATOR = "robust_sandwich"
CLUSTER_COL = "match_id"

# Primary Cox outcome: generalized helping to others, excluding self-directed accepts
# and (separately tabulated) edits. Matches the prior AllData / answers_comments estimand.
PRIMARY_HELP_TYPES = ["answer", "comment"]

CONTINUOUS_COVARIATES = [
    "hasAnswer_response_time_interaction",
    "treated_response_time_interaction",
    "treated_post_question_response_time_interaction",
]

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
OBSERVABLE_CONTROL_COVARIATES = [
    "postHour", "postDayOfWeek", "numTags",
    "viewCount", "bodyLenChars", "titleLenChars", "ownerReputation",
]
COVARIATES_MAIN_OBSERVABLE = COVARIATES_MAIN + OBSERVABLE_CONTROL_COVARIATES

# ISS-06: answer-quality robustness (speed spec extension)
QUALITY_COVARIATES = ["hasAcceptedAnswer", "firstAnswerScore", "firstAnswerBodyLenChars"]
COVARIATES_SPEED_QUALITY = COVARIATES_SPEED + QUALITY_COVARIATES

CONTINUOUS_COVARIATES_EXTENDED = CONTINUOUS_COVARIATES + OBSERVABLE_CONTROL_COVARIATES + [
    "firstAnswerScore", "firstAnswerBodyLenChars",
]

RT_BIN_EDGES_HOURS = [0, 0.25, 0.5, 1, 2, 4, 8, 12, 24, 72, float("inf")]
RT_BIN_LABELS = [
    "0-15 min", "15-30 min", "30-60 min", "1-2 hr", "2-4 hr", "4-8 hr", "8-12 hr",
    "12-24 hr", "1-3 days", ">3 days",
]
