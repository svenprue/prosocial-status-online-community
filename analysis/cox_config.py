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

RT_BIN_EDGES_HOURS = [0, 0.25, 0.5, 1, 2, 4, 8, 12]
RT_BIN_LABELS = [
    "0-15 min", "15-30 min", "30-60 min", "1-2 hr", "2-4 hr", "4-8 hr", "8-12 hr",
]
