"""
fit_cox_models.py — Fit Cox survival models and cache results.

Loads event-history data, builds survival intervals, fits Model A (main) and
Model B (speed) per tenure bucket, plus pooled all-data and response-time bin models.
Outputs: model_cache/*.csv and model_cache/descriptives.pkl.

Usage:
    python fit_cox_models.py [--input <path>] [--sample N] [--no-cache] [--n-jobs N]
"""
import os
import sys
import pickle
import argparse

# Ensure analysis dir is on path when run from project root
_ANALYSIS_DIR = os.path.dirname(os.path.abspath(__file__))
if _ANALYSIS_DIR not in sys.path:
    sys.path.insert(0, _ANALYSIS_DIR)

from cox_config import CACHE_DIR
from cox_data import load_and_prepare
from cox_fit import fit_all_models, fit_all_data_models, fit_response_time_bin_models


def main():
    default_input = os.path.normpath(os.path.join(_ANALYSIS_DIR, "..", "data", "event_history"))
    parser = argparse.ArgumentParser(description="Fit Cox survival models per tenure bucket")
    parser.add_argument("--input", default=default_input, help="Input data folder")
    parser.add_argument("--sample", type=int, default=None, help="Subsample N matched pairs")
    parser.add_argument("--no-cache", action="store_true", help="Ignore cached models")
    parser.add_argument("--n-jobs", type=int, default=None, help="Parallel jobs for tenure-bucket fits")
    args = parser.parse_args()

    model_df, descriptives = load_and_prepare(args.input, sample_size=args.sample)

    os.makedirs(CACHE_DIR, exist_ok=True)
    with open(os.path.join(CACHE_DIR, "descriptives.pkl"), "wb") as f:
        pickle.dump(descriptives, f)
    print(f"✓ Saved descriptives to {os.path.abspath(os.path.join(CACHE_DIR, 'descriptives.pkl'))}")

    fit_all_models(model_df, use_cache=not args.no_cache, n_jobs=args.n_jobs)
    fit_all_data_models(model_df, use_cache=not args.no_cache)
    fit_response_time_bin_models(model_df, use_cache=not args.no_cache, n_jobs=args.n_jobs)


if __name__ == "__main__":
    main()
