"""
Delete cached data and model outputs so the pipeline rebuilds from scratch.

Use after changing tenure bucketing, covariates, or when you want fresh fits.

Usage:
    python clear_caches.py [--data-only | --models-only]
"""
import os
import sys
import argparse
import glob

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_CACHE_DIR = os.path.join(_SCRIPT_DIR, "data_cache")
MODEL_CACHE_DIR = os.path.join(_SCRIPT_DIR, "model_cache")


def main():
    parser = argparse.ArgumentParser(description="Delete analysis caches")
    parser.add_argument("--data-only", action="store_true", help="Only delete data_cache (intervals, descriptives)")
    parser.add_argument("--models-only", action="store_true", help="Only delete model_cache (fits, results CSVs)")
    args = parser.parse_args()

    deleted = []
    if not args.models_only:
        for pattern in [os.path.join(DATA_CACHE_DIR, "*.parquet"), os.path.join(DATA_CACHE_DIR, "*.pkl")]:
            for path in glob.glob(pattern):
                try:
                    os.remove(path)
                    deleted.append(path)
                except OSError as e:
                    print(f"  skip {path}: {e}", file=sys.stderr)
    if not args.data_only:
        for pattern in [
            os.path.join(MODEL_CACHE_DIR, "*.pkl"),
            os.path.join(MODEL_CACHE_DIR, "*.csv"),
        ]:
            for path in glob.glob(pattern):
                try:
                    os.remove(path)
                    deleted.append(path)
                except OSError as e:
                    print(f"  skip {path}: {e}", file=sys.stderr)

    if deleted:
        print(f"Removed {len(deleted)} file(s):")
        for p in deleted:
            print(f"  {p}")
    else:
        print("No cache files found to remove.")


if __name__ == "__main__":
    main()
