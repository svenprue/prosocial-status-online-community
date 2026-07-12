"""
Matched-pair bootstrap uncertainty for Cox time-varying models.

lifelines 0.27.x exposes ``robust=True`` on CoxTimeVaryingFitter but raises
NotImplementedError, and it does not wire a public ``cluster_col`` argument for
time-varying fits. This script implements the reviewer-requested matched-pair
alternative by resampling whole match_id pairs and refitting Model A.

The bootstrapped statistic (base_coef / base_hr and the CIs) is the SUMMED DiD
contrast, treated_post_question + is_treated_active — the treatment effect, not
the is_treated_active increment over the waiting-period term.

Replicates run in a process pool with independent RNGs via SeedSequence.spawn
(not byte-identical to a serial Generator stream; statistically equivalent).
After each replicate the coef is flushed to a checkpoint under
``analysis/model_cache/``. Re-running the same (scope, seed, n-bootstrap, …)
config resumes from finished replicates so an OOM/kill only loses in-progress draws.

Outputs:
  analysis/model_cache/results_pair_bootstrap.csv
  analysis/model_cache/pair_bootstrap_checkpoint_<scope>.csv  (resume state)

Usage examples:
  python pair_bootstrap_se.py --scope all --n-bootstrap 200
  python pair_bootstrap_se.py --scope all --sample 10000 --max-pairs 5000 --n-bootstrap 20
  python pair_bootstrap_se.py --scope all --n-bootstrap 200 --n-jobs 4
  python pair_bootstrap_se.py --scope all --n-bootstrap 200 --no-resume  # ignore checkpoint
"""
import argparse
import gc
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import cpu_count

import numpy as np
import pandas as pd

_ANALYSIS_DIR = os.path.dirname(os.path.abspath(__file__))
if _ANALYSIS_DIR not in sys.path:
    sys.path.insert(0, _ANALYSIS_DIR)

from cox_config import (
    BUCKET_ORDER,
    CACHE_DIR,
    COVARIATES_MAIN,
    MAX_FIT_ROWS,
    MAX_FIT_WORKERS,
    PRIMARY_HELP_TYPES,
)
from cox_data import load_and_prepare


DROP_FOR_MODEL_A = [
    "tenure_bucket",
    "response_time_hours",
    "response_time_bin",
    "treated_bin2",
    "treated_bin3",
]

# Sampler id in checkpoint meta — bump if draw logic changes incompatibly.
_SAMPLER_ID = "seed_sequence_parallel_v1"

# Worker globals (set in _init_boot_worker; avoid pickling the full frame per task).
_BOOT_SCOPED: pd.DataFrame | None = None
_BOOT_UNIQUE_IDS: np.ndarray | None = None
_BOOT_MAX_PAIRS: int | None = None


def _scope_slug(scope: str) -> str:
    return (
        str(scope)
        .replace(" ", "_")
        .replace("<", "lt")
        .replace(">", "gt")
        .replace("/", "-")
    )


def _checkpoint_paths(scope: str) -> tuple[str, str]:
    slug = _scope_slug(scope)
    base = os.path.join(CACHE_DIR, f"pair_bootstrap_checkpoint_{slug}")
    return base + ".csv", base + ".meta.json"


def _checkpoint_meta(
    scope: str,
    seed: int,
    n_bootstrap: int,
    max_pairs: int | None,
    sample_size: int | None,
) -> dict:
    return {
        "scope": scope,
        "seed": int(seed),
        "n_bootstrap": int(n_bootstrap),
        "max_pairs": None if max_pairs is None else int(max_pairs),
        "sample_size": None if sample_size is None else int(sample_size),
        "sampler": _SAMPLER_ID,
    }


def _atomic_write_json(path: str, payload: dict) -> None:
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)
        f.write("\n")
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def _atomic_write_csv(path: str, df: pd.DataFrame) -> None:
    tmp = path + ".tmp"
    df.to_csv(tmp, index=False)
    # Ensure bytes hit disk before rename (critical for kill/OOM safety).
    with open(tmp, "rb+") as f:
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def _load_checkpoint(
    scope: str,
    seed: int,
    n_bootstrap: int,
    max_pairs: int | None,
    sample_size: int | None,
) -> dict[int, float | None]:
    """Return {1-based replicate: coef_or_None} for a compatible checkpoint."""
    csv_path, meta_path = _checkpoint_paths(scope)
    if not (os.path.exists(csv_path) and os.path.exists(meta_path)):
        return {}
    with open(meta_path, encoding="utf-8") as f:
        meta = json.load(f)
    expected = _checkpoint_meta(scope, seed, n_bootstrap, max_pairs, sample_size)
    if meta != expected:
        print(
            "  ⚠ Checkpoint meta mismatch — starting fresh "
            f"(found {meta}, expected {expected})."
        )
        return {}
    df = pd.read_csv(csv_path)
    if df.empty or "replicate" not in df.columns:
        return {}
    completed: dict[int, float | None] = {}
    for _, row in df.iterrows():
        rep = int(row["replicate"])
        ok = bool(row.get("ok", True))
        if ok and pd.notna(row.get("coef")):
            completed[rep] = float(row["coef"])
        else:
            completed[rep] = None
    last = max(completed)
    print(
        f"  ✓ Resuming from checkpoint: {len(completed)} replicate(s) done "
        f"(next pending among 1..{n_bootstrap}; last finished = {last})"
    )
    return completed


def _save_checkpoint(
    scope: str,
    seed: int,
    n_bootstrap: int,
    max_pairs: int | None,
    sample_size: int | None,
    completed: dict[int, float | None],
) -> None:
    os.makedirs(CACHE_DIR, exist_ok=True)
    csv_path, meta_path = _checkpoint_paths(scope)
    rows = [
        {
            "replicate": rep,
            "coef": coef if coef is not None else np.nan,
            "ok": coef is not None and np.isfinite(coef),
        }
        for rep, coef in sorted(completed.items())
    ]
    _atomic_write_csv(csv_path, pd.DataFrame(rows))
    _atomic_write_json(
        meta_path,
        _checkpoint_meta(scope, seed, n_bootstrap, max_pairs, sample_size),
    )


def _scope_subset(model_df: pd.DataFrame, scope: str) -> pd.DataFrame:
    if scope == "all":
        return model_df.copy()
    if scope not in BUCKET_ORDER:
        raise ValueError(f"Unknown scope '{scope}'. Use 'all' or one of: {', '.join(BUCKET_ORDER)}")
    return model_df.loc[model_df["tenure_bucket"] == scope].copy()


def _model_a_frame(df: pd.DataFrame) -> pd.DataFrame:
    return df.drop(columns=DROP_FOR_MODEL_A, errors="ignore").copy()


def _draw_match_ids(
    unique_ids: np.ndarray,
    rng: np.random.Generator,
    max_pairs: int | None = None,
) -> np.ndarray:
    """Resample match_ids with replacement from a precomputed unique-id pool."""
    n_pairs = len(unique_ids) if max_pairs is None else min(int(max_pairs), len(unique_ids))
    return rng.choice(unique_ids, size=n_pairs, replace=True)


def _bootstrap_sample_by_match(
    df: pd.DataFrame,
    unique_ids: np.ndarray,
    rng: np.random.Generator,
    max_pairs: int | None = None,
) -> pd.DataFrame:
    """Resample whole pairs; prefix unique_id so with-replacement draws stay distinct.

    match_id is left unchanged: fit_cox_cached(robust=False) drops it before the
    MAX_FIT_ROWS downsample (which uses unique_id), so rebuilding match_id is pure waste.
    """
    sampled = _draw_match_ids(unique_ids, rng, max_pairs=max_pairs)
    n_pairs = len(sampled)
    draws = pd.DataFrame({"match_id": sampled, "_boot_draw": np.arange(n_pairs, dtype=np.int64)})
    boot = draws.merge(df, on="match_id", how="left", sort=False)
    draw_prefix = boot["_boot_draw"].astype(str)
    boot["unique_id"] = draw_prefix + "_" + boot["unique_id"].astype(str)
    return boot.drop(columns=["_boot_draw"])


def _coef_from_result(result) -> float | None:
    """Return the SUMMED DiD coefficient (treated_post_question + is_treated_active).

    This is the treatment effect (post-answer vs. pre-question baseline, treated vs.
    control); the bootstrap therefore quantifies uncertainty in the effect itself, not
    in the is_treated_active increment over the waiting-period term.
    """
    if result is None or "is_treated_active" not in result.summary_df.index:
        return None
    s = result.summary_df
    try:
        from cox_fit import DID_TERMS
    except Exception:
        DID_TERMS = ["treated_post_question", "is_treated_active"]
    terms = [t for t in DID_TERMS if t in s.index]
    return float(s.loc[terms, "coef"].sum())


def _init_boot_worker(
    scoped: pd.DataFrame,
    unique_ids: np.ndarray,
    max_pairs: int | None,
) -> None:
    global _BOOT_SCOPED, _BOOT_UNIQUE_IDS, _BOOT_MAX_PAIRS
    _BOOT_SCOPED = scoped
    _BOOT_UNIQUE_IDS = unique_ids
    _BOOT_MAX_PAIRS = max_pairs


def _run_one_replicate(payload: tuple[int, object, str]) -> tuple[int, float | None]:
    """Worker: one bootstrap replicate. payload = (rep, child_seed, scope)."""
    rep, child_seed, scope = payload
    from cox_fit import fit_cox_cached

    rng = np.random.default_rng(child_seed)
    boot_df = _bootstrap_sample_by_match(
        _BOOT_SCOPED, _BOOT_UNIQUE_IDS, rng, max_pairs=_BOOT_MAX_PAIRS
    )
    result = fit_cox_cached(
        boot_df,
        f"ModelA_Bootstrap_{scope}_{rep}",
        COVARIATES_MAIN,
        use_cache=False,
        save_cache=False,
    )
    coef = _coef_from_result(result)
    del boot_df, result
    if coef is not None and np.isfinite(coef):
        return rep, float(coef)
    return rep, None


def _default_n_jobs(n_pending: int) -> int:
    """Bound workers by CPU, pending work, and RAM (each fit ≤ MAX_FIT_ROWS)."""
    cpus = cpu_count() or 4
    return max(1, min(cpus, n_pending, MAX_FIT_WORKERS))


def run_pair_bootstrap(
    input_folder: str,
    scope: str,
    n_bootstrap: int,
    sample_size: int | None = None,
    max_pairs: int | None = None,
    seed: int = 42,
    use_cache: bool = True,
    resume: bool = True,
    n_jobs: int | None = None,
) -> pd.DataFrame:
    try:
        from cox_fit import fit_cox_cached
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "Cox fitting dependencies are unavailable in this environment; "
            "install the project requirements, including lifelines, before running pair bootstrap fits."
        ) from exc

    model_df, _ = load_and_prepare(
        input_folder, sample_size=sample_size, event_help_types=PRIMARY_HELP_TYPES
    )
    scoped = _model_a_frame(_scope_subset(model_df, scope))
    del model_df
    gc.collect()

    n_questions = int(scoped["unique_id"].nunique())
    n_events = int(scoped["event_occurred"].sum())
    if len(scoped) < 100 or n_events < 10:
        raise ValueError(f"Scope '{scope}' has too few rows/events for bootstrap fitting")

    unique_ids = pd.unique(scoped["match_id"].to_numpy())

    base = fit_cox_cached(
        scoped,
        f"ModelA_BootstrapBase_{scope}",
        COVARIATES_MAIN,
        use_cache=use_cache,
        save_cache=use_cache,
    )
    base_coef = _coef_from_result(base)
    if base_coef is None:
        raise RuntimeError(f"Base fit failed for scope '{scope}'")

    completed: dict[int, float | None] = {}
    if resume:
        completed = _load_checkpoint(scope, seed, n_bootstrap, max_pairs, sample_size)
    else:
        csv_path, meta_path = _checkpoint_paths(scope)
        for path in (csv_path, meta_path):
            if os.path.exists(path):
                os.remove(path)
                print(f"  Removed old checkpoint {path}")

    pending = [rep for rep in range(1, int(n_bootstrap) + 1) if rep not in completed]
    if not pending:
        print(f"  ✓ All {n_bootstrap} replicates already checkpointed")
    else:
        ss = np.random.SeedSequence(seed)
        child_seeds = ss.spawn(int(n_bootstrap))
        workers = n_jobs if n_jobs is not None else _default_n_jobs(len(pending))
        # Rough RAM guard: each concurrent fit holds up to MAX_FIT_ROWS.
        print(
            f"  Running {len(pending)} remaining replicate(s) with n_jobs={workers} "
            f"(MAX_FIT_ROWS={MAX_FIT_ROWS:,}, MAX_FIT_WORKERS={MAX_FIT_WORKERS})"
        )
        tasks = [(rep, child_seeds[rep - 1], scope) for rep in pending]

        if workers <= 1:
            _init_boot_worker(scoped, unique_ids, max_pairs)
            for task in tasks:
                rep, coef = _run_one_replicate(task)
                completed[rep] = coef
                status = "ok" if coef is not None else "skipped"
                _save_checkpoint(scope, seed, n_bootstrap, max_pairs, sample_size, completed)
                print(f"  Bootstrap {rep}/{n_bootstrap}: {status} (checkpointed)", flush=True)
        else:
            with ProcessPoolExecutor(
                max_workers=workers,
                initializer=_init_boot_worker,
                initargs=(scoped, unique_ids, max_pairs),
            ) as pool:
                futures = {pool.submit(_run_one_replicate, t): t[0] for t in tasks}
                for fut in as_completed(futures):
                    rep, coef = fut.result()
                    completed[rep] = coef
                    status = "ok" if coef is not None else "skipped"
                    _save_checkpoint(
                        scope, seed, n_bootstrap, max_pairs, sample_size, completed
                    )
                    n_done = sum(1 for r in pending if r in completed)
                    print(
                        f"  Bootstrap {rep}/{n_bootstrap}: {status} "
                        f"(checkpointed; {n_done}/{len(pending)} this batch)",
                        flush=True,
                    )

    boot_coefs = [
        c for _, c in sorted(completed.items())
        if c is not None and np.isfinite(c)
    ]
    if not boot_coefs:
        raise RuntimeError(f"No successful bootstrap fits for scope '{scope}'")

    coefs = np.asarray(boot_coefs, dtype=float)
    row = {
        "scope": scope,
        "n_questions": n_questions,
        "n_events": n_events,
        "n_bootstrap_requested": int(n_bootstrap),
        "n_bootstrap_success": int(len(coefs)),
        "max_pairs_per_replicate": max_pairs if max_pairs is not None else "",
        "base_coef": base_coef,
        "base_hr": float(np.exp(base_coef)),
        "bootstrap_se_coef": float(coefs.std(ddof=1)) if len(coefs) > 1 else np.nan,
        "bootstrap_coef_ci_lo": float(np.percentile(coefs, 2.5)),
        "bootstrap_coef_ci_hi": float(np.percentile(coefs, 97.5)),
        "bootstrap_hr_ci_lo": float(np.exp(np.percentile(coefs, 2.5))),
        "bootstrap_hr_ci_hi": float(np.exp(np.percentile(coefs, 97.5))),
    }
    return pd.DataFrame([row])


def main():
    default_input = os.path.normpath(os.path.join(_ANALYSIS_DIR, "..", "data", "event_history"))
    parser = argparse.ArgumentParser(description="Matched-pair bootstrap SEs for Model A")
    parser.add_argument("--input", default=default_input, help="Folder with study_timelines.parquet and study_events.parquet")
    parser.add_argument("--scope", default="all", help="Scope to fit: 'all' or an exact tenure bucket label")
    parser.add_argument("--sample", type=int, default=None, help="Subsample N matched pairs before fitting")
    parser.add_argument("--max-pairs", type=int, default=None, help="Cap pairs resampled per bootstrap replicate")
    parser.add_argument("--n-bootstrap", type=int, default=200, help="Number of bootstrap replicates")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--n-jobs",
        type=int,
        default=None,
        help=f"Parallel workers (default: min(cpu, pending, MAX_FIT_WORKERS={MAX_FIT_WORKERS}))",
    )
    parser.add_argument("--no-cache", action="store_true", help="Ignore cached base model")
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="Ignore/delete existing checkpoint and start from replicate 1",
    )
    args = parser.parse_args()

    result = run_pair_bootstrap(
        input_folder=args.input,
        scope=args.scope,
        n_bootstrap=args.n_bootstrap,
        sample_size=args.sample,
        max_pairs=args.max_pairs,
        seed=args.seed,
        use_cache=not args.no_cache,
        resume=not args.no_resume,
        n_jobs=args.n_jobs,
    )
    os.makedirs(CACHE_DIR, exist_ok=True)
    out = os.path.join(CACHE_DIR, "results_pair_bootstrap.csv")
    if os.path.exists(out):
        existing = pd.read_csv(out)
        result = pd.concat([existing, result], ignore_index=True)
        result = result.drop_duplicates(subset=["scope", "n_bootstrap_requested", "max_pairs_per_replicate"], keep="last")
    result.to_csv(out, index=False)
    print(f"✓ Wrote {out}")


if __name__ == "__main__":
    main()
