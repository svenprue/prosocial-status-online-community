#!/bin/bash
#SBATCH --job-name=fd_stage
#SBATCH --partition=normal
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=256G
#SBATCH --output=logs/slurm_fd_%x_%j.out
#SBATCH --error=logs/slurm_fd_%x_%j.err
#
# 2026-07-16 full-data point-estimate refit (fix for the subsample-fragile pooled
# estimate). Runs ONE stage on the full answers-only frame with no MAX_FIT_ROWS
# subsample (COX_MAX_FIT_ROWS raised above the ~62M frame). Each stage writes the
# results_*.csv that create_figures.py reads. Cache names are tagged _r100M (see
# cox_fit rows_tag), so these never collide with the 8M-subsample pickles or the
# matched-pair bootstrap (which stays at the 8M default).
#
# Submit one job per stage, concurrently:
#   for S in pooled buckets rtbins observable quality newcomer rt_bins_quality cohort; do
#     sbatch --job-name="fd_$S" --export=ALL,STAGE=$S preprocessing/slurm_fulldata_stage.sbatch
#   done
#
# All stages use the answers-only interval cache (intervals_full_answer_rev2ao,
# already built). The composite-decomposition stage is handled separately (it needs
# other help-type interval caches).

set -euo pipefail
ROOT="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")/.." && pwd)}"
cd "$ROOT"
mkdir -p logs

PYTHON="${PYTHON:-$ROOT/.venv/bin/python}"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export COX_MAX_FIT_ROWS=100000000   # above the ~62M pooled frame -> no subsample

STAGE="${STAGE:?STAGE unset (pooled|buckets|rtbins|observable|quality|newcomer|rt_bins_quality|cohort)}"
echo "=== $(date -Is) host=$(hostname) job=${SLURM_JOB_ID:-local} STAGE=$STAGE ==="
cd "$ROOT/analysis"
"$PYTHON" -c "from cox_config import MAX_FIT_ROWS, PRIMARY_HELP_TYPES; print('MAX_FIT_ROWS', MAX_FIT_ROWS, 'help_types', PRIMARY_HELP_TYPES)"

"$PYTHON" - <<'PY'
import os
stage = os.environ["STAGE"]
inp = "../data/event_history"
if stage in ("pooled", "buckets", "rtbins"):
    from cox_config import PRIMARY_HELP_TYPES
    from cox_data import load_and_prepare
    from cox_fit import fit_all_models, fit_all_data_models, fit_response_time_bin_models
    model_df, _ = load_and_prepare(inp, event_help_types=PRIMARY_HELP_TYPES)
    if stage == "pooled":
        fit_all_data_models(model_df, use_cache=True)
    elif stage == "buckets":
        fit_all_models(model_df, use_cache=True)
    elif stage == "rtbins":
        fit_response_time_bin_models(model_df, use_cache=True)
elif stage == "observable":
    import revision_robustness as r; r.run_observable_controls(inp, use_cache=True)
elif stage == "quality":
    import revision_robustness as r; r.run_answer_quality(inp, use_cache=True)
elif stage == "newcomer":
    import revision_robustness as r; r.run_newcomer_bucket_checks(inp, use_cache=True)
elif stage == "rt_bins_quality":
    import revision_robustness as r; r.run_rt_bins_quality(inp, use_cache=True)
elif stage == "cohort":
    from cohort_robustness import run_cohort_robustness; run_cohort_robustness(inp, use_cache=True)
else:
    raise SystemExit(f"unknown STAGE {stage!r}")
print(f"=== stage {stage} done ===", flush=True)
PY
echo "=== $(date -Is) STAGE=$STAGE complete ==="
