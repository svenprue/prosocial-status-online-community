#!/bin/bash
#SBATCH --job-name=fd_probe
#SBATCH --partition=normal
#SBATCH --time=8:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=900G
#SBATCH --output=logs/slurm_fd_probe_%j.out
#SBATCH --error=logs/slurm_fd_probe_%j.err
#
# 2026-07-16 feasibility probe for the full-data pooled Model A fit (the fix for the
# subsample-fragile pooled point estimate). Fits ONLY pooled Model A on the full
# answers-only analysis frame (~62M rows, no MAX_FIT_ROWS subsample), timing it and
# capturing peak RSS, and prints beta_4 / HR / model-based CI. Does not write any CSV
# or cache. If this completes in reasonable time/memory, the full point-estimate
# pipeline follows.

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

echo "=== $(date -Is) host=$(hostname) job=${SLURM_JOB_ID:-local} full-data probe ==="
cd "$ROOT/analysis"
"$PYTHON" - <<'PY'
import time, resource
import numpy as np
from cox_config import PRIMARY_HELP_TYPES, COVARIATES_MAIN, MAX_FIT_ROWS
print("MAX_FIT_ROWS =", MAX_FIT_ROWS, flush=True)
from cox_data import load_and_prepare
import cox_fit

model_df, _ = load_and_prepare("../data/event_history", event_help_types=PRIMARY_HELP_TYPES)
scoped = model_df.drop(columns=["tenure_bucket","response_time_hours","response_time_bin","treated_bin2","treated_bin3"], errors="ignore")
del model_df
print(f"pooled frame rows={len(scoped):,}", flush=True)

t0 = time.time()
res = cox_fit.fit_cox_cached(scoped, "ModelA_AllData_FULLDATA_PROBE", COVARIATES_MAIN, use_cache=False, save_cache=False)
dt = time.time() - t0
s = res.summary_df
b4 = float(s.loc["is_treated_active","coef"]); se = float(s.loc["is_treated_active","se(coef)"])
b2 = float(s.loc["treated_post_question","coef"])
peak_gb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024**2)
print(f"=== RESULT b4={b4:.6f} HR={np.exp(b4):.6f} CI=[{np.exp(b4-1.96*se):.4f},{np.exp(b4+1.96*se):.4f}] b2={b2:.6f}", flush=True)
print(f"=== FIT_TIME_S={dt:.1f} PEAK_RSS_GB={peak_gb:.1f}", flush=True)
PY
echo "=== $(date -Is) probe done ==="
