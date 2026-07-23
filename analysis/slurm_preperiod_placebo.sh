#!/bin/bash
#SBATCH --job-name=pre_placebo
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=200G
#SBATCH --time=2:00:00
#SBATCH --output=/yen/home/users/lenardst/prosocial-status-online-community/analysis/logs/slurm_pre_placebo_%j.log

# ISS-38 (Audit #8, OPTIONAL): pre-period (T_Q - 1 day) placebo for the newcomer
# (< 1 Week) stratum, Model A + observable-controls spec. Single full-data fit per spec
# (~13-14M rows after the -24h cutpoint split; cap raised so no subsampling). Results are
# printed to this log; no caches or result files are written.
set -euo pipefail

REPO=/yen/home/users/lenardst/prosocial-status-online-community
cd "$REPO/analysis"

# Newcomer stratum is ~12.6M interval rows; the -24h split adds ~4M pre-question rows.
# Raise the fit-row cap above that so the point estimates come from a full-data fit
# (no matched-pair subsampling), matching the tabled newcomer robustness fits.
export COX_MAX_FIT_ROWS=16000000
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1

"$REPO/.venv/bin/python" newcomer_preperiod_placebo.py --input "$REPO/data/event_history"
