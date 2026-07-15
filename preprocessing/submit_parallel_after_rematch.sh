#!/usr/bin/env bash
# Parallel rematch-downstream DAG (with-replacement matching already running or done).
#
# Usage:
#   AFTER_REMATCH=<jobid> bash preprocessing/submit_parallel_after_rematch.sh
#   bash preprocessing/submit_parallel_after_rematch.sh   # if rematch already finished
#
# Fan-out after clear/warm barrier:
#   cox_main ∥ rev(observable|quality|placebo|newcomer) ∥ boot_shards ∥
#   (eh_alltypes → iss04_fit) ∥ question_loss
# Then after cox: selection + cohort
# Final merge waits on bootstrap + all analysis jobs.

set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
mkdir -p logs

export N_BOOTSTRAP="${N_BOOTSTRAP:-100}"
export N_JOBS="${N_JOBS:-4}"
export SCOPE="${SCOPE:-all}"
export SEED="${SEED:-42}"

DEP_REMATCH=()
if [ -n "${AFTER_REMATCH:-}" ]; then
  DEP_REMATCH=(--dependency="afterok:${AFTER_REMATCH}")
  echo "Waiting on rematch job: $AFTER_REMATCH"
else
  echo "No AFTER_REMATCH set — submitting immediately (matched parquet must already be current)."
fi

echo "=== Submitting parallel post-rematch DAG ==="

J_EH=$(sbatch --parsable "${DEP_REMATCH[@]}" preprocessing/slurm_eh_default.sbatch)
echo "eh_default (histories+clear+warm): $J_EH"

J_ALL=$(sbatch --parsable "${DEP_REMATCH[@]}" preprocessing/slurm_eh_alltypes.sbatch)
echo "eh_alltypes (∥ with eh_default):  $J_ALL"

J_QLOSS=$(sbatch --parsable "${DEP_REMATCH[@]}" preprocessing/slurm_question_loss.sbatch)
echo "question_loss:                    $J_QLOSS"

# After clear/warm
J_COX=$(sbatch --parsable --dependency="afterok:${J_EH}" preprocessing/slurm_cox_main.sbatch)
echo "cox_main:                         $J_COX (afterok:$J_EH)"

submit_rev() {
  local step="$1" name="$2"
  sbatch --parsable \
    --job-name="$name" \
    --dependency="afterok:${J_EH}" \
    --export=ALL,REV_STEP="$step" \
    preprocessing/slurm_revision_step.sbatch
}

J_ISS02=$(submit_rev observable rev_iss02)
echo "rev observable (ISS-02):          $J_ISS02"
J_ISS06=$(submit_rev quality rev_iss06)
echo "rev quality (ISS-06):             $J_ISS06"
# ISS-16 ViewCount placebo is opt-in only (cumulative dump ViewCount not identified).
# J_ISS16=$(submit_rev placebo rev_iss16)
J_NEW=$(submit_rev newcomer rev_newcom)
echo "rev newcomer:                     $J_NEW"

J_ISS04=$(sbatch --parsable \
  --dependency="afterok:${J_EH},afterok:${J_ALL}" \
  preprocessing/slurm_iss04_fit.sbatch)
echo "iss04_fit:                        $J_ISS04 (afterok:$J_EH+$J_ALL)"

J_BOOT=$(sbatch --parsable \
  --dependency="afterok:${J_EH}" \
  --export=ALL,N_BOOTSTRAP,N_JOBS,SCOPE,SEED \
  preprocessing/slurm_pair_bootstrap_shard_full.sbatch)
echo "boot_shard[0-4]:                  $J_BOOT (afterok:$J_EH)"

# After cox
J_SEL=$(sbatch --parsable --dependency="afterok:${J_COX}" \
  preprocessing/slurm_selection_sensitivity.sbatch)
echo "selection_sensitivity:            $J_SEL (afterok:$J_COX)"

J_COHORT=$(sbatch --parsable --dependency="afterok:${J_COX}" \
  preprocessing/slurm_iss10_after_cox.sbatch)
echo "iss10_cohort:                     $J_COHORT (afterok:$J_COX)"

# Final: merge bootstrap + figures/PDF once everything is in
J_MERGE=$(sbatch --parsable \
  --dependency="afterok:${J_BOOT},afterok:${J_ISS04},afterok:${J_COX},afterok:${J_ISS02},afterok:${J_ISS06},afterok:${J_NEW},afterok:${J_SEL},afterok:${J_COHORT},afterok:${J_QLOSS}" \
  --export=ALL,N_BOOTSTRAP,SCOPE,SEED \
  preprocessing/slurm_pair_bootstrap_merge_and_rest.sbatch)
echo "boot_merge+figures+PDF:           $J_MERGE"

cat <<EOF

DAG summary:
  rematch(${AFTER_REMATCH:-done})
    ├─ eh_default($J_EH) ──┬─ cox($J_COX) ──┬─ sel($J_SEL)
    │                      │               └─ cohort($J_COHORT)
    │                      ├─ iss02($J_ISS02) iss06($J_ISS06) newcomer($J_NEW)
    │                      └─ boot($J_BOOT)
    ├─ eh_alltypes($J_ALL) ── iss04($J_ISS04)  [also waits eh_default/clear]
    └─ q_loss($J_QLOSS)
  merge($J_MERGE) waits on all analysis + boot
  (ISS-16 ViewCount placebo is opt-in only; not in this DAG)

Monitor:
  squeue -u \$USER -j ${J_EH},${J_ALL},${J_COX},${J_ISS02},${J_ISS06},${J_NEW},${J_ISS04},${J_BOOT},${J_SEL},${J_COHORT},${J_QLOSS},${J_MERGE}
EOF
