#!/usr/bin/env bash
# Submit 4 parallel Slurm shard jobs for pair-bootstrap reps 21-100, then a
# merge+figures job that waits for the array to finish.
#
# Usage:
#   bash preprocessing/submit_pair_bootstrap_shards.sh
#   N_JOBS=2 bash preprocessing/submit_pair_bootstrap_shards.sh   # fewer workers / RAM
#   SKIP_MERGE=1 bash preprocessing/submit_pair_bootstrap_shards.sh
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
mkdir -p logs

export N_BOOTSTRAP="${N_BOOTSTRAP:-100}"
export N_JOBS="${N_JOBS:-4}"
export SCOPE="${SCOPE:-all}"
export SEED="${SEED:-42}"

echo "Submitting bootstrap shards (reps 21-40, 41-60, 61-80, 81-100; keep 1-20 from main checkpoint)"
echo "  N_BOOTSTRAP=$N_BOOTSTRAP  N_JOBS=$N_JOBS  SCOPE=$SCOPE  SEED=$SEED"

ARRAY_JOB=$(sbatch --parsable \
  --export=ALL,N_BOOTSTRAP,N_JOBS,SCOPE,SEED \
  preprocessing/slurm_pair_bootstrap_shard.sbatch)
echo "Array job id: $ARRAY_JOB"

if [ "${SKIP_MERGE:-0}" = "1" ]; then
  echo "SKIP_MERGE=1 — not submitting merge job. When shards finish:"
  echo "  sbatch --dependency=afterok:${ARRAY_JOB} preprocessing/slurm_pair_bootstrap_merge_and_rest.sbatch"
  exit 0
fi

MERGE_JOB=$(sbatch --parsable \
  --dependency="afterok:${ARRAY_JOB}" \
  --export=ALL,N_BOOTSTRAP,SCOPE,SEED \
  preprocessing/slurm_pair_bootstrap_merge_and_rest.sbatch)
echo "Merge+rest job id: $MERGE_JOB (afterok:$ARRAY_JOB)"
echo "Monitor: squeue -u \$USER -j ${ARRAY_JOB},${MERGE_JOB}"
echo "Logs:    logs/slurm_boot_shard_${ARRAY_JOB}_*.out  logs/slurm_boot_merge_${MERGE_JOB}.out"
