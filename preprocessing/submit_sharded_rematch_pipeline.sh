#!/usr/bin/env bash
# Stage 4 sharded rematch → then the parallel post-match analysis DAG.
#
# Usage:
#   bash preprocessing/submit_sharded_rematch_pipeline.sh
#   N_SHARDS=8 bash preprocessing/submit_sharded_rematch_pipeline.sh
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
mkdir -p logs

export N_SHARDS="${N_SHARDS:-6}"
export N_BOOTSTRAP="${N_BOOTSTRAP:-100}"
export N_JOBS="${N_JOBS:-4}"
export SCOPE="${SCOPE:-all}"
export SEED="${SEED:-42}"

echo "=== Submitting sharded Stage 4 (N_SHARDS=$N_SHARDS) + parallel analysis ==="

J_PREP=$(sbatch --parsable preprocessing/slurm_match_prepare.sbatch)
echo "match_prepare:  $J_PREP"

J_SHARD=$(sbatch --parsable \
  --dependency="afterok:${J_PREP}" \
  --array="0-$((N_SHARDS-1))" \
  --export=ALL,N_SHARDS \
  preprocessing/slurm_match_shard.sbatch)
echo "match_shard[0-$((N_SHARDS-1))]: $J_SHARD (afterok:$J_PREP)"

J_MERGE=$(sbatch --parsable \
  --dependency="afterok:${J_SHARD}" \
  --export=ALL,N_SHARDS \
  preprocessing/slurm_match_merge.sbatch)
echo "match_merge:    $J_MERGE (afterok:$J_SHARD)"

echo
echo "=== Submitting parallel analysis DAG after match_merge ==="
AFTER_REMATCH="$J_MERGE" bash preprocessing/submit_parallel_after_rematch.sh
