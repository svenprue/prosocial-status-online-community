#!/usr/bin/env bash
# Rematch Stage 4 → Stages 5–7 → full bootstrap + ISS-04 → merge/figures/PDF.
#
# Usage:
#   bash preprocessing/submit_rematch_full_pipeline.sh
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
mkdir -p logs

export N_BOOTSTRAP="${N_BOOTSTRAP:-100}"
export N_JOBS="${N_JOBS:-4}"
export SCOPE="${SCOPE:-all}"
export SEED="${SEED:-42}"

echo "=== Submitting rematch full pipeline ==="

J4=$(sbatch --parsable preprocessing/slurm_rematch_stage4.sbatch)
echo "Stage 4 rematch:          $J4"

J57=$(sbatch --parsable --dependency="afterok:${J4}" \
  preprocessing/slurm_pipeline_stage5to7.sbatch)
echo "Stages 5→7:               $J57 (afterok:$J4)"

JBOOT=$(sbatch --parsable --dependency="afterok:${J57}" \
  --export=ALL,N_BOOTSTRAP,N_JOBS,SCOPE,SEED \
  preprocessing/slurm_pair_bootstrap_shard_full.sbatch)
echo "Bootstrap shards 1–100:   $JBOOT (afterok:$J57)"

JISS04=$(sbatch --parsable --dependency="afterok:${J57}" \
  preprocessing/slurm_iss04_composite.sbatch)
echo "ISS-04 composite:         $JISS04 (afterok:$J57)"

JMERGE=$(sbatch --parsable \
  --dependency="afterok:${JBOOT},afterok:${JISS04}" \
  --export=ALL,N_BOOTSTRAP,SCOPE,SEED \
  preprocessing/slurm_pair_bootstrap_merge_and_rest.sbatch)
echo "Merge+figures+PDF:        $JMERGE (afterok:$JBOOT+$JISS04)"

echo
echo "Monitor:"
echo "  squeue -u \$USER -j ${J4},${J57},${JBOOT},${JISS04},${JMERGE}"
echo "  tail -f logs/slurm_rematch_s4_${J4}.out"
