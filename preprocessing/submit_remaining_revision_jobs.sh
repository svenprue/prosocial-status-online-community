#!/usr/bin/env bash
# Submit remaining revision jobs in parallel with (or after) bootstrap shards:
#   - ISS-04 composite (event_history_alltypes + Cox decomposition)
#   - matching plot refresh (common_support / love_plot)
# Then (re)submit merge+figures+PDF with afterok on shards + these jobs.
#
# Usage:
#   bash preprocessing/submit_remaining_revision_jobs.sh
#   BOOT_ARRAY=370659 bash preprocessing/submit_remaining_revision_jobs.sh
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
mkdir -p logs

BOOT_ARRAY="${BOOT_ARRAY:-}"
if [ -z "$BOOT_ARRAY" ]; then
  # Prefer the newest boot_shard array still in queue / recently submitted.
  BOOT_ARRAY=$(squeue -u "$USER" -h -o '%i %j' | awk '/boot_shard/ {print $1; exit}' | sed 's/_.*//')
fi
if [ -z "$BOOT_ARRAY" ]; then
  echo "⚠ No boot_shard array found in queue; set BOOT_ARRAY=<id> explicitly."
  echo "  Will submit ISS-04 + plots only (no merge dependency)."
fi

# Drop any pending merge that only waits on the array (we'll replace it).
for jid in $(squeue -u "$USER" -h -o '%i %j %T' | awk '$2=="boot_merge" && $3=="PENDING" {print $1}'); do
  echo "Cancelling stale pending merge job $jid"
  scancel "$jid" || true
done

ISS04=$(sbatch --parsable preprocessing/slurm_iss04_composite.sbatch)
PLOTS=$(sbatch --parsable preprocessing/slurm_refresh_matching_plots.sbatch)
echo "ISS-04 job:        $ISS04"
echo "Matching plots:    $PLOTS"

DEPS="$ISS04:$PLOTS"
if [ -n "$BOOT_ARRAY" ]; then
  DEPS="${BOOT_ARRAY}:${DEPS}"
fi

MERGE=$(sbatch --parsable \
  --dependency="afterok:${DEPS}" \
  preprocessing/slurm_pair_bootstrap_merge_and_rest.sbatch)
echo "Merge+rest job:    $MERGE (afterok:$DEPS)"
echo
squeue -u "$USER" -o '%.18i %.12j %.2t %.10M %R'
echo
echo "Logs:"
echo "  logs/slurm_iss04_${ISS04}.out"
echo "  logs/slurm_match_plots_${PLOTS}.out"
echo "  logs/slurm_boot_merge_${MERGE}.out"
