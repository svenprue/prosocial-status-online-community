#!/bin/bash
#SBATCH --job-name=q_loss
#SBATCH --partition=normal
#SBATCH --time=4:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=128G
#SBATCH --output=logs/slurm_q_loss_%j.out
#SBATCH --error=logs/slurm_q_loss_%j.err

set -euo pipefail
ROOT="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")/.." && pwd)}"
cd "$ROOT"
mkdir -p logs

PYTHON="${PYTHON:-$ROOT/.venv/bin/python}"
export PYTHONUNBUFFERED=1

echo "=== $(date -Is) host=$(hostname) job=${SLURM_JOB_ID:-local} question_loss ==="
"$PYTHON" preprocessing/investigate_question_loss.py
echo "=== $(date -Is) q_loss done ==="
