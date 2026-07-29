#!/bin/bash
#SBATCH --job-name=adoption_figs_regen
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --output=/yen/home/users/lenardst/prosocial-status-online-community/analysis/logs/adoption_figs_regen_%j.log
#SBATCH --error=/yen/home/users/lenardst/prosocial-status-online-community/analysis/logs/adoption_figs_regen_%j.log

# dirname $0 resolves to the slurm SPOOL dir, so hardcode REPO (run-state gotcha)
REPO=/yen/home/users/lenardst/prosocial-status-online-community
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1

cd "$REPO/analysis" || exit 1

echo "host=$(hostname) start=$(date -Is)"
/usr/bin/python3 -c "import pandas,numpy,matplotlib; print('deps ok', pandas.__version__)" || exit 1
/usr/bin/python3 help_rate_over_time.py \
  --input ../data/event_history \
  --output-dir output_figures \
  --adoption-only \
  --adoption-bin-hours 0.1 \
  --adoption-max-hours 6.0
echo "exit=$? end=$(date -Is)"
