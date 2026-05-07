#!/bin/bash
#SBATCH --job-name=ledidi_ag_rescore
#SBATCH --output=sbatch_out/ledidi_comparison/%j_ag_stdout.out
#SBATCH --error=sbatch_out/ledidi_comparison/%j_ag_stderr.out
#SBATCH --time=01:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --gres=gpu:1
#SBATCH --qos=fast
#SBATCH --partition=gpuq
#SBATCH --constraint=h100
#SBATCH --exclude=bamgpu29
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=anonymous@example.com

# Post-hoc AG K562 scoring on LEDIDI matched CSVs + ISM trajectories.
# Uses scripts/alphagenome/score_sequences.py (JAX, alphagenome env).

export PATH="${HOME}/.conda/envs/alphagenome/bin:$PATH"

set -euo pipefail
PROJECT_DIR="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")/../.." && pwd)}"
cd "$PROJECT_DIR"
mkdir -p sbatch_out/ledidi_comparison

# 1) ISM trajectories from run_ism_baseline.py
for uid in 50178 181692; do
    in="results/ism_baseline/ism_trajectory_uid${uid}.csv"
    out="results/ism_baseline/ism_trajectory_uid${uid}_ag.csv"
    if [[ -f "$in" ]]; then
        echo "=== ISM uid=${uid} ==="
        python scripts/alphagenome/score_sequences.py \
            --input "$in" \
            --output "$out" \
            --seq_column sequence \
            --cell_types k562 \
            --batch_size 64
    else
        echo "(skip, missing: $in)"
    fi
done

# 2) LEDIDI matched-seed CSVs
shopt -s nullglob
for csv in results/ledidi_comparison/matched/uid*/ledidi_uid*_t*_l*.csv; do
    out="${csv%.csv}_ag.csv"
    echo "=== LEDIDI $(basename "$csv") ==="
    python scripts/alphagenome/score_sequences.py \
        --input "$csv" \
        --output "$out" \
        --seq_column edited_sequence \
        --cell_types k562 \
        --batch_size 64
done

echo "LEDIDI + ISM AG post-hoc scoring done."
