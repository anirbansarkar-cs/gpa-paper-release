#!/bin/bash
#SBATCH --job-name=ledidi_matched
#SBATCH --output=sbatch_out/ledidi_comparison/%j_ledidi_stdout.out
#SBATCH --error=sbatch_out/ledidi_comparison/%j_ledidi_stderr.out
#SBATCH --time=02:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --qos=fast
#SBATCH --partition=gpuq
#SBATCH --constraint=h100
#SBATCH --exclude=bamgpu29
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=anonymous@example.com

# LEDIDI matched-seed sweep: 2 seeds x 5 l x 2 target x 50 copies ~ 1 h on H100.

source ${CONDA_BASE}/etc/profile.d/conda.sh
conda activate d3_cuda118

set -euo pipefail
PROJECT_DIR="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")/../.." && pwd)}"
cd "$PROJECT_DIR"
mkdir -p sbatch_out/ledidi_comparison

for UID_ in 50178 181692; do
    SEED_H5="results/k562_mdlm_gpa/single_seed_ism_${UID_}.h5"
    OUT_DIR="results/ledidi_comparison/matched/uid${UID_}"
    if [[ ! -f "$SEED_H5" ]]; then
        echo "SKIP uid=${UID_}: $SEED_H5 missing"
        continue
    fi
    mkdir -p "$OUT_DIR"
    echo "=== LEDIDI uid=${UID_} ==="
    python scripts/ledidi_comparison/run_ledidi_matched_seeds.py \
        --seed_h5 "$SEED_H5" \
        --uid "$UID_" \
        --output_dir "$OUT_DIR" \
        --n_copies 50 \
        --l_grid 0.02 0.05 0.15 0.4 1.0 \
        --target_grid 10.0 15.0 \
        --tau 1.0 --lr 1.0 --max_iter 1000 \
        --ledidi_batch_size 64 --early_stop 100
done

echo "LEDIDI matched-seed rerun complete."
