#!/bin/bash
#SBATCH --job-name=ism_baseline
#SBATCH --output=sbatch_out/k562_mdlm_gpa/%j_ism_stdout.out
#SBATCH --error=sbatch_out/k562_mdlm_gpa/%j_ism_stderr.out
#SBATCH --time=00:30:00
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

# ISM greedy hill-climb baseline vs GPA on matched seeds uid=50178, uid=181692.

source ${CONDA_BASE}/etc/profile.d/conda.sh
conda activate d3_cuda118

set -euo pipefail
PROJECT_DIR="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")/../.." && pwd)}"
cd "$PROJECT_DIR"
mkdir -p sbatch_out/k562_mdlm_gpa

OUT_DIR="results/ism_baseline"
mkdir -p "$OUT_DIR"

for UID_ in 50178 181692; do
    SEED_H5="results/k562_mdlm_gpa/single_seed_ism_${UID_}.h5"
    if [[ ! -f "$SEED_H5" ]]; then
        echo "SKIP: $SEED_H5 missing"
        continue
    fi
    echo "=== ISM uid=${UID_} ==="
    python scripts/k562_mdlm_gpa/run_ism_baseline.py \
        --seed_h5 "$SEED_H5" \
        --uid "$UID_" \
        --total_steps 115 \
        --batch_size 1024 \
        --output_dir "$OUT_DIR"
done

echo "Done. ISM trajectories -> $OUT_DIR"
