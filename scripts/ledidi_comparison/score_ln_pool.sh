#!/bin/bash
#SBATCH --job-name=ln_v2_rescore
#SBATCH --output=sbatch_out/gpa_vs_ism_ledidi_v2/ln_rescore_%j.out
#SBATCH --error=sbatch_out/gpa_vs_ism_ledidi_v2/ln_rescore_%j.err
#SBATCH --time=01:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --qos=fast
#SBATCH --partition=gpuq
#SBATCH --constraint=h100
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=anonymous@example.com

# ===========================================================================
# LegNet K562 rescore for v2 outputs (GPA + ISM + LEDIDI bundled)
# ===========================================================================
# Writes both fwd-only and fwd+RC-averaged LegNet scores per the LegNet paper
# (Penzar 2023) test-time-augmentation convention. GPA h5 datasets are
# ``ln_k562_scores_v2{,_rc}``; CSV columns are ``k562_ln{,_rc}``.
#
# Usage:
#   sbatch scripts/ledidi_comparison/score_ln_pool.sh
#   FILTER=pool_A sbatch scripts/ledidi_comparison/score_ln_pool.sh
#   bash scripts/ledidi_comparison/score_ln_pool.sh DRY  # interactive dry-run
# ===========================================================================
set -euo pipefail
PROJECT_DIR="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")/../.." && pwd)}"
cd "$PROJECT_DIR"
mkdir -p sbatch_out/gpa_vs_ism_ledidi_v2

source ${CONDA_BASE}/etc/profile.d/conda.sh
conda activate d3_cuda118

V2_ROOT="${V2_ROOT:-results/ledidi_comparison/gpa_vs_ism_ledidi_v2}"

CMD=(python scripts/ledidi_comparison/score_ln_pool.py
     --v2_root "$V2_ROOT"
     --batch_size "${BATCH_SIZE:-256}")

if [[ -n "${FILTER:-}" ]]; then
    CMD+=(--filter "$FILTER")
fi

if [[ "${1:-}" == "DRY" ]]; then
    CMD+=(--dry_run)
    echo "Dry-run discovery (no scoring):"
    "${CMD[@]}"
    exit 0
fi

echo "========================================================================"
echo "LegNet v2 rescore (fwd + RC averaging)"
echo "  V2_ROOT: $V2_ROOT"
echo "  Filter:  ${FILTER:-<none>}"
echo "========================================================================"
"${CMD[@]}"
