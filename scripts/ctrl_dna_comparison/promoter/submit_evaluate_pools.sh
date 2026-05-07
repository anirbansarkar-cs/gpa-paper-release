#!/bin/bash
# Run the 6-metric evaluation on one or more promoter result pools.
# Usage:
#   bash submit_evaluate_pools.sh [POOL_GLOB ...]
# If no arguments are given, evaluates all CtrlDNA & GPA full-run dirs
# under results/ctrl_dna_comparison/promoter/.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
cd "${PROJECT_DIR}"

if [[ $# -eq 0 ]]; then
    POOLS=(
        "results/ctrl_dna_comparison/promoter/ctrldna_full_*_seed*"
        "results/ctrl_dna_comparison/promoter/gpa_full_*_seed*"
    )
else
    POOLS=("$@")
fi

mkdir -p sbatch_out/ctrl_dna_comparison

JOB_NAME="evaluate_promoter_pools"
JID=$(sbatch --parsable <<JOB_EOF
#!/bin/bash
#SBATCH --job-name=${JOB_NAME}
#SBATCH --output=sbatch_out/ctrl_dna_comparison/${JOB_NAME}_%j.out
#SBATCH --error=sbatch_out/ctrl_dna_comparison/${JOB_NAME}_%j.err
#SBATCH --time=01:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=8G
#SBATCH --gres=gpu:1
#SBATCH --qos=koolab_shared
#SBATCH --partition=gpuq
#SBATCH --constraint=h100
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=anonymous@example.com

source ${CONDA_BASE}/etc/profile.d/conda.sh
conda activate d3_cuda118
set -euo pipefail
cd ${PROJECT_DIR}

python scripts/ctrl_dna_comparison/promoter/evaluate_promoter_pools.py \
    --pools ${POOLS[@]} \
    --output results/ctrl_dna_comparison/promoter/eval_summary.csv
JOB_EOF
)
echo "${JOB_NAME}: job ${JID}"
echo "Pools: ${POOLS[*]}"
