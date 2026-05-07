#!/bin/bash
# Evaluate 3 promoter oracles on val + test splits (Pearson/Spearman/MSE).
set -euo pipefail

PROJECT_DIR="${GPA_REPO_ROOT}"
cd "$PROJECT_DIR"

COMPARISON_DIR="scripts/ctrl_dna_comparison"
PROMOTER_DIR="${COMPARISON_DIR}/promoter"

mkdir -p sbatch_out/ctrl_dna_comparison

TMPDIR_SCRIPTS=$(mktemp -d)
SCRIPT="${TMPDIR_SCRIPTS}/eval_promoter_oracles.sh"
cat > "${SCRIPT}" <<'SCRIPT_EOF'
#!/bin/bash
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=8G
#SBATCH --gres=gpu:1
#SBATCH --partition=gpuq
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=anonymous@example.com

source ${CONDA_BASE}/etc/profile.d/conda.sh
conda activate d3_cuda118
set -euo pipefail
SCRIPT_EOF

cat >> "${SCRIPT}" << EOF
cd ${PROJECT_DIR}

python ${PROMOTER_DIR}/eval_oracles.py \\
    --data_csv ${PROMOTER_DIR}/data/finetuning_data.csv \\
    --ckpt_dir ${PROMOTER_DIR}/checkpoints
EOF

JID=$(sbatch --parsable \
    --qos="qos_long" \
    --time="01:00:00" \
    --job-name="eval_promoter_oracles" \
    --output="sbatch_out/ctrl_dna_comparison/eval_promoter_oracles_%j.out" \
    --error="sbatch_out/ctrl_dna_comparison/eval_promoter_oracles_%j.err" \
    "${SCRIPT}")
echo "eval_promoter_oracles: job ${JID}"

rm -rf "${TMPDIR_SCRIPTS}"
