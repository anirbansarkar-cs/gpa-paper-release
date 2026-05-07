#!/bin/bash
# Train 3 per-cell regLM EnformerModel oracles on Reddy promoter MPRA data.
# Produces human_paired_{jurkat,k562,THP1}.ckpt used by Ctrl-DNA promoter runs
# and by the GPA promoter oracle wrapper (B4).
set -euo pipefail

PROJECT_DIR="${GPA_REPO_ROOT}"
cd "$PROJECT_DIR"

COMPARISON_DIR="scripts/ctrl_dna_comparison"
PROMOTER_DIR="${COMPARISON_DIR}/promoter"
DATA_CSV="${PROMOTER_DIR}/data/finetuning_data.csv"
OUT_DIR="${PROMOTER_DIR}/checkpoints"

mkdir -p sbatch_out/ctrl_dna_comparison "${OUT_DIR}"

TMPDIR_SCRIPTS=$(mktemp -d)

echo "========================================================================"
echo "Train 3 promoter oracles (JURKAT / K562 / THP1) on Reddy MPRA"
echo "========================================================================"

for CELL in JURKAT K562 THP1; do
    JOB_NAME="train_promoter_oracle_${CELL}"
    SCRIPT="${TMPDIR_SCRIPTS}/${JOB_NAME}.sh"

    cat > "${SCRIPT}" <<'SCRIPT_EOF'
#!/bin/bash
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
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

python ${PROMOTER_DIR}/train_oracles.py \\
    --cell ${CELL} \\
    --data_csv ${DATA_CSV} \\
    --out_dir ${OUT_DIR} \\
    --seq_len 250 \\
    --epochs 20 \\
    --batch_size 128 \\
    --lr 1e-4 \\
    --num_workers 4 \\
    --pretrained
EOF

    JID=$(sbatch --parsable \
        --qos="qos_long" \
        --time="24:00:00" \
        --job-name="${JOB_NAME}" \
        --output="sbatch_out/ctrl_dna_comparison/${JOB_NAME}_%j.out" \
        --error="sbatch_out/ctrl_dna_comparison/${JOB_NAME}_%j.err" \
        "${SCRIPT}")
    echo "  ${JOB_NAME}: job ${JID}"
done

rm -rf "${TMPDIR_SCRIPTS}"
echo ""
echo "Done. 3 jobs submitted (qos_long, 24H, any gpu)."
