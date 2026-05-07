#!/bin/bash
# DRAKES-protocol scoring for all Phase C pools.
#   gpa_output_pool.h5 -> drakes_protocol/<name>.json
# Auto-skips already-scored. Spreads across {default, koolab_shared, fast}.
set -euo pipefail

H5_NAME="gpa_output_pool.h5"
JSON_SUFFIX=""
WALL="06:00:00"

PROJECT_DIR="${GPA_REPO_ROOT}"
cd "$PROJECT_DIR"

RESULTS_DIR="results/rerd_comparison"
OUT_DIR="${RESULTS_DIR}/drakes_protocol"
SCRIPT_DIR="scripts/rerd_comparison"

mkdir -p "${OUT_DIR}" sbatch_out/rerd_comparison
TMPDIR_SCRIPTS=$(mktemp -d)

QOSES=("default" "koolab_shared" "fast")
# Per-QOS time = WALL but fast capped at 04:00:00
QOS_TIME=("06:00:00" "06:00:00" "04:00:00")

i=0
n_submitted=0
n_skipped=0
for POOL_DIR in "${RESULTS_DIR}"/run_*/; do
    POOL_NAME=$(basename "${POOL_DIR}")
    H5="${POOL_DIR}${H5_NAME}"
    JSON="${OUT_DIR}/${POOL_NAME}${JSON_SUFFIX}.json"

    if [ ! -f "${H5}" ]; then
        continue   # sampling not done yet
    fi
    if [ -f "${JSON}" ]; then
        n_skipped=$((n_skipped + 1))
        continue
    fi

    JOB_NAME="drakesC${JSON_SUFFIX}_${POOL_NAME}"
    # Skip if this pool's scoring job is already queued (running or pending)
    if squeue -u $(whoami) -o "%.50j" 2>/dev/null | grep -q "^[[:space:]]*${JOB_NAME}\$"; then
        n_skipped=$((n_skipped + 1))
        continue
    fi
    SCRIPT="${TMPDIR_SCRIPTS}/${JOB_NAME}.sh"

    q_idx=$((i % ${#QOSES[@]}))
    QOS="${QOSES[$q_idx]}"
    TIME="${QOS_TIME[$q_idx]}"

    cat > "${SCRIPT}" <<'SCRIPT_EOF'
#!/bin/bash
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --mem-per-cpu=8G
#SBATCH --gres=gpu:1
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=anonymous@example.com

source ${CONDA_BASE}/etc/profile.d/conda.sh
conda activate d3_cuda118
set -euo pipefail
SCRIPT_EOF

    cat >> "${SCRIPT}" << EOF
cd ${PROJECT_DIR}
python ${SCRIPT_DIR}/score_drakes_protocol.py \\
    --input_h5 ${H5} \\
    --output_json ${JSON}
EOF

    JID=$(sbatch --parsable \
        --partition=gpuq \
        --qos="${QOS}" \
        --time="${TIME}" \
        --job-name="${JOB_NAME}" \
        --output="sbatch_out/rerd_comparison/${JOB_NAME}_%j.out" \
        --error="sbatch_out/rerd_comparison/${JOB_NAME}_%j.err" \
        "${SCRIPT}" 2>/dev/null) || { echo "  [LIMIT] hit submit cap at i=${i}"; break; }
    printf '  [%-13s] %-50s : %s\n' "${QOS}" "${POOL_NAME}" "${JID}"
    i=$((i + 1))
    n_submitted=$((n_submitted + 1))
done

rm -rf "${TMPDIR_SCRIPTS}"
echo
echo "Submitted: ${n_submitted}, skipped (already scored): ${n_skipped}"
