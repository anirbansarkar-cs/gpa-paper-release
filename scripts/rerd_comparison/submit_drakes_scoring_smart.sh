#!/bin/bash
# Size-aware pool scoring for Phase C pools.
# Picks wall time and QOS per pool size; never routes to fast (4h is too short for large pools >100K).
# Skips pools whose JSON exists or whose scoring job is already queued.
# Supports DRY_RUN=1 to print plan without submitting.
set -euo pipefail

PROJECT_DIR="${GPA_REPO_ROOT}"
cd "$PROJECT_DIR"

RESULTS_DIR="results/rerd_comparison"
OUT_DIR="${RESULTS_DIR}/drakes_protocol"
SCRIPT_DIR="scripts/rerd_comparison"

mkdir -p "${OUT_DIR}" sbatch_out/rerd_comparison
TMPDIR_SCRIPTS=$(mktemp -d)

DRY_RUN="${DRY_RUN:-0}"
# bio_ai budget for this run.
#   "auto"  → query current free slots (MaxSubmitPU=16 minus current R+PD)
#   <int>   → cap at that number
#   0       → never use bio_ai (default)
BIO_AI_BUDGET="${BIO_AI_BUDGET:-auto}"
if [ "${BIO_AI_BUDGET}" = "auto" ]; then
    used=$(squeue -u $(whoami) -h -o "%q %t" 2>/dev/null | awk '$1=="bio_ai"' | wc -l)
    BIO_AI_BUDGET=$(( 16 - used ))
    [ "${BIO_AI_BUDGET}" -lt 0 ] && BIO_AI_BUDGET=0
    echo "[auto] bio_ai free slots: ${BIO_AI_BUDGET}"
fi
BIO_AI_USED=0

# ── Pool-size → (wall, qos) tiering ──────────────────────────────────
# Big jobs (>=180K) prefer qos_long (48h cap); biggest (>=280K) can use bio_ai if budget allows.
# Default (12h cap) handles medium jobs.
pick_tier() {
    local N="$1"
    if   [ "$N" -lt  80000 ]; then echo "06:00:00 default"
    elif [ "$N" -lt 180000 ]; then echo "10:00:00 default"
    elif [ "$N" -lt 280000 ]; then echo "14:00:00 qos_long"
    elif [ "$N" -lt 400000 ]; then echo "18:00:00 qos_long"
    else                           echo "30:00:00 qos_long"
    fi
}

# ── Get current queued pool job names ─────────────────────────────
QUEUED=$(squeue -u $(whoami) -h -o "%j" 2>/dev/null | grep "^drakesC__pool_run_" || true)

n_submitted=0; n_skip_done=0; n_skip_queued=0
echo "==============================================================="
echo "Smart pool scoring — DRY_RUN=${DRY_RUN}"
echo "==============================================================="
printf '%-58s %8s  %s\n' "RECIPE" "N_POOL" "WALL  QOS  ACTION"

for POOL_DIR in "${RESULTS_DIR}"/run_*/; do
    POOL_NAME=$(basename "${POOL_DIR}")
    H5="${POOL_DIR}gpa_output_pool.h5"
    JSON="${OUT_DIR}/${POOL_NAME}__pool.json"
    [ ! -f "${H5}" ] && continue

    if [ -f "${JSON}" ]; then
        n_skip_done=$((n_skip_done+1)); continue
    fi

    JOB_NAME="drakesC__pool_${POOL_NAME}"
    if echo "${QUEUED}" | grep -qx "${JOB_NAME}"; then
        n_skip_queued=$((n_skip_queued+1)); continue
    fi

    # Get pool seq count (cheap: just dataset metadata)
    N=$(python3 -c "
import h5py
with h5py.File('${H5}','r') as f:
    k = 'sequences' if 'sequences' in f else list(f.keys())[0]
    print(len(f[k]))
" 2>/dev/null || echo 0)
    [ "$N" -le 0 ] && { echo "  [SKIP] ${POOL_NAME}: pool empty/unreadable"; continue; }

    read WALL QOS <<< "$(pick_tier "$N")"

    printf '%-58s %8s  %s  %s  SUBMIT\n' "${POOL_NAME}" "${N}" "${WALL}" "${QOS}"

    if [ "${DRY_RUN}" = "1" ]; then
        n_submitted=$((n_submitted+1)); continue
    fi

    SCRIPT="${TMPDIR_SCRIPTS}/${JOB_NAME}.sh"
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

    submit_to() {
        local Q="$1" W="$2"
        sbatch --parsable \
            --partition=gpuq \
            --qos="${Q}" \
            --time="${W}" \
            --job-name="${JOB_NAME}" \
            --output="sbatch_out/rerd_comparison/${JOB_NAME}_%j.out" \
            --error="sbatch_out/rerd_comparison/${JOB_NAME}_%j.err" \
            "${SCRIPT}" 2>/dev/null
    }
    # Try primary, then alt. If both full and bio_ai budget remains, try bio_ai.
    # Walls are capped per QOS: default=12h, qos_long=48h, bio_ai=48h.
    cap_wall() {
        local Q="${1-}" W="${2-}" wh
        wh=${W%%:*}
        if [ "$Q" = "default" ] && [ "$wh" -gt 12 ] 2>/dev/null; then echo "12:00:00"; return; fi
        echo "$W"
    }
    PWALL="$(cap_wall "${QOS}" "${WALL}")"
    JID=$(submit_to "${QOS}" "${PWALL}" || true)
    USED_QOS="${QOS}"; USED_WALL="${WALL}"
    if [ -z "${JID}" ]; then
        # primary full → try the other of {default, qos_long}
        if [ "${QOS}" = "default" ]; then ALT_QOS="qos_long"; else ALT_QOS="default"; fi
        ALT_WALL="$(cap_wall "${ALT_QOS}" "${WALL}")"
        JID=$(submit_to "${ALT_QOS}" "${ALT_WALL}" || true)
        USED_QOS="${ALT_QOS}"; USED_WALL="${ALT_WALL}"
    fi
    if [ -z "${JID}" ] && [ "${BIO_AI_BUDGET}" -gt 0 ] && [ "${BIO_AI_USED}" -lt "${BIO_AI_BUDGET}" ]; then
        # both full → try bio_ai (cap 48h, biggest jobs)
        BWALL="${WALL}"
        bh=${BWALL%%:*}
        if [ "$bh" -gt 48 ] 2>/dev/null; then BWALL="48:00:00"; fi
        JID=$(submit_to "bio_ai" "${BWALL}" || true)
        USED_QOS="bio_ai"; USED_WALL="${BWALL}"
        [ -n "${JID}" ] && BIO_AI_USED=$((BIO_AI_USED+1))
    fi
    if [ -z "${JID}" ]; then
        printf '    → DEFERRED (all QOSes full / budgets exhausted)\n'
        n_deferred=$((${n_deferred:-0}+1))
        continue
    fi
    printf '    → JID=%s [%s, %s]\n' "${JID}" "${USED_QOS}" "${USED_WALL}"
    n_submitted=$((n_submitted+1))
done

rm -rf "${TMPDIR_SCRIPTS}"
echo
echo "Submitted: ${n_submitted}  |  skipped (JSON exists): ${n_skip_done}  |  skipped (in queue): ${n_skip_queued}  |  deferred (cap): ${n_deferred:-0}"
