#!/bin/bash
#SBATCH --job-name=ledidi_ag
#SBATCH --output=${GPA_REPO_ROOT}/sbatch_out/ledidi_comparison/ag_%j_stdout.out
#SBATCH --error=${GPA_REPO_ROOT}/sbatch_out/ledidi_comparison/ag_%j_stderr.out
#SBATCH --time=02:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --mem-per-cpu=10G
#SBATCH --gres=gpu:1
#SBATCH --qos=fast
#SBATCH --partition=gpuq
#SBATCH --constraint=h100
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=anonymous@example.com

# Score all LEDIDI outputs through AlphaGenome eval oracle.
# Uses explicit conda env python paths to avoid env mixing issues.

set -euo pipefail

SOCKET="/tmp/ag_oracle_ledidi_${SLURM_JOB_ID}.sock"
AG_SERVER=${GPA_REPO_ROOT}/scripts/k562_mdlm_gpa/ag_oracle_server.py
SCORE_SCRIPT=${GPA_REPO_ROOT}/scripts/ledidi_comparison/score_with_alphagenome.py
RESULTS_BASE=${GPA_REPO_ROOT}/results/ledidi_comparison

AG_PYTHON=${HOME}/.conda/envs/alphagenome/bin/python
D3_PYTHON=${HOME}/.conda/envs/d3_cuda118/bin/python

# ---- Start AG server (alphagenome env) ----
echo "=== Starting AG oracle server ==="
${AG_PYTHON} ${AG_SERVER} --socket ${SOCKET} --batch_size 64 &
AG_PID=$!

# Wait for server ready
echo "Waiting for AG server..."
READY_FILE="${SOCKET}.ready"
for i in $(seq 1 180); do
    if [ -f "${READY_FILE}" ]; then
        echo "  AG server ready (${i}s)"
        break
    fi
    # Check if server died
    if ! kill -0 ${AG_PID} 2>/dev/null; then
        echo "ERROR: AG server process died"
        exit 1
    fi
    sleep 1
done
if [ ! -f "${READY_FILE}" ]; then
    echo "ERROR: AG server not ready after 180s"
    kill ${AG_PID} 2>/dev/null
    exit 1
fi

# ---- Score each run ----
for MODE in ism random_init gpa_seed; do
    INPUT_DIR="${RESULTS_BASE}/${MODE}"
    if [ ! -d "${INPUT_DIR}" ]; then
        echo "Skipping ${MODE}: no results dir"
        continue
    fi

    # Check if any CSV files exist
    if ! ls ${INPUT_DIR}/ledidi_edited_*.csv 1>/dev/null 2>&1; then
        echo "Skipping ${MODE}: no CSV files"
        continue
    fi

    echo ""
    echo "=== Scoring ${MODE} ==="
    ${D3_PYTHON} ${SCORE_SCRIPT} \
        --socket ${SOCKET} \
        --input_dir ${INPUT_DIR} \
        --output ${INPUT_DIR}/ledidi_ag_scores.csv
done

# ---- Shutdown AG server ----
echo ""
echo "=== Shutting down AG server ==="
${D3_PYTHON} -c "
import socket, struct
sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
sock.connect('${SOCKET}')
sock.sendall(struct.pack('<i', 0))
sock.close()
"
wait ${AG_PID} 2>/dev/null || true

echo ""
echo "=== Done ==="
