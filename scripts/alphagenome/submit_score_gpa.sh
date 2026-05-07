#!/bin/bash
# =============================================================================
# Submit AlphaGenome scoring jobs for key GPA runs + ISM sequences
# =============================================================================

set -euo pipefail
cd "$(dirname "$0")/../.."

SCORE_SCRIPT="scripts/alphagenome/score_gpa.sh"
RESULTS_BASE="results/rerd_comparison"
OUT_BASE="results/alphagenome"

mkdir -p "$OUT_BASE"
mkdir -p sbatch_out/alphagenome

# ---- v9 best-mean run (pw=0, hard4555) --------------------------------------
echo "Submitting: v9 mean (pw=0, hard4555)"
JID_V9_MEAN=$(
    INPUT_H5="${RESULTS_BASE}/run_v9_pw0_smooth01_hard4555/gpa_output.h5" \
    OUTPUT_CSV="${OUT_BASE}/v9_pw0_hard4555_scores.csv" \
    APPEND_H5=1 \
    sbatch --parsable "$SCORE_SCRIPT")
echo "  Job ID: $JID_V9_MEAN"

# ---- v7 best-spec run (pw=0.35, smooth03, beta100) --------------------------
echo "Submitting: v7 spec (pw=0.35, smooth03, beta100)"
JID_V7_SPEC=$(
    INPUT_H5="${RESULTS_BASE}/run_v7_cascade_pw035_beta100/gpa_output.h5" \
    OUTPUT_CSV="${OUT_BASE}/v7_pw035_beta100_scores.csv" \
    APPEND_H5=1 \
    sbatch --parsable "$SCORE_SCRIPT")
echo "  Job ID: $JID_V7_SPEC"

# ---- ISM sequences (CSV input) ----------------------------------------------
ISM_CSV="results/ism_high_oracle_scored.csv"
if [[ -f "$ISM_CSV" ]]; then
    echo "Submitting: ISM sequences"
    JID_ISM=$(
        INPUT_H5="$ISM_CSV" \
        SEQ_COLUMN=sequence \
        OUTPUT_CSV="${OUT_BASE}/ism_scores.csv" \
        sbatch --parsable "$SCORE_SCRIPT")
    echo "  Job ID: $JID_ISM"
else
    echo "Skipping ISM: $ISM_CSV not found"
fi

echo ""
echo "All jobs submitted. Monitor with: squeue -u \$USER"
echo "Results will be in: $OUT_BASE/"
