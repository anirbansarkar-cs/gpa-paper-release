#!/bin/bash
# =============================================================================
# Submit AlphaGenome scoring for all v24 and v25 GPA runs
# =============================================================================

set -euo pipefail
cd "$(dirname "$0")/../.."

SCORE_SCRIPT="scripts/alphagenome/score_gpa.sh"
RESULTS_BASE="results/rerd_comparison"
OUT_BASE="results/alphagenome"

mkdir -p "$OUT_BASE"
mkdir -p sbatch_out/alphagenome

count=0
for h5 in ${RESULTS_BASE}/run_v24*/gpa_output.h5 ${RESULTS_BASE}/run_v25*/gpa_output.h5; do
    run=$(basename "$(dirname "$h5")")

    # Skip pareto variants (subsets of main runs)
    if [[ "$run" == *_pareto* ]]; then continue; fi

    out_csv="${OUT_BASE}/${run}_scores.csv"

    # Skip if already scored
    if [[ -f "$out_csv" ]]; then
        echo "SKIP (exists): $run"
        continue
    fi

    echo -n "Submitting: $run ... "
    JID=$(
        INPUT_H5="$h5" \
        OUTPUT_CSV="$out_csv" \
        APPEND_H5=1 \
        sbatch --parsable "$SCORE_SCRIPT")
    echo "Job $JID"
    count=$((count + 1))
done

echo ""
echo "Submitted $count jobs. Monitor with: squeue -u \$USER"
echo "Results will be in: $OUT_BASE/"
