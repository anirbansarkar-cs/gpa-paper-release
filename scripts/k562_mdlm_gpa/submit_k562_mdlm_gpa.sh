#!/bin/bash
# ===========================================================================
# Submit initial K562 LentiMPRA GPA+DPS jobs with MDLM backbone
# ===========================================================================
#
# 4 configurations:
#   1. baseline_nodps:    GPA-only, no DPS, random init
#   2. mean_dps_gc4555:   DPS eta=3000, hard GC [0.45,0.55] (v9-style)
#   3. mean_dps_gc4851:   DPS eta=3000, hard GC [0.48,0.51] (v10-style)
#   4. mean_dps_noGC:     DPS eta=3000, no GC constraint (pure oracle)

set -euo pipefail

cd "$(dirname "$0")/../.."
GPA_SCRIPT="scripts/k562_mdlm_gpa/run_k562_mdlm_gpa.sh"

# ---- Validate files ----
SGDD_DIR="${HOME}/SGDD"
MDLM_CKPT="${SGDD_DIR}/applications/drakes_dna/data_and_model/mdlm/outputs_lentimpra/2026.03.09/150346/checkpoints/best.ckpt"
ORACLE_CKPT="${GPA_REPO_ROOT}/model_zoo/lentimpra/oracle_models/best_model-epoch=24-val_pearson=0.814.ckpt"

if [[ ! -f "$MDLM_CKPT" ]]; then
    echo "ERROR: MDLM checkpoint not found: $MDLM_CKPT"
    exit 1
fi
if [[ ! -f "$ORACLE_CKPT" ]]; then
    echo "ERROR: Oracle checkpoint not found: $ORACLE_CKPT"
    exit 1
fi

echo "MDLM:   $MDLM_CKPT"
echo "Oracle: $ORACLE_CKPT"
echo ""

# ---- Common environment ----
COMMON_ENV=(
    SGDD_DIR="$SGDD_DIR"
    MDLM_CKPT="$MDLM_CKPT"
    ORACLE_CKPT="$ORACLE_CKPT"
    POPULATION_SIZE=5000
    MAX_BETA=200.0
    NOISE_FRACTION=0.05
    FROM_RANDOM=true
    HARD_MAX_STEPS=200
    EXTEND_STEPS=10
    EXTEND_ON_DELTA_BETA=0.5
)

# ===========================================================================
# Job 1: baseline_nodps (GPA-only, no DPS, no GC constraint)
# ===========================================================================
RUN_TAG="baseline_nodps"
JID1=$(env "${COMMON_ENV[@]}" \
    USE_DPS=false \
    RUN_TAG="$RUN_TAG" \
    OUTPUT_DIR="results/k562_mdlm_gpa/run_${RUN_TAG}" \
    sbatch --parsable "$GPA_SCRIPT")
echo "  $RUN_TAG: $JID1"

# ===========================================================================
# Job 2: mean_dps_gc4555 (DPS + hard GC [0.45, 0.55])
# ===========================================================================
RUN_TAG="mean_dps_gc4555"
JID2=$(env "${COMMON_ENV[@]}" \
    USE_DPS=true \
    DPS_ETA=3000.0 \
    BIO_FILTER=1 \
    GC_LOW=0.45 \
    GC_HIGH=0.55 \
    GC_FITNESS_WEIGHT=10.0 \
    GC_FITNESS_LOW=0.45 \
    GC_FITNESS_HIGH=0.55 \
    RUN_TAG="$RUN_TAG" \
    OUTPUT_DIR="results/k562_mdlm_gpa/run_${RUN_TAG}" \
    sbatch --parsable "$GPA_SCRIPT")
echo "  $RUN_TAG: $JID2"

# ===========================================================================
# Job 3: mean_dps_gc4851 (DPS + tight GC [0.48, 0.51])
# ===========================================================================
RUN_TAG="mean_dps_gc4851"
JID3=$(env "${COMMON_ENV[@]}" \
    USE_DPS=true \
    DPS_ETA=3000.0 \
    BIO_FILTER=1 \
    GC_LOW=0.48 \
    GC_HIGH=0.51 \
    GC_FITNESS_WEIGHT=10.0 \
    GC_FITNESS_LOW=0.48 \
    GC_FITNESS_HIGH=0.51 \
    RUN_TAG="$RUN_TAG" \
    OUTPUT_DIR="results/k562_mdlm_gpa/run_${RUN_TAG}" \
    sbatch --parsable "$GPA_SCRIPT")
echo "  $RUN_TAG: $JID3"

# ===========================================================================
# Job 4: mean_dps_noGC (DPS, no GC constraint)
# ===========================================================================
RUN_TAG="mean_dps_noGC"
JID4=$(env "${COMMON_ENV[@]}" \
    USE_DPS=true \
    DPS_ETA=3000.0 \
    RUN_TAG="$RUN_TAG" \
    OUTPUT_DIR="results/k562_mdlm_gpa/run_${RUN_TAG}" \
    sbatch --parsable "$GPA_SCRIPT")
echo "  $RUN_TAG: $JID4"

echo ""
echo "Submitted 4 jobs: $JID1 $JID2 $JID3 $JID4"
