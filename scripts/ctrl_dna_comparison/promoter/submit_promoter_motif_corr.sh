#!/bin/bash
# Build per-cell promoter FIMO reference cache (50/50 percentile rule, paper §4.1),
# then score all GPA + CtrlDNA promoter pools the supervisor doc references.
# Cpuq, fast QOS, parallel.
set -euo pipefail

PROJECT_DIR="${GPA_REPO_ROOT}"
cd "$PROJECT_DIR"

LOGDIR="sbatch_out/ctrl_dna_comparison/promoter"
OUTDIR="results/ctrl_dna_comparison/promoter/motif_corr_paper"
mkdir -p "$LOGDIR" "$OUTDIR"

CONDA="source ${CONDA_BASE}/etc/profile.d/conda.sh && conda activate d3_cuda118"
SCRIPT="scripts/ctrl_dna_comparison/promoter/score_motif_corr_promoter.py"

# ── Step 1: build all 3 reference caches in one job (FIMO is the slow step,
# ~3-5 min per cell × 3 cells = ~10-15 min) ─────────────────────────────────
BUILD_JID=$(sbatch --parsable \
    --job-name="motif_paper_buildref" \
    --output="$LOGDIR/motif_paper_buildref_%j.out" \
    --error="$LOGDIR/motif_paper_buildref_%j.err" \
    --time=00:30:00 --ntasks=1 --cpus-per-task=4 --mem-per-cpu=6G \
    --partition=cpuq --qos=fast \
    --wrap "$CONDA && set -euo pipefail && cd $PROJECT_DIR && export PYTHONUNBUFFERED=1 && python -u $SCRIPT build-ref")
echo "Ref-cache build: $BUILD_JID"

# ── Step 2: score all GPA + CtrlDNA pools (depends on cache) ────────────────
GPA_RECIPES=(
    nodpsA_K8_div1p0           # universal
    argmax_K8_nodpsA           # K=8 baseline (K562/THP1)
    argmax_K8_nodpsA_JK        # K=8 baseline (JURKAT)
    nodpsA_K8_div0p3 nodpsA_K8_div0p5 nodpsA_K8_div1p5
    nodpsA_K8_div2p0 nodpsA_K8_div3p0 nodpsA_K8_div10p0
    univ_K1 univ_K4
    univ_K8_substeps2 univ_K8_substeps3 univ_K8_substeps4
    argmax_K8_POP20k argmax_K8
)

submit_score_job() {
    local label="$1"
    local pool_dir="$2"
    local cell="$3"
    local out_json="$4"
    local extra_args="${5:-}"
    sbatch --dependency=afterok:$BUILD_JID \
        --job-name="motif_paper_$label" \
        --output="$LOGDIR/motif_paper_${label}_%j.out" \
        --error="$LOGDIR/motif_paper_${label}_%j.err" \
        --time=00:15:00 --ntasks=1 --cpus-per-task=2 --mem-per-cpu=6G \
        --partition=cpuq --qos=fast \
        --wrap "$CONDA && set -euo pipefail && cd $PROJECT_DIR && export PYTHONUNBUFFERED=1 && python -u $SCRIPT score-pool --pool_dir '$pool_dir' --target_cell '$cell' --output_json '$out_json' $extra_args"
}

n_gpa=0
for recipe in "${GPA_RECIPES[@]}"; do
    for cell in JURKAT K562 THP1; do
        for seed in 0 1 2 3 4; do
            d="results/ctrl_dna_comparison/promoter/gpa_full_${cell}_nodps_seed${seed}_${recipe}"
            # Some recipes are dps not nodps (e.g. argmax_K8_*)
            if [[ ! -d "$d" ]]; then
                d="results/ctrl_dna_comparison/promoter/gpa_full_${cell}_dps_seed${seed}_${recipe}"
            fi
            if [[ ! -d "$d" ]]; then continue; fi
            out="$OUTDIR/gpa_${cell}_seed${seed}_${recipe}.json"
            if [[ -f "$out" ]]; then continue; fi
            submit_score_job "gpa_${cell}_s${seed}_${recipe}" "$d" "$cell" "$out" "--method gpa"
            n_gpa=$((n_gpa+1))
        done
    done
done
echo "GPA scoring: $n_gpa jobs submitted"

n_ctrl=0
for cell in JURKAT K562 THP1; do
    for seed in 0 1 2 3 4; do
        d="results/ctrl_dna_comparison/promoter/ctrldna_full_${cell}_seed${seed}_e15"
        out="$OUTDIR/ctrldna_${cell}_seed${seed}.json"
        if [[ ! -d "$d" ]]; then continue; fi
        if [[ -f "$out" ]]; then continue; fi
        submit_score_job "ctrldna_${cell}_s${seed}" "$d" "$cell" "$out" "--method ctrldna --seed $seed"
        n_ctrl=$((n_ctrl+1))
    done
done
echo "CtrlDNA scoring: $n_ctrl jobs submitted"
echo "Total chained on dependency afterok:$BUILD_JID"
