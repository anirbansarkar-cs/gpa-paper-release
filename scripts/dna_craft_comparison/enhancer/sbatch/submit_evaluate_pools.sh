#!/bin/bash
#SBATCH --job-name=dnacraft_eval
#SBATCH --partition=gpuq
#SBATCH --qos=fast
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=03:45:00
#SBATCH --output=${GPA_REPO_ROOT}/sbatch_out/dna_craft_comparison/eval_%j.out
#SBATCH --error=${GPA_REPO_ROOT}/sbatch_out/dna_craft_comparison/eval_%j.err
#SBATCH --exclude=gpunode29

set -euo pipefail

REPO="${GPA_REPO_ROOT}"
EVAL_DIR="${REPO}/results/dna_craft_comparison/enhancer/eval"
SCRIPTS="${REPO}/scripts/dna_craft_comparison/enhancer"
PYBIN="${HOME}/.conda/envs/d3_cuda118/bin/python"

mkdir -p "${REPO}/sbatch_out/dna_craft_comparison" "${EVAL_DIR}"

echo "[eval] start: $(date)"
echo "[eval] node: $(hostname)"
echo "[eval] gpu:  $(nvidia-smi --query-gpu=name --format=csv,noheader | tr '\n' ',')"
echo "[eval] python: ${PYBIN}"

# Phase 1: build anchors + kmer refs + top_indices_{cell}.npy.
# motif_corr will be NaN here because motif_ref_*.npy doesn't exist yet.
echo "[eval] === Phase 1: anchors + kmer refs + top-99.9% indices ==="
"${PYBIN}" "${SCRIPTS}/evaluate_dnacraft_pools.py"

# Phase 2: build per-cell FIMO motif references from the dumped indices.
# Skip cells whose motif_ref_{cell}.npy already exists (idempotent).
echo "[eval] === Phase 2: FIMO motif references ==="
for cell in hepg2 k562 sknsh; do
    refs="${EVAL_DIR}/top_indices_${cell}.npy"
    out="${EVAL_DIR}/motif_ref_${cell}.npy"
    if [[ ! -f "${refs}" ]]; then
        echo "[eval]   missing ${refs} — phase 1 did not produce it; aborting"
        exit 1
    fi
    if [[ -f "${out}" ]]; then
        echo "[eval]   ${cell}: motif_ref already cached"
        continue
    fi
    "${PYBIN}" "${SCRIPTS}/fimo_motif_corr.py" \
        --mode build_ref --cell "${cell}" --ref_indices_npy "${refs}"
done

# Phase 3: re-run evaluate (anchors/kmer caches hit, motif_corr now populated).
echo "[eval] === Phase 3: per-run metrics with motif_corr ==="
"${PYBIN}" "${SCRIPTS}/evaluate_dnacraft_pools.py"

echo "[eval] === Phase 4: final markdown table ==="
"${PYBIN}" "${SCRIPTS}/final_table.py"

echo "[eval] done: $(date)"
