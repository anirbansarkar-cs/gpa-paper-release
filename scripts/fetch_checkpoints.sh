#!/bin/bash
# Download the checkpoints we trained from the Hugging Face Hub and put them
# where the run scripts look for them.
#
#   export GPA_DATA_ROOT=/path/to/datasets
#   bash scripts/fetch_checkpoints.sh
#
# Needs `huggingface_hub` (in environment.yml). The repo is public, so no token
# is required. Override HF_REPO or HF_REVISION to pin a fork or an older commit.
#
# Not included: our AlphaGenome-derived encoder, the held-out evaluator for the
# cross-oracle experiment. The README lists the public repositories you can use
# to fine-tune an equivalent one.
set -euo pipefail
cd "$(dirname "$0")/.."

HF_REPO="${HF_REPO:-tataiani/gpa-checkpoints}"
HF_REVISION="${HF_REVISION:-main}"
: "${GPA_DATA_ROOT:?set GPA_DATA_ROOT (where the non-repo checkpoints should live)}"

PROMOTER_CKPT="scripts/ctrl_dna_comparison/promoter/checkpoints"
mkdir -p "${PROMOTER_CKPT}/hyenadna_promoter_e3" "${GPA_DATA_ROOT}/lentimpra"

# hub filename -> destination path
FILES=(
  "gpa_promoter_oracle_JURKAT.ckpt|${PROMOTER_CKPT}/human_paired_jurkat.ckpt"
  "gpa_promoter_oracle_K562.ckpt|${PROMOTER_CKPT}/human_paired_k562.ckpt"
  "gpa_promoter_oracle_THP1.ckpt|${PROMOTER_CKPT}/human_paired_THP1.ckpt"
  "gpa_hyenadna_promoter_stage1.ckpt|${PROMOTER_CKPT}/hyenadna_promoter_e3/best.ckpt"
  "gpa_legnet_k562.ckpt|${GPA_DATA_ROOT}/lentimpra/legnet_k562.ckpt"
  "gpa_legnet_k562_config.json|${GPA_DATA_ROOT}/lentimpra/config.json"
)

for ENTRY in "${FILES[@]}"; do
  NAME="${ENTRY%%|*}"
  OUT="${ENTRY##*|}"
  if [ -f "${OUT}" ]; then
    echo "  [skip] ${OUT} already present"
    continue
  fi
  echo "  [get ] ${NAME}"
  mkdir -p "$(dirname "${OUT}")"
  SRC="$(python3 - "$HF_REPO" "$HF_REVISION" "$NAME" <<'PY'
import sys
from huggingface_hub import hf_hub_download
print(hf_hub_download(repo_id=sys.argv[1], revision=sys.argv[2],
                      filename=sys.argv[3], repo_type="model"))
PY
)"
  # hf_hub_download returns a path inside its cache; copy so the cache stays intact.
  cp "${SRC}" "${OUT}"
done

echo
echo "Done. Promoter checkpoints are in ${PROMOTER_CKPT}/;"
echo "LegNet is under ${GPA_DATA_ROOT}/lentimpra/."
echo "Source: https://huggingface.co/${HF_REPO}"
