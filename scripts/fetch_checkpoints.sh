#!/bin/bash
# Download the checkpoints we trained from the GitHub release and put them where
# the run scripts look for them.
#
#   export GPA_DATA_ROOT=/path/to/datasets
#   bash scripts/fetch_checkpoints.sh
#
# Needs the GitHub CLI (`gh auth login`), or set USE_CURL=1 to fetch over plain
# HTTPS instead. Override RELEASE_TAG to pin an older release.
#
# Not included: the AlphaGenome-derived encoder used as the held-out evaluator.
# Its loaders depend on packages that are not part of this release, so the
# checkpoint alone would not load — see the README.
set -euo pipefail
cd "$(dirname "$0")/.."

REPO="${REPO:-anirbansarkar-cs/gpa-paper-release}"
RELEASE_TAG="${RELEASE_TAG:-checkpoints-v1}"
: "${GPA_DATA_ROOT:?set GPA_DATA_ROOT (where the non-repo checkpoints should live)}"

PROMOTER_CKPT="scripts/ctrl_dna_comparison/promoter/checkpoints"
mkdir -p "${PROMOTER_CKPT}/hyenadna_promoter_e3" \
         "${GPA_DATA_ROOT}/lentimpra" \
         "${GPA_DATA_ROOT}/enformer_oracles"

# asset name -> destination path
declare -A DEST=(
  [gpa_promoter_oracle_JURKAT.ckpt]="${PROMOTER_CKPT}/human_paired_jurkat.ckpt"
  [gpa_promoter_oracle_K562.ckpt]="${PROMOTER_CKPT}/human_paired_k562.ckpt"
  [gpa_promoter_oracle_THP1.ckpt]="${PROMOTER_CKPT}/human_paired_THP1.ckpt"
  [gpa_hyenadna_promoter_stage1.ckpt]="${PROMOTER_CKPT}/hyenadna_promoter_e3/best.ckpt"
  [gpa_legnet_k562.ckpt]="${GPA_DATA_ROOT}/lentimpra/legnet_k562.ckpt"
)
for CELL in hepg2 k562 sknsh; do
  for HALF in design eval; do
    DEST[gpa_enformer_${CELL}_${HALF}.ckpt]="${GPA_DATA_ROOT}/enformer_oracles/enformer_${CELL}_${HALF}/best.ckpt"
  done
done

TMP="$(mktemp -d)"
trap 'rm -rf "${TMP}"' EXIT

for ASSET in "${!DEST[@]}"; do
  OUT="${DEST[$ASSET]}"
  if [ -f "${OUT}" ]; then
    echo "  [skip] ${OUT} already present"
    continue
  fi
  echo "  [get ] ${ASSET}"
  mkdir -p "$(dirname "${OUT}")"
  if [ "${USE_CURL:-0}" = "1" ]; then
    curl -fL --retry 3 -o "${TMP}/${ASSET}" \
      "https://github.com/${REPO}/releases/download/${RELEASE_TAG}/${ASSET}"
  else
    gh release download "${RELEASE_TAG}" --repo "${REPO}" \
      --pattern "${ASSET}" --dir "${TMP}" --clobber
  fi
  mv "${TMP}/${ASSET}" "${OUT}"
done

echo
echo "Done. Promoter checkpoints are in ${PROMOTER_CKPT}/;"
echo "the rest are under ${GPA_DATA_ROOT}/."
