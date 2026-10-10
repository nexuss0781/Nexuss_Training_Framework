#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../../.." && pwd)"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-${REPO_ROOT}/checkpoints}"
if [[ "${CHECKPOINT_DIR}" != /* ]]; then CHECKPOINT_DIR="${REPO_ROOT}/${CHECKPOINT_DIR}"; fi
TOKENIZER_DIR="${TOKENIZER_DIR:-${CHECKPOINT_DIR}/EleutherAI/pythia-14m}"
if [[ "${TOKENIZER_DIR}" != /* ]]; then TOKENIZER_DIR="${REPO_ROOT}/${TOKENIZER_DIR}"; fi

if ! command -v litgpt >/dev/null 2>&1; then
  echo "litgpt is not installed. Create/activate a virtual environment and install training/litgpt/requirements.txt first." >&2
  exit 1
fi

if [[ -d "${TOKENIZER_DIR}" ]] && find "${TOKENIZER_DIR}" -type f -print -quit | grep -q .; then
  echo "Tokenizer files already exist at ${TOKENIZER_DIR}; nothing to download."
  exit 0
fi

# LitGPT appends the Hugging Face repo ID to checkpoint_dir.
EXPECTED_TOKENIZER_DIR="${CHECKPOINT_DIR}/EleutherAI/pythia-14m"
if [[ "${TOKENIZER_DIR}" != "${EXPECTED_TOKENIZER_DIR}" ]]; then
  echo "The tokenizer download target must be CHECKPOINT_DIR/EleutherAI/pythia-14m." >&2
  echo "Set CHECKPOINT_DIR to the parent directory or place tokenizer files at ${TOKENIZER_DIR} yourself." >&2
  exit 1
fi

cd "${REPO_ROOT}"
echo "Downloading tokenizer files only (no pretrained model weights and no training)."
litgpt download EleutherAI/pythia-14m --tokenizer_only true --checkpoint_dir "${CHECKPOINT_DIR}"

if [[ ! -d "${TOKENIZER_DIR}" ]] || ! find "${TOKENIZER_DIR}" -type f -print -quit | grep -q .; then
  echo "LitGPT did not create tokenizer files at ${TOKENIZER_DIR}. Check the LitGPT download output." >&2
  exit 1
fi

echo "Tokenizer ready at ${TOKENIZER_DIR}."
