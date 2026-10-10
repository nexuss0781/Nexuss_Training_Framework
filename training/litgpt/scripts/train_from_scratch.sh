#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
TRAINING_DIR="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
REPO_ROOT="$(cd -- "${TRAINING_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

if ! command -v litgpt >/dev/null 2>&1; then
  echo "litgpt is not installed. Create/activate a virtual environment and install training/litgpt/requirements.txt first." >&2
  exit 1
fi

TRAIN_DATA_DIR="${TRAIN_DATA_DIR:-${TRAINING_DIR}/data/sample_text}"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-${REPO_ROOT}/checkpoints}"
if [[ "${CHECKPOINT_DIR}" != /* ]]; then CHECKPOINT_DIR="${REPO_ROOT}/${CHECKPOINT_DIR}"; fi
TOKENIZER_DIR="${TOKENIZER_DIR:-${CHECKPOINT_DIR}/EleutherAI/pythia-14m}"
MAX_TOKENS="${MAX_TOKENS:-256}"
DEVICES="${DEVICES:-1}"
NUM_NODES="${NUM_NODES:-1}"
RUN_ID="$(date -u +%Y%m%dT%H%M%SZ)"
OUT_DIR="${OUT_DIR:-${TRAINING_DIR}/out/pythia-14m-${RUN_ID}}"

if [[ ! -d "${TRAIN_DATA_DIR}" ]]; then
  echo "Training text directory does not exist: ${TRAIN_DATA_DIR}" >&2
  exit 1
fi
TEXT_FILE_COUNT="$(find "${TRAIN_DATA_DIR}" -maxdepth 1 -type f -name '*.txt' | wc -l)"
if [[ "${TEXT_FILE_COUNT}" -lt 2 ]]; then
  echo "TextFiles needs at least two top-level .txt files: the first sorted file is held out for validation." >&2
  exit 1
fi
if [[ ! -d "${TOKENIZER_DIR}" ]] || ! find "${TOKENIZER_DIR}" -type f -print -quit | grep -q .; then
  echo "Tokenizer files not found at ${TOKENIZER_DIR}. Run training/litgpt/scripts/prepare_tokenizer.sh first." >&2
  exit 1
fi
if [[ ! "${MAX_TOKENS}" =~ ^[1-9][0-9]*$ ]]; then
  echo "MAX_TOKENS must be a positive integer." >&2
  exit 1
fi
if [[ -e "${OUT_DIR}" ]]; then
  echo "Output path already exists; choose a fresh OUT_DIR to avoid overwriting it: ${OUT_DIR}" >&2
  exit 1
fi

# Resolve relative paths from the repository root, independent of the caller's cwd.
if [[ "${TRAIN_DATA_DIR}" != /* ]]; then TRAIN_DATA_DIR="${REPO_ROOT}/${TRAIN_DATA_DIR}"; fi
if [[ "${TOKENIZER_DIR}" != /* ]]; then TOKENIZER_DIR="${REPO_ROOT}/${TOKENIZER_DIR}"; fi
if [[ "${OUT_DIR}" != /* ]]; then OUT_DIR="${REPO_ROOT}/${OUT_DIR}"; fi

echo "Starting LitGPT pretraining from random initialization."
echo "Model architecture: Pythia-14M; tokenizer: ${TOKENIZER_DIR} (tokenizer only)."
echo "Data: ${TRAIN_DATA_DIR}; max tokens: ${MAX_TOKENS}; output: ${OUT_DIR}"
echo "This command now launches a real training run; this script is never called by GitHub Actions."

exec litgpt pretrain pythia-14m \
  --config "${TRAINING_DIR}/configs/pythia-14m-from-scratch.yaml" \
  --tokenizer_dir "${TOKENIZER_DIR}" \
  --data TextFiles \
  --data.train_data_path "${TRAIN_DATA_DIR}" \
  --out_dir "${OUT_DIR}" \
  --train.max_tokens "${MAX_TOKENS}" \
  --devices "${DEVICES}" \
  --num_nodes "${NUM_NODES}"
