# LitGPT from-scratch trainer

This optional training component adds [Lightning AI LitGPT](https://github.com/Lightning-AI/litgpt) to NTF as a **real, opt-in trainer**. It uses the Pythia-14M architecture and initializes its weights randomly: the configuration deliberately leaves `initial_checkpoint_dir` empty and disables resume. The preparation script downloads **tokenizer files only**, never model weights.

LitGPT is an Apache-2.0 dependency; this repository's existing MIT license remains unchanged. No LitGPT source is vendored here. The dependency is pinned to LitGPT `0.5.13` for a reproducible CLI baseline; revisit that pin intentionally when upgrading.

> **The checked-in defaults are a tiny pipeline smoke configuration, not a useful language model.** It processes at most 256 training tokens from two short, synthetic text files. Replace the demo corpus and increase the token budget only after choosing licensed, cleaned data and reviewing compute, storage, and validation needs.

## Layout

```text
training/litgpt/
├── configs/pythia-14m-from-scratch.yaml  # random-init architecture + conservative smoke defaults
├── data/sample_text/                    # tiny synthetic validation/train examples
├── requirements.txt                     # pinned LitGPT dependency
├── scripts/prepare_tokenizer.sh         # downloads tokenizer only
└── scripts/train_from_scratch.sh        # starts training only when explicitly invoked
```

## Local quick start

Requires Python 3.10+ (Python 3.12 is a good default). A CPU can exercise this tiny smoke configuration, but it is not a practical target for meaningful pretraining. For a CUDA run, install a PyTorch build compatible with the machine's driver and accelerator before starting the trainer; hardware and precision choices are not universal.

From the repository root:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r training/litgpt/requirements.txt

# Downloads only the tokenizer; does not download weights or train.
./training/litgpt/scripts/prepare_tokenizer.sh

# This is the point that starts an actual training run.
./training/litgpt/scripts/train_from_scratch.sh
```

The run writes a unique output directory under `training/litgpt/out/`; downloaded model/tokenizer files go under `checkpoints/`. Those paths and LitGPT's tokenized `train/` and `val/` caches are ignored by Git.

## Use your own data and budget

`TextFiles` consumes top-level UTF-8 `.txt` files. With `val_data_path` omitted, LitGPT reserves the first alphabetically sorted text file for validation and trains on the remaining files, so provide at least two files. Keep private or licensed corpora outside Git and point the script at them:

```bash
TRAIN_DATA_DIR=/absolute/path/to/clean-text-files \
MAX_TOKENS=1000000 \
./training/litgpt/scripts/train_from_scratch.sh
```

`MAX_TOKENS` defaults to `256` solely for the checked-in smoke example. Increase it deliberately for actual pretraining; review sequence length, global/micro batch sizes, learning-rate schedule, corpus size, checkpoints, and validation for the specific experiment. Set `CHECKPOINT_DIR` to change the checkpoint root for both tokenizer preparation and training; `TOKENIZER_DIR` can point the trainer at tokenizer files in a custom location. Override `OUT_DIR`, `DEVICES`, or `NUM_NODES` only when their paths and distributed runtime are configured intentionally. The trainer refuses to reuse an existing output directory; set a fresh `OUT_DIR` to start a separate run.

The Pythia-14M architecture is a small, convenient starting point—not a claim that the demo data or tiny token budget produces a capable model. For a larger model or meaningful corpus, plan hardware, runtime, checkpoints, evaluation, and repeatability first. See [LitGPT's official pretraining tutorial](https://github.com/Lightning-AI/litgpt/blob/main/tutorials/pretrain.md) and [official repository](https://github.com/Lightning-AI/litgpt).

## Distributed-training and Actions boundary

Synchronous Transformer pretraining needs workers participating in one distributed process group, exchanging gradients (for example, through an all-reduce), and following a coordinated optimizer/step schedule. [PyTorch `DistributedDataParallel`](https://pytorch.org/docs/stable/generated/torch.nn.parallel.DistributedDataParallel.html) synchronizes gradients among its replicas. NTF's **19 independent GitHub Actions matrix jobs do none of that**: each has an isolated workspace/process, and uploaded artifacts are not a gradient collective. They are not a substitute for DDP/FSDP or a synchronized multi-node training cluster; do not average their checkpoints and call it pretraining.

This scaffold launches locally when the trainer script is explicitly invoked. No GitHub Actions workflow installs LitGPT or starts a training run. The existing `ubuntu-latest` NTF demo remains a separate CPU-friendly, manual fan-out/fan-in example. GitHub-hosted standard runners are resource- and plan-limited; GPU/larger runners are a separate offering, concurrency can queue, and each hosted job has a six-hour execution limit. Check [current Actions limits](https://docs.github.com/en/actions/reference/limits) and [runner hardware](https://docs.github.com/en/actions/reference/runners/github-hosted-runners) before assigning any compute-intensive job to Actions.
