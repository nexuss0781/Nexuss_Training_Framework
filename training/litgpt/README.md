# Legacy LitGPT standalone smoke example

This directory is retained from the original `Training` branch as a separate optional example. It is **not used by the v2.0.0-candidate FedAvg workflow** in `.github/workflows/ntf.yml`; the current shared model architecture and iterative coordinator live in `training/fedavg/`.

The pinned LitGPT 0.5.13 scaffold starts a small Pythia-14M model from random initialization and is not a pretrained checkpoint, federated implementation, or useful trained model. Its 256-token budget exists only as a tiny smoke example. It does not communicate between workers, synchronize rounds, or aggregate model updates.

Do not use the legacy `train_from_scratch.sh` script to test the FedAvg pipeline. That command starts real standalone model training. The v2.0.0-candidate test suite uses only synthetic tensors and never invokes it.
