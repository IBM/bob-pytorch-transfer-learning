# AGENTS.md

This file provides guidance to agents when working with code in this repository.

## Project Overview

PyTorch demo comparing a **base ResNet-18 model trained from scratch** vs a **fine-tuned torchvision pretrained ResNet-18** on the CIFAR-10 dataset. The goal is demonstrating the benefits of transfer learning.

`compare_models.py` reads both training-stats JSON files and both inference JSON files to produce **two separate bar charts** and a summary table:
- `results/training_accuracy_comparison.png` — training accuracy, base vs fine-tuned
- `results/inference_accuracy_comparison.png` — inference (test) accuracy, base vs fine-tuned
- Each chart annotates: absolute accuracy per model, **absolute gain (pp)**, and **relative gain (%)**.
- `results/model_comparison.json` — combined stats including gains for both metrics.

## Stack

- **Language**: Python 3 (venv)
- **ML Framework**: PyTorch + torchvision
- **Dataset**: CIFAR-10 (32×32 images)
- **Visualization**: matplotlib / seaborn (bar charts, accuracy curves)
- **Device priority**: MPS → CUDA → CPU (always check in this order)

## File Structure

```
.
├── project_blueprint.txt          # Source requirements
├── requirements.txt               # pip dependencies
├── venv/                          # Virtual environment (do not commit)
├── train_base_model.py            # Build, train, test ResNet-18 from scratch
├── infer_base_model.py            # Base model inferencing
├── train_finetune_model.py        # Fine-tune pretrained torchvision ResNet-18
├── infer_finetune_model.py        # Fine-tuned model inferencing
├── compare_models.py              # Side-by-side performance comparison
├── checkpoints/                   # Saved model checkpoints (best + final)
├── results/                       # Saved inference results & comparison outputs
├── QUICKSTART.md                  # Step-by-step install & run guide
└── README.md                      # Demo overview
```

## Commands

```bash
# 1. Create venv (do NOT install yet — verify requirements.txt first)
python3 -m venv venv
source venv/bin/activate    # macOS/Linux

# 2. Verify requirements.txt, then install manually
pip install -r requirements.txt

# 3. Run scripts in order
python train_base_model.py
python infer_base_model.py
python train_finetune_model.py
python infer_finetune_model.py
python compare_models.py
```

## Critical Non-Obvious Rules

1. **Never auto-install packages** — always create `requirements.txt` and stop; the user must verify contents and run `pip install` manually.
2. **First conv layer must be patched** in the fine-tune script — torchvision ResNet-18 expects 224×224 (ImageNet); CIFAR-10 is 32×32, so replace `model.conv1` with a `3×3` kernel, `stride=1`, `padding=1` before loading weights.
3. **Unfreeze all layers by default** in `train_finetune_model.py`; expose a `--freeze-backbone` CLI flag to optionally freeze them — do not make freezing the default.
4. **Epochs = 3** for all training scripts (saves demo time); do not increase without explicit instruction.
5. **Checkpoints**: save both `best_<model>.pth` (best val accuracy) and `final_<model>.pth` at end of training into `checkpoints/`.
6. **Training stats JSON** (`results/base_model_train_stats.json`, `results/finetune_model_train_stats.json`) must include `train_accuracy` (last-epoch training accuracy) so `compare_models.py` can build the training chart.
7. **Comparison charts**: `compare_models.py` produces **two separate bar charts** — one for training accuracy, one for inference accuracy. Each chart has a gain annotation band showing absolute gain (percentage-point) and relative gain. Do **not** combine into a single dual-subplot chart. Do **not** add inference time as a chart subplot.
8. **Inference and comparison outputs** must be written to `results/` (CSV or JSON for stats, PNG for charts).
9. **Optimizer & learning rate**: use `torch.optim.SGD(momentum=0.9, weight_decay=5e-4)` in both training scripts with **different LRs per script**:
   - `train_base_model.py`: `lr=0.1` (standard from-scratch CIFAR-10 rate) + `CosineAnnealingLR(T_max=EPOCHS)`
   - `train_finetune_model.py`: `lr=0.01` (lower to protect pretrained weights) + `CosineAnnealingLR(T_max=EPOCHS)`
   Call `scheduler.step()` once per epoch, after the epoch print. Do **not** use Adam. Do **not** add warmup, AMP, or gradient accumulation.
10. **`num_workers=0` in all DataLoaders** — MPS on Apple Silicon triggers a PyTorch bug (`torch.Stream.is_capturing` undefined) with `num_workers > 0`. Set `num_workers=0` in all four scripts (`train_base_model.py`, `train_finetune_model.py`, `infer_base_model.py`, `infer_finetune_model.py`).
11. **Verify syntax** of all `.py` files inside the activated venv at the end: `python -m py_compile <file>.py`.
