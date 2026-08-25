# PyTorch Demo Implementation Plan

## Overview

Build a live-demo project that shows three things side-by-side on the CIFAR-10 dataset:
1. Training a ResNet-18 architecture **from scratch** (base model)
2. **Fine-tuning** torchvision's pretrained ResNet-18 (ImageNet weights, adapted for 32×32 input)
3. A **comparison** of both models' accuracy and inference time via visualizations

The project is intentionally minimal — readable by a live audience, 3 epochs, no advanced optimizations. Each script is standalone and run in sequence. All output is persisted to `checkpoints/` and `results/`.

---

## Sub-Task 1 — Project Scaffolding & Environment Setup

**Status**: `[x] done`

### Intent
Create the physical directory structure, virtual environment, and `requirements.txt`. The venv is created here so it is available for Sub-Task 8's syntax check, but **package installation is not automated** — the user must verify `requirements.txt` and run `pip install` manually.

### Expected Outcomes
- `venv/` is created and activates cleanly (packages not yet installed — user installs manually)
- `requirements.txt` lists all required packages
- Empty `checkpoints/` and `results/` directories exist
- No Python scripts yet — just the shell of the project

### Todo List
1. Create `checkpoints/` and `results/` directories (add `.gitkeep` in each)
2. Create `requirements.txt` with:
   - `torch` (latest stable, CPU/MPS/CUDA universal wheel)
   - `torchvision`
   - `matplotlib`
   - `numpy`
3. Create the venv: `python3 -m venv venv`
4. **Stop here** — do NOT run `pip install`. Instruct the user to verify `requirements.txt` and install manually: `source venv/bin/activate && pip install -r requirements.txt`
5. Document venv setup in a stub `QUICKSTART.md` (full content added in Sub-Task 7)

### Relevant Context
- Do NOT include `torchaudio` or other unused packages
- `venv/` must not be committed — add it to `.gitignore`
- **Never auto-install** — blueprint explicitly requires the user to verify and install manually
- Sub-Task 8 assumes the user has already run `pip install` before syntax checking

---

## Sub-Task 2 — Base Model: Train & Test (`train_base_model.py`)

**Status**: `[x] done`

### Intent
Implement the ResNet-18 architecture from scratch in PyTorch and train it on CIFAR-10. This script is the "before" side of the demo — showing what it takes to train a model with no pretrained knowledge.

### Expected Outcomes
- `train_base_model.py` runs end-to-end without error
- Prints per-epoch: epoch number, train loss, train accuracy, val accuracy, elapsed time (one line per epoch)
- Saves `checkpoints/best_base_model.pth` whenever val accuracy improves
- Saves `checkpoints/final_base_model.pth` unconditionally at the last epoch
- Total training completes in 3 epochs

### Todo List
1. Implement `ResNet18` class from scratch using `nn.Module` (BasicBlock + layer stack matching torchvision architecture)
2. Load CIFAR-10 via `torchvision.datasets.CIFAR10`; apply standard normalization `mean=(0.4914, 0.4822, 0.4465), std=(0.2023, 0.1994, 0.2010)`
3. Define device selection: MPS → CUDA → CPU
4. Set up `CrossEntropyLoss` + `Adam` optimizer; no LR scheduler
5. Training loop: 3 epochs; track best val accuracy; save both checkpoints
6. Print per-epoch stats: epoch, train loss, train acc, val acc, elapsed seconds
7. Save final training stats (accuracy, total time) to `results/base_model_train_stats.json`

### Relevant Context
- Architecture must match torchvision's ResNet-18 layer structure (so that fine-tuning script can load weights into the same shape)
- No dropout, no weight decay unless clearly needed to run — keep it minimal
- CIFAR-10 has 10 classes; the final `fc` layer must output 10

---

## Sub-Task 3 — Base Model: Inference (`infer_base_model.py`)

**Status**: `[x] done`

### Intent
Run inference on the CIFAR-10 test set using the saved base model checkpoint. Print and persist accuracy + timing so the comparison script can read them later.

### Expected Outcomes
- Loads `checkpoints/best_base_model.pth`
- Prints test accuracy and total inference time to stdout
- Writes `results/base_model_inference.json` with keys: `test_accuracy`, `inference_time_sec`

### Todo List
1. Load `checkpoints/best_base_model.pth` into the same `ResNet18` class defined in Sub-Task 2 (import or duplicate the class)
2. Load CIFAR-10 **test** split with the same normalization as training
3. Run inference (no gradients); accumulate correct predictions
4. Print: `Test Accuracy: XX.XX% | Inference Time: X.XXs`
5. Write `results/base_model_inference.json`

### Relevant Context
- Must use `torch.no_grad()` context
- Reuse the same normalization constants as `train_base_model.py`
- Inference loads `best_base_model.pth` (not `final_`)

---

## Sub-Task 4 — Fine-Tuned Model: Train (`train_finetune_model.py`)

**Status**: `[x] done`

### Intent
Load torchvision's pretrained ResNet-18, adapt it for CIFAR-10's 32×32 input, and fine-tune on the CIFAR-10 training set. This is the "after" side of the demo.

### Expected Outcomes
- Loads torchvision `resnet18(weights=ResNet18_Weights.DEFAULT)`
- `model.conv1` is replaced **before any forward pass** with `Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)`
- `model.maxpool` is replaced with `nn.Identity()` (avoids spatial collapse on 32×32)
- `model.fc` is replaced with `Linear(512, 10)`
- By default all layers are trainable; `--freeze-backbone` CLI flag freezes all layers except `fc`
- Saves `checkpoints/best_finetune_model.pth` and `checkpoints/final_finetune_model.pth`
- Saves `results/finetune_model_train_stats.json`

### Todo List
1. Load pretrained ResNet-18 via torchvision
2. Patch `model.conv1` → `Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)`
3. Replace `model.maxpool` with `nn.Identity()`
4. Replace `model.fc` with `Linear(512, 10)`
5. Add `argparse` with `--freeze-backbone` flag (default `False`); when `True`: freeze all params, then unfreeze `model.fc.parameters()`
6. Load CIFAR-10 with the same normalization constants
7. Training loop: 3 epochs, same structure as Sub-Task 2
8. Save both checkpoints; save training stats JSON

### Relevant Context
- **Order matters**: patch `conv1` and `maxpool` BEFORE replacing `fc` — the pretrained `fc` still has 1000 outputs; replace it last
- **Freeze order**: freeze ALL params first, then explicitly unfreeze `fc` — reversing this leaves `fc` frozen
- AGENTS.md rule: unfreeze is the default; `--freeze-backbone` is opt-in
- Same normalization constants as base model (CIFAR-10 stats, not ImageNet stats)

---

## Sub-Task 5 — Fine-Tuned Model: Inference (`infer_finetune_model.py`)

**Status**: `[x] done`

### Intent
Run inference on the CIFAR-10 test set using the fine-tuned model. Mirror the structure of `infer_base_model.py` for consistency.

### Expected Outcomes
- Loads `checkpoints/best_finetune_model.pth`
- Prints test accuracy and inference time
- Writes `results/finetune_model_inference.json` with keys: `test_accuracy`, `inference_time_sec`

### Todo List
1. Recreate the patched model architecture (same patches as Sub-Task 4: `conv1`, `maxpool`, `fc`)
2. Load checkpoint weights
3. Run inference on CIFAR-10 test split
4. Print and save results to `results/finetune_model_inference.json`

### Relevant Context
- The model architecture at load time must exactly match the saved checkpoint — apply the same 3 patches before `load_state_dict`
- Loads `best_finetune_model.pth` (not `final_`)

---

## Sub-Task 6 — Model Comparison (`compare_models.py`)

**Status**: `[x] done`

### Intent
Read the persisted inference results from both models and produce a visual bar-chart comparison. This is the final demo payoff — showing the benefit of fine-tuning at a glance.

### Expected Outcomes
- Reads `results/base_model_inference.json` and `results/finetune_model_inference.json`
- Prints a side-by-side summary table to stdout
- Saves `results/model_comparison.png` — a dual bar chart: one for accuracy, one for inference time
- Does NOT re-run any training or inference

### Todo List
1. Load both JSON result files
2. Print comparison table: model name, test accuracy, inference time
3. Create a matplotlib figure with 2 subplots:
   - Subplot 1: bar chart of test accuracy (Base vs Fine-Tuned)
   - Subplot 2: bar chart of inference time (Base vs Fine-Tuned)
4. Save figure to `results/model_comparison.png`
5. Keep chart styling minimal (no seaborn required; plain matplotlib is sufficient)

### Relevant Context
- This script assumes both inference JSONs already exist — it does NOT call the inference scripts
- "Don't overload visualization" — two subplots in one figure is the ceiling
- No model loading happens in this script

---

## Sub-Task 7 — Documentation (`README.md`, `QUICKSTART.md`)

**Status**: `[x] done`

### Intent
Write the two documentation files that a live-demo attendee or first-time user needs to understand and run the project.

### Expected Outcomes
- `README.md` explains what the demo shows, the architecture choices, and the expected results
- `QUICKSTART.md` lists every command in order: venv create, activate, pip install, then each python script in sequence

### Todo List
1. Write `QUICKSTART.md`:
   - venv create + activate
   - `pip install -r requirements.txt`
   - Run scripts in order: `train_base_model.py` → `infer_base_model.py` → `train_finetune_model.py` → `infer_finetune_model.py` → `compare_models.py`
   - Note the `--freeze-backbone` optional flag for `train_finetune_model.py`
2. Write `README.md`:
   - What the demo shows (base vs fine-tuned)
   - Dataset: CIFAR-10, 10 classes, 32×32
   - Architecture: ResNet-18 from scratch vs pretrained ResNet-18 adapted for 32×32
   - Output files produced (checkpoints, results)
   - Where to find the comparison chart

### Relevant Context
- QUICKSTART must show `source venv/bin/activate` before any python command
- README is for demo attendees — keep it brief and outcome-focused

---

## Sub-Task 8 — Syntax Verification

**Status**: `[x] done`

### Intent
Validate all Python scripts **inside the activated venv** to catch any syntax errors before a live demo runs them. Using the venv (with packages installed by the user) ensures the same Python interpreter and installed packages are used for checking as for running.

### Expected Outcomes
- All 5 `.py` scripts pass `python -m py_compile` with no errors inside the venv
- Any errors found are fixed before this sub-task is marked done

### Todo List
1. Confirm the user has activated the venv and installed packages (`pip install -r requirements.txt`) before proceeding
2. Activate the venv: `source venv/bin/activate`
3. Run `python -m py_compile` on all scripts at once:
   ```bash
   python -m py_compile train_base_model.py infer_base_model.py train_finetune_model.py infer_finetune_model.py compare_models.py
   ```
4. Fix any reported syntax errors
5. Re-run until all five pass with exit code 0

### Relevant Context
- The venv was created in Sub-Task 1; packages must be installed by the user before this step runs
- Must activate the venv first — running with the system Python may mask import issues
- This is the final gate before the demo is considered ready

---

## Execution Order

```
Sub-Task 1  →  Sub-Task 2  →  Sub-Task 3
                                  ↓
             Sub-Task 4  →  Sub-Task 5
                                  ↓
                           Sub-Task 6  →  Sub-Task 7  →  Sub-Task 8
```

Sub-Tasks 2–3 (base model) and Sub-Tasks 4–5 (fine-tuned model) can be built in parallel, but Sub-Task 6 depends on both inference scripts being complete.
