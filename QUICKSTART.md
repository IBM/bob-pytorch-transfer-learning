# QUICKSTART

Step-by-step guide to set up and run the PyTorch Transfer Learning demo.

---

## 1. Create & activate the virtual environment

```bash
python3 -m venv venv
source venv/bin/activate    # macOS / Linux
# venv\Scripts\activate     # Windows
```

## 2. Verify and install dependencies

> **Before installing**, open `requirements.txt` and confirm the listed libraries are correct.
> If you added or changed any libraries, update `requirements.txt` first, then run:

```bash
pip install -r requirements.txt
```

---

## 3. Run the scripts in order

### Step 1 — Train the base model
Trains ResNet-18 from scratch on CIFAR-10 for 3 epochs and saves checkpoints.
```bash
python train_base_model.py
```

### Step 2 — Infer with the base model
Runs test-set inference and writes `results/base_model_inference.json`.
```bash
python infer_base_model.py
```

### Step 3 — Fine-tune the pretrained model
Loads torchvision ResNet-18 (ImageNet weights), adapts it for 32×32 input, and fine-tunes on CIFAR-10.
```bash
python train_finetune_model.py
```

> **Optional — freeze the backbone:**
> ```bash
> python train_finetune_model.py --freeze-backbone
> ```
> Freezes all layers except the final `fc` layer. Default is fully unfrozen.

### Step 4 — Infer with the fine-tuned model
Runs test-set inference and writes `results/finetune_model_inference.json`.
```bash
python infer_finetune_model.py
```

### Step 5 — Compare both models
Reads both result JSONs, prints a summary table, and saves two bar charts in `results/`.
```bash
python compare_models.py
```

---

## 4. Outputs

| Location | Content |
|---|---|
| `checkpoints/` | Saved model weights (`best_*.pth`, `final_*.pth`) |
| `results/training_accuracy_comparison.png` | Bar chart: training accuracy, base vs fine-tuned |
| `results/inference_accuracy_comparison.png` | Bar chart: inference accuracy, base vs fine-tuned |
| `results/*.json` | Per-model test accuracy and inference time |

> **Note**: All scripts auto-detect the best available device: MPS → CUDA → CPU.
