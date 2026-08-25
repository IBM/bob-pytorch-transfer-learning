# PyTorch Transfer Learning Demo — Base vs Fine-Tuned ResNet-18

Demonstrates the benefit of transfer learning by training two models on the CIFAR-10 dataset and comparing their
accuracy — built end-to-end using IBM Bob, IBM's agentic AI coding assistant.

---

## What the demo shows

| Model | Description |
|---|---|
| **Base model** | ResNet-18 built from scratch, trained on CIFAR-10 for 3 epochs with no pretrained weights |
| **Fine-tuned model** | torchvision ResNet-18 (ImageNet pretrained), adapted for 32×32 input and fine-tuned on CIFAR-10 for 3 epochs |

The fine-tuned model typically reaches higher accuracy in the same number of epochs, illustrating why pretrained weights matter.

---

## Dataset

**CIFAR-10** — 60,000 colour images across 10 classes (airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck).  
Image size: 32×32 pixels. Split: 50,000 train / 10,000 test.

---

## Architecture

**Base model** — ResNet-18 implemented with `nn.Module` from scratch. The final fully-connected layer outputs 10 classes.

**Fine-tuned model** — torchvision `resnet18(weights=ResNet18_Weights.DEFAULT)` with three adaptations for 32×32 input:
- `conv1` replaced with `Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)`
- `maxpool` replaced with `nn.Identity()` (avoids spatial collapse)
- `fc` replaced with `Linear(512, 10)`

Both models train with `CrossEntropyLoss` + `SGD` (momentum=0.9, weight_decay=5e-4) + `CosineAnnealingLR`, 3 epochs.

---

## Scripts

Run them in this order:

| # | Script | What it does |
|---|---|---|
| 1 | `train_base_model.py` | Trains ResNet-18 from scratch on CIFAR-10; saves checkpoints |
| 2 | `infer_base_model.py` | Runs test-set inference with the base model; writes JSON results |
| 3 | `train_finetune_model.py` | Fine-tunes pretrained ResNet-18 on CIFAR-10; saves checkpoints |
| 4 | `infer_finetune_model.py` | Runs test-set inference with the fine-tuned model; writes JSON results |
| 5 | `compare_models.py` | Reads both result JSONs; prints a summary and saves a bar chart |

### Optional flag

```bash
python train_finetune_model.py --freeze-backbone
```

Freezes all layers except the final `fc` layer. Default is fully unfrozen (all layers trainable).

---

## Outputs

| Location | Content |
|---|---|
| `checkpoints/best_base_model.pth` | Best base model weights (by val accuracy) |
| `checkpoints/final_base_model.pth` | Base model weights after the last epoch |
| `checkpoints/best_finetune_model.pth` | Best fine-tuned model weights (by val accuracy) |
| `checkpoints/final_finetune_model.pth` | Fine-tuned model weights after the last epoch |
| `results/base_model_inference.json` | Base model test accuracy + inference time |
| `results/finetune_model_inference.json` | Fine-tuned model test accuracy + inference time |
| `results/training_accuracy_comparison.png` | Bar chart: training accuracy, base vs fine-tuned |
| `results/inference_accuracy_comparison.png` | Bar chart: inference accuracy, base vs fine-tuned |

---

## Quick start

See [`QUICKSTART.md`](QUICKSTART.md) for the full step-by-step setup and run instructions.

> All scripts auto-select the best available device: **MPS → CUDA → CPU**.

---

## About IBM Bob

IBM Bob is an agentic AI assistant that goes beyond traditional chatbots by:

- **Taking autonomous actions** to complete tasks
- **Using tools** to read, write, and modify files
- **Breaking down complex problems** into executable steps
- **Iterating and refining** based on results and feedback

Bob represents the future of AI-assisted development and productivity.

---

## Acknowledgements & Licenses

| Component | Source | License |
|---|---|---|
| **CIFAR-10 dataset** | Krizhevsky, A., Nair, V., & Hinton, G. (2009). [*Learning Multiple Layers of Features from Tiny Images*](https://www.cs.toronto.edu/~kriz/learning-features-2009-TR.pdf). University of Toronto. | MIT |
| **ResNet-18 pretrained weights** | [pytorch/vision](https://github.com/pytorch/vision) — ImageNet-1K weights distributed by torchvision | BSD 3-Clause |
| **PyTorch & torchvision** | [pytorch/pytorch](https://github.com/pytorch/pytorch) / [pytorch/vision](https://github.com/pytorch/vision) | BSD 3-Clause |

All third-party components used in this demo are permissively licensed and free for research, educational, and commercial use. No modifications were made to the pretrained weights or the dataset.
