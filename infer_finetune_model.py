"""
infer_finetune_model.py
Run inference on the CIFAR-10 test set using the best fine-tuned model checkpoint.
Prints test accuracy and inference time; saves results/finetune_model_inference.json.
"""

import os
import time
import json
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torchvision.models import resnet18, ResNet18_Weights

if __name__ == '__main__':
    # -------------------------------------------------------------------------
    # Device
    # -------------------------------------------------------------------------

    device = torch.device("mps" if torch.backends.mps.is_available()
                           else "cuda" if torch.cuda.is_available()
                           else "cpu")
    print(f"Using device: {device}")

    # -------------------------------------------------------------------------
    # Data — test split only
    # -------------------------------------------------------------------------

    MEAN = (0.4914, 0.4822, 0.4465)
    STD  = (0.2023, 0.1994, 0.2010)

    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ])

    test_dataset = torchvision.datasets.CIFAR10(root="./data", train=False,
                                                download=True, transform=test_transform)
    test_loader  = torch.utils.data.DataLoader(test_dataset, batch_size=128,
                                               shuffle=False, num_workers=0)

    # -------------------------------------------------------------------------
    # Build patched model — same 3 patches as train_finetune_model.py, same order
    # -------------------------------------------------------------------------

    model = resnet18(weights=ResNet18_Weights.DEFAULT)

    # Step A: replace conv1 to accept 32×32 CIFAR-10 input (3×3 kernel, stride 1)
    model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)

    # Step B: remove maxpool to avoid spatial collapse on small feature maps
    model.maxpool = nn.Identity()

    # Step C: replace final classifier for 10 CIFAR-10 classes
    model.fc = nn.Linear(512, 10)

    # -------------------------------------------------------------------------
    # Load weights
    # -------------------------------------------------------------------------

    model.load_state_dict(torch.load("checkpoints/best_finetune_model.pth",
                                     map_location=device))
    model.to(device)
    model.eval()

    # -------------------------------------------------------------------------
    # Inference
    # -------------------------------------------------------------------------

    correct = 0
    total   = 0

    start = time.time()
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs        = model(images)
            _, predicted   = outputs.max(1)
            correct       += predicted.eq(labels).sum().item()
            total         += images.size(0)
    elapsed = time.time() - start

    test_accuracy  = 100.0 * correct / total
    inference_time = round(elapsed, 2)

    print(f"Test Accuracy: {test_accuracy:.2f}% | Inference Time: {inference_time:.2f}s")

    # -------------------------------------------------------------------------
    # Save results
    # -------------------------------------------------------------------------

    os.makedirs("results", exist_ok=True)
    results = {
        "test_accuracy":      round(test_accuracy, 4),
        "inference_time_sec": inference_time,
    }
    with open("results/finetune_model_inference.json", "w") as f:
        json.dump(results, f, indent=2)

    print("Results saved to results/finetune_model_inference.json")
