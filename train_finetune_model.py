"""
train_finetune_model.py
Load torchvision's pretrained ResNet-18 (ImageNet weights), adapt it for
CIFAR-10's 32×32 input, and fine-tune on CIFAR-10 for 3 epochs.

Patch order (STRICT): conv1 → maxpool → fc
Freeze order (STRICT): freeze all params FIRST, then unfreeze fc

Saves checkpoints/best_finetune_model.pth and checkpoints/final_finetune_model.pth.
Saves results/finetune_model_train_stats.json.
"""

import argparse
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
    # Args
    # -------------------------------------------------------------------------

    parser = argparse.ArgumentParser(description="Fine-tune ResNet-18 on CIFAR-10")
    parser.add_argument(
        "--freeze-backbone",
        action="store_true",
        default=False,
        help="Freeze all layers except the final fc layer (default: all layers trainable)",
    )
    args = parser.parse_args()

    # -------------------------------------------------------------------------
    # Model — load pretrained, then patch in STRICT order: conv1 → maxpool → fc
    # -------------------------------------------------------------------------

    model = resnet18(weights=ResNet18_Weights.DEFAULT)

    # Step A: replace conv1 so 32×32 inputs don't collapse spatially
    model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)

    # Step B: remove maxpool to preserve spatial resolution on 32×32
    model.maxpool = nn.Identity()

    # Step C: replace fc head for 10 CIFAR-10 classes
    model.fc = nn.Linear(512, 10)

    # -------------------------------------------------------------------------
    # Optional backbone freeze — order STRICT: freeze all FIRST, then unfreeze fc
    # -------------------------------------------------------------------------

    if args.freeze_backbone:
        for p in model.parameters():
            p.requires_grad = False
        for p in model.fc.parameters():
            p.requires_grad = True

    # -------------------------------------------------------------------------
    # Device
    # -------------------------------------------------------------------------

    device = torch.device(
        "mps"  if torch.backends.mps.is_available() else
        "cuda" if torch.cuda.is_available()          else
        "cpu"
    )
    print(f"Using device: {device}")
    if args.freeze_backbone:
        print("Backbone frozen — only fc layer will be trained.")

    model = model.to(device)

    # -------------------------------------------------------------------------
    # Data — same CIFAR-10 normalization constants as train_base_model.py
    # -------------------------------------------------------------------------

    MEAN = (0.4914, 0.4822, 0.4465)
    STD  = (0.2023, 0.1994, 0.2010)

    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ])

    val_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ])

    train_dataset = torchvision.datasets.CIFAR10(root="./data", train=True,  download=True, transform=train_transform)
    val_dataset   = torchvision.datasets.CIFAR10(root="./data", train=False, download=True, transform=val_transform)

    train_loader  = torch.utils.data.DataLoader(train_dataset, batch_size=128, shuffle=True,  num_workers=0)
    val_loader    = torch.utils.data.DataLoader(val_dataset,   batch_size=128, shuffle=False, num_workers=0)

    # -------------------------------------------------------------------------
    # Loss / optimizer
    # -------------------------------------------------------------------------

    criterion = nn.CrossEntropyLoss()
    # lr=0.01 is appropriate for fine-tuning — avoids overwriting pretrained weights too aggressively
    optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, model.parameters()),
                                lr=0.01, momentum=0.9, weight_decay=5e-4)

    os.makedirs("checkpoints", exist_ok=True)
    os.makedirs("results",     exist_ok=True)

    # -------------------------------------------------------------------------
    # Training loop — 3 epochs
    # -------------------------------------------------------------------------

    EPOCHS = 3
    # Cosine annealing decays lr smoothly from 0.01 → ~0 over EPOCHS
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
    best_val_acc = 0.0
    total_start  = time.time()

    for epoch in range(1, EPOCHS + 1):
        epoch_start = time.time()

        # --- train ---
        model.train()
        train_loss = 0.0
        correct    = 0
        total      = 0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss    = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * images.size(0)
            _, predicted = outputs.max(1)
            correct += predicted.eq(labels).sum().item()
            total   += images.size(0)

        train_loss /= total
        train_acc   = 100.0 * correct / total

        # --- validate ---
        model.eval()
        val_correct = 0
        val_total   = 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs        = model(images)
                _, predicted   = outputs.max(1)
                val_correct   += predicted.eq(labels).sum().item()
                val_total     += images.size(0)

        val_acc = 100.0 * val_correct / val_total
        elapsed = time.time() - epoch_start

        print(f"Epoch [{epoch}/{EPOCHS}] | Train Loss: {train_loss:.4f} | "
              f"Train Acc: {train_acc:.2f}% | Val Acc: {val_acc:.2f}% | "
              f"Time: {elapsed:.0f}s")

        scheduler.step()

        # --- checkpoints ---
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), "checkpoints/best_finetune_model.pth")

    torch.save(model.state_dict(), "checkpoints/final_finetune_model.pth")

    total_time = time.time() - total_start

    # -------------------------------------------------------------------------
    # Save training stats
    # -------------------------------------------------------------------------

    stats = {
        "best_val_accuracy":       best_val_acc,
        "final_val_accuracy":      val_acc,
        "train_accuracy":          round(train_acc, 4),
        "total_training_time_sec": round(total_time, 2),
    }
    with open("results/finetune_model_train_stats.json", "w") as f:
        json.dump(stats, f, indent=2)

    print(f"\nTraining complete. Best val acc: {best_val_acc:.2f}% | "
          f"Total time: {total_time:.0f}s")
    print("Checkpoints saved to checkpoints/")
    print("Stats saved to results/finetune_model_train_stats.json")
