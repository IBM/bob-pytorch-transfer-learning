"""
train_base_model.py
ResNet-18 built from scratch, trained on CIFAR-10 for 3 epochs.
Saves checkpoints/best_base_model.pth and checkpoints/final_base_model.pth.
Saves results/base_model_train_stats.json.
"""

import os
import time
import json
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms

# ---------------------------------------------------------------------------
# ResNet-18 architecture
# ---------------------------------------------------------------------------

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn1   = nn.BatchNorm2d(out_channels)
        self.relu  = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2   = nn.BatchNorm2d(out_channels)

        self.downsample = None
        if stride != 1 or in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1,
                          stride=stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )

    def forward(self, x):
        identity = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        return self.relu(out)


class ResNet18(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        # For CIFAR-10 (32×32) use a 3×3 conv with stride 1 — no spatial collapse
        self.conv1   = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1     = nn.BatchNorm2d(64)
        self.relu    = nn.ReLU(inplace=True)

        self.layer1  = self._make_layer(64,  64,  stride=1)
        self.layer2  = self._make_layer(64,  128, stride=2)
        self.layer3  = self._make_layer(128, 256, stride=2)
        self.layer4  = self._make_layer(256, 512, stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc      = nn.Linear(512, num_classes)

    def _make_layer(self, in_channels, out_channels, stride):
        return nn.Sequential(
            BasicBlock(in_channels, out_channels, stride=stride),
            BasicBlock(out_channels, out_channels, stride=1),
        )

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        return self.fc(x)


# ---------------------------------------------------------------------------
# Data, training, and entry point
# ---------------------------------------------------------------------------

MEAN = (0.4914, 0.4822, 0.4465)
STD  = (0.2023, 0.1994, 0.2010)

if __name__ == '__main__':
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
    # Device / model / loss / optimizer
    # -------------------------------------------------------------------------

    device = torch.device("mps" if torch.backends.mps.is_available()
                           else "cuda" if torch.cuda.is_available()
                           else "cpu")
    print(f"Using device: {device}")

    model     = ResNet18(num_classes=10).to(device)
    criterion = nn.CrossEntropyLoss()
    # lr=0.1 is the standard SGD starting point for training from scratch on CIFAR-10
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1,
                                momentum=0.9, weight_decay=5e-4)

    os.makedirs("checkpoints", exist_ok=True)
    os.makedirs("results",     exist_ok=True)

    # -------------------------------------------------------------------------
    # Training loop
    # -------------------------------------------------------------------------

    EPOCHS = 3
    # Cosine annealing decays lr smoothly from 0.1 → ~0 over EPOCHS
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
            torch.save(model.state_dict(), "checkpoints/best_base_model.pth")

    torch.save(model.state_dict(), "checkpoints/final_base_model.pth")

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
    with open("results/base_model_train_stats.json", "w") as f:
        json.dump(stats, f, indent=2)

    print(f"\nTraining complete. Best val acc: {best_val_acc:.2f}% | "
          f"Total time: {total_time:.0f}s")
    print("Checkpoints saved to checkpoints/")
    print("Stats saved to results/base_model_train_stats.json")
