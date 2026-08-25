"""
infer_base_model.py
Run inference on the CIFAR-10 test set using the best base model checkpoint.
Prints test accuracy and inference time; saves results/base_model_inference.json.
"""

import os
import time
import json
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms

# ---------------------------------------------------------------------------
# ResNet-18 architecture (identical to train_base_model.py)
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
    # Load model
    # -------------------------------------------------------------------------

    model = ResNet18(num_classes=10).to(device)
    model.load_state_dict(torch.load("checkpoints/best_base_model.pth",
                                     map_location=device))
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
    with open("results/base_model_inference.json", "w") as f:
        json.dump(results, f, indent=2)

    print("Results saved to results/base_model_inference.json")
