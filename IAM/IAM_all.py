import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torch.nn.functional as F
import numpy as np
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
import torch.optim.lr_scheduler as lr_scheduler
import os

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# WRN-16-4 Model Definition
class BasicBlock(nn.Module):
    def __init__(self, in_planes, out_planes, stride, dropRate=0.0):
        super(BasicBlock, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_planes)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_planes, out_planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.dropRate = dropRate
        self.equalInOut = (in_planes == out_planes)
        self.convShortcut = (not self.equalInOut) and nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, padding=0, bias=False) or None

    def forward(self, x):
        if not self.equalInOut:
            x = self.relu1(self.bn1(x))
        else:
            out = self.relu1(self.bn1(x))
        out = self.relu2(self.bn2(self.conv1(out if self.equalInOut else x)))
        if self.dropRate > 0:
            out = F.dropout(out, p=self.dropRate, training=self.training)
        out = self.conv2(out)
        return torch.add(x if self.equalInOut else self.convShortcut(x), out)

class NetworkBlock(nn.Module):
    def __init__(self, nb_layers, in_planes, out_planes, block, stride, dropRate=0.0):
        super(NetworkBlock, self).__init__()
        self.layer = self._make_layer(block, in_planes, out_planes, nb_layers, stride, dropRate)

    def _make_layer(self, block, in_planes, out_planes, nb_layers, stride, dropRate):
        layers = []
        for i in range(int(nb_layers)):
            layers.append(block(in_planes if i == 0 else out_planes, out_planes, stride if i == 0 else 1, dropRate))
        return nn.Sequential(*layers)

    def forward(self, x):
        return self.layer(x)

class WideResNet(nn.Module):
    def __init__(self, depth, num_classes, widen_factor=1, dropRate=0.0):
        super(WideResNet, self).__init__()
        nChannels = [16, 16 * widen_factor, 32 * widen_factor, 64 * widen_factor]
        assert ((depth - 4) % 6 == 0), "Depth must be 6n+4!"
        n = (depth - 4) // 6
        block = BasicBlock
        self.conv1 = nn.Conv2d(3, nChannels[0], kernel_size=3, stride=1, padding=1, bias=False)
        self.block1 = NetworkBlock(n, nChannels[0], nChannels[1], block, 1, dropRate)
        self.block2 = NetworkBlock(n, nChannels[1], nChannels[2], block, 2, dropRate)
        self.block3 = NetworkBlock(n, nChannels[2], nChannels[3], block, 2, dropRate)
        self.bn1 = nn.BatchNorm2d(nChannels[3])
        self.relu = nn.ReLU(inplace=True)
        self.fc = nn.Linear(nChannels[3], num_classes)
        self.nChannels = nChannels[3]

    def forward(self, x):
        out = self.conv1(x)
        out = self.block1(out)
        out = self.block2(out)
        out = self.block3(out)
        out = self.relu(self.bn1(out))
        out = F.avg_pool2d(out, 8)
        out = out.view(-1, self.nChannels)
        return self.fc(out)


def evaluate(model):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = outputs.max(1)
            correct += predicted.eq(labels).sum().item()
            total += labels.size(0)
    return 100. * correct / total


# Data preparation (CIFAR-10)
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261)),
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261)),
])

train_dataset = datasets.CIFAR10(root='../data', train=True, download=True, transform=transform_train)
test_dataset = datasets.CIFAR10(root='../data', train=False, download=True, transform=transform_test)

train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True, num_workers=4)
test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False, num_workers=4)

# Inconsistency Loss
model_prime = WideResNet(depth=28, widen_factor=10, num_classes=10).to(device)
criterion_kl = nn.KLDivLoss(reduction="batchmean")

def inconsistencyLoss(model, image, pred, label, criterion, k = 1, beta = 1.0):
    # Weight Initialization
    model_prime.load_state_dict(model.state_dict())
    with torch.no_grad():
      for param in model_prime.parameters():
        param.add(0.1 * torch.normal(0, 1, size=param.shape, device=device))

    # Optimizar Initialization
    optimizer = optim.SGD(model_prime.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)

    # Gradient Descent
    model_prime.train()
    for _ in range(k):
        optimizer.zero_grad()
        with torch.enable_grad():
            loss_kl = -1 * criterion_kl(F.log_softmax(model_prime(image), dim=1),
                                        F.softmax(pred, dim=1))
        loss_kl.backward(retain_graph=True)
        optimizer.step()

    inconsistency_loss = beta * criterion_kl(F.log_softmax(model_prime(image), dim=1),
                                                               F.softmax(pred, dim=1))
    return inconsistency_loss


# Training and Evaluation
epochs = 300

model = WideResNet(depth=28, widen_factor=10, num_classes=10).to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)
scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)


if not os.path.exists("./checkpoints"):
    os.makedirs("./checkpoints")

# 학습 기록을 저장할 리스트
loss_history = []
error_history = []
inconsistency_history = []

for epoch in range(epochs):
    scheduler.step()
    model.train()
    
    total_loss = 0.0
    total_inconsistency = 0.0
    
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)

        # Loss 및 Inconsistency Loss 계산
        inconsistency = inconsistencyLoss(model, images, outputs, labels, criterion)
        loss = criterion(outputs, labels) + inconsistency
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        total_inconsistency += inconsistency.item()

    # 평균 loss 계산
    avg_loss = total_loss / len(train_loader)
    avg_inconsistency = total_inconsistency / len(train_loader)

    # Evaluation (Accuracy 계산 후 Test Error 저장)
    acc = evaluate(model)
    test_error = 100 - acc

    # 기록 저장
    loss_history.append(avg_loss)
    error_history.append(test_error)
    inconsistency_history.append(avg_inconsistency)

    print(f"Epoch {epoch+1}/{epochs}: Test Error: {test_error:.2f}%, Inconsistency: {avg_inconsistency:}")

    # 모델 저장
    if (epoch + 1) % 10 == 0:
        torch.save(model.state_dict(), f"./checkpoints/model_epoch{epoch+1}.pth")

plt.figure(figsize=(12, 5))

# Loss 그래프
plt.subplot(1, 3, 1)
plt.plot(range(1, epochs + 1), loss_history, label="Loss", linestyle="-", color="blue")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.title("Training Loss per Epoch")
plt.legend()

# Test Error 그래프
plt.subplot(1, 3, 2)
plt.plot(range(1, epochs + 1), error_history, label="Test Error", linestyle="-", color="red")
plt.xlabel("Epochs")
plt.ylabel("Test Error (%)")
plt.title("Test Error per Epoch")
plt.legend()

# Inconsistency Loss 그래프
plt.subplot(1, 3, 3)
plt.plot(range(1, epochs + 1), inconsistency_history, label="Inconsistency Loss", linestyle="-", color="green")
plt.xlabel("Epochs")
plt.ylabel("Inconsistency Loss")
plt.title("Inconsistency Loss per Epoch")
plt.legend()

plt.tight_layout()
plt.savefig("training_results.png", dpi=300)