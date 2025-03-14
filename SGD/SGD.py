import torch
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
import torch.nn as nn
import matplotlib.pyplot as plt

import sys; sys.path.append("..")
from model import WideResNet
from data import get_cifar10_loaders

# CUDA 설정
device = "cuda" if torch.cuda.is_available() else "cpu"

# 데이터 로드
train_loader, test_loader = get_cifar10_loaders()

# Model Initialization
model = WideResNet(depth=28, widen_factor=10, num_classes=10).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)
scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=300)

loss_history = []
error_history = []

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

# 학습 루프
for epoch in range(300):
    model.train()
    total_loss = 0

    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    scheduler.step()
    acc = evaluate(model)
    error = 100 - acc

    loss_history.append(total_loss / len(train_loader))
    error_history.append(error)

    print(f"Epoch {epoch+1}: SGD Test Error: {error:.2f}%")

    if (epoch + 1) % 50 == 0:
        torch.save(model.state_dict(), f"./checkpoints/sgd_epoch{epoch+1}.pth")
        print(f"Model saved at epoch {epoch+1}")

# 결과 그래프 저장
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(range(1, 301), loss_history, label="SGD Loss", linestyle="--")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.title("Training Loss per Epoch")
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(range(1, 301), error_history, label="SGD Test Error", linestyle="--")
plt.xlabel("Epochs")
plt.ylabel("Test Error (%)")
plt.title("Test Error per Epoch")
plt.legend()

plt.tight_layout()
plt.savefig("training_results_sgd.png", dpi=300)