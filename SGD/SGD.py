import torch
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
import torch.nn as nn
import matplotlib.pyplot as plt
import torch.nn.functional as F
import os

import sys; sys.path.append("..")
from model import WideResNet
from data import get_cifar10_loaders

# CUDA 설정
device = "cuda" if torch.cuda.is_available() else "cpu"

# 데이터 로드
train_loader, test_loader = get_cifar10_loaders()

# Model Initialization
model = WideResNet(depth=28, width_factor=10, dropout=0.0, in_channels=3, labels=10).to(device)
model_prime = WideResNet(depth=28, width_factor=10, dropout=0.0, in_channels=3, labels=10).to(device)
criterion = nn.CrossEntropyLoss()
criterion_kl = nn.KLDivLoss(reduction="batchmean")
optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)
# scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=300)
scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[60, 120, 160], gamma=0.2)

loss_history = []
error_history = []
inconsistency_history = []

def inconsistencyLoss(model, image, pred, label, criterion, k = 1, beta = 1.0):
    # Weight Initialization
    model_prime.load_state_dict(model.state_dict())
    with torch.no_grad():
      for param in model_prime.parameters():
        param.add(0.1 * torch.normal(0, 1, size=param.shape, device=device)) #0.1 수치가 애매함 -> normalization (gradient 없애려고 하는건데 0.1은 약간 클수도)

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

if not os.path.exists("./checkpoints"):
    os.makedirs("./checkpoints")

# 학습 루프
for epoch in range(300):
    model.train()
    total_loss = 0.0
    total_inconsistency = 0.0

    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        inconsistency = inconsistencyLoss(model, images, outputs, labels, criterion)
        loss.backward()
        optimizer.step()

        total_inconsistency += inconsistency.item()
        total_loss += loss.item()

    scheduler.step()
    acc = evaluate(model)
    error = 100 - acc
    avg_inconsistency = total_inconsistency / len(train_loader)

    loss_history.append(total_loss / len(train_loader))
    inconsistency_history.append(avg_inconsistency)
    error_history.append(error)

    print(f"Epoch {epoch+1}: SGD Test Error: {error:.2f}% Inconsistency: {avg_inconsistency}")

    if (epoch + 1) % 50 == 0:
        torch.save(model.state_dict(), f"./checkpoints/sgd_epoch{epoch+1}.pth")
        print(f"Model saved at epoch {epoch+1}")

# 결과 그래프 저장
plt.figure(figsize=(12, 5))

plt.subplot(1, 3, 1)
plt.plot(range(1, 301), loss_history, label="SGD Loss", linestyle="--")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.title("Training Loss per Epoch")
plt.legend()

plt.subplot(1, 3, 2)
plt.plot(range(1, 301), error_history, label="SGD Test Error", linestyle="--")
plt.xlabel("Epochs")
plt.ylabel("Test Error (%)")
plt.title("Test Error per Epoch")
plt.legend()

# Inconsistency  Graph
plt.subplot(1, 3, 3)
plt.plot(range(1, epochs + 1), inconsistency_history, label="Inconsistency", linestyle="-", color="green")
plt.xlabel("Epochs")
plt.ylabel("Inconsistency")
plt.title("Inconsistency per Epoch")
plt.legend()

plt.tight_layout()
plt.savefig("training_results_sgd.png", dpi=300)