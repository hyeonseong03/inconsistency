import torch
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
import torch.nn as nn
import matplotlib.pyplot as plt
import torch.nn.functional as F

import sys; sys.path.append("..")
from model import WideResNet
from data import get_cifar10_loaders

# CUDA 설정
device = "cuda" if torch.cuda.is_available() else "cpu"

# 데이터 로드
train_loader, test_loader = get_cifar10_loaders()

# Model Initialization
model = WideResNet(depth=28, widen_factor=10, num_classes=10).to(device)
model_prime = WideResNet(depth=28, widen_factor=10, num_classes=10).to(device)
criterion = nn.CrossEntropyLoss()
criterion_kl = nn.KLDivLoss(reduction="batchmean")
optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)
scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=300)

loss_history = []
error_history = []
inconsistency_history = []

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

#Training Loop
epochs = 300

for epoch in range(epochs):
    model.train()
    
    total_loss = 0.0
    total_inconsistency = 0.0
    
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)

        # Loss
        inconsistency = inconsistencyLoss(model, images, outputs, labels, criterion)
        loss = criterion(outputs, labels) + inconsistency
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        total_inconsistency += inconsistency.item()

    scheduler.step()

    # Average Loss
    avg_loss = total_loss / len(train_loader)
    avg_inconsistency = total_inconsistency / len(train_loader)

    # Evaluation
    acc = evaluate(model)
    test_error = 100 - acc

    loss_history.append(avg_loss)
    error_history.append(test_error)
    inconsistency_history.append(avg_inconsistency)

    print(f"Epoch {epoch+1}/{epochs}: IAM Test Error: {test_error:.2f}%, Inconsistency: {avg_inconsistency}")

    # 모델 저장
    if (epoch + 1) % 50 == 0:
        torch.save(model.state_dict(), f"./checkpoints/iam_epoch{epoch+1}.pth")
        print(f"Model saved at epoch {epoch+1}")

plt.figure(figsize=(12, 5))

# Loss Graph
plt.subplot(1, 3, 1)
plt.plot(range(1, epochs + 1), loss_history, label="IAM Loss", linestyle="-", color="blue")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.title("Training Loss per Epoch")
plt.legend()

# Test Error Graph
plt.subplot(1, 3, 2)
plt.plot(range(1, epochs + 1), error_history, label="IAM Test Error", linestyle="-", color="red")
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
plt.savefig("training_results_iam.png", dpi=300)