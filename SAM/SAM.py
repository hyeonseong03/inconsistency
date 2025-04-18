import torch
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
import torch.nn as nn
import matplotlib.pyplot as plt
import os
import wandb

import sys; sys.path.append("..")
from model import WideResNet
from data import get_cifar10_loaders
from stepLR import StepLR

epochs = 200
lr = 0.1
dropout = 0.0
eval_mode = True if dropout > 0.0 else False
rho = 0.05

# Start a new wandb run to track this script.
run = wandb.init(
    # Set the wandb entity where your project will be logged (generally your team name).
    entity="hyeonseong03-hanyang-university",
    # Set the wandb project where this run will be logged.
    project="IAM",
    name="SAM_basic",
    # Track hyperparameters and run metadata.
    config={
        "learning_rate": lr,
        "architecture": "WRN-28-10",
        "dataset": "CIFAR-10",
        "epochs": epochs,
        "optimizer": "SAM",
        "dropout": dropout,
        "augmentation": "cutout",
        "scheduler": "stepLR",
        "ascent": rho,
        "eval": eval_mode,
    },
)

# CUDA 설정
device = "cuda" if torch.cuda.is_available() else "cpu"

# 데이터 로드
train_loader, test_loader = get_cifar10_loaders()

# Model Initialization
model = WideResNet(depth=28, width_factor=10, dropout=dropout, in_channels=3, labels=10).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)
# scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
# scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[60, 120, 160], gamma=0.2)
scheduler = StepLR(optimizer, lr, epochs)

loss_history = []
error_history = []

def SAMLoss(model, image, label, criterion, rho = rho):
    pred = model(image)
    loss = criterion(pred, label)
    loss.backward(retain_graph=True)

    grads = [param.grad.clone() for param in model.parameters() if param.requires_grad]
    wgrads = [torch.norm(param.grad, p=2) for param in model.parameters() if param.requires_grad]
    norm = torch.norm(torch.stack(wgrads), p=2) + 1e-12

    with torch.no_grad():
        for param, grad in zip(model.parameters(), grads):
          if param is not None:
            param.data = param.data + rho * grad / norm

    sam_pred = model(image)
    sam_loss = criterion(sam_pred, label)

    with torch.no_grad():
      for param, grad in zip(model.parameters(), grads):
        if param is not None:
          param.data = param.data - rho * grad / norm

    return sam_loss

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
for epoch in range(epochs):
    model.train()
    total_loss = 0

    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        loss = SAMLoss(model, images, labels, criterion)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    # scheduler.step()
    scheduler(epoch)
    acc = evaluate(model)
    error = 100 - acc

    loss_history.append(total_loss / len(train_loader))
    error_history.append(error)

    run.log({"Test Error":error},)

    if (epoch + 1) % 50 == 0:
        torch.save(model.state_dict(), f"./checkpoints/sam_epoch{epoch+1}.pth")
        print(f"Model saved at epoch {epoch+1}")

run.finish()

# 결과 그래프 저장
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(range(1, 301), loss_history, label="SAM Loss", linestyle="-")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.title("Training Loss per Epoch")
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(range(1, 301), error_history, label="SAM Test Error", linestyle="-")
plt.xlabel("Epochs")
plt.ylabel("Test Error (%)")
plt.title("Test Error per Epoch")
plt.legend()

plt.tight_layout()
plt.savefig("training_results_sam.png", dpi=300)