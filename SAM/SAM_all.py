import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torch.nn.functional as F
from tqdm.auto import tqdm
import os
import torch.optim.lr_scheduler as lr_scheduler
import matplotlib.pyplot as plt

# Wide ResNet
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

# SAM Loss implementation
def SAMLoss(model, image, pred, label, criterion, rho = 0.05):
    loss = criterion(pred, label)

    loss.backward(retain_graph=True)

    with torch.no_grad():
        grads = []
        for param in model.parameters():
          if param is not None:
            grads.append(param.grad.clone())

        for param, grad in zip(model.parameters(), grads):
          if param is not None:
            param.add(rho * grad / torch.norm(grad + 1e-12))

    sam_pred = model(image)
    sam_loss = criterion(sam_pred, label)

    with torch.no_grad():
      for param, grad in zip(model.parameters(), grads):
        if param is not None:
          param.sub(rho * grad / torch.norm(grad + 1e-12))

    return sam_loss

# Training and Evaluation
device = 'cuda'
epochs = 300

# model_sam = WideResNet(depth=28, widen_factor=10, num_classes=10).to(device)
model_sgd = WideResNet(depth=28, widen_factor=10, num_classes=10).to(device)

criterion = nn.CrossEntropyLoss()

# optimizer_sam = optim.SGD(model_sam.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)
optimizer_sgd = optim.SGD(model_sgd.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)

# scheduler_sam = lr_scheduler.CosineAnnealingLR(optimizer_sam, T_max=epochs)
scheduler_sgd = lr_scheduler.CosineAnnealingLR(optimizer_sgd, T_max=epochs)

# scheduler_sam = lr_scheduler.CosineAnnealingLR(optimizer_sam, T_max=epochs)
scheduler_sgd = lr_scheduler.CosineAnnealingLR(optimizer_sgd, T_max=epochs)

if not os.path.exists("./checkpoints"):
    os.makedirs("./checkpoints")

loss_history_sam = []
loss_history_sgd = []
error_history_sam = []
error_history_sgd = []

for epoch in range(epochs):
    # scheduler_sam.step()
    scheduler_sgd.step()

    # Training SAM
    # model_sam.train()
    # total_loss_sam = 0
    # for images, labels in train_loader:
    #     images, labels = images.to(device), labels.to(device)
    #     optimizer_sam.zero_grad()
    #     outputs = model_sam(images)
    #     loss = SAMLoss(model_sam, images, outputs, labels, criterion)
    #     loss.backward()
    #     optimizer_sam.step()
    #     total_loss_sam += loss.item()  # loss accumulation

    # Training SGD
    model_sgd.train()
    total_loss_sgd = 0
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        optimizer_sgd.zero_grad()
        outputs = model_sgd(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer_sgd.step()
        total_loss_sgd += loss.item()  # loss accumulation

    # Evaluation
    # acc_sam = evaluate(model_sam)
    acc_sgd = evaluate(model_sgd)
    # error_sam = 100 - acc_sam
    error_sgd = 100 - acc_sgd

    # 평균 loss 계산
    # avg_loss_sam = total_loss_sam / len(train_loader)
    avg_loss_sgd = total_loss_sgd / len(train_loader)

    # 기록 저장
    # loss_history_sam.append(avg_loss_sam)
    loss_history_sgd.append(avg_loss_sgd)
    # error_history_sam.append(error_sam)
    error_history_sgd.append(error_sgd)

    print(f"Epoch {epoch+1}/{epochs}: SGD Test Error: {error_sgd:.2f}%")
    #  SGD Accuracy: {acc_sgd:.2f}%

    # Save models every 10 epochs
    if (epoch + 1) % 10 == 0:
        # torch.save(model_sam.state_dict(), f"./checkpoints/model_sam_epoch{epoch+1}.pth")
        torch.save(model_sgd.state_dict(), f"./checkpoints/model_sgd_epoch{epoch+1}.pth")
        print(f"Models saved at epoch {epoch+1}")

plt.figure(figsize=(12, 5))

# Loss 그래프
plt.subplot(1, 2, 1)
# plt.plot(range(1, epochs + 1), loss_history_sam, label="SAM Loss", linestyle="-")
plt.plot(range(1, epochs + 1), loss_history_sgd, label="SGD Loss", linestyle="--")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.title("Training Loss per Epoch")
plt.legend()

# Test Error 그래프
plt.subplot(1, 2, 2)
# plt.plot(range(1, epochs + 1), error_history_sam, label="SAM Test Error", linestyle="-")
plt.plot(range(1, epochs + 1), error_history_sgd, label="SGD Test Error", linestyle="--")
plt.xlabel("Epochs")
plt.ylabel("Test Error (%)")
plt.title("Test Error per Epoch")
plt.legend()

plt.tight_layout()
plt.savefig("training_results.png", dpi=300)