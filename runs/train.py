import torch
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
import torch.nn as nn
import matplotlib.pyplot as plt
import argparse
import os
import wandb

import sys; sys.path.append("..")
from model import WideResNet
from data import get_cifar10_loaders
from stepLR import StepLR
from IAM import inconsistencyLoss
from SAM import SAMLoss

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

if __name__ == "__main__":
    os.environ["TMPDIR"] = "/home/intern/tmp"

    parser = argparse.ArgumentParser()
    parser.add_argument("--optimizer", default="IAM", type=str)
    parser.add_argument("--dropout", default=0.0, type=float)
    parser.add_argument("--ascent", default=0.05, type=float)
    parser.add_argument("--epochs", default=200, type=int)
    parser.add_argument("--lr", default=0.1, type=float)
    parser.add_argument("--beta", default=1.0, type=float)
    args = parser.parse_args()

    # eval_mode 설정 (IAM용)
    eval_mode = False

    # CUDA 설정
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Start a new wandb run to track this script.
    run = wandb.init(
        # Set the wandb entity where your project will be logged (generally your team name).
        entity="hyeonseong03-hanyang-university",
        # Set the wandb project where this run will be logged.
        project="IAM",
        name=args.optimizer+"_grad_clip_(SGD)",
        # Track hyperparameters and run metadata.
        config={
            "learning_rate": args.lr,
            "architecture": "WRN-28-10",
            "dataset": "CIFAR-10",
            "epochs": args.epochs,
            "optimizer": args.optimizer,
            "dropout": args.dropout,
            "augmentation": "basic",
            "scheduler": "multistep",
            "beta": args.beta,
            "ascent": args.ascent,
            "eval": eval_mode,
            "noise": "adaptive",
        },
    )

    # Initialize dataloader
    train_loader, test_loader = get_cifar10_loaders()

    model = WideResNet(depth=28, width_factor=10, dropout=args.dropout, in_channels=3, labels=10).to(device)
    model_prime = WideResNet(depth=28, width_factor=10, dropout=args.dropout, in_channels=3, labels=10).to(device)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
    # scheduler = StepLR(optimizer, args.lr, args.epochs)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[60, 120, 160], gamma=0.2)
    # scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    loss_history = []
    error_history = []
    inconsistency_history = []

    if not os.path.exists("./checkpoints"):
        os.makedirs("./checkpoints")

    #Training Loop
    for epoch in range(args.epochs):
        
        total_loss = 0.0
        total_inconsistency = 0.0
        
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            loss = 0.0
            inconsistency = 0.0

            if args.optimizer == "IAM":
                model.train()
                outputs = model(images)
                inconsistency = inconsistencyLoss(model, model_prime, images, labels, outputs, criterion, rho=args.ascent, eval_mode = eval_mode)
                loss = criterion(outputs, labels) + inconsistency
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                optimizer.step()
                total_inconsistency += inconsistency.item()
            elif args.optimizer == "SAM":
                loss = SAMLoss(model, images, labels, criterion, optimizer, args.ascent)
            elif args.optimizer == "SGD":
                model.train()
                outputs = model(images)
                loss = criterion(outputs, labels)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            total_loss += loss.item()
        scheduler.step()
        # scheduler(epoch)

        # Average Loss
        avg_loss = total_loss / len(train_loader)
        acc = evaluate(model)
        error = 100 - acc

        loss_history.append(avg_loss)
        error_history.append(error)

        if args.optimizer == "IAM":
            avg_inconsistency = total_inconsistency / len(train_loader)
            inconsistency_history.append(avg_inconsistency)
            run.log({"Test Error": error, "Inconsistency": avg_inconsistency, "Loss": avg_loss,})
        else:
            run.log({"Test Error": error, "Loss": avg_loss,})

        # 모델 저장
        if (epoch + 1) % 50 == 0:
            torch.save(model.state_dict(), "./checkpoints/"+args.optimizer+f"_epoch{epoch+1}.pth")
            print(f"Model saved at epoch {epoch+1}")

    run.finish()