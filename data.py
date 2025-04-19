import torch
import torchvision
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

class Cutout:
    def __init__(self, size=16, p=0.5):
        self.size = size
        self.half_size = size // 2
        self.p = p

    def __call__(self, image):
        if torch.rand(1).item() > self.p:
            return image

        c, h, w = image.shape
        left = torch.randint(-self.half_size, w - self.half_size, [1]).item()
        top = torch.randint(-self.half_size, h - self.half_size, [1]).item()
        right = min(w, left + self.size)
        bottom = min(h, top + self.size)

        image[:, max(0, left): right, max(0, top): bottom] = 0
        return image

def get_cifar10_loaders(batch_size=128, num_workers=4):
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        # Cutout()
        transforms.ToTensor(),
        # transforms.Normalize(mean, std),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        # transforms.Normalize(mean, std)
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    train_dataset = datasets.CIFAR10(root='../data', train=True, download=True, transform=transform_train)
    test_dataset = datasets.CIFAR10(root='../data', train=False, download=True, transform=transform_test)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True, persistent_workers=True, prefetch_factor=2)
    test_loader = DataLoader(test_dataset, batch_size=1000, shuffle=False, num_workers=num_workers)

    return train_loader, test_loader