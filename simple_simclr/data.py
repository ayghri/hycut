import torch
from torchvision import transforms
from torchvision.datasets import CIFAR10, CIFAR100
from PIL import Image, ImageFilter
import random

class GaussianBlur(object):
    """Gaussian blur augmentation in SimCLR https://arxiv.org/abs/2002.05709"""

    def __init__(self, sigma=[.1, 2.]):
        self.sigma = sigma

    def __call__(self, x):
        sigma = random.uniform(self.sigma[0], self.sigma[1])
        x = x.filter(ImageFilter.GaussianBlur(radius=sigma))
        return x

class SimCLRTransform:
    def __init__(self, size, s=1):
        color_jitter = transforms.ColorJitter(0.8 * s, 0.8 * s, 0.8 * s, 0.2 * s)
        self.train_transform = transforms.Compose([
            transforms.RandomResizedCrop(size=size),
            transforms.RandomHorizontalFlip(),
            transforms.RandomApply([color_jitter], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.RandomApply([GaussianBlur([.1, 2.])], p=0.5),
            transforms.ToTensor(),
        ])
        
        self.test_transform = transforms.Compose([
            transforms.Resize(size),
            transforms.ToTensor(),
        ])

    def __call__(self, x):
        return self.train_transform(x), self.train_transform(x)

class CIFAR10SimCLR(CIFAR10):
    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]
        img = Image.fromarray(img)

        if self.transform is not None:
            img1, img2 = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img1, img2, target

class CIFAR100SimCLR(CIFAR100):
    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]
        img = Image.fromarray(img)

        if self.transform is not None:
            img1, img2 = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img1, img2, target

def get_dataloaders(dataset_name, batch_size, num_workers, root='./data'):
    if dataset_name == 'cifar10':
        train_dataset = CIFAR10SimCLR(
            root, train=True, transform=SimCLRTransform(32), download=True
        )
        test_dataset = CIFAR10(
            root, train=False, transform=transforms.Compose([
                transforms.ToTensor()
            ]), download=True
        )
    elif dataset_name == 'cifar100':
        train_dataset = CIFAR100SimCLR(
            root, train=True, transform=SimCLRTransform(32), download=True
        )
        test_dataset = CIFAR100(
            root, train=False, transform=transforms.Compose([
                transforms.ToTensor()
            ]), download=True
        )
    else:
        raise ValueError("Dataset not supported")

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True, drop_last=True
    )
    
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True
    )

    return train_loader, test_loader
