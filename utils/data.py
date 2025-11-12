from __future__ import annotations
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import torch


# Optional: tqdm
try:
    from tqdm.auto import tqdm
except Exception:
    tqdm = None


def build_cifar10_loaders(data_root: str, batch_size=256, num_workers=4):
    mean = (0.4914, 0.4822, 0.4465); std = (0.2023, 0.1994, 0.2010)
    try:
        from torchvision.transforms import TrivialAugmentWide, RandomErasing
        aug = [TrivialAugmentWide()]
        erasing = RandomErasing(p=0.25)
    except Exception:
        aug = []
        erasing = transforms.RandomErasing(p=0.25)
    train_tf = transforms.Compose(
        aug + [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
            erasing,
        ]
    )
    pin = torch.cuda.is_available()
    test_tf = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean, std)])
    train_set = datasets.CIFAR10(root=data_root, train=True,  download=True, transform=train_tf)
    test_set  = datasets.CIFAR10(root=data_root, train=False, download=True, transform=test_tf)
    dl_kwargs = dict(batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=pin)
    if num_workers and num_workers > 0:
        dl_kwargs.update(dict(persistent_workers=True, prefetch_factor=6))
    train_loader = DataLoader(train_set, **dl_kwargs)
    test_loader  = DataLoader(test_set, batch_size=batch_size, shuffle=False,
                              num_workers=num_workers, pin_memory=pin)
    return train_loader, test_loader
