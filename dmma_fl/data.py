from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, Subset, TensorDataset


def load_dataset(
    name: str,
    samples: int,
    num_classes: int,
    seed: int,
    root: str = "data",
    download: bool = False,
) -> tuple[Dataset, Dataset, tuple[int, int, int]]:
    if name == "synthetic":
        train = make_synthetic_dataset(samples, num_classes, (1, 28, 28), seed)
        test = make_synthetic_dataset(max(samples // 5, 200), num_classes, (1, 28, 28), seed + 1)
        return train, test, (1, 28, 28)

    try:
        from torchvision import datasets, transforms
    except ImportError as exc:
        raise RuntimeError("torchvision is required for MNIST/CIFAR-10") from exc

    if name == "mnist":
        transform = transforms.Compose([transforms.ToTensor()])
        return (
            datasets.MNIST(root=root, train=True, download=download, transform=transform),
            datasets.MNIST(root=root, train=False, download=download, transform=transform),
            (1, 28, 28),
        )
    if name == "cifar10":
        transform = transforms.Compose([transforms.ToTensor()])
        return (
            datasets.CIFAR10(root=root, train=True, download=download, transform=transform),
            datasets.CIFAR10(root=root, train=False, download=download, transform=transform),
            (3, 32, 32),
        )
    raise ValueError(f"Unsupported dataset: {name}")


def make_synthetic_dataset(samples: int, num_classes: int, image_shape: tuple[int, int, int], seed: int) -> TensorDataset:
    rng = np.random.default_rng(seed)
    channels, height, width = image_shape
    labels = rng.integers(0, num_classes, size=samples)
    prototypes = rng.normal(0.0, 1.0, size=(num_classes, channels, height, width)).astype(np.float32)
    x = prototypes[labels] + rng.normal(0.0, 0.7, size=(samples, channels, height, width)).astype(np.float32)
    y = labels.astype(np.int64)
    return TensorDataset(torch.from_numpy(x), torch.from_numpy(y))


def split_clients(dataset: Dataset, num_devices: int, iid: bool, seed: int) -> list[Subset]:
    rng = np.random.default_rng(seed)
    indices = np.arange(len(dataset))
    if iid:
        rng.shuffle(indices)
    else:
        labels = np.asarray([int(dataset[i][1]) for i in range(len(dataset))])
        indices = indices[np.argsort(labels)]
        chunks = np.array_split(indices, num_devices * 2)
        rng.shuffle(chunks)
        indices = np.concatenate(chunks)
    return [Subset(dataset, idx.tolist()) for idx in np.array_split(indices, num_devices)]


def loader_for(dataset: Dataset, batch_size: int, shuffle: bool = True) -> DataLoader:
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=0)
