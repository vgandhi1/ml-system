"""IID partition of MNIST across virtual clients."""

from __future__ import annotations

from pathlib import Path

import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms


def _project_data_dir() -> Path:
    root = Path(__file__).resolve().parent.parent / "data"
    root.mkdir(parents=True, exist_ok=True)
    return root


def ensure_mnist_downloaded() -> None:
    """Download MNIST once (call from a single process before spawning clients)."""
    tfm = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
    )
    data_root = _project_data_dir()
    datasets.MNIST(str(data_root), train=True, download=True, transform=tfm)
    datasets.MNIST(str(data_root), train=False, download=True, transform=tfm)


def load_client_dataloaders(
    *,
    num_clients: int,
    client_id: int,
    batch_size: int,
) -> tuple[DataLoader, DataLoader]:
    """Return train/test loaders for one client (IID shard of MNIST)."""
    if not (0 <= client_id < num_clients):
        raise ValueError("client_id must be in [0, num_clients)")

    tfm = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
    )
    data_root = _project_data_dir()
    train_full = datasets.MNIST(
        str(data_root), train=True, download=False, transform=tfm
    )
    test_full = datasets.MNIST(
        str(data_root), train=False, download=False, transform=tfm
    )

    n_train = len(train_full)
    per = n_train // num_clients
    start = client_id * per
    end = (client_id + 1) * per if client_id < num_clients - 1 else n_train
    train_idx = list(range(start, end))
    train_subset: Subset[datasets.MNIST] = Subset(train_full, train_idx)

    train_loader = DataLoader(
        train_subset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
    )
    test_loader = DataLoader(
        test_full,
        batch_size=256,
        shuffle=False,
        num_workers=0,
    )
    return train_loader, test_loader
