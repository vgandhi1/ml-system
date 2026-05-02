#!/usr/bin/env python3
"""Reproducible centralized vs in-process FedAvg MNIST benchmark for README tables."""

from __future__ import annotations

import sys
import time
from collections import OrderedDict
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader, Subset

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from fed_ml.data import ensure_mnist_downloaded, load_client_dataloaders
from fed_ml.model import MnistCNN, get_device


def _train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
) -> None:
    model.train()
    for x, y in loader:
        x = x.to(device)
        y = y.to(device)
        optimizer.zero_grad()
        criterion(model(x), y).backward()
        optimizer.step()


def _evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> float:
    model.eval()
    criterion = nn.CrossEntropyLoss()
    correct = 0
    total = 0
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)
            pred = model(x).argmax(dim=1)
            correct += (pred == y).sum().item()
            total += y.size(0)
    return correct / max(total, 1)


def _fedavg_merge(
    state_dicts: list[OrderedDict[str, torch.Tensor]], weights: list[int]
) -> OrderedDict[str, torch.Tensor]:
    total_w = float(sum(weights))
    keys = list(state_dicts[0].keys())
    out: OrderedDict[str, torch.Tensor] = OrderedDict()
    for k in keys:
        acc = torch.zeros_like(state_dicts[0][k], dtype=torch.float64)
        for sd, w in zip(state_dicts, weights, strict=True):
            acc += sd[k].to(dtype=torch.float64) * (w / total_w)
        out[k] = acc.to(dtype=state_dicts[0][k].dtype)
    return out


def run_centralized(
    *,
    device: torch.device,
    num_epochs: int,
    batch_size: int,
) -> tuple[float, float]:
    """Full MNIST train set; returns (wall_seconds, test_accuracy)."""
    from torchvision import datasets, transforms

    tfm = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
    )
    data_root = ROOT / "data"
    data_root.mkdir(parents=True, exist_ok=True)
    train_full = datasets.MNIST(str(data_root), train=True, download=False, transform=tfm)
    test_full = datasets.MNIST(str(data_root), train=False, download=False, transform=tfm)
    train_loader = DataLoader(
        train_full, batch_size=batch_size, shuffle=True, num_workers=0
    )
    test_loader = DataLoader(test_full, batch_size=256, shuffle=False, num_workers=0)

    model = MnistCNN().to(device)
    opt = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    crit = nn.CrossEntropyLoss()

    t0 = time.perf_counter()
    for _ in range(num_epochs):
        _train_one_epoch(model, train_loader, device, opt, crit)
    elapsed = time.perf_counter() - t0
    acc = _evaluate(model, test_loader, device)
    return elapsed, acc


def run_federated_sim(
    *,
    device: torch.device,
    num_clients: int,
    num_rounds: int,
    batch_size: int,
    local_epochs: int,
) -> tuple[float, float]:
    """In-process FedAvg (same shards as fed_ml.data); returns (wall_seconds, test_accuracy)."""
    crit = nn.CrossEntropyLoss()
    global_model = MnistCNN().to(device)
    # One shared test loader (full test set) via client 0's test_loader
    _, test_loader = load_client_dataloaders(
        num_clients=num_clients, client_id=0, batch_size=batch_size
    )

    t0 = time.perf_counter()
    for _ in range(num_rounds):
        state_dicts: list[OrderedDict[str, torch.Tensor]] = []
        weights: list[int] = []
        for cid in range(num_clients):
            train_loader, _ = load_client_dataloaders(
                num_clients=num_clients, client_id=cid, batch_size=batch_size
            )
            n = len(train_loader.dataset)
            local = MnistCNN().to(device)
            local.load_state_dict(global_model.state_dict())
            opt = torch.optim.SGD(local.parameters(), lr=0.01, momentum=0.9)
            for _ in range(local_epochs):
                _train_one_epoch(local, train_loader, device, opt, crit)
            state_dicts.append(OrderedDict(local.state_dict()))
            weights.append(n)
        merged = _fedavg_merge(state_dicts, weights)
        global_model.load_state_dict(merged)
    elapsed = time.perf_counter() - t0
    acc = _evaluate(global_model, test_loader, device)
    return elapsed, acc


def main() -> None:
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)

    ensure_mnist_downloaded()
    device = get_device()

    num_clients = 3
    batch_size = 64
    fed_rounds = 10
    local_epochs = 1
    # Match aggregate passes over data order-of-magnitude: ~10 federated rounds
    central_epochs = 10

    t_cen, acc_cen = run_centralized(
        device=device, num_epochs=central_epochs, batch_size=batch_size
    )
    t_fed, acc_fed = run_federated_sim(
        device=device,
        num_clients=num_clients,
        num_rounds=fed_rounds,
        batch_size=batch_size,
        local_epochs=local_epochs,
    )
    ratio = t_fed / t_cen if t_cen > 0 else float("nan")

    print(f"device={device}")
    print(f"centralized_epochs={central_epochs} time_s={t_cen:.3f} test_acc={acc_cen:.4f}")
    print(
        f"federated_rounds={fed_rounds} clients={num_clients} local_epochs={local_epochs} "
        f"time_s={t_fed:.3f} test_acc={acc_fed:.4f} time_vs_central={ratio:.2f}x"
    )


if __name__ == "__main__":
    main()
