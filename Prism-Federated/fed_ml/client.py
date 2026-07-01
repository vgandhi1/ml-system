"""Flower NumPyClient wrapping local MNIST training."""

from __future__ import annotations

from collections import OrderedDict

import flwr as fl
import numpy as np
import torch
from flwr.common import Scalar
from torch import nn
from torch.utils.data import DataLoader

from fed_ml.data import load_client_dataloaders
from fed_ml.model import MnistCNN, get_device


class MnistFlowerClient(fl.client.NumPyClient):
    def __init__(
        self,
        *,
        client_id: int,
        num_clients: int,
        batch_size: int,
        local_epochs: int,
    ) -> None:
        self.device = get_device()
        self.model = MnistCNN().to(self.device)
        self.train_loader, self.test_loader = load_client_dataloaders(
            num_clients=num_clients,
            client_id=client_id,
            batch_size=batch_size,
        )
        self.criterion = nn.CrossEntropyLoss()
        self.local_epochs = local_epochs

    def get_parameters(self, config: dict[str, Scalar]) -> list[np.ndarray]:
        del config
        return [val.cpu().numpy() for _, val in self.model.state_dict().items()]

    def set_parameters(self, parameters: list[np.ndarray]) -> None:
        keys = list(self.model.state_dict().keys())
        state_dict = OrderedDict(
            {
                k: torch.tensor(v, device=self.device)
                for k, v in zip(keys, parameters, strict=True)
            }
        )
        self.model.load_state_dict(state_dict, strict=True)

    def fit(
        self, parameters: list[np.ndarray], config: dict[str, Scalar]
    ) -> tuple[list[np.ndarray], int, dict[str, Scalar]]:
        del config
        self.set_parameters(parameters)
        self.model.train()
        opt = torch.optim.SGD(self.model.parameters(), lr=0.01, momentum=0.9)
        for _ in range(self.local_epochs):
            for x, y in self.train_loader:
                x = x.to(self.device)
                y = y.to(self.device)
                opt.zero_grad()
                loss = self.criterion(self.model(x), y)
                loss.backward()
                opt.step()
        n = len(self.train_loader.dataset)
        return self.get_parameters({}), n, {}

    def evaluate(
        self, parameters: list[np.ndarray], config: dict[str, Scalar]
    ) -> tuple[float, int, dict[str, Scalar]]:
        del config
        self.set_parameters(parameters)
        self.model.eval()
        loss_sum = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for x, y in self.test_loader:
                x = x.to(self.device)
                y = y.to(self.device)
                logits = self.model(x)
                loss_sum += self.criterion(logits, y).item() * y.size(0)
                pred = logits.argmax(dim=1)
                correct += (pred == y).sum().item()
                total += y.size(0)
        return loss_sum / max(total, 1), total, {"accuracy": correct / max(total, 1)}
