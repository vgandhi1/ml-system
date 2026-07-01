import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.models.lstm_forecaster import LSTMForecaster


def test_lstm_forward_pass():
    model = LSTMForecaster(input_dim=13, pred_len=4)
    x = torch.randn(2, 8, 13)
    out = model(x)
    assert out.shape == (2, 4)
