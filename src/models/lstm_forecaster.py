import pytorch_lightning as pl
import torch
import torch.nn as nn
from torchmetrics import MeanAbsolutePercentageError, MeanSquaredError

from src.features.dataset_builder import DEFAULT_FEATURE_COLS


class AttentionLayer(nn.Module):
    """Scaled dot-product attention over LSTM hidden states."""

    def __init__(self, hidden_dim: int):
        super().__init__()
        self.attention = nn.Linear(hidden_dim, 1)

    def forward(self, lstm_out: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        scores = self.attention(lstm_out)
        weights = torch.softmax(scores, dim=1)
        context = (lstm_out * weights).sum(dim=1)
        return context, weights


class LSTMForecaster(pl.LightningModule):
    """
    Stacked bidirectional LSTM with attention for multi-step forecasting.
    Input → BiLSTM → Attention → Dropout → Dense → Output
    """

    def __init__(
        self,
        input_dim: int = len(DEFAULT_FEATURE_COLS),
        hidden_dim: int = 128,
        num_layers: int = 2,
        pred_len: int = 4,
        dropout: float = 0.2,
        lr: float = 1e-3,
        weight_decay: float = 1e-5,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.pred_len = pred_len

        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            batch_first=True,
            bidirectional=True,
        )
        self.attention = AttentionLayer(hidden_dim * 2)
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Sequential(
            nn.Linear(hidden_dim * 2, 64),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(64, pred_len),
        )

        self.train_mape = MeanAbsolutePercentageError()
        self.val_mape = MeanAbsolutePercentageError()
        self.val_rmse = MeanSquaredError(squared=False)
        self.test_mape = MeanAbsolutePercentageError()
        self.test_rmse = MeanSquaredError(squared=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        lstm_out, _ = self.lstm(x)
        context, _ = self.attention(lstm_out)
        context = self.dropout(context)
        return self.head(context)

    def _shared_step(self, batch, stage: str):
        x, y = batch
        y_hat = self(x)
        loss = nn.functional.huber_loss(y_hat, y, delta=1.0)
        self.log(f"{stage}/loss", loss, on_epoch=True, prog_bar=True)

        if stage == "train":
            self.train_mape(y_hat, y)
            self.log("train/mape", self.train_mape, on_epoch=True)
        elif stage == "val":
            self.val_mape(y_hat, y)
            self.val_rmse(y_hat, y)
            self.log("val/mape", self.val_mape, on_epoch=True, prog_bar=True)
            self.log("val/rmse", self.val_rmse, on_epoch=True)
        elif stage == "test":
            self.test_mape(y_hat, y)
            self.test_rmse(y_hat, y)
            self.log("test/mape", self.test_mape, on_epoch=True)
            self.log("test/rmse", self.test_rmse, on_epoch=True)

        return loss

    def training_step(self, batch, _):
        return self._shared_step(batch, "train")

    def validation_step(self, batch, _):
        return self._shared_step(batch, "val")

    def test_step(self, batch, _):
        return self._shared_step(batch, "test")

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.hparams.lr,
            weight_decay=self.hparams.weight_decay,
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50)
        return {"optimizer": optimizer, "lr_scheduler": scheduler}
