import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader, Dataset
from typing import Optional, Tuple

DEFAULT_FEATURE_COLS = [
    "total_units",
    "total_revenue",
    "unique_customers",
    "avg_order_value",
    "rolling_4w_avg_units",
    "rolling_4w_std_units",
    "units_lag_1w",
    "units_lag_2w",
    "units_lag_4w",
    "wow_growth_pct",
    "norm_revenue",
    "is_jan_effect",
    "is_valentines_season",
]


class TimeSeriesDataset(Dataset):
    """Sliding window dataset for LSTM / sequence models."""

    def __init__(
        self,
        data: pd.DataFrame,
        target_col: str = "total_units",
        feature_cols: list | None = None,
        seq_len: int = 8,
        pred_len: int = 4,
        scaler: Optional[MinMaxScaler] = None,
    ):
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.target_col = target_col
        self.feature_cols = feature_cols or DEFAULT_FEATURE_COLS

        values = data[self.feature_cols].fillna(0).values
        if scaler is None:
            self.scaler = MinMaxScaler(feature_range=(0, 1))
            self.data = self.scaler.fit_transform(values).astype(np.float32)
        else:
            self.scaler = scaler
            self.data = scaler.transform(values).astype(np.float32)

        self.target_idx = self.feature_cols.index(target_col)

    def __len__(self) -> int:
        return len(self.data) - self.seq_len - self.pred_len + 1

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        x = self.data[idx : idx + self.seq_len]
        y = self.data[
            idx + self.seq_len : idx + self.seq_len + self.pred_len, self.target_idx
        ]
        return torch.tensor(x), torch.tensor(y)


class SalesForecastDataModule(pl.LightningDataModule):
    """PyTorch Lightning DataModule — manages train/val/test splits."""

    def __init__(
        self,
        data: pd.DataFrame,
        seq_len: int = 8,
        pred_len: int = 4,
        train_frac: float = 0.70,
        val_frac: float = 0.15,
        batch_size: int = 32,
        num_workers: int = 0,
        feature_cols: list | None = None,
    ):
        super().__init__()
        self.data = data
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.train_frac = train_frac
        self.val_frac = val_frac
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.feature_cols = feature_cols or DEFAULT_FEATURE_COLS

    def setup(self, stage: str | None = None):
        n = len(self.data)
        i_train = int(n * self.train_frac)
        i_val = int(n * (self.train_frac + self.val_frac))

        train_df = self.data.iloc[:i_train]
        val_df = self.data.iloc[i_train:i_val]
        test_df = self.data.iloc[i_val:]

        self.train_ds = TimeSeriesDataset(
            train_df,
            feature_cols=self.feature_cols,
            seq_len=self.seq_len,
            pred_len=self.pred_len,
        )
        self.val_ds = TimeSeriesDataset(
            val_df,
            feature_cols=self.feature_cols,
            seq_len=self.seq_len,
            pred_len=self.pred_len,
            scaler=self.train_ds.scaler,
        )
        self.test_ds = TimeSeriesDataset(
            test_df,
            feature_cols=self.feature_cols,
            seq_len=self.seq_len,
            pred_len=self.pred_len,
            scaler=self.train_ds.scaler,
        )

    def train_dataloader(self):
        return DataLoader(
            self.train_ds,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=False,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_ds,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_ds,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
        )


def load_sample_feature_mart(n_weeks: int = 80) -> pd.DataFrame:
    """
    Weekly feature mart based on the original ARIMA project (Jan–Mar 2017),
    extended to `n_weeks` for train/val/test splits with seq_len=8.
    """
    base_units = np.array(
        [1824, 1761, 2089, 1992, 2198, 2311, 2404, 2278, 2512, 2688, 2574, 2812],
        dtype=float,
    )
    rng = np.random.default_rng(42)
    weeks = np.arange(n_weeks)
    seasonal = 120 * np.sin(2 * np.pi * weeks / 52)
    trend = 15 * weeks
    noise = rng.normal(0, 40, n_weeks)
    total_units = np.maximum(
        1500,
        np.round(
            np.interp(weeks, np.linspace(0, n_weeks - 1, len(base_units)), base_units)
            + seasonal
            + trend
            + noise
        ),
    ).astype(int)

    total_revenue = np.round(total_units * rng.uniform(17.5, 21.0, n_weeks), 2)
    unique_customers = np.maximum(350, np.round(total_units / 4.5 + rng.normal(0, 20, n_weeks))).astype(int)
    avg_order_value = np.round(total_revenue / unique_customers, 2)

    s = pd.Series(total_units, dtype=float)
    rolling_4w_avg = s.rolling(4, min_periods=1).mean().shift(1).fillna(s.iloc[0])
    rolling_4w_std = s.rolling(4, min_periods=1).std().shift(1).fillna(0)

    rev_min, rev_max = total_revenue.min(), total_revenue.max()
    norm_revenue = (total_revenue - rev_min) / max(rev_max - rev_min, 1)

    return pd.DataFrame(
        {
            "total_units": total_units,
            "total_revenue": total_revenue,
            "unique_customers": unique_customers,
            "avg_order_value": avg_order_value,
            "rolling_4w_avg_units": rolling_4w_avg.values,
            "rolling_4w_std_units": rolling_4w_std.values,
            "units_lag_1w": s.shift(1).fillna(s.iloc[0]).values,
            "units_lag_2w": s.shift(2).fillna(s.iloc[0]).values,
            "units_lag_4w": s.shift(4).fillna(s.iloc[0]).values,
            "wow_growth_pct": (s.pct_change().fillna(0) * 100).round(2).values,
            "norm_revenue": norm_revenue.round(6),
            "is_jan_effect": ((weeks % 52) < 2).astype(int),
            "is_valentines_season": (((weeks % 52) >= 5) & ((weeks % 52) <= 7)).astype(int),
        }
    )
