#!/usr/bin/env python3
"""End-to-end: load features → train LSTM → log to MLflow."""

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import mlflow
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint

from src.features.dataset_builder import (
    DEFAULT_FEATURE_COLS,
    SalesForecastDataModule,
    load_sample_feature_mart,
)
from src.models.lstm_forecaster import LSTMForecaster


def main():
    os.makedirs("checkpoints", exist_ok=True)
    os.makedirs("data", exist_ok=True)
    os.makedirs("artifacts", exist_ok=True)

    max_epochs = int(os.environ.get("MAX_EPOCHS", "40"))

    df = load_sample_feature_mart()
    df.to_parquet("data/mart_ts_features.parquet", index=False)

    dm = SalesForecastDataModule(
        df,
        seq_len=8,
        pred_len=4,
        batch_size=16,
        num_workers=0,
        feature_cols=DEFAULT_FEATURE_COLS,
    )
    dm.setup()

    mlflow.set_tracking_uri(os.environ.get("MLFLOW_TRACKING_URI", "file:./mlruns"))
    mlflow.set_experiment("ecommerce-forecasting-v2")

    with mlflow.start_run(run_name="lstm-attention-v2"):
        model = LSTMForecaster(
            input_dim=len(DEFAULT_FEATURE_COLS),
            hidden_dim=128,
            num_layers=2,
            pred_len=4,
        )
        trainer = pl.Trainer(
            max_epochs=max_epochs,
            callbacks=[
                EarlyStopping("val/mape", patience=12, mode="min"),
                ModelCheckpoint(
                    "checkpoints/",
                    monitor="val/mape",
                    mode="min",
                    filename="lstm-best",
                ),
            ],
            accelerator="auto",
            devices=1,
            gradient_clip_val=0.5,
            enable_progress_bar=True,
        )
        trainer.fit(model, datamodule=dm)
        trainer.test(model, datamodule=dm, ckpt_path="best")

        mlflow.log_params(dict(model.hparams))
        mlflow.pytorch.log_model(model, "lstm_model", registered_model_name="lstm-ecommerce-forecaster")

        export_path = ROOT / "artifacts" / "lstm_model"
        mlflow.pytorch.save_model(model, str(export_path))

        val_mape = trainer.callback_metrics.get("val/mape")
        print(f"\nBest val MAPE: {val_mape}")
        print("\nTraining complete. Model logged to MLflow.")
        print("  MLflow UI:  mlflow ui --backend-store-uri ./mlruns")
        print("  API:        uvicorn src.serving.api:app --reload --port 8000")


if __name__ == "__main__":
    main()
