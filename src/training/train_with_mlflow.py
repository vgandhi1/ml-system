import os

import mlflow
import mlflow.pytorch
import optuna
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, LearningRateMonitor, ModelCheckpoint

from src.features.dataset_builder import DEFAULT_FEATURE_COLS
from src.models.lstm_forecaster import LSTMForecaster


def train_lstm_with_mlflow(
    datamodule: pl.LightningDataModule,
    trial: optuna.Trial | None = None,
    experiment_name: str = "ecommerce-forecasting",
    max_epochs: int = 100,
) -> dict:
    """Full training loop with MLflow tracking."""
    tracking_uri = os.environ.get("MLFLOW_TRACKING_URI", "file:./mlruns")
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment_name)

    hparams = {
        "input_dim": len(DEFAULT_FEATURE_COLS),
        "hidden_dim": trial.suggest_int("hidden_dim", 64, 256, step=64) if trial else 128,
        "num_layers": trial.suggest_int("num_layers", 1, 3) if trial else 2,
        "dropout": trial.suggest_float("dropout", 0.1, 0.4) if trial else 0.2,
        "lr": trial.suggest_float("lr", 1e-4, 1e-2, log=True) if trial else 1e-3,
        "weight_decay": trial.suggest_float("weight_decay", 1e-6, 1e-4, log=True) if trial else 1e-5,
    }
    if trial:
        datamodule.batch_size = trial.suggest_categorical("batch_size", [16, 32, 64])

    with mlflow.start_run():
        mlflow.log_params(hparams)
        mlflow.log_param("model_type", "LSTM-Attention-Bidirectional")
        mlflow.log_param("dataset", "UCI Online Retail Weekly")

        model = LSTMForecaster(**hparams)

        callbacks = [
            EarlyStopping(monitor="val/mape", patience=10, mode="min", verbose=True),
            ModelCheckpoint(
                monitor="val/mape",
                mode="min",
                save_top_k=1,
                dirpath="checkpoints/",
                filename="lstm-best-{epoch:02d}",
            ),
            LearningRateMonitor(logging_interval="epoch"),
        ]

        trainer = pl.Trainer(
            max_epochs=max_epochs,
            callbacks=callbacks,
            accelerator="auto",
            devices=1,
            gradient_clip_val=0.5,
            log_every_n_steps=1,
            enable_progress_bar=True,
        )

        trainer.fit(model, datamodule=datamodule)
        test_results = trainer.test(model, datamodule=datamodule, ckpt_path="best")

        metrics = {
            "best_val_mape": float(trainer.callback_metrics.get("val/mape", 0)),
        }
        if test_results:
            metrics["test_mape"] = float(test_results[0].get("test/mape", 0))
            metrics["test_rmse"] = float(test_results[0].get("test/rmse", 0))

        mlflow.log_metrics(metrics)
        mlflow.pytorch.log_model(
            model,
            artifact_path="model",
            registered_model_name="lstm-ecommerce-forecaster",
        )

        return {"mape": metrics.get("test_mape", metrics["best_val_mape"])}


def run_optuna_hpo(datamodule, n_trials: int = 30) -> dict:
    """Bayesian hyperparameter search with Optuna."""
    study = optuna.create_study(
        direction="minimize",
        sampler=optuna.samplers.TPESampler(seed=42),
        pruner=optuna.pruners.MedianPruner(n_warmup_steps=5),
    )

    study.optimize(
        lambda trial: train_lstm_with_mlflow(datamodule, trial, max_epochs=30)["mape"],
        n_trials=n_trials,
        n_jobs=1,
        show_progress_bar=True,
    )

    print(f"\nBest trial: MAPE = {study.best_value:.4f}%")
    print(f"Best params: {study.best_params}")
    return study.best_params
