"""
MLflow experiment tracking for training runs (optional).
Enable with --mlflow or set MLFLOW_TRACKING_URI to a server or file: URI.
Does not log raw transaction rows or PII — only params, aggregate metrics, and artifact files.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass
class TrainRunPayload:
    """Summary of a training job suitable for experiment tracking."""

    data_path: Path
    artifacts_dir: Path
    seed: int
    target_fpr: float
    n_train: int
    n_val: int
    scale_pos_weight: float
    roc_auc_val: float
    pr_auc_val: float
    decision_threshold: float
    xgb_params: dict[str, Any]
    feature_columns: list[str]


def _file_tracking_uri(project_mlruns: Path) -> str:
    return project_mlruns.resolve().as_uri()


def resolve_tracking_uri(
    *,
    use_mlflow: bool,
    no_mlflow: bool,
    project_root: Path,
    explicit_uri: str | None,
) -> str | None:
    """Return tracking URI to use, or None if MLflow logging is disabled."""
    if no_mlflow:
        return None
    env_uri = (os.environ.get("MLFLOW_TRACKING_URI") or "").strip()
    if explicit_uri:
        return explicit_uri
    if env_uri:
        return env_uri
    if use_mlflow:
        return _file_tracking_uri(project_root / "mlruns")
    return None


def log_training_run(
    payload: TrainRunPayload,
    tracking_uri: str,
    experiment_name: str,
) -> str | None:
    """
    Log one MLflow run. Returns run_id if successful, else None.
    """
    import mlflow

    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment_name)

    tags = {
        "domain": "sentinel_stream",
        "risk_tier": "real_time_financial_defense",
    }
    if os.environ.get("GITHUB_REPOSITORY"):
        tags["github.repository"] = os.environ["GITHUB_REPOSITORY"]
    if os.environ.get("GITHUB_SHA"):
        tags["github.sha"] = os.environ["GITHUB_SHA"]
    if os.environ.get("GITHUB_WORKFLOW"):
        tags["github.workflow"] = os.environ["GITHUB_WORKFLOW"]

    with mlflow.start_run(run_name="train_sentinel_stream") as run:
        mlflow.set_tags(tags)
        mlflow.log_params(
            {
                "seed": payload.seed,
                "target_fpr": payload.target_fpr,
                "n_train": payload.n_train,
                "n_val": payload.n_val,
                "scale_pos_weight": round(payload.scale_pos_weight, 4),
                "data_path_suffix": payload.data_path.name,
            }
        )
        for k, v in payload.xgb_params.items():
            if v is None or isinstance(v, (str, int, float, bool)):
                mlflow.log_param(f"xgb__{k}", v)

        mlflow.log_metrics(
            {
                "val_roc_auc": payload.roc_auc_val,
                "val_pr_auc": payload.pr_auc_val,
                "decision_threshold": payload.decision_threshold,
            }
        )

        art = payload.artifacts_dir
        mlflow.log_artifact(str(art / "model.xgb.json"), artifact_path="model_bundle")
        mlflow.log_artifact(str(art / "scaler.joblib"), artifact_path="model_bundle")
        mlflow.log_artifact(str(art / "feature_columns.json"), artifact_path="model_bundle")
        mlflow.log_artifact(str(art / "metadata.json"), artifact_path="model_bundle")
        lineage_path = art / "lineage.json"
        if lineage_path.is_file():
            mlflow.log_artifact(str(lineage_path), artifact_path="model_bundle")

        run_id = run.info.run_id
    return run_id
