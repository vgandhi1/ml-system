"""
Train Sentinel-Stream XGBoost classifier; write model + scaler + metadata to artifacts/.
Optional MLflow experiment tracking (--mlflow or MLFLOW_TRACKING_URI).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import xgboost as xgb

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import (
    ARTIFACTS_DIR,
    DATA_DIR,
    FEATURE_COLUMNS,
    LABEL_COLUMN,
)
from src.mlflow_tracking import (
    TrainRunPayload,
    log_training_run,
    resolve_tracking_uri,
)


def _default_threshold(y_true: np.ndarray, proba: np.ndarray, target_fpr: float) -> float:
    """Pick threshold targeting approximate FPR on validation negatives."""
    neg_mask = y_true == 0
    pos_scores = proba[neg_mask]
    if pos_scores.size == 0:
        return 0.5
    q = min(0.9999, max(0.0001, 1.0 - target_fpr))
    return float(np.quantile(pos_scores, q))


def _write_lineage(artifacts_dir: Path) -> None:
    """Lightweight lineage for audit (no PII)."""
    lineage = {
        "git_sha": os.environ.get("GITHUB_SHA", ""),
        "github_run_id": os.environ.get("GITHUB_RUN_ID", ""),
        "github_workflow": os.environ.get("GITHUB_WORKFLOW", ""),
        "feature_columns": list(FEATURE_COLUMNS),
    }
    (artifacts_dir / "lineage.json").write_text(
        json.dumps(lineage, indent=2),
        encoding="utf-8",
    )


def train(
    data_path: Path,
    artifacts_dir: Path,
    seed: int,
    target_fpr: float,
) -> TrainRunPayload:
    df = pd.read_parquet(data_path)
    missing = [c for c in FEATURE_COLUMNS + [LABEL_COLUMN] if c not in df.columns]
    if missing:
        raise ValueError(f"Dataset missing columns: {missing}")

    X = df[FEATURE_COLUMNS].astype(np.float64)
    y = df[LABEL_COLUMN].astype(np.int32).values

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=seed, stratify=y
    )

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_val_s = scaler.transform(X_val)

    neg = int((y_train == 0).sum())
    pos = int((y_train == 1).sum())
    scale_pos_weight = max(neg / max(pos, 1), 1.0)

    xgb_kw: dict = {
        "n_estimators": 200,
        "max_depth": 6,
        "learning_rate": 0.08,
        "subsample": 0.9,
        "colsample_bytree": 0.9,
        "reg_lambda": 1.0,
        "objective": "binary:logistic",
        "eval_metric": "aucpr",
        "scale_pos_weight": scale_pos_weight,
        "random_state": seed,
        "n_jobs": -1,
        "early_stopping_rounds": 25,
    }
    clf = xgb.XGBClassifier(**xgb_kw)
    clf.fit(
        X_train_s,
        y_train,
        eval_set=[(X_val_s, y_val)],
        verbose=False,
    )

    val_proba = clf.predict_proba(X_val_s)[:, 1]
    roc_auc = float(roc_auc_score(y_val, val_proba))
    pr_auc = float(average_precision_score(y_val, val_proba))
    threshold = _default_threshold(y_val, val_proba, target_fpr)

    artifacts_dir.mkdir(parents=True, exist_ok=True)
    model_path = artifacts_dir / "model.xgb.json"
    scaler_path = artifacts_dir / "scaler.joblib"
    features_path = artifacts_dir / "feature_columns.json"
    meta_path = artifacts_dir / "metadata.json"

    clf.save_model(str(model_path))
    joblib.dump(scaler, scaler_path)
    with open(features_path, "w", encoding="utf-8") as f:
        json.dump(FEATURE_COLUMNS, f, indent=2)
    meta = {
        "roc_auc_val": roc_auc,
        "pr_auc_val": pr_auc,
        "decision_threshold": threshold,
        "target_fpr_for_threshold": target_fpr,
        "n_features": len(FEATURE_COLUMNS),
        "label": LABEL_COLUMN,
    }
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    _write_lineage(artifacts_dir)

    xgb_logged_params = {
        "n_estimators": xgb_kw["n_estimators"],
        "max_depth": xgb_kw["max_depth"],
        "learning_rate": xgb_kw["learning_rate"],
        "subsample": xgb_kw["subsample"],
        "colsample_bytree": xgb_kw["colsample_bytree"],
        "reg_lambda": xgb_kw["reg_lambda"],
        "objective": str(xgb_kw["objective"]),
    }

    payload = TrainRunPayload(
        data_path=data_path,
        artifacts_dir=artifacts_dir,
        seed=seed,
        target_fpr=target_fpr,
        n_train=int(len(y_train)),
        n_val=int(len(y_val)),
        scale_pos_weight=scale_pos_weight,
        roc_auc_val=roc_auc,
        pr_auc_val=pr_auc,
        decision_threshold=threshold,
        xgb_params=xgb_logged_params,
        feature_columns=list(FEATURE_COLUMNS),
    )

    print(f"Saved model to {model_path}")
    print(f"Val ROC-AUC={roc_auc:.4f} PR-AUC={pr_auc:.4f} threshold={threshold:.4f}")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data",
        type=Path,
        default=DATA_DIR / "transactions.parquet",
    )
    parser.add_argument("--artifacts", type=Path, default=ARTIFACTS_DIR)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--target-fpr",
        type=float,
        default=0.01,
        help="Approximate false-positive rate on validation negatives for default threshold.",
    )
    parser.add_argument(
        "--mlflow",
        action="store_true",
        help="Log this run to MLflow using local mlruns/ unless MLFLOW_TRACKING_URI is set.",
    )
    parser.add_argument(
        "--no-mlflow",
        action="store_true",
        help="Disable MLflow even when MLFLOW_TRACKING_URI is set.",
    )
    parser.add_argument(
        "--mlflow-experiment",
        default="sentinel_stream_classifier",
        help="MLflow experiment name.",
    )
    parser.add_argument(
        "--mlflow-tracking-uri",
        default=None,
        help="Override MLFLOW_TRACKING_URI for this process only.",
    )
    args = parser.parse_args()
    if not args.data.is_file():
        raise SystemExit(
            f"Data not found: {args.data}. Run: python scripts/generate_data.py"
        )

    payload = train(args.data, args.artifacts, args.seed, args.target_fpr)

    tracking_uri = resolve_tracking_uri(
        use_mlflow=args.mlflow,
        no_mlflow=args.no_mlflow,
        project_root=ROOT,
        explicit_uri=(args.mlflow_tracking_uri or "").strip() or None,
    )
    if tracking_uri:
        run_id = log_training_run(
            payload,
            tracking_uri,
            experiment_name=args.mlflow_experiment,
        )
        if run_id:
            print(f"MLflow run_id={run_id}")


if __name__ == "__main__":
    main()
