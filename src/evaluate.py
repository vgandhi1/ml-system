"""
Evaluate saved model on a parquet dataset (same schema as training).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    roc_auc_score,
)
import xgboost as xgb

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import ARTIFACTS_DIR, DATA_DIR, FEATURE_COLUMNS, LABEL_COLUMN


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data",
        type=Path,
        default=DATA_DIR / "transactions.parquet",
    )
    parser.add_argument("--artifacts", type=Path, default=ARTIFACTS_DIR)
    args = parser.parse_args()

    model_path = args.artifacts / "model.xgb.json"
    scaler_path = args.artifacts / "scaler.joblib"
    meta_path = args.artifacts / "metadata.json"
    for p in (model_path, scaler_path, meta_path):
        if not p.is_file():
            raise SystemExit(f"Missing artifact: {p}. Train first: python -m src.train")

    booster = xgb.XGBClassifier()
    booster.load_model(str(model_path))
    scaler = joblib.load(scaler_path)
    with open(meta_path, encoding="utf-8") as f:
        meta = json.load(f)
    threshold = float(meta["decision_threshold"])

    df = pd.read_parquet(args.data)
    X = df[FEATURE_COLUMNS].astype(np.float64)
    y = df[LABEL_COLUMN].astype(np.int32).values
    proba = booster.predict_proba(scaler.transform(X))[:, 1]
    pred = (proba >= threshold).astype(int)

    print(f"ROC-AUC: {roc_auc_score(y, proba):.4f}")
    print(f"PR-AUC:  {average_precision_score(y, proba):.4f}")
    print(f"Threshold (from metadata): {threshold:.4f}")
    tn, fp, fn, tp = confusion_matrix(y, pred).ravel()
    print(f"Confusion [tn fp; fn tp]: {tn} {fp} {fn} {tp}")


if __name__ == "__main__":
    main()
