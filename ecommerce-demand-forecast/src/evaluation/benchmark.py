from typing import Dict

import numpy as np
import pandas as pd


def full_evaluation_report(
    actuals: np.ndarray,
    model_predictions: Dict[str, np.ndarray],
) -> pd.DataFrame:
    """Generate evaluation table: MAPE, SMAPE, MAE, RMSE, R², directional accuracy."""
    rows = []
    for model_name, preds in model_predictions.items():
        a, p = np.array(actuals), np.array(preds)
        errors = a - p
        pct_err = errors / np.maximum(np.abs(a), 1e-8)

        mape = np.mean(np.abs(pct_err)) * 100
        smape = np.mean(2 * np.abs(errors) / (np.abs(a) + np.abs(p) + 1e-8)) * 100
        mae = np.mean(np.abs(errors))
        rmse = np.sqrt(np.mean(errors**2))
        r2 = 1 - np.sum(errors**2) / np.sum((a - a.mean()) ** 2)

        dir_acc = np.mean(np.sign(np.diff(a)) == np.sign(np.diff(p))) * 100 if len(a) > 1 else 0.0

        rows.append(
            {
                "Model": model_name,
                "MAPE (%)": round(mape, 2),
                "SMAPE (%)": round(smape, 2),
                "MAE": round(mae, 1),
                "RMSE": round(rmse, 1),
                "R²": round(r2, 4),
                "Dir Acc %": round(dir_acc, 1),
            }
        )

    return pd.DataFrame(rows).sort_values("MAPE (%)")
