from typing import Dict

import numpy as np


class WeightedEnsemble:
    """Weighted average ensemble of LSTM, TFT, and N-BEATS predictions."""

    def __init__(self, weights: Dict[str, float] | None = None):
        self.weights = weights or {"lstm": 0.30, "tft": 0.50, "nbeats": 0.20}

    def predict(self, predictions: Dict[str, np.ndarray]) -> np.ndarray:
        result = np.zeros_like(list(predictions.values())[0])
        for model_name, preds in predictions.items():
            result += self.weights[model_name] * preds
        return result

    def optimize_weights(
        self,
        val_predictions: Dict[str, np.ndarray],
        val_actuals: np.ndarray,
    ) -> Dict[str, float]:
        from scipy.optimize import minimize

        model_names = list(val_predictions.keys())
        pred_matrix = np.stack([val_predictions[m] for m in model_names], axis=1)

        def objective(w):
            w = np.abs(w) / np.abs(w).sum()
            ensemble = (pred_matrix * w).sum(axis=1)
            return np.mean(np.abs((val_actuals - ensemble) / val_actuals)) * 100

        x0 = np.array([1 / len(model_names)] * len(model_names))
        result = minimize(objective, x0, method="Nelder-Mead", options={"maxiter": 1000, "xatol": 1e-6})

        optimal_w = np.abs(result.x) / np.abs(result.x).sum()
        self.weights = dict(zip(model_names, optimal_w.tolist()))
        return self.weights
