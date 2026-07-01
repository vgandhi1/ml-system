import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional

import numpy as np
import torch
from fastapi import BackgroundTasks, FastAPI, HTTPException
from pydantic import BaseModel, Field

from src.features.dataset_builder import DEFAULT_FEATURE_COLS, load_sample_feature_mart
from src.models.lstm_forecaster import LSTMForecaster

app = FastAPI(
    title="eCommerce Sales Forecast API",
    description="LSTM + TFT ensemble forecast service",
    version="2.0.0",
)

_model: Optional[LSTMForecaster] = None
_cache = None
CACHE_TTL = 3600
SEQ_LEN = 8
PRED_LEN = 4


def _get_redis():
    global _cache
    if _cache is not None:
        return _cache
    try:
        import redis

        url = os.environ.get("REDIS_URL", "redis://localhost:6379/0")
        _cache = redis.from_url(url, decode_responses=True)
        _cache.ping()
    except Exception:
        _cache = False
    return _cache if _cache is not False else None


def _load_model() -> LSTMForecaster:
    global _model
    if _model is not None:
        return _model

    uri = os.environ.get("MLFLOW_MODEL_URI")
    if uri:
        try:
            import mlflow.pytorch

            _model = mlflow.pytorch.load_model(uri)
            _model.eval()
            return _model
        except Exception:
            pass

    checkpoint = os.environ.get("MODEL_CHECKPOINT", "checkpoints/lstm-best.ckpt")
    if Path(checkpoint).exists():
        _model = LSTMForecaster.load_from_checkpoint(checkpoint)
        _model.eval()
        return _model

    artifacts = Path("artifacts/lstm_model")
    if artifacts.exists():
        try:
            import mlflow.pytorch

            _model = mlflow.pytorch.load_model(str(artifacts))
            _model.eval()
            return _model
        except Exception:
            pass

    _model = LSTMForecaster()
    _model.eval()
    return _model


def _build_input_tensor(history: List[float]) -> torch.Tensor:
    """Build (1, seq_len, n_features) tensor from recent history + feature mart."""
    df = load_sample_feature_mart()
    row = df.iloc[-1].copy()
    if len(history) >= SEQ_LEN:
        recent = history[-SEQ_LEN:]
        for i, val in enumerate(recent):
            df.loc[df.index[-SEQ_LEN + i], "total_units"] = val

    values = df[DEFAULT_FEATURE_COLS].fillna(0).values.astype(np.float32)
    from sklearn.preprocessing import MinMaxScaler

    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled = scaler.fit_transform(values)
    x = scaled[-SEQ_LEN:]
    return torch.tensor(x, dtype=torch.float32).unsqueeze(0)


class ForecastRequest(BaseModel):
    series_id: str = Field(..., description="Product/segment identifier")
    history: List[float] = Field(..., min_length=8, description="Historical weekly units (min 8 weeks)")
    features: dict = Field(default_factory=dict, description="Optional covariate features")
    horizon: int = Field(default=4, ge=1, le=12, description="Forecast weeks")


class ForecastResponse(BaseModel):
    series_id: str
    forecast_at: str
    horizon: int
    point_forecast: List[float]
    lower_95: List[float]
    upper_95: List[float]
    model_version: str


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": _model is not None}


@app.post("/forecast", response_model=ForecastResponse)
async def forecast(req: ForecastRequest, background_tasks: BackgroundTasks):
    cache_key = hashlib.md5(
        json.dumps(
            {"sid": req.series_id, "hist": req.history[-8:], "h": req.horizon},
            sort_keys=True,
        ).encode()
    ).hexdigest()

    redis_client = _get_redis()
    if redis_client:
        cached = redis_client.get(f"forecast:{cache_key}")
        if cached:
            return ForecastResponse(**json.loads(cached))

    try:
        model = _load_model()
        x_tensor = _build_input_tensor(req.history)

        with torch.no_grad():
            pred = model(x_tensor).squeeze().numpy()

        model.train()
        mc_preds = []
        for _ in range(20):
            with torch.no_grad():
                mc_preds.append(model(x_tensor).squeeze().numpy())
        model.eval()

        mc_array = np.stack(mc_preds)
        lower_95 = np.percentile(mc_array, 2.5, axis=0).tolist()
        upper_95 = np.percentile(mc_array, 97.5, axis=0).tolist()

        if np.ndim(pred) == 0:
            pred = [float(pred)] * req.horizon
        elif len(np.atleast_1d(pred)) < req.horizon:
            pred = np.resize(pred, req.horizon)

        response = ForecastResponse(
            series_id=req.series_id,
            forecast_at=datetime.now(timezone.utc).isoformat(),
            horizon=req.horizon,
            point_forecast=np.atleast_1d(pred)[: req.horizon].tolist(),
            lower_95=lower_95[: req.horizon],
            upper_95=upper_95[: req.horizon],
            model_version=os.environ.get("MODEL_VERSION", "lstm-v2.0"),
        )

        if redis_client:
            background_tasks.add_task(
                redis_client.setex,
                f"forecast:{cache_key}",
                CACHE_TTL,
                response.model_dump_json(),
            )

        return response

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e
