"""
Sentinel-Stream inference: production ML score with optional circuit-breaker heuristic
and optional shadow (challenger) model for live comparison — no user-facing behavior change
from shadow scores alone.
"""

from __future__ import annotations

import json
import logging
import os
import time
from pathlib import Path

import joblib
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import xgboost as xgb

from src.heuristic_fallback import heuristic_fraud_score

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

ROOT = Path(__file__).resolve().parents[1]
_ARTIFACT_ENV = (
    os.environ.get("SENTINEL_ARTIFACTS_DIR")
    or os.environ.get("FRAUD_ARTIFACTS_DIR")
    or str(ROOT / "artifacts")
)
ARTIFACTS = Path(_ARTIFACT_ENV)
_SHADOW_ENV = os.environ.get("SENTINEL_SHADOW_ARTIFACTS_DIR", "").strip()
SHADOW_ARTIFACTS = Path(_SHADOW_ENV) if _SHADOW_ENV else None

_CIRCUIT_MS = float(os.environ.get("SENTINEL_CIRCUIT_THRESHOLD_MS", "60"))
_INJECT_MS = float(os.environ.get("SENTINEL_INJECT_LATENCY_MS", "0"))


class ScoreRequest(BaseModel):
    amount: float = Field(..., ge=0)
    hour_of_day: float = Field(..., ge=0, le=23)
    days_since_last_txn: float = Field(..., ge=0)
    txn_count_24h: float = Field(..., ge=0)
    distinct_merchants_24h: float = Field(..., ge=0)
    avg_amount_7d: float = Field(..., ge=0)
    channel_online: float = Field(..., ge=0, le=1)


class ScoreResponse(BaseModel):
    fraud_probability: float
    decision: str
    threshold: float
    circuit_breaker_state: str = Field(
        ...,
        description="closed = ML path; open = heuristic fallback due to latency budget",
    )
    shadow_fraud_probability: float | None = Field(
        default=None,
        description="Challenger model score when SENTINEL_SHADOW_ARTIFACTS_DIR is configured",
    )


class _ModelBundle:
    __slots__ = ("booster", "scaler", "feature_columns", "threshold")

    def __init__(
        self,
        booster: xgb.XGBClassifier,
        scaler,
        feature_columns: list[str],
        threshold: float,
    ) -> None:
        self.booster = booster
        self.scaler = scaler
        self.feature_columns = feature_columns
        self.threshold = threshold


_primary: _ModelBundle | None = None
_shadow: _ModelBundle | None = None


def _load_bundle(artifacts_dir: Path) -> _ModelBundle:
    model_path = artifacts_dir / "model.xgb.json"
    scaler_path = artifacts_dir / "scaler.joblib"
    features_path = artifacts_dir / "feature_columns.json"
    meta_path = artifacts_dir / "metadata.json"
    for p in (model_path, scaler_path, features_path, meta_path):
        if not p.is_file():
            raise FileNotFoundError(str(p))
    booster = xgb.XGBClassifier()
    booster.load_model(str(model_path))
    scaler = joblib.load(scaler_path)
    with open(features_path, encoding="utf-8") as f:
        feature_columns = json.load(f)
    with open(meta_path, encoding="utf-8") as f:
        meta = json.load(f)
    threshold = float(meta["decision_threshold"])
    return _ModelBundle(booster, scaler, feature_columns, threshold)


def _row_vector(body: ScoreRequest, feature_columns: list[str]) -> np.ndarray:
    return np.array([[getattr(body, name) for name in feature_columns]], dtype=np.float64)


def _predict_proba_ms(bundle: _ModelBundle, body: ScoreRequest) -> tuple[float, float]:
    row = _row_vector(body, bundle.feature_columns)
    x = bundle.scaler.transform(row)
    t0 = time.perf_counter()
    proba = float(bundle.booster.predict_proba(x)[0, 1])
    elapsed_ms = (time.perf_counter() - t0) * 1000.0
    return proba, elapsed_ms


app = FastAPI(
    title="Sentinel-Stream Scoring API",
    version="0.2.0",
    description="Real-time fraud scoring with optional circuit breaker and shadow challenger.",
)


@app.on_event("startup")
def startup() -> None:
    global _primary, _shadow
    _primary = _load_bundle(ARTIFACTS)
    logger.info("Loaded production model from %s", ARTIFACTS)
    if SHADOW_ARTIFACTS and SHADOW_ARTIFACTS.is_dir():
        try:
            _shadow = _load_bundle(SHADOW_ARTIFACTS)
            logger.info("Shadow (challenger) model loaded from %s", SHADOW_ARTIFACTS)
        except OSError:
            logger.exception("Failed to load shadow artifacts from %s", SHADOW_ARTIFACTS)
            _shadow = None
    else:
        _shadow = None


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok", "service": "sentinel-stream"}


@app.post("/v1/score", response_model=ScoreResponse, response_model_exclude_none=True)
def score(body: ScoreRequest) -> ScoreResponse:
    if _primary is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    if _INJECT_MS > 0:
        time.sleep(_INJECT_MS / 1000.0)

    force_open = os.environ.get("SENTINEL_FORCE_CIRCUIT_OPEN", "").lower() in (
        "1",
        "true",
        "yes",
    )

    proba_ml, ml_ms = _predict_proba_ms(_primary, body)
    circuit_open = force_open or ml_ms > _CIRCUIT_MS

    if circuit_open:
        proba = heuristic_fraud_score(body)
        state = "open"
        logger.warning(
            "circuit_breaker_open ml_latency_ms=%.2f threshold_ms=%.0f",
            ml_ms,
            _CIRCUIT_MS,
        )
    else:
        proba = proba_ml
        state = "closed"

    decision = "review" if proba >= _primary.threshold else "approve"

    shadow_p: float | None = None
    if _shadow is not None:
        try:
            shadow_p, _ = _predict_proba_ms(_shadow, body)
        except Exception:
            logger.exception("Shadow inference failed; omitting shadow score")

    return ScoreResponse(
        fraud_probability=proba,
        decision=decision,
        threshold=_primary.threshold,
        circuit_breaker_state=state,
        shadow_fraud_probability=shadow_p,
    )
