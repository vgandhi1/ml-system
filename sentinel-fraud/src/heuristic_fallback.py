"""
High-speed heuristic fallback when ML inference exceeds latency budget (circuit breaker).
Uses only aggregate-safe signals already on the request — no external DB calls.
"""

from __future__ import annotations

from pydantic import BaseModel


def heuristic_fraud_score(body: BaseModel) -> float:
    """
    Return a synthetic fraud probability in [0, 1] for payment-path continuity.
    Tuned to be conservative on obvious velocity / amount spikes (not a replacement for ML).
    """
    amount = float(getattr(body, "amount", 0))
    txn_24h = float(getattr(body, "txn_count_24h", 0))
    merchants = float(getattr(body, "distinct_merchants_24h", 0))
    channel = float(getattr(body, "channel_online", 0))
    days_since = float(getattr(body, "days_since_last_txn", 99))

    score = 0.05
    if amount > 2500:
        score += 0.18
    if amount > 8000:
        score += 0.15
    if txn_24h > 15:
        score += 0.22
    if merchants > 8 and txn_24h > 8:
        score += 0.12
    if days_since < 0.05 and amount > 200:
        score += 0.15
    if channel >= 0.5 and amount > 1500:
        score += 0.08
    return max(0.0, min(1.0, score))
