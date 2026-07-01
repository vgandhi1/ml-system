"""
Generate a synthetic fraud-labeled dataset for offline training.
No external downloads; safe for CI and local demos.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import DATA_DIR, FEATURE_COLUMNS, LABEL_COLUMN


def _sample_legit(rng: np.random.Generator, n: int) -> pd.DataFrame:
    amount = rng.lognormal(mean=3.5, sigma=1.0, size=n)
    hour_of_day = rng.integers(0, 24, size=n)
    days_since_last_txn = rng.exponential(scale=2.0, size=n).clip(0.01, 30)
    txn_count_24h = rng.poisson(lam=2.0, size=n)
    distinct_merchants_24h = np.minimum(
        txn_count_24h, rng.poisson(lam=1.5, size=n) + 1
    )
    avg_amount_7d = rng.lognormal(mean=3.2, sigma=0.8, size=n)
    channel_online = rng.binomial(1, 0.35, size=n)
    df = pd.DataFrame(
        {
            "amount": amount,
            "hour_of_day": hour_of_day.astype(float),
            "days_since_last_txn": days_since_last_txn,
            "txn_count_24h": txn_count_24h.astype(float),
            "distinct_merchants_24h": distinct_merchants_24h.astype(float),
            "avg_amount_7d": avg_amount_7d,
            "channel_online": channel_online.astype(float),
        }
    )
    df[LABEL_COLUMN] = 0
    return df


def _sample_fraud(rng: np.random.Generator, n: int) -> pd.DataFrame:
    # Fraudulent rows: larger amounts, burst activity, odd hours more often.
    amount = rng.lognormal(mean=5.2, sigma=1.1, size=n)
    hour_of_day = rng.integers(0, 24, size=n)
    fraud_night = rng.random(size=n) < 0.45
    hour_of_day = np.where(fraud_night, rng.integers(0, 6, size=n), hour_of_day)
    days_since_last_txn = rng.exponential(scale=0.15, size=n).clip(0.01, 7)
    txn_count_24h = rng.poisson(lam=14.0, size=n)
    distinct_merchants_24h = np.minimum(
        txn_count_24h, rng.poisson(lam=8.0, size=n) + rng.integers(1, 4, size=n)
    )
    avg_amount_7d = rng.lognormal(mean=3.0, sigma=0.9, size=n)
    channel_online = rng.binomial(1, 0.72, size=n)
    df = pd.DataFrame(
        {
            "amount": amount,
            "hour_of_day": hour_of_day.astype(float),
            "days_since_last_txn": days_since_last_txn,
            "txn_count_24h": txn_count_24h.astype(float),
            "distinct_merchants_24h": distinct_merchants_24h.astype(float),
            "avg_amount_7d": avg_amount_7d,
            "channel_online": channel_online.astype(float),
        }
    )
    df[LABEL_COLUMN] = 1
    return df


def build_dataset(
    n_legit: int,
    n_fraud: int,
    seed: int,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    legit = _sample_legit(rng, n_legit)
    fraud = _sample_fraud(rng, n_fraud)
    out = pd.concat([legit, fraud], ignore_index=True)
    out = out.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    # Light feature noise + rare label flip so offline metrics are not trivially 1.0.
    for c in FEATURE_COLUMNS:
        sigma = float(out[c].std() * 0.08 + 1e-6)
        out[c] = out[c] + rng.normal(0, sigma, size=len(out))
        out[c] = out[c].clip(lower=0)
    flip = rng.random(len(out)) < 0.004
    out.loc[flip, LABEL_COLUMN] = 1 - out.loc[flip, LABEL_COLUMN]
    for c in FEATURE_COLUMNS:
        if c not in out.columns:
            raise KeyError(f"Missing feature column: {c}")
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate synthetic fraud dataset.")
    parser.add_argument("--n-legit", type=int, default=48_000)
    parser.add_argument("--n-fraud", type=int, default=2_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output parquet path (default: data/transactions.parquet)",
    )
    args = parser.parse_args()
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    output = args.output or (DATA_DIR / "transactions.parquet")
    output.parent.mkdir(parents=True, exist_ok=True)
    df = build_dataset(args.n_legit, args.n_fraud, args.seed)
    df.to_parquet(output, index=False)
    fraud_rate = df[LABEL_COLUMN].mean()
    print(f"Wrote {len(df)} rows to {output} (fraud_rate={fraud_rate:.4f})")


if __name__ == "__main__":
    main()
