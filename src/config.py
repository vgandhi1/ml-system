from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
ARTIFACTS_DIR = ROOT / "artifacts"

FEATURE_COLUMNS = [
    "amount",
    "hour_of_day",
    "days_since_last_txn",
    "txn_count_24h",
    "distinct_merchants_24h",
    "avg_amount_7d",
    "channel_online",
]

LABEL_COLUMN = "is_fraud"
