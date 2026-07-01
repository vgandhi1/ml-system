# MLOps — Sentinel-Stream

This folder holds **governance and automation** assets for a fintech-style ML lifecycle: experiment tracking, promotion rules, and documentation templates.

## Components

| Asset | Purpose |
| --- | --- |
| [`promotion_policy.yaml`](./promotion_policy.yaml) | Minimum validation **PR-AUC** and **ROC-AUC** before a build is considered safe to promote (CI enforces this). |
| [`model_card_template.md`](./model_card_template.md) | Fill-in template for **model risk**, limitations, and monitoring hooks (SR 11-7 / internal policy alignment). |
| [`docker-compose.mlflow.yml`](./docker-compose.mlflow.yml) | Optional local **MLflow Tracking** server + UI (Postgres backend). |

## Experiment tracking (MLflow)

Training logs **parameters**, **aggregate validation metrics**, and **artifact bundle** (XGBoost JSON, scaler, metadata, lineage). It does **not** log raw transaction payloads or account identifiers.

**Local file store (default with `--mlflow`):**

```bash
cd Sentinel-Stream
source .venv/bin/activate
export MLFLOW_TRACKING_URI=""   # optional; --mlflow uses ./mlruns
python -m src.train --mlflow --mlflow-experiment sentinel_stream_classifier
```

**Central tracking server:** set `MLFLOW_TRACKING_URI` to your server (often backed by Postgres + artifact store in production).

```bash
export MLFLOW_TRACKING_URI="https://mlflow.internal.example.com"
python -m src.train --no-mlflow   # skip
# or omit --no-mlflow to log remotely
```

**UI (local file store):**

```bash
mlflow ui --backend-store-uri file:$(pwd)/mlruns --host 127.0.0.1 --port 5000
```

**UI (Docker Compose):** see `docker-compose.mlflow.yml`.

## Promotion gate

After training, CI (or a release job) runs:

```bash
python scripts/check_promotion.py \
  --metadata artifacts/metadata.json \
  --policy mlops/promotion_policy.yaml
```

Exit code **1** blocks merge/deploy when metrics fall below policy floors. Adjust thresholds per environment and baseline fraud rate.

## GitHub Actions

Workflow (repo root): [`../.github/workflows/sentinel-stream-mlops.yml`](../.github/workflows/sentinel-stream-mlops.yml) — runs train, promotion gate, and evaluate on push/PR.

## Suggested next steps (production)

- Model **registry** with staged transitions (`Staging` → `Production`) on a managed MLflow or equivalent.
- **Data contracts** and drift checks (Evidently, custom KS tests on score distributions).
- **Signed** artifacts and immutable build provenance (SLSA, container image digests).
- **Separate** promotion policies per jurisdiction / product line.
