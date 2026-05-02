# Sentinel-Stream — System Design

## Tagline

**A high-throughput, sub-100ms ML engine for real-time financial defense.**

## 1. Problem statement

Most fraud models are trained and evaluated in **batch**, yet decisions happen at the **moment of authorization**. The hard part is **fresh features**: knowing that a user made five purchases in ten minutes when the warehouse row is stale by seconds. Sentinel-Stream targets **event-time correctness** and **sub-60ms** scoring so the payment path never depends on a cold database alone.

## 2. Pitch (why Sentinel-Stream)

> While most fraud detection models are built for batch processing, Sentinel-Stream is built for the moment of transaction. It addresses the **cold-start feature** problem: how do you know a user has made five purchases in the last ten minutes if your database has not updated yet? Using a **Kappa-style** pipeline, the design processes, aggregates, and scores transactions in **under ~60ms** end-to-end at the inference hop (SLO; validate under your load model).

## 3. Key capabilities

### 3.1 Shadow deployment pattern

Run a **challenger** model **alongside** production on live traffic: the gateway still uses the **champion** decision for customers, while responses (or async logs) carry **shadow scores** for drift and uplift analysis before promotion.

- **In this repo:** optional `SENTINEL_SHADOW_ARTIFACTS_DIR` loads a second bundle; `POST /v1/score` returns `shadow_fraud_probability` when loaded (production `decision` unchanged).
- **Production:** dual consume from Kafka, shadow path to observability store, gated promotion in MLflow / registry.

### 3.2 Stateful sliding windows (Apache Flink)

**Velocity**, **geo-divergence**, and **session aggregates** are maintained as **living state** in Flink (or Flink SQL / ksqlDB), updated per event—not polled from OLTP on the critical path.

- **In this repo:** Phase 0 uses engineered columns in Parquet to **simulate** aggregates; Flink jobs are the **Phase 2** target in this document.
- **Production:** keyed state, sliding windows, side outputs to Redis / DynamoDB online store.

### 3.3 Circuit breaker logic

If **ML inference latency** exceeds budget (spikes, GC, model overload), the path **falls back** to a **high-speed heuristic** ruleset so the gateway **does not hang** waiting on the model.

- **In this repo:** `SENTINEL_CIRCUIT_THRESHOLD_MS` (default **60**); when open, `circuit_breaker_state` is `open` and `fraud_probability` comes from heuristics. `SENTINEL_FORCE_CIRCUIT_OPEN` and `SENTINEL_INJECT_LATENCY_MS` exist for **tests only** (never set inject in production).
- **Production:** same pattern at the sidecar / mesh level with metrics to Prometheus and alert on sustained `open` state.

## 4. Requirements

- **Latency:** Authorization scoring **under 50–100ms** at p99 under declared RPS.
- **Throughput:** Peak **10,000+** scored events per second (horizontal scale).
- **Freshness:** Feature state updated **per event** in the streaming layer.
- **Availability:** **99.99%** where required; **fail-open vs fail-closed** is a **business** policy—document and test both.

## 5. High-level architecture (production)

Kappa-style **streaming** pipeline:

1. **Ingestion:** Transactions hit an edge API → **Kafka** (or managed equivalent).
2. **Stream processing:** **Apache Flink** maintains keyed state (windows, velocity, geo).
3. **Feature store:** **Redis** / **DynamoDB** online; **Feast** (or similar) for offline/online parity.
4. **Inference:** **FastAPI** (or Go) loads online features + payload → **XGBoost / LightGBM** (or ensemble) → score + action.
5. **Sink:** Decisions back to Kafka + audit store for **approve / step-up / deny** workflows.

## 6. Tech stack

| Component | Technology |
| :--- | :--- |
| Message broker | Apache Kafka / Confluent |
| Stream processing | Apache Flink |
| Feature store | Redis (online), Feast (management) |
| Model serving | BentoML / NVIDIA Triton / FastAPI + XGBoost |
| MLOps | MLflow, promotion YAML gates, CI |
| Monitoring | Prometheus & Grafana |

## 7. Repository layout (Phase 0 in this repo)

| Path | Purpose |
| :--- | :--- |
| `scripts/generate_data.py` | Synthetic labeled transactions |
| `src/train.py` | Train XGBoost, write `artifacts/` |
| `src/evaluate.py` | Offline metrics |
| `src/heuristic_fallback.py` | Circuit-breaker heuristic |
| `api/main.py` | Scoring API (circuit breaker + optional shadow) |
| `mlops/` | Promotion policy, model card template, MLflow compose |

## 8. How to run (Phase 0)

```bash
cd Sentinel-Stream
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python scripts/generate_data.py
python -m src.train
uvicorn api.main:app --host 0.0.0.0 --port 8000
```

Example score:

```bash
curl -s -X POST http://127.0.0.1:8000/v1/score \
  -H "Content-Type: application/json" \
  -d '{"amount": 420.5, "hour_of_day": 3, "days_since_last_txn": 0.1, "txn_count_24h": 12, "distinct_merchants_24h": 9, "avg_amount_7d": 55.0, "channel_online": 1}'
```

## 9. Metrics and model policy

- Prefer **PR-AUC** and **recall at fixed FPR**; ROC-AUC alone is weak on rare fraud.
- Tune **threshold** per environment; shadow runs validate challenger thresholds before cutover.

## 10. Roadmap

1. **Phase 0 (here):** Batch train, FastAPI score, MLflow, promotion gate, circuit breaker + shadow hooks.
2. **Phase 1:** Warehouse-scale training, feature catalog, PCI-safe pipelines.
3. **Phase 2:** Flink stateful windows → Redis; end-to-end latency tests.
4. **Phase 3:** Hardened serving, autoscaling, SLO dashboards, chaos tests on circuit breaker.
5. **Phase 4:** Full shadow traffic percentage, A/B and champion–challenger automation.

## 11. Hardening

- AuthN/Z on inference; rate limits; generic client errors (no stack traces).
- No PII in logs or MLflow params; redact where needed.
- Model versioning, drift monitoring, signed artifacts.
