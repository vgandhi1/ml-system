# eCommerce Sales Forecasting (ARIMA → Deep Learning)

**Time-series demand forecasting** upgraded from classical **ARIMA(2,1,1)** (R · SQL Server) to an ensemble **LSTM + TFT + N-BEATS** platform with **dbt**, **MLflow**, **FastAPI**, and **Docker Compose**.

[![Live Presentation](https://img.shields.io/badge/presentation-live-brightgreen?style=flat-square)](https://vgandhi1.github.io/ecommerce-demand-forecast/)
[![View Slides](https://img.shields.io/badge/slides-presentation.html-a855f7?style=flat-square)](presentation.html)

📊 **[Live Presentation](https://vgandhi1.github.io/ecommerce-demand-forecast/)** · [Static slides](presentation.html) · [Full upgrade doc](forecast-upgrade.md)

*(Pages: **Settings → Pages → Deploy from branch → `gh-pages` / (root)** — not `main`; run **Deploy GitHub Pages** workflow after push)*

---

## Quick start

```bash
pip install -r requirements.txt
pytest tests/ -q

docker compose up --build   # forecast-api :8000, MLflow :5000
```

```bash
cd dbt && dbt run && dbt test
```

---

## Repository layout

| Path | Purpose |
|------|---------|
| [`presentation.html`](presentation.html) | Static slide deck (GitHub Pages) |
| [`forecast-upgrade.md`](forecast-upgrade.md) | Architecture, phases, and implementation roadmap |
| [`src/models/`](src/models/) | LSTM, TFT, N-BEATS, ensemble |
| [`src/serving/api.py`](src/serving/api.py) | FastAPI forecast service |
| [`src/training/`](src/training/) | MLflow-tracked training |
| [`dbt/`](dbt/) | Staging → marts feature pipeline |
| [`docker-compose.yml`](docker-compose.yml) | API, MLflow, Redis, Kafka |
| [`.github/workflows/ci.yml`](.github/workflows/ci.yml) | Unit tests on push |

---

## Performance targets

| Metric | ARIMA baseline | LSTM target | TFT target |
|--------|----------------|-------------|------------|
| MAPE | 4.2% | &lt; 3.0% | &lt; 2.5% |
| RMSE | 148 units | &lt; 110 | &lt; 90 |
| Latency | Batch | &lt; 100ms | &lt; 150ms |

See [forecast-upgrade.md](forecast-upgrade.md) for the full architecture diagram and code walkthrough.
