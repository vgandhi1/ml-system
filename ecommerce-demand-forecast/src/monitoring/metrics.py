import time
from functools import wraps

from prometheus_client import Counter, Gauge, Histogram

FORECAST_REQUESTS = Counter(
    "forecast_requests_total", "Total forecast API requests", ["series_id", "status"]
)
FORECAST_LATENCY = Histogram(
    "forecast_latency_seconds",
    "Forecast request latency",
    buckets=[0.01, 0.05, 0.1, 0.25, 0.5, 1.0],
)
MODEL_MAPE = Gauge("model_mape_current", "Current model MAPE on production", ["model_name"])
CACHE_HIT_RATE = Gauge("cache_hit_rate", "Redis cache hit rate")
ACTIVE_SERIES = Gauge("active_forecast_series", "Number of series being forecast")


def track_forecast(func):
    """Decorator to auto-instrument forecast endpoints."""

    @wraps(func)
    def wrapper(*args, **kwargs):
        start = time.time()
        try:
            result = func(*args, **kwargs)
            FORECAST_REQUESTS.labels(series_id="all", status="success").inc()
            return result
        except Exception:
            FORECAST_REQUESTS.labels(series_id="all", status="error").inc()
            raise
        finally:
            FORECAST_LATENCY.observe(time.time() - start)

    return wrapper
