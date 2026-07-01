import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from fastapi.testclient import TestClient

from src.serving.api import app

client = TestClient(app)


def test_health():
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_forecast_endpoint():
    history = [1824, 1761, 2089, 1992, 2198, 2311, 2404, 2278, 2512, 2688]
    r = client.post(
        "/forecast",
        json={"series_id": "uk_ecommerce", "history": history, "horizon": 4},
    )
    assert r.status_code == 200
    body = r.json()
    assert len(body["point_forecast"]) == 4
    assert len(body["lower_95"]) == 4
