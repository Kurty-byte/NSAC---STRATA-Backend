from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)


def test_health_check():
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json()["status"] == "ok"


def test_post_prescribe_success_all_crops():
    for crop in ("corn", "rice", "tobacco", "tabacco"):
        resp = client.post(
            "/api/v1/prescribe",
            json={
                "crop": crop,
                "time_start": "2026-11-15",
                "time_end": "2026-11-20",
                "lat": 7.9064,
                "long": 125.0942,
            },
        )
        assert resp.status_code == 200
        body = resp.json()
        assert body["planting_likelihood"] in ("high", "moderate", "low")
        assert 0.0 <= body["probability"] <= 1.0
        assert 0.0 <= body["confidence"] <= 1.0
        assert "annual_climate_baseline" in body["telemetry"]
        assert "future_planting_date_historical_analogues" in body["telemetry"]


def test_post_prescribe_validation_errors():
    # Invalid crop
    r1 = client.post(
        "/api/v1/prescribe",
        json={
            "crop": "wheat",
            "time_start": "2026-11-15",
            "time_end": "2026-11-20",
            "lat": 7.9064,
            "long": 125.0942,
        },
    )
    assert r1.status_code == 422

    # time_end before time_start
    r2 = client.post(
        "/api/v1/prescribe",
        json={
            "crop": "corn",
            "time_start": "2026-11-25",
            "time_end": "2026-11-20",
            "lat": 7.9064,
            "long": 125.0942,
        },
    )
    assert r2.status_code == 422

    # Invalid GPS coordinates
    r3 = client.post(
        "/api/v1/prescribe",
        json={
            "crop": "corn",
            "time_start": "2026-11-15",
            "time_end": "2026-11-20",
            "lat": 105.0,
            "long": 125.0942,
        },
    )
    assert r3.status_code == 422
