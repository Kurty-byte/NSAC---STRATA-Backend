from datetime import date
import pytest
from app.schemas import PrescribeRequest
from app.services.prescribe import PrescribeEngine


@pytest.fixture(scope="module")
def engine() -> PrescribeEngine:
    return PrescribeEngine()


def test_prescribe_corn_aliases_and_summary(engine: PrescribeEngine):
    req = PrescribeRequest(
        crop="CORN",
        time_start=date(2026, 11, 15),
        time_end=date(2026, 11, 20),
        lat=7.9064,
        long=125.0942,
    )
    res = engine.prescribe(req)
    assert res.crop == "corn"
    assert res.scientific_name == "Zea mays"
    assert res.planting_likelihood in ("high", "moderate", "low")
    assert 0.0 <= res.probability <= 1.0
    assert 0.0 <= res.confidence <= 1.0
    assert res.best_planting_date.startswith("2026-11-")
    assert res.telemetry["annual_climate_baseline"]["reference_year_count"] == 10


def test_prescribe_tabacco_typo_and_outside_valencia_coords(engine: PrescribeEngine):
    req = PrescribeRequest(
        crop="tabacco",
        time_start=date(2026, 1, 15),
        time_end=date(2026, 1, 20),
        lat=14.5995,
        long=120.9842,
    )
    res = engine.prescribe(req)
    assert res.crop == "tobacco"
    assert res.scientific_name == "Nicotiana tabacum"
    assert res.telemetry["requested_coordinates"] == {"lat": 14.5995, "long": 120.9842}
    assert res.telemetry["reference_station"]["name"] == "Valencia City, Bukidnon"


def test_prescribe_handles_long_span_and_past_dates(engine: PrescribeEngine):
    req = PrescribeRequest(
        crop="rice",
        time_start=date(2024, 6, 1),
        time_end=date(2024, 9, 29),
        lat=7.9064,
        long=125.0942,
    )
    res = engine.prescribe(req)
    assert res.crop == "rice"
    assert res.telemetry["requested_window"]["requested_span_days"] == 120
    assert len(res.telemetry["future_planting_date_historical_analogues"]) == 31
