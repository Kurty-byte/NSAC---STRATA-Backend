from pathlib import Path
from app.strata_v4.strata_maize_model_v4 import StrataMaizeV4, load_power_csv
from app.strata_v4.strata_palay_model_v4 import StrataPalayV4
from app.strata_v4.strata_tobacco_model_v4 import StrataTobaccoV4


def test_models_load_and_assess_valencia_csv():
    csv_path = Path("data/dataset (1).csv")
    climate = load_power_csv(csv_path)
    assert len(climate) >= 3900

    maize = StrataMaizeV4()
    res_m = maize.assess(
        start_date="2026-11-15",
        end_date="2026-11-17",
        latitude=7.9065,
        longitude=125.0939,
        crop="corn",
        climate=climate,
    )
    assert res_m["annual_climate_baseline"]["reference_year_count"] == 10
    assert len(res_m["future_planting_date_historical_analogues"]) == 3

    palay = StrataPalayV4()
    res_p = palay.assess(
        start_date="2026-11-15",
        end_date="2026-11-17",
        latitude=7.9065,
        longitude=125.0939,
        crop="rice",
        climate=climate,
    )
    assert res_p["request"]["crop"] == "Oryza sativa"

    tobacco = StrataTobaccoV4()
    res_t = tobacco.assess(
        start_date="2026-11-15",
        end_date="2026-11-17",
        latitude=7.9065,
        longitude=125.0939,
        crop="tobacco",
        climate=climate,
    )
    assert res_t["request"]["crop"] == "Nicotiana tabacum"
