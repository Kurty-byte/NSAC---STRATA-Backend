from __future__ import annotations

from datetime import date, timedelta
from pathlib import Path
from typing import Any
import numpy as np
import pandas as pd

from app.schemas import PrescribeRequest, PrescribeResponse
from app.strata_v4 import (
    StrataMaizeV4,
    StrataPalayV4,
    StrataTobaccoV4,
    load_power_csv,
)

CROP_ALIASES = {
    "corn": "corn",
    "maize": "corn",
    "mais": "corn",
    "zea mays": "corn",
    "rice": "rice",
    "palay": "rice",
    "oryza sativa": "rice",
    "tobacco": "tobacco",
    "tabacco": "tobacco",
    "nicotiana tabacum": "tobacco",
}

SCIENTIFIC_NAMES = {
    "corn": "Zea mays",
    "rice": "Oryza sativa",
    "tobacco": "Nicotiana tabacum",
}

VALENCIA_STATION = {
    "name": "Valencia City, Bukidnon",
    "lat": 7.9065,
    "long": 125.0939,
    "dataset": "data/dataset (1).csv",
    "reference_years": [2016, 2025],
}


class PrescribeEngine:
    def __init__(self, dataset_path: str | Path = "data/dataset (1).csv"):
        self.dataset_path = Path(dataset_path)
        self.climate_data: pd.DataFrame = load_power_csv(self.dataset_path)
        self.maize_model = StrataMaizeV4()
        self.palay_model = StrataPalayV4()
        self.tobacco_model = StrataTobaccoV4()

    def normalize_crop(self, crop_input: str) -> str:
        key = crop_input.strip().casefold()
        if key not in CROP_ALIASES:
            raise ValueError(
                f"Unsupported crop '{crop_input}'. Supported crops: corn, rice, tobacco."
            )
        return CROP_ALIASES[key]

    def prescribe(self, req: PrescribeRequest) -> PrescribeResponse:
        norm_crop = self.normalize_crop(req.crop)
        sci_name = SCIENTIFIC_NAMES[norm_crop]

        # Prevent upstream error when end_date - start_date >= 31 days
        max_eval_span = timedelta(days=30)
        eval_end = min(req.time_end, req.time_start + max_eval_span)

        # Model routing & assessment
        if norm_crop == "corn":
            model = self.maize_model
            raw_crop_arg = "corn"
            assessment = model.assess(
                start_date=req.time_start,
                end_date=eval_end,
                latitude=req.lat,
                longitude=req.long,
                crop=raw_crop_arg,
                climate=self.climate_data,
                today=req.time_start,
                soil_ph=req.soil_ph,
                irrigation=req.irrigation,
            )
        elif norm_crop == "rice":
            model = self.palay_model
            raw_crop_arg = "rice"
            assessment = model.assess(
                start_date=req.time_start,
                end_date=eval_end,
                latitude=req.lat,
                longitude=req.long,
                crop=raw_crop_arg,
                climate=self.climate_data,
                today=req.time_start,
                soil_ph=req.soil_ph,
                irrigation=req.irrigation,
            )
        else:  # tobacco
            model = self.tobacco_model
            raw_crop_arg = "tobacco"
            assessment = model.assess(
                start_date=req.time_start,
                end_date=eval_end,
                latitude=req.lat,
                longitude=req.long,
                crop=raw_crop_arg,
                climate=self.climate_data,
                today=req.time_start,
                soil_ph=req.soil_ph,
                irrigation=req.irrigation,
                soil_drainage=req.soil_drainage,
            )

        analogues = assessment.get("future_planting_date_historical_analogues", [])
        details = assessment.get("historical_window_details", [])
        annual_baseline = assessment.get("annual_climate_baseline", {})

        # 1. Window Temperature Score (T_win, 0-100)
        temp_medians = [
            a["historical_temperature_compatibility"]["median"]
            for a in analogues
            if a.get("historical_temperature_compatibility")
        ]
        t_win = float(np.mean(temp_medians)) if temp_medians else 50.0

        # 2. Annual Baseline Score (A_base, 0-100)
        screening_score = annual_baseline.get("climate_screening_score") or {}
        a_base = float(screening_score.get("median", 50.0))

        # 3. Stress diagnostics from candidate dates / window details
        dry_spells = [
            a["historical_max_dry_spell_days"]["median"]
            for a in analogues
            if a.get("historical_max_dry_spell_days")
        ]
        dry_spell_med = float(np.median(dry_spells)) if dry_spells else 0.0

        heavy_rains = [
            d["heavy_rain_days_ge_30mm"]
            for d in details
            if "heavy_rain_days_ge_30mm" in d
        ]
        heavy_rain_med = float(np.median(heavy_rains)) if heavy_rains else 0.0

        # Heat exceedance keys differ slightly across crops
        heat_key = next(
            (k for k in (analogues[0] if analogues else {}) if "tmax_over" in k),
            None,
        )
        if heat_key:
            heat_excs = [
                a[heat_key]["median"]
                for a in analogues
                if a.get(heat_key)
            ]
            heat_exc_med = float(np.median(heat_excs)) if heat_excs else 0.0
        else:
            heat_exc_med = 0.0

        # Stress penalties
        p_dry = min(15.0, max(0.0, (dry_spell_med - 10.0) * 1.0))
        p_rain = min(10.0, heavy_rain_med * 2.0)
        p_heat = min(10.0, heat_exc_med * 2.5)
        p_stress = p_dry + p_rain + p_heat

        # 4. Probability (0.0 - 1.0)
        raw_prob = (0.65 * t_win) + (0.35 * a_base) - p_stress
        probability = round(float(np.clip(raw_prob / 100.0, 0.0, 1.0)), 2)

        # 5. Confidence (0.0 - 1.0)
        analogue_counts = [a.get("historical_analogue_year_count", 0) for a in analogues]
        mean_counts = float(np.mean(analogue_counts)) if analogue_counts else 0.0
        sample_ratio = min(1.0, mean_counts / 10.0)

        spreads = [
            (a["historical_temperature_compatibility"]["p90"] - a["historical_temperature_compatibility"]["p10"])
            for a in analogues
            if a.get("historical_temperature_compatibility")
        ]
        mean_spread = (float(np.mean(spreads)) / 100.0) if spreads else 0.0
        confidence = round(float(np.clip(sample_ratio * (1.0 - 0.5 * mean_spread), 0.0, 1.0)), 2)

        # 6. Planting Likelihood
        if probability >= 0.70:
            likelihood = "high"
        elif probability >= 0.45:
            likelihood = "moderate"
        else:
            likelihood = "low"

        # 7. Best Planting Date in Window
        best_candidate = req.time_start.isoformat()
        if analogues:
            sorted_analogues = sorted(
                analogues,
                key=lambda a: (
                    a.get("historical_temperature_compatibility", {}).get("median", 0.0),
                    -a.get("historical_max_dry_spell_days", {}).get("median", 999.0),
                ),
                reverse=True,
            )
            best_candidate = sorted_analogues[0]["proposed_planting_date"]

        # 8. Telemetry Construction
        telemetry: dict[str, Any] = {
            "requested_coordinates": {"lat": req.lat, "long": req.long},
            "reference_station": VALENCIA_STATION,
            "requested_window": {
                "time_start": req.time_start.isoformat(),
                "time_end": req.time_end.isoformat(),
                "requested_span_days": (req.time_end - req.time_start).days,
                "evaluated_candidate_start": req.time_start.isoformat(),
                "evaluated_candidate_end": eval_end.isoformat(),
                "model_cycle_days": int(model.profile["model_cycle_days"]),
            },
            "score_breakdown": {
                "window_temperature_compatibility_median": round(t_win, 2),
                "annual_climate_baseline_median": round(a_base, 2),
                "dry_spell_days_median": round(dry_spell_med, 2),
                "heavy_rain_days_median": round(heavy_rain_med, 2),
                "heat_exceedance_days_median": round(heat_exc_med, 2),
            },
        }
        # Merge complete v4 assessment keys into telemetry
        for k, v in assessment.items():
            telemetry[k] = v

        return PrescribeResponse(
            crop=norm_crop,  # type: ignore[arg-type]
            scientific_name=sci_name,
            planting_likelihood=likelihood,
            probability=probability,
            confidence=confidence,
            best_planting_date=best_candidate,
            telemetry=telemetry,
        )
