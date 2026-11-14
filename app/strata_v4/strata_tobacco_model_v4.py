"""Strata Tobacco v4: EcoCrop-informed historical climate screening (NOT trained ML).

This deliberately separates FAO ECOCROP *annual* rainfall suitability from
historical crop-window temperature suitability and window rainfall diagnostics.
Scores are screening indices, not probabilities of yield or crop success.

Dependencies: pandas, numpy, requests (only for NASA API retrieval).
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo
import json
import math
import hashlib
import numpy as np
import pandas as pd
import requests

VERSION = "4.0.0"
SCIENTIFIC_NAME = "Nicotiana tabacum"
SOURCE_ECOCROP = "FAO ECOCROP Nicotiana tabacum datasheet (values supplied by project team)"
SOURCE_POWER = "https://power.larc.nasa.gov/docs/services/api/temporal/daily/"
CORE_COLUMNS = ("T2M", "T2M_MAX", "T2M_MIN", "PRECTOTCORR")
OPTIONAL_COLUMNS = ("GWETROOT",)

DEFAULT_PROFILE: dict[str, Any] = {
    "schema_version": 1,
    "scientific_name": SCIENTIFIC_NAME,
    "common_name": "Tobacco",
    "source_url": None,
    "source_reference": SOURCE_ECOCROP,
    "temperature_c": {"absolute_min": 7.0, "optimal_min": 15.0,
                      "optimal_max": 30.0, "absolute_max": 35.0},
    "annual_rainfall_mm": {"absolute_min": 350.0, "optimal_min": 500.0,
                           "optimal_max": 750.0, "absolute_max": 3000.0},
    "soil_ph": {"absolute_min": 4.5, "optimal_min": 5.0,
                "optimal_max": 6.5, "absolute_max": 7.5},
    "ecocrop_cycle_days": {"min": 70, "max": 150},
    "model_cycle_days": 110,
    "temperature_exceedance_c": 35.0,
    "rain_day_threshold_mm": 1.0,
    "heavy_rain_diagnostic_mm_per_day": 30.0,
    "soil_drainage_preference": "well-drained (dry spells)",
    "soil_depth_preference": "deep (>150 cm)",
    "soil_depth_absolute": "medium (50-150 cm)",
    "soil_texture_optimal": ["medium", "light"],
    "soil_texture_absolute": ["heavy", "medium", "light"],
    "soil_salinity": "low (<4 dS/m)",
    "altitude_absolute_max_m": 1800,
    "early_growth_killing_temperature_c": -1.0,
    "wind_susceptibility": True,
    "hail_susceptibility": True,
    "soil_fertility": "moderate",
    "light_intensity_optimal": "very bright",
    "light_intensity_absolute": "very bright to light shade",
    "photoperiod": ["short-day (<12 hours)", "neutral-day (12-14 hours)"],
    "killing_temperature_during_rest_c": -3.0,
    "climate_zones_listed": ["Aw", "Ar", "Bs", "Cf", "Cs", "Cw", "Do", "Dc", "Df", "Dw"],
    "reference_start_year": 2016,
    "reference_end_year": 2025,
    "notes": [
        "ECOCROP rainfall bounds refer to annual totals, not the crop cycle.",
        "The 110-day growing window is illustrative within ECOCROP 70–150 days, not a cultivar-specific duration.",
        "Daily Tmax > 35 C is an ECOCROP upper-limit exposure diagnostic, not a validated crop-failure threshold.",
        "30 mm daily precipitation is a descriptive diagnostic threshold, not a flooding or crop-failure threshold.",
        "Soil drainage, depth, texture, salinity and wind exposure are crop requirements but cannot be inferred reliably from NASA POWER daily climate alone."
    ],
}


def today_manila() -> date:
    return datetime.now(ZoneInfo("Asia/Manila")).date()


def validate_coordinates(latitude: float, longitude: float) -> tuple[float, float]:
    try:
        lat, lon = float(latitude), float(longitude)
    except (ValueError, TypeError) as exc:
        raise ValueError("latitude and longitude must be numerical GPS coordinates") from exc
    if not (math.isfinite(lat) and math.isfinite(lon)) or not (-90 <= lat <= 90 and -180 <= lon <= 180):
        raise ValueError("GPS coordinates are outside accepted geographic ranges")
    return lat, lon


def validate_dates(start_date: str | date, end_date: str | date, *, today: date | None = None,
                   max_span_days: int = 31) -> tuple[date, date]:
    def parse(x: str | date) -> date:
        if x is None:
            raise ValueError("Both start_date and end_date are required")
        if isinstance(x, datetime):
            return x.date()
        if isinstance(x, date):
            return x
        try:
            return date.fromisoformat(str(x))
        except (ValueError, TypeError) as exc:
            raise ValueError("Use YYYY-MM-DD for the start and end dates") from exc
    s, e = parse(start_date), parse(end_date)
    if s < (today or today_manila()):
        raise ValueError("Planting-date range must begin today or in the future")
    if e < s:
        raise ValueError("end_date must be on or after start_date")
    if (e - s).days >= max_span_days:
        raise ValueError(f"At most {max_span_days} candidate planting dates per request")
    return s, e


def validate_profile(profile: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(profile, dict):
        raise ValueError("Crop profile must be a dictionary")
    if profile.get("scientific_name") != SCIENTIFIC_NAME:
        raise ValueError(f"Tobacco-only model requires scientific_name={SCIENTIFIC_NAME}")
    for name in ("temperature_c", "annual_rainfall_mm", "soil_ph"):
        values = profile.get(name)
        if not isinstance(values, dict):
            raise ValueError(f"Missing profile range: {name}")
        try:
            seq = [float(values[k]) for k in ("absolute_min", "optimal_min", "optimal_max", "absolute_max")]
        except (ValueError, KeyError, TypeError) as exc:
            raise ValueError(f"Malformed {name} range") from exc
        if not all(math.isfinite(v) for v in seq) or not (seq[0] < seq[1] <= seq[2] < seq[3]):
            raise ValueError(f"{name} range must satisfy abs_min < opt_min <= opt_max < abs_max")
    period = profile.get("ecocrop_cycle_days", {})
    d = int(profile.get("model_cycle_days", 0))
    if not int(period.get("min", -1)) <= d <= int(period.get("max", -1)):
        raise ValueError("model_cycle_days outside supplied ECOCROP broad crop cycle")
    a = int(profile.get("reference_start_year", 0))
    b = int(profile.get("reference_end_year", 0))
    if not (1981 <= a <= b <= 2100):
        raise ValueError("Invalid reference years")
    return dict(profile)


def trapezoid(x: float, bounds: dict[str, Any]) -> float:
    a, b, c, d = [float(bounds[k]) for k in ("absolute_min", "optimal_min", "optimal_max", "absolute_max")]
    if not np.isfinite(x):
        return float("nan")
    if x <= a or x >= d:
        return 0.0
    if b <= x <= c:
        return 100.0
    if x < b:
        return float(np.clip((x - a) / (b - a) * 100.0, 0.0, 100.0))
    return float(np.clip((d - x) / (d - c) * 100.0, 0.0, 100.0))


def normalize_history(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or len(df) == 0:
        raise ValueError("Climate history is empty")
    data = df.copy()
    if "Date" in data.columns:
        idx = pd.to_datetime(data.pop("Date"), errors="coerce")
        if idx.isna().any():
            raise ValueError("Invalid Date values in climate data")
        data.index = pd.DatetimeIndex(idx)
    elif {"YEAR", "DOY"}.issubset(data.columns):
        years = pd.to_numeric(data.pop("YEAR"), errors="raise").astype(int).astype(str)
        doy = pd.to_numeric(data.pop("DOY"), errors="raise").astype(int).astype(str).str.zfill(3)
        data.index = pd.to_datetime(years + doy, format="%Y%j", errors="raise")
    elif not isinstance(data.index, pd.DatetimeIndex):
        raise ValueError("Climate must include Date or YEAR/DOY or have DatetimeIndex")
    data.index = pd.DatetimeIndex(data.index).normalize()
    if data.index.has_duplicates:
        raise ValueError("Duplicate date rows must be resolved before modeling")
    for c in CORE_COLUMNS:
        if c not in data.columns:
            raise ValueError(f"NASA climate data missing required column {c}")
    for c in (*CORE_COLUMNS, *OPTIONAL_COLUMNS):
        if c in data.columns:
            data[c] = pd.to_numeric(data[c], errors="coerce").replace(-999, np.nan)
    data.loc[data["PRECTOTCORR"] < 0, "PRECTOTCORR"] = np.nan
    for c in ("T2M", "T2M_MAX", "T2M_MIN"):
        data.loc[(data[c] < -80) | (data[c] > 65), c] = np.nan
    # Reject conflicting temperatures instead of treating broken records as complete.
    broken = (data["T2M_MAX"] < data["T2M_MIN"]) | (data["T2M"] < data["T2M_MIN"] - 2) | (data["T2M"] > data["T2M_MAX"] + 2)
    data.loc[broken, list(CORE_COLUMNS[:3])] = np.nan
    data = data.sort_index()
    data = data.reindex(pd.date_range(data.index.min(), data.index.max(), freq="D", name="Date"))
    return data


def load_power_csv(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(path)
    with path.open(encoding="utf-8-sig") as f:
        first_row = next((i for i, line in enumerate(f) if line.strip().startswith("Date,") or line.strip().startswith("YEAR,DOY,")), None)
    if first_row is None:
        raise ValueError("Cannot find Date or YEAR,DOY CSV header")
    return normalize_history(pd.read_csv(path, skiprows=first_row))


def fetch_power(latitude: float, longitude: float, *, start_year: int = 2016, end_year: int = 2026,
               cache_dir: str | Path | None = None, timeout: int = 90) -> pd.DataFrame:
    lat, lon = validate_coordinates(latitude, longitude)
    # Fetch a full extra follow-on year when possible, needed for end-of-year analogues.
    last = min(date(end_year, 12, 31), today_manila() - timedelta(days=10))
    first = date(start_year, 1, 1)
    if last < first:
        raise ValueError("POWER date request is before its available data")
    cache_file: Path | None = None
    if cache_dir is not None:
        key = hashlib.sha256(f"v4|{lat:.5f}|{lon:.5f}|{first:%Y%m%d}|{last:%Y%m%d}|LST".encode()).hexdigest()[:24]
        cache_file = Path(cache_dir) / f"power_{key}.csv"
        if cache_file.exists():
            return load_power_csv(cache_file)
    args = {"parameters": ",".join(CORE_COLUMNS), "community": "AG", "longitude": lon,
            "latitude": lat, "start": first.strftime("%Y%m%d"), "end": last.strftime("%Y%m%d"),
            "format": "JSON", "time-standard": "LST"}
    try:
        res = requests.get("https://power.larc.nasa.gov/api/temporal/daily/point", params=args, timeout=timeout)
        res.raise_for_status()
        raw = res.json()["properties"]["parameter"]
        if not all(c in raw for c in CORE_COLUMNS):
            raise ValueError("NASA response omitted a required climate parameter")
        dates = sorted(set().union(*(raw[c].keys() for c in CORE_COLUMNS)))
        frame = pd.DataFrame({"Date": pd.to_datetime(dates, format="%Y%m%d")})
        for c in CORE_COLUMNS:
            frame[c] = [raw[c].get(d, np.nan) for d in dates]
        out = normalize_history(frame)
    except (requests.RequestException, ValueError, KeyError, TypeError) as exc:
        raise RuntimeError(f"Could not get NASA POWER data for selected GPS coordinates: {exc}") from exc
    if cache_file is not None:
        cache_file.parent.mkdir(parents=True, exist_ok=True)
        out.reset_index().to_csv(cache_file, index=False)
    return out


def _longest_dry_spell(arr: np.ndarray, cutoff: float) -> int:
    longest = run = 0
    for val in arr:
        run = run + 1 if val < cutoff else 0
        longest = max(longest, run)
    return int(longest)


def _percentiles(arr: list[float]) -> dict[str, float] | None:
    if not arr:
        return None
    values = np.asarray(arr, dtype=float)
    return {"p10": round(float(np.percentile(values, 10)), 2),
            "median": round(float(np.median(values)), 2),
            "p90": round(float(np.percentile(values, 90)), 2)}


@dataclass
class StrataTobaccoV4:
    profile: dict[str, Any] | None = None

    def __post_init__(self):
        self.profile = validate_profile(self.profile or DEFAULT_PROFILE.copy())

    def annual_screening(self, climate: pd.DataFrame) -> list[dict[str, Any]]:
        history = normalize_history(climate)
        p = self.profile
        entries: list[dict[str, Any]] = []
        for y in range(int(p["reference_start_year"]), int(p["reference_end_year"]) + 1):
            dates = pd.date_range(date(y, 1, 1), date(y, 12, 31), freq="D")
            block = history.reindex(dates)
            if block[list(CORE_COLUMNS)].isna().any().any():
                continue
            temp = float(block["T2M"].mean())
            annual_rainfall = float(block["PRECTOTCORR"].sum())
            ts = trapezoid(temp, p["temperature_c"])
            rs = trapezoid(annual_rainfall, p["annual_rainfall_mm"])
            entries.append({"year": int(y), "mean_annual_temperature_c": round(temp, 3),
                            "actual_annual_precipitation_mm": round(annual_rainfall, 3),
                            "temperature_score": round(ts, 3), "annual_rainfall_score": round(rs, 3),
                            "same_year_climate_screening_score": round(min(ts, rs), 3)})
        return entries

    def assess(self, *, start_date: str | date, end_date: str | date,
               latitude: float, longitude: float, crop: str,
               climate: pd.DataFrame, today: date | None = None,
               soil_ph: float | None = None, irrigation: str = "unknown",
               soil_drainage: str = "unknown") -> dict[str, Any]:
        s, e = validate_dates(start_date, end_date, today=today)
        lat, lon = validate_coordinates(latitude, longitude)
        if not isinstance(crop, str) or crop.strip().casefold() not in ("nicotiana tabacum", "tobacco"):
            raise ValueError("v4 supports only Tobacco (Nicotiana tabacum)")
        if soil_drainage not in ("unknown", "well", "moderate", "poor"):
            raise ValueError("soil_drainage must be unknown, well, moderate or poor")
        if irrigation not in ("unknown", "rainfed", "irrigated"):
            raise ValueError("irrigation must be unknown, rainfed or irrigated")
        if soil_ph is not None and (not math.isfinite(float(soil_ph)) or not (0 <= float(soil_ph) <= 14)):
            raise ValueError("soil_ph must be a pH measurement between 0 and 14")
        history = normalize_history(climate)
        ann = self.annual_screening(history)
        if len(ann) < 3:
            raise ValueError("At least three complete reference years are required")
        p = self.profile
        cycle = int(p["model_cycle_days"])
        result_rows = []
        detail_rows = []
        for candidate in pd.date_range(s, e, freq="D"):
            analogues = []
            for y in range(int(p["reference_start_year"]), int(p["reference_end_year"]) + 1):
                try:
                    analogue_date = candidate.date().replace(year=y)
                except ValueError:
                    continue  # leap day in non-leap year
                dates = pd.date_range(analogue_date, periods=cycle, freq="D")
                block = history.reindex(dates)
                if block[list(CORE_COLUMNS)].isna().any().any():
                    continue
                daily_temp = block["T2M"].to_numpy(dtype=float)
                temp_scores = [trapezoid(v, p["temperature_c"]) for v in daily_temp]
                rain = block["PRECTOTCORR"].to_numpy(dtype=float)
                tmax = block["T2M_MAX"].to_numpy(dtype=float)
                tmin = block["T2M_MIN"].to_numpy(dtype=float)
                highs = int(np.count_nonzero(tmax > float(p["temperature_exceedance_c"])))
                lows = int(np.count_nonzero(tmin < float(p["temperature_c"]["absolute_min"])))
                rainfall = float(np.sum(rain))
                biggest = np.sort(rain)[-5:]
                # Rainfall concentration indicates vulnerability to concentrated storms;
                # these are indicators, not calibrated numeric suitability penalties.
                row = {
                    "proposed_planting_date": candidate.date().isoformat(),
                    "reference_year": y,
                    "historical_analogue_start": analogue_date.isoformat(),
                    "growing_window_days": cycle,
                    "temperature_compatibility": round(float(np.mean(temp_scores)), 3),
                    "rainfall_during_window_mm": round(rainfall, 3),
                    "longest_dry_spell_days_lt_1mm": _longest_dry_spell(rain, float(p["rain_day_threshold_mm"])),
                    "daily_max_over_35c_count": highs,
                    "daily_min_below_7c_count": lows,
                    "heaviest_five_days_rain_share_pct": round(float(np.sum(biggest) / rainfall * 100), 2) if rainfall > 0 else None,
                    "rain_days_ge_1mm": int(np.count_nonzero(rain >= 1.0)),
                    "heavy_rain_days_ge_30mm": int(np.count_nonzero(rain >= float(p["heavy_rain_diagnostic_mm_per_day"]))),
                }
                analogues.append(row)
                detail_rows.append(row)
            scores = [r["temperature_compatibility"] for r in analogues]
            rainfall_window = [r["rainfall_during_window_mm"] for r in analogues]
            dryspells = [r["longest_dry_spell_days_lt_1mm"] for r in analogues]
            hotdays = [r["daily_max_over_35c_count"] for r in analogues]
            concentration = [r["heaviest_five_days_rain_share_pct"] for r in analogues if r["heaviest_five_days_rain_share_pct"] is not None]
            result_rows.append({
                "proposed_planting_date": candidate.date().isoformat(),
                "historical_analogue_year_count": len(analogues),
                "historical_temperature_compatibility": _percentiles(scores),
                "historical_window_rain_mm": _percentiles(rainfall_window),
                "historical_max_dry_spell_days": _percentiles(dryspells),
                "historical_days_tmax_over_35c": _percentiles(hotdays),
                "historical_heaviest_five_rain_days_share_pct": _percentiles(concentration),
                "limited_analogue_sample": len(analogues) < 5,
            })
        annual_s = [x["same_year_climate_screening_score"] for x in ann]
        pH_score = trapezoid(float(soil_ph), p["soil_ph"]) if soil_ph is not None else None
        return {
            "model_version": VERSION,
            "model_type": "EcoCrop-informed deterministic climate screening, NOT trained ML",
            "request": {"crop": SCIENTIFIC_NAME, "latitude": lat, "longitude": lon,
                        "start_date": s.isoformat(), "end_date": e.isoformat(),
                        "irrigation": irrigation, "measured_soil_ph": soil_ph,
                        "soil_drainage": soil_drainage},
            "annual_climate_baseline": {
                "period": "Calendar-year observations; no seasonal rainfall annualization",
                "reference_year_count": len(ann),
                "actual_annual_temperature_and_rain_joint_same_year": True,
                "climate_screening_score": _percentiles(annual_s),
                "yearly": ann,
            },
            "future_planting_date_historical_analogues": result_rows,
            "soil_drainage_note": ("Reported drainage is poor; tobacco prefers well-drained soils. This is a qualitative warning, not a calibrated score." if soil_drainage == "poor" else
                                   "Reported well-drained conditions align with the crop reference." if soil_drainage == "well" else
                                   "Field drainage suitability not verified; this factor can affect crop success."),
            "historical_window_details": detail_rows,
            "measured_soil_ph_compatibility_score": round(pH_score, 2) if pH_score is not None else None,
            "limitations": [
                "No future complete growing-season forecast is generated.",
                "Future candidate dates compare same-calendar-date historical 110-day windows.",
                "Window rainfall/dry spell/heavy-rain metrics are descriptive, not agronomic penalties or seasonal crop water requirements.",
                "Tobacco requires well-drained soil; drainage must be verified separately from NASA POWER climate data.",
                "Light, soil texture, salinity, altitude, wind/hail risk and photoperiod are not automatically scored.",
                "ECOCROP rainfall ranges are annual; only true full-year rainfall is scored against them.",
                "Climate baseline and date-window scores are separate; not averaged into a fake future suitability probability.",
                "Temperature extremes reported relative to ECOCROP broad limits, not a validated phenological crop-failure model.",
                "Irrigation metadata is explanatory; no adjustment of climate score without evidence.",
                "NASA POWER values represent source grid cells, not measured farm sensor values.",
                "No learned parameters, supervised labels, crop yields, geographic validation, or accuracy claim.",
            ]
        }


def load_model(path: str | Path) -> StrataTobaccoV4:
    """Portable JSON-first loader. Source code must be installed; JSON has no pickle code execution."""
    return StrataTobaccoV4(json.loads(Path(path).read_text(encoding="utf-8")))


def export_artifacts(folder: str | Path, profile: dict[str, Any] | None = None) -> dict[str, str]:
    """Portable builtins-only joblib (optional) plus canonical JSON profile.

    Python logic remains separately versioned. Joblib uses pickle and MUST NOT
    load arbitrary user uploads; JSON is recommended for Go and Python apps.
    """
    import joblib
    out = Path(folder)
    out.mkdir(parents=True, exist_ok=True)
    p = validate_profile(profile or DEFAULT_PROFILE)
    path_json = out / "strata_tobacco_profile_v4.json"
    path_joblib = out / "strata_tobacco_config_v4.joblib"
    manifest = out / "strata_tobacco_manifest_v4.json"
    path_json.write_text(json.dumps(p, indent=2), encoding="utf-8")
    # Only built-in dict/str/int/float/list values; never custom class, numpy array, DataFrame.
    joblib.dump({"kind": "configuration_only_not_trained_ml", "model_version": VERSION, "profile": p}, path_joblib)
    manifest.write_text(json.dumps({"model_version": VERSION,
                                   "purpose": "Tobacco EcoCrop-informed historical climate suitability screening",
                                   "trained_ml": False,
                                   "joblib_contains": "builtins-only config (not executable algorithm)",
                                   "python_module": "strata_tobacco_model_v4.py",
                                   "required_fields": ["start_date", "end_date", "latitude", "longitude", "crop"],
                                   "sources": {"crop": SOURCE_ECOCROP, "climate": SOURCE_POWER}}, indent=2), encoding="utf-8")
    return {"profile_json": str(path_json), "joblib_config": str(path_joblib), "manifest": str(manifest)}
