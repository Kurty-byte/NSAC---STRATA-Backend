# STRATA DSS Backend Specification (`POST /api/v1/prescribe`)

## 1. Overview & Intent
STRATA is a Decision Support System (DSS) backend that evaluates whether a farmer-supplied time window is suitable to plant **corn (maize)**, **rice (palay)**, or **tobacco** at a given GPS coordinate `(lat, long)`.

Under the hood, the service wraps the deterministic **Strata v4 EcoCrop-informed screening models** (`StrataMaizeV4`, `StrataPalayV4`, `StrataTobaccoV4`) and preloads the **2016–2025 Valencia City, Bukidnon NASA POWER daily dataset** (`data/dataset (1).csv`) into memory at startup so prescriptions execute offline in `< 15 ms` with zero external NASA API latency.

## 2. Goals & Non-Goals
### Goals
- Expose `POST /api/v1/prescribe` accepting `crop`, `time_start`, `time_end`, `lat`, and `long`.
- Return a two-tier response:
  1. **Farmer-facing summary**: `planting_likelihood` (`"high"` | `"moderate"` | `"low"`), `probability` (`0.0–1.0`), `confidence` (`0.0–1.0`), and `best_planting_date`.
  2. **Expandable `telemetry`**: The full Strata v4 assessment (`annual_climate_baseline`, `future_planting_date_historical_analogues`, `historical_window_details`, `limitations`, and Valencia City station provenance) for the frontend "Why?" trigger.
- Accept coordinates outside Valencia City for presentation/demo purposes while always evaluating against the preloaded Valencia City NASA POWER dataset.
- Support both short candidate planting windows (`<= 30 days`) and full-season planting-to-harvest windows (`> 30 days`) without failing validation.

### Non-Goals
- Live runtime calls to the external NASA POWER HTTP endpoint during `POST /api/v1/prescribe`.
- Training or claiming supervised ML accuracy/yield prediction (`trained_ml: False`).

## 3. API Contract

### Request: `POST /api/v1/prescribe`
```json
{
  "crop": "corn",
  "time_start": "2026-11-15",
  "time_end": "2026-11-21",
  "lat": 7.9064,
  "long": 125.0942,
  "soil_ph": null,
  "irrigation": "unknown",
  "soil_drainage": "unknown"
}
```

#### Field Validation Rules
- `crop` (required, string, case-insensitive):
  - **Maize**: `"corn"`, `"maize"`, `"mais"`, `"zea mays"`
  - **Palay**: `"rice"`, `"palay"`, `"oryza sativa"`
  - **Tobacco**: `"tobacco"`, `"tabacco"`, `"nicotiana tabacum"`
- `time_start`, `time_end` (required, ISO `YYYY-MM-DD` date strings):
  - Must satisfy `time_end >= time_start`.
  - Past, present, and future dates are accepted for presentation flexibility.
  - If `(time_end - time_start).days > 30`, the engine evaluates the first 31 candidate planting days starting at `time_start` (each projecting forward by the crop's 110/120-day growth cycle) and records the original requested span in `telemetry.requested_window`.
- `lat` (required, float): `-90.0 <= lat <= 90.0`.
- `long` (required, float): `-180.0 <= long <= 180.0`.
- `soil_ph` (optional, float | null): `0.0 <= soil_ph <= 14.0`.
- `irrigation` (optional, string): `"unknown"` (default), `"rainfed"`, `"irrigated"`.
- `soil_drainage` (optional, string): `"unknown"` (default), `"well"`, `"moderate"`, `"poor"`.

### Response: `200 OK`
```json
{
  "crop": "corn",
  "scientific_name": "Zea mays",
  "planting_likelihood": "high",
  "probability": 0.85,
  "confidence": 0.92,
  "best_planting_date": "2026-11-15",
  "telemetry": {
    "requested_coordinates": { "lat": 7.9064, "long": 125.0942 },
    "reference_station": {
      "name": "Valencia City, Bukidnon",
      "lat": 7.9065,
      "long": 125.0939,
      "dataset": "data/dataset (1).csv",
      "reference_years": [2016, 2025]
    },
    "requested_window": {
      "time_start": "2026-11-15",
      "time_end": "2026-11-21",
      "requested_span_days": 6,
      "evaluated_candidate_start": "2026-11-15",
      "evaluated_candidate_end": "2026-11-21",
      "model_cycle_days": 110
    },
    "score_breakdown": {
      "window_temperature_compatibility_median": 99.1,
      "annual_climate_baseline_median": 68.5,
      "dry_spell_days_median": 5.5,
      "heavy_rain_days_median": 1.0,
      "heat_exceedance_days_median": 0.0
    },
    "model_version": "4.0.0",
    "model_type": "EcoCrop-informed deterministic climate screening, NOT trained ML",
    "request": {},
    "annual_climate_baseline": {},
    "future_planting_date_historical_analogues": [],
    "historical_window_details": [],
    "measured_soil_ph_compatibility_score": null,
    "limitations": []
  }
}
```

## 4. Summary Scoring Derivation (From Strata v4 Output)
For each request, the v4 model produces `annual_climate_baseline` and `future_planting_date_historical_analogues` (with `0–100` trapezoid scores across 2016–2025):
1. **Window Temperature Score (`T_win`, `0–100`)**: Mean of `historical_temperature_compatibility.median` across candidate dates.
2. **Annual Baseline Score (`A_base`, `0–100`)**: `annual_climate_baseline.climate_screening_score.median`.
3. **Stress Penalty (`P_stress`, `0–30`)**:
   - Dry-spell penalty: `min(15.0, max(0.0, (median_dry_spell_days - 10.0) * 1.0))`
   - Heavy-rain penalty: `min(10.0, median_heavy_rain_days * 2.0)`
   - Heat-exceedance penalty: `min(10.0, median_heat_exceedance_days * 2.5)`
4. **`probability` (`0.0–1.0`)**:
   - `raw = 0.65 * T_win + 0.35 * A_base - P_stress`
   - `probability = round(clamp(raw / 100.0, 0.0, 1.0), 2)`
5. **`confidence` (`0.0–1.0`)**:
   - `sample_ratio = min(1.0, mean_analogue_years / 10.0)`
   - `spread = mean(p90 - p10 of temperature_compatibility) / 100.0`
   - `confidence = round(clamp(sample_ratio * (1.0 - 0.5 * spread), 0.0, 1.0), 2)`
6. **`planting_likelihood`**:
   - `"high"` if `probability >= 0.70`
   - `"moderate"` if `0.45 <= probability < 0.70`
   - `"low"` if `probability < 0.45`
