# STRATA Decision Support System (DSS) Backend

FastAPI backend providing crop planting prescriptions and climate telemetry for **Corn (Maize)**, **Rice (Palay)**, and **Tobacco**.

The engine evaluates requested planting timeframes against the **2016–2025 Valencia City, Bukidnon NASA POWER dataset** and deterministic **FAO EcoCrop-informed screening models (Strata v4)** in under 15 ms offline.

---

## Quick Start

### 1. Install Dependencies
```powershell
.\.venv\Scripts\pip install -r requirements.txt
```

### 2. Run the API Server
```powershell
.\.venv\Scripts\uvicorn.exe app.main:app --reload --host 127.0.0.1 --port 8000
```
- Interactive API Docs (Swagger UI): `http://127.0.0.1:8000/docs`
- Health Check: `http://127.0.0.1:8000/health`

### 3. Run Tests
```powershell
.\.venv\Scripts\python.exe -m pytest -v
```

---

## API Specification

### `POST /api/v1/prescribe`

Evaluates whether the requested planting timeframe is suitable for the selected crop.

#### Request Body
```json
{
  "crop": "corn",
  "time_start": "2026-11-15",
  "time_end": "2026-11-20",
  "lat": 7.9064,
  "long": 125.0942,
  "soil_ph": null,
  "irrigation": "unknown",
  "soil_drainage": "unknown"
}
```

* **`crop`** (required): `"corn"` / `"maize"`, `"rice"` / `"palay"`, or `"tobacco"` / `"tabacco"` (case-insensitive).
* **`time_start`**, **`time_end`** (required): ISO dates (`YYYY-MM-DD`). Full-season windows are automatically handled.
* **`lat`**, **`long`** (required): GPS coordinates (`lat` -90 to 90, `long` -180 to 180). Coords outside Valencia City are accepted for presentation and mapped to the reference station baseline.
* **`soil_ph`** (optional): Measured field pH (0–14).
* **`irrigation`** (optional): `"unknown"`, `"rainfed"`, or `"irrigated"`.
* **`soil_drainage`** (optional): `"unknown"`, `"well"`, `"moderate"`, or `"poor"`.

#### Response Body (`200 OK`)
```json
{
  "crop": "corn",
  "scientific_name": "Zea mays",
  "planting_likelihood": "high",
  "probability": 0.85,
  "confidence": 0.95,
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
      "time_end": "2026-11-20",
      "requested_span_days": 5,
      "evaluated_candidate_start": "2026-11-15",
      "evaluated_candidate_end": "2026-11-20",
      "model_cycle_days": 110
    },
    "score_breakdown": {
      "window_temperature_compatibility_median": 99.89,
      "annual_climate_baseline_median": 69.31,
      "dry_spell_days_median": 6.0,
      "heavy_rain_days_median": 1.0,
      "heat_exceedance_days_median": 0.0
    },
    "annual_climate_baseline": { ... },
    "future_planting_date_historical_analogues": [ ... ],
    "historical_window_details": [ ... ],
    "limitations": [ ... ]
  }
}
```

---

## Project Structure
```
NSAC/
├── app/
│   ├── config.py              # App settings & dataset path
│   ├── main.py                # FastAPI entrypoint, routes, CORS
│   ├── schemas.py             # Pydantic v2 request/response models
│   ├── services/
│   │   └── prescribe.py       # PrescribeEngine & scoring aggregator
│   └── strata_v4/             # Extracted v4 EcoCrop rule engines
│       ├── strata_maize_model_v4.py
│       ├── strata_palay_model_v4.py
│       └── strata_tobacco_model_v4.py
├── data/
│   ├── dataset (1).csv        # 2016-2026 NASA POWER daily weather
│   └── *.joblib / *.zip       # Original configurations and bundles
├── docs/
│   ├── PRD.md                 # Product Requirements Document
│   └── SPEC.md                # Technical Specification
├── tests/
│   ├── test_api_prescribe.py  # FastAPI integration tests
│   ├── test_prescribe_service.py # Service unit tests
│   └── test_strata_v4_models.py  # Engine verification tests
└── requirements.txt
```
