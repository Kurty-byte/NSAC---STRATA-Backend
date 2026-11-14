# STRATA Product Requirements Document (PRD)

## 1. Problem Statement
Smallholder farmers in the Philippines (starting with Valencia City, Bukidnon) face shifting climate patterns that make traditional planting calendars unreliable. Without actionable agro-climatic guidance, farmers risk planting during unseasonal heat spikes, severe dry spells, or destructive heavy-rain periods.

## 2. Product Vision
STRATA DSS provides real-time, science-informed planting prescriptions via a unified API. A farmer or extension worker specifies:
1. Target crop (Corn / Maize, Rice / Palay, Tobacco)
2. Proposed planting timeframe (`time_start` to `time_end`)
3. Field GPS location (`lat`, `long`)

The backend computes planting likelihood, probability, confidence, and best planting date in `< 15 ms` offline, accompanied by full climate telemetry explaining the agronomic reasons behind the decision.

## 3. Key User Personas
- **Farmer / Mobile App User**: Needs immediate, intuitive answers: "Should I plant this week? What is the likelihood?"
- **Agronomist / Extension Worker**: Needs full telemetry: historical temperature trapezoid scores, annual baseline, dry-spell risk, heavy-rain frequency, and heat exceedance days.

## 4. Requirements
- **Unified Endpoint**: `POST /api/v1/prescribe`
- **Offline Reliability**: Bundled 2016–2025 NASA POWER historical daily records with zero external API latency.
- **Crop Support**: Corn (110-day cycle), Rice (120-day cycle), Tobacco (110-day cycle).
- **Two-Tier UX**: Instant high-level decision (`planting_likelihood`, `probability`, `confidence`) + expandable `telemetry`.
