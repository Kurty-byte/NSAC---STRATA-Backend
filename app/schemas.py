from __future__ import annotations

from datetime import date
from typing import Any, Literal
from pydantic import BaseModel, Field, model_validator


class PrescribeRequest(BaseModel):
    crop: str = Field(..., min_length=1, description="Target crop: corn, rice, or tobacco")
    time_start: date = Field(..., description="Start of planting timeframe (YYYY-MM-DD)")
    time_end: date = Field(..., description="End of planting timeframe (YYYY-MM-DD)")
    lat: float = Field(..., ge=-90.0, le=90.0, description="GPS Latitude (-90 to 90)")
    long: float = Field(..., ge=-180.0, le=180.0, description="GPS Longitude (-180 to 180)")
    soil_ph: float | None = Field(default=None, ge=0.0, le=14.0, description="Optional soil pH measurement")
    irrigation: Literal["unknown", "rainfed", "irrigated"] = "unknown"
    soil_drainage: Literal["unknown", "well", "moderate", "poor"] = "unknown"

    @model_validator(mode="after")
    def validate_date_order(self) -> PrescribeRequest:
        if self.time_end < self.time_start:
            raise ValueError("time_end must be on or after time_start")
        return self


class PrescribeResponse(BaseModel):
    crop: Literal["corn", "rice", "tobacco"]
    scientific_name: str
    planting_likelihood: Literal["high", "moderate", "low"]
    probability: float = Field(..., ge=0.0, le=1.0)
    confidence: float = Field(..., ge=0.0, le=1.0)
    best_planting_date: str
    telemetry: dict[str, Any]
