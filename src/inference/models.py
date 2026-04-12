"""Request / response Pydantic schemas for the inference API."""

from __future__ import annotations

from pydantic import BaseModel, Field


class PredictionResult(BaseModel):
    driver_number: str
    lap_number: int
    stint_number: int
    compound: str | None = None
    tire_age: int | None = None
    predicted_laps_to_cliff: float
    confidence_lower: float
    confidence_upper: float


class DriverStateResponse(BaseModel):
    driver_number: str
    stint_number: int
    compound: str | None = None
    tire_age: int | None = None
    lap_number: int
    latest_prediction: PredictionResult | None = None


class RaceStateResponse(BaseModel):
    session_id: str
    race_control_status: str = "CLEAR"
    drivers: list[DriverStateResponse] = Field(default_factory=list)
