"""Pydantic message schemas for Kafka streaming topics."""

from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, Field


class LapCompletedEvent(BaseModel):
    """Published to the ``lap_completed`` topic for every lap completion."""

    session_id: str
    driver_number: str
    lap_number: int
    lap_time: float | None = Field(default=None, description="Lap time in seconds")
    sector_times: list[float | None] = Field(
        default_factory=lambda: [None, None, None],
        description="Sector 1/2/3 times in seconds",
    )
    compound: str | None = None
    tire_age: int | None = Field(default=None, description="Tire age in laps (TyreLife)")
    track_temp: float | None = Field(default=None, description="Track temperature in Celsius")
    gap_ahead: float | None = Field(
        default=None,
        description="Gap to car ahead in seconds (None for P1)",
    )


class RaceControlEventType(str, Enum):
    SAFETY_CAR = "SAFETY_CAR"
    VSC = "VSC"
    RED_FLAG = "RED_FLAG"
    CLEAR = "CLEAR"


TRACK_STATUS_TO_EVENT: dict[str, RaceControlEventType] = {
    "4": RaceControlEventType.SAFETY_CAR,
    "5": RaceControlEventType.RED_FLAG,
    "6": RaceControlEventType.VSC,
    "7": RaceControlEventType.VSC,
}


class RaceControlEvent(BaseModel):
    """Published to the ``race_control`` topic on track status changes."""

    session_id: str
    lap_number: int
    event_type: RaceControlEventType
    timestamp: float = Field(description="Session time in seconds when the event occurred")
