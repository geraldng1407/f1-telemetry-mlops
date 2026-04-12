"""Load a historical race session from Parquet and build a replay timeline.

Reads raw lap, weather, and metadata files produced by the ingestion pipeline,
then constructs an ordered sequence of ``(delay_seconds, event)`` tuples that
the producer can replay through Kafka.
"""

from __future__ import annotations

from pathlib import Path
from typing import Union

import pandas as pd
from loguru import logger

from src.ingestion.constants import RAW_DATA_DIR
from src.streaming.models import (
    TRACK_STATUS_TO_EVENT,
    LapCompletedEvent,
    RaceControlEvent,
    RaceControlEventType,
)

TimelineEntry = tuple[float, Union[LapCompletedEvent, RaceControlEvent]]


def _read_parquet_safe(path: Path) -> pd.DataFrame | None:
    if not path.exists():
        logger.debug("File not found: {}", path)
        return None
    df = pd.read_parquet(path, engine="pyarrow")
    return df if not df.empty else None


def _join_track_temp(laps: pd.DataFrame, weather: pd.DataFrame | None) -> pd.DataFrame:
    """Attach nearest track temperature to each lap via merge_asof."""
    laps = laps.copy()
    if weather is None or "LapStartTime" not in laps.columns:
        laps["track_temp"] = float("nan")
        return laps

    weather = weather.sort_values("Time").copy()
    valid = laps["LapStartTime"].notna()
    if not valid.any() or weather.empty:
        laps["track_temp"] = float("nan")
        return laps

    lap_times = laps.loc[valid, ["LapStartTime"]].copy()
    merged = pd.merge_asof(
        lap_times.sort_values("LapStartTime"),
        weather[["Time", "TrackTemp"]].rename(columns={"TrackTemp": "track_temp"}),
        left_on="LapStartTime",
        right_on="Time",
        direction="nearest",
    )
    laps.loc[valid, "track_temp"] = merged["track_temp"].values
    laps["track_temp"] = laps.get("track_temp", float("nan"))
    return laps


def _compute_gap_ahead(laps: pd.DataFrame) -> pd.DataFrame:
    """Compute gap to car ahead from Position and Time columns."""
    laps = laps.copy()
    laps["gap_ahead"] = float("nan")

    if "Position" not in laps.columns or "Time" not in laps.columns:
        return laps

    has_data = laps["Position"].notna() & laps["Time"].notna()
    usable = laps.loc[has_data, ["LapNumber", "Position", "Time"]].copy()
    usable["Position"] = usable["Position"].astype(int)

    lap_pos_time = usable.set_index(["LapNumber", "Position"])["Time"]
    lap_pos_time = lap_pos_time[~lap_pos_time.index.duplicated(keep="first")]

    gaps = []
    for _, row in usable.iterrows():
        lap, pos = row["LapNumber"], row["Position"]
        if pos <= 1:
            gaps.append(float("nan"))
            continue
        ahead_key = (lap, pos - 1)
        if ahead_key in lap_pos_time.index:
            gaps.append(abs(row["Time"] - lap_pos_time[ahead_key]))
        else:
            gaps.append(float("nan"))

    usable["_gap"] = gaps
    laps.loc[has_data, "gap_ahead"] = usable["_gap"].values
    return laps


def _detect_race_control_events(
    laps: pd.DataFrame, session_id: str
) -> list[tuple[float, RaceControlEvent]]:
    """Detect TrackStatus transitions and emit RaceControlEvents."""
    if "TrackStatus" not in laps.columns:
        return []

    per_lap = (
        laps.dropna(subset=["TrackStatus"])
        .sort_values("Time")
        .drop_duplicates(subset=["LapNumber"], keep="first")
    )
    if per_lap.empty:
        return []

    events: list[tuple[float, RaceControlEvent]] = []
    prev_status: str | None = None

    for _, row in per_lap.iterrows():
        raw = str(row["TrackStatus"]).strip()
        status_chars = set(raw)

        current_type: RaceControlEventType | None = None
        for char in status_chars:
            if char in TRACK_STATUS_TO_EVENT:
                current_type = TRACK_STATUS_TO_EVENT[char]
                break

        if current_type is None and prev_status is not None:
            current_type = RaceControlEventType.CLEAR

        status_key = raw
        if current_type is not None and status_key != prev_status:
            timestamp = row["Time"] if pd.notna(row["Time"]) else 0.0
            events.append((
                timestamp,
                RaceControlEvent(
                    session_id=session_id,
                    lap_number=int(row["LapNumber"]),
                    event_type=current_type,
                    timestamp=timestamp,
                ),
            ))
        prev_status = status_key

    return events


def load_session_timeline(
    season: int,
    round_num: int,
    raw_data_dir: Path = RAW_DATA_DIR,
) -> list[TimelineEntry]:
    """Build an ordered replay timeline for a Race session.

    Returns a list of ``(delay_seconds, event)`` tuples sorted by the
    real-world time at which each event occurred.  The first entry has
    ``delay_seconds = 0``.
    """
    round_dir = raw_data_dir / str(season) / str(round_num)
    session_id = f"{season}_{round_num}_Race"

    laps_df = _read_parquet_safe(round_dir / "laps_Race.parquet")
    if laps_df is None:
        raise FileNotFoundError(
            f"No lap data at {round_dir / 'laps_Race.parquet'}"
        )

    weather_df = _read_parquet_safe(round_dir / "weather_Race.parquet")

    laps_df = _join_track_temp(laps_df, weather_df)
    laps_df = _compute_gap_ahead(laps_df)

    required = {"Time", "DriverNumber", "LapNumber"}
    missing = required - set(laps_df.columns)
    if missing:
        raise ValueError(f"Lap data missing required columns: {missing}")

    valid_laps = laps_df.dropna(subset=["Time"]).sort_values("Time").copy()

    lap_events: list[tuple[float, LapCompletedEvent]] = []
    for _, row in valid_laps.iterrows():
        sector_times = [
            row.get("Sector1Time") if pd.notna(row.get("Sector1Time")) else None,
            row.get("Sector2Time") if pd.notna(row.get("Sector2Time")) else None,
            row.get("Sector3Time") if pd.notna(row.get("Sector3Time")) else None,
        ]
        tire_age_raw = row.get("TyreLife")
        tire_age = int(tire_age_raw) if pd.notna(tire_age_raw) else None

        event = LapCompletedEvent(
            session_id=session_id,
            driver_number=str(row["DriverNumber"]),
            lap_number=int(row["LapNumber"]),
            lap_time=row.get("LapTime") if pd.notna(row.get("LapTime")) else None,
            sector_times=sector_times,
            compound=row.get("Compound") if pd.notna(row.get("Compound")) else None,
            tire_age=tire_age,
            track_temp=row.get("track_temp") if pd.notna(row.get("track_temp")) else None,
            gap_ahead=row.get("gap_ahead") if pd.notna(row.get("gap_ahead")) else None,
        )
        lap_events.append((float(row["Time"]), event))

    rc_events = _detect_race_control_events(valid_laps, session_id)

    all_events: list[tuple[float, LapCompletedEvent | RaceControlEvent]] = []
    all_events.extend(lap_events)
    all_events.extend(rc_events)
    all_events.sort(key=lambda x: x[0])

    timeline: list[TimelineEntry] = []
    prev_time = all_events[0][0] if all_events else 0.0
    for abs_time, event in all_events:
        delay = max(0.0, abs_time - prev_time)
        timeline.append((delay, event))
        prev_time = abs_time

    logger.info(
        "Built timeline: {} lap events, {} race control events, {:.1f}s total duration",
        len(lap_events),
        len(rc_events),
        sum(d for d, _ in timeline),
    )
    return timeline
