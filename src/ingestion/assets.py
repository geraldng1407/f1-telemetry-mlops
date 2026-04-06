from pathlib import Path

import pandas as pd
from dagster import AssetExecutionContext, asset
from loguru import logger

from src.ingestion.constants import (
    LAP_COLUMNS,
    RAW_DATA_DIR,
    SESSION_TYPES_SPRINT,
    SESSION_TYPES_STANDARD,
    TELEMETRY_COLUMNS,
    TIMEDELTA_COLUMNS,
    WEATHER_COLUMNS,
)
from src.ingestion.partitions import season_round_partitions
from src.ingestion.resources import FastF1Resource


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _parse_partition_key(context: AssetExecutionContext) -> tuple[int, int]:
    """Extract (season, round) integers from the job partition (multi-dimensional)."""
    keys = context.multi_partition_key.keys_by_dimension
    return int(keys["season"]), int(keys["round"])


def _round_dir(season: int, round_num: int) -> Path:
    return RAW_DATA_DIR / str(season) / str(round_num)


def _get_session_types(event_format: str) -> list[str]:
    if event_format in ("sprint", "sprint_shootout", "sprint_qualifying"):
        return SESSION_TYPES_SPRINT
    return SESSION_TYPES_STANDARD


def _round_exists(fastf1_res: FastF1Resource, season: int, round_num: int) -> bool:
    """Check whether a given round number exists in the season calendar."""
    try:
        schedule = fastf1_res.get_event_schedule(season)
        valid_rounds = schedule["RoundNumber"].tolist()
        return round_num in valid_rounds
    except Exception:
        return False


def _timedelta_cols_to_seconds(df: pd.DataFrame) -> pd.DataFrame:
    """Convert any timedelta columns present in the DataFrame to float seconds."""
    df = df.copy()
    for col in TIMEDELTA_COLUMNS:
        if col in df.columns and pd.api.types.is_timedelta64_dtype(df[col]):
            df[col] = df[col].dt.total_seconds()
    return df


def _write_parquet(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False, engine="pyarrow")
    logger.info("Wrote {} rows to {}", len(df), path)


# ---------------------------------------------------------------------------
# Asset 1: raw_session_metadata
# ---------------------------------------------------------------------------


@asset(
    partitions_def=season_round_partitions,
    group_name="raw_ingestion",
    kinds={"python", "parquet"},
)
def raw_session_metadata(context: AssetExecutionContext, fastf1: FastF1Resource) -> None:
    """Extract event-level metadata for every session in a race weekend."""
    season, round_num = _parse_partition_key(context)
    fastf1_res = fastf1
    output_path = _round_dir(season, round_num) / "session_metadata.parquet"

    if output_path.exists() and not fastf1_res.force_reload:
        logger.info("Already ingested – skipping {}", output_path)
        return

    if not _round_exists(fastf1_res, season, round_num):
        logger.warning("Round {} does not exist in season {}", round_num, season)
        return

    rows: list[dict] = []
    event_format: str | None = None

    for session_type in SESSION_TYPES_STANDARD + ["Sprint Qualifying", "Sprint"]:
        try:
            session = fastf1_res.get_session_metadata(season, round_num, session_type)
        except (ValueError, KeyError, Exception) as exc:
            logger.debug("Session {}/{}/{} unavailable: {}", season, round_num, session_type, exc)
            continue

        event = session.event
        if event_format is None:
            event_format = str(event.get("EventFormat", "conventional"))

        rows.append(
            {
                "season": season,
                "round_number": int(event["RoundNumber"]),
                "event_name": str(event["EventName"]),
                "official_name": str(event.get("OfficialEventName", "")),
                "country": str(event["Country"]),
                "location": str(event["Location"]),
                "event_date": str(event["EventDate"]),
                "event_format": event_format,
                "session_type": session_type,
            }
        )

    if not rows:
        logger.warning("No metadata collected for {}/{}", season, round_num)
        return

    _write_parquet(pd.DataFrame(rows), output_path)


# ---------------------------------------------------------------------------
# Asset 2: raw_lap_data
# ---------------------------------------------------------------------------


@asset(
    partitions_def=season_round_partitions,
    group_name="raw_ingestion",
    kinds={"python", "parquet"},
)
def raw_lap_data(context: AssetExecutionContext, fastf1: FastF1Resource) -> None:
    """Extract lap-by-lap timing for all drivers across every session in a race weekend."""
    season, round_num = _parse_partition_key(context)
    fastf1_res = fastf1
    base_dir = _round_dir(season, round_num)

    if not _round_exists(fastf1_res, season, round_num):
        logger.warning("Round {} does not exist in season {}", round_num, season)
        return

    # Determine session types from event format
    try:
        event_schedule = fastf1_res.get_event_schedule(season)
        event_row = event_schedule[event_schedule["RoundNumber"] == round_num].iloc[0]
        event_format = str(event_row.get("EventFormat", "conventional"))
    except Exception:
        event_format = "conventional"

    session_types = _get_session_types(event_format)

    for session_type in session_types:
        output_path = base_dir / f"laps_{session_type.replace(' ', '_')}.parquet"

        if output_path.exists() and not fastf1_res.force_reload:
            logger.info("Already ingested – skipping {}", output_path)
            continue

        try:
            session = fastf1_res.load_session(season, round_num, session_type)
        except (ValueError, KeyError, Exception) as exc:
            logger.debug(
                "Session {}/{}/{} unavailable: {}", season, round_num, session_type, exc
            )
            continue

        laps = session.laps
        if laps is None or laps.empty:
            logger.warning("No lap data for {}/{}/{}", season, round_num, session_type)
            continue

        available_cols = [c for c in LAP_COLUMNS if c in laps.columns]
        df = laps[available_cols].copy()
        df["session_type"] = session_type
        df = _timedelta_cols_to_seconds(df)

        _write_parquet(df, output_path)


# ---------------------------------------------------------------------------
# Asset 3: raw_telemetry_data
# ---------------------------------------------------------------------------


@asset(
    partitions_def=season_round_partitions,
    group_name="raw_ingestion",
    kinds={"python", "parquet"},
)
def raw_telemetry_data(context: AssetExecutionContext, fastf1: FastF1Resource) -> None:
    """Extract ~4 Hz car telemetry for every lap, written per-driver per-session."""
    season, round_num = _parse_partition_key(context)
    fastf1_res = fastf1
    base_dir = _round_dir(season, round_num) / "telemetry"

    if not _round_exists(fastf1_res, season, round_num):
        logger.warning("Round {} does not exist in season {}", round_num, season)
        return

    try:
        event_schedule = fastf1_res.get_event_schedule(season)
        event_row = event_schedule[event_schedule["RoundNumber"] == round_num].iloc[0]
        event_format = str(event_row.get("EventFormat", "conventional"))
    except Exception:
        event_format = "conventional"

    session_types = _get_session_types(event_format)

    for session_type in session_types:
        session_dir = base_dir / session_type.replace(" ", "_")

        try:
            session = fastf1_res.load_session(season, round_num, session_type)
        except (ValueError, KeyError, Exception) as exc:
            logger.debug(
                "Session {}/{}/{} unavailable: {}", season, round_num, session_type, exc
            )
            continue

        laps = session.laps
        if laps is None or laps.empty:
            logger.warning("No laps for telemetry in {}/{}/{}", season, round_num, session_type)
            continue

        driver_numbers = laps["DriverNumber"].unique()

        for driver_num in driver_numbers:
            output_path = session_dir / f"{driver_num}.parquet"

            if output_path.exists() and not fastf1_res.force_reload:
                logger.debug("Already ingested – skipping {}", output_path)
                continue

            driver_laps = laps.pick_drivers(driver_num)
            telemetry_frames: list[pd.DataFrame] = []

            for _, lap in driver_laps.iterlaps():
                try:
                    tel = lap.get_telemetry()
                except Exception as exc:
                    logger.debug(
                        "No telemetry for driver {} lap {}: {}",
                        driver_num,
                        lap.get("LapNumber", "?"),
                        exc,
                    )
                    continue

                if tel is None or tel.empty:
                    continue

                available_cols = [c for c in TELEMETRY_COLUMNS if c in tel.columns]
                chunk = tel[available_cols].copy()
                chunk["driver_number"] = str(driver_num)
                chunk["lap_number"] = int(lap["LapNumber"])
                telemetry_frames.append(chunk)

            if not telemetry_frames:
                logger.debug("No telemetry collected for driver {} in {}", driver_num, session_type)
                continue

            df = pd.concat(telemetry_frames, ignore_index=True)
            df = _timedelta_cols_to_seconds(df)
            _write_parquet(df, output_path)


# ---------------------------------------------------------------------------
# Asset 4: raw_weather_data
# ---------------------------------------------------------------------------


@asset(
    partitions_def=season_round_partitions,
    group_name="raw_ingestion",
    kinds={"python", "parquet"},
)
def raw_weather_data(context: AssetExecutionContext, fastf1: FastF1Resource) -> None:
    """Extract weather data for each session in a race weekend."""
    season, round_num = _parse_partition_key(context)
    fastf1_res = fastf1
    base_dir = _round_dir(season, round_num)

    if not _round_exists(fastf1_res, season, round_num):
        logger.warning("Round {} does not exist in season {}", round_num, season)
        return

    try:
        event_schedule = fastf1_res.get_event_schedule(season)
        event_row = event_schedule[event_schedule["RoundNumber"] == round_num].iloc[0]
        event_format = str(event_row.get("EventFormat", "conventional"))
    except Exception:
        event_format = "conventional"

    session_types = _get_session_types(event_format)

    for session_type in session_types:
        output_path = base_dir / f"weather_{session_type.replace(' ', '_')}.parquet"

        if output_path.exists() and not fastf1_res.force_reload:
            logger.info("Already ingested – skipping {}", output_path)
            continue

        try:
            session = fastf1_res.load_session(season, round_num, session_type)
        except (ValueError, KeyError, Exception) as exc:
            logger.debug(
                "Session {}/{}/{} unavailable: {}", season, round_num, session_type, exc
            )
            continue

        weather = session.weather_data
        if weather is None or weather.empty:
            logger.warning("No weather data for {}/{}/{}", season, round_num, session_type)
            continue

        available_cols = [c for c in WEATHER_COLUMNS if c in weather.columns]
        df = weather[available_cols].copy()
        df["session_type"] = session_type
        df = _timedelta_cols_to_seconds(df)

        _write_parquet(df, output_path)
