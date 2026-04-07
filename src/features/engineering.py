"""Feature engineering pipeline for tire degradation prediction.

Transforms raw Parquet lap/weather data into ML-ready features. Operates as
standalone functions (no Dagster wiring) — reads from data/raw/ and writes to
data/processed/.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from loguru import logger

from src.ingestion.constants import RAW_DATA_DIR
from src.features.constants import (
    CIRCUIT_REF_PATH,
    DIRTY_AIR_THRESHOLD_S,
    FUEL_CORRECTION_PER_LAP,
    PROCESSED_DATA_DIR,
    ROLLING_WINDOWS,
    SC_TRACK_STATUS_CODES,
    SECTOR_TIME_COLUMNS,
    SESSION_HOUR_OFFSETS,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _write_parquet(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False, engine="pyarrow")
    logger.info("Wrote {} rows to {}", len(df), path)


def _read_parquet_safe(path: Path) -> pd.DataFrame | None:
    if not path.exists():
        logger.debug("File not found: {}", path)
        return None
    df = pd.read_parquet(path, engine="pyarrow")
    if df.empty:
        return None
    return df


# ---------------------------------------------------------------------------
# Feature Group 1: Stint-Level Aggregation
# ---------------------------------------------------------------------------


def _add_stint_aggregation(df: pd.DataFrame) -> pd.DataFrame:
    """Add stint_lap_number: 1-indexed lap position within each (Driver, Stint)."""
    df = df.copy()
    df["stint_lap_number"] = df.groupby(["Driver", "Stint"])["LapNumber"].transform(
        lambda s: s.rank(method="dense").astype(int)
    )
    return df


# ---------------------------------------------------------------------------
# Feature Group 2: Rolling Statistics
# ---------------------------------------------------------------------------


def _add_rolling_stats(df: pd.DataFrame) -> pd.DataFrame:
    """Add rolling mean and variance of LapTime within each (Driver, Stint)."""
    df = df.copy()
    df = df.sort_values(["Driver", "Stint", "stint_lap_number"])

    for window in ROLLING_WINDOWS:
        mean_col = f"rolling_mean_laptime_{window}"
        var_col = f"rolling_var_laptime_{window}"

        df[mean_col] = df.groupby(["Driver", "Stint"])["LapTime"].transform(
            lambda s: s.rolling(window, min_periods=1).mean()
        )
        df[var_col] = df.groupby(["Driver", "Stint"])["LapTime"].transform(
            lambda s: s.rolling(window, min_periods=2).var()
        )

    return df


# ---------------------------------------------------------------------------
# Feature Group 3: Sector Deltas
# ---------------------------------------------------------------------------


def _add_sector_deltas(df: pd.DataFrame) -> pd.DataFrame:
    """Add per-sector time delta from stint-best for each (Driver, Stint)."""
    df = df.copy()
    for sector_col in SECTOR_TIME_COLUMNS:
        sector_num = sector_col.replace("Sector", "").replace("Time", "")
        delta_col = f"sector{sector_num}_delta_from_best"

        stint_best = df.groupby(["Driver", "Stint"])[sector_col].transform("min")
        df[delta_col] = df[sector_col] - stint_best

    return df


# ---------------------------------------------------------------------------
# Feature Group 4: Tire Age Features
# ---------------------------------------------------------------------------


def _add_tire_age_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add tire_age_laps (from TyreLife) and fuel-corrected lap time."""
    df = df.copy()

    df["tire_age_laps"] = df["TyreLife"]

    # Fuel correction: add back the time gained from fuel burn within the stint.
    # As fuel burns, the car gets ~FUEL_CORRECTION_PER_LAP faster each lap.
    # Normalizing to start-of-stint fuel means any remaining increase is tire deg.
    df["fuel_corrected_laptime"] = (
        df["LapTime"] + FUEL_CORRECTION_PER_LAP * df["stint_lap_number"]
    )

    return df


# ---------------------------------------------------------------------------
# Feature Group 5: Environmental Features
# ---------------------------------------------------------------------------


def _add_environmental_features(
    df: pd.DataFrame, weather_df: pd.DataFrame | None
) -> pd.DataFrame:
    """Join weather data by nearest timestamp and compute track evolution index."""
    df = df.copy()

    if weather_df is not None and "LapStartTime" in df.columns:
        weather = weather_df.sort_values("Time").copy()
        lap_weather = df[["LapStartTime"]].copy()
        lap_weather = lap_weather.dropna(subset=["LapStartTime"])

        if not lap_weather.empty and not weather.empty:
            merged = pd.merge_asof(
                lap_weather.sort_values("LapStartTime"),
                weather.rename(
                    columns={
                        "TrackTemp": "track_temp_c",
                        "AirTemp": "air_temp_c",
                        "Humidity": "humidity",
                        "Rainfall": "rainfall",
                    }
                )[["Time", "track_temp_c", "air_temp_c", "humidity", "rainfall"]],
                left_on="LapStartTime",
                right_on="Time",
                direction="nearest",
            )
            merged = merged.drop(columns=["Time"])
            for col in ["track_temp_c", "air_temp_c", "humidity", "rainfall"]:
                df[col] = merged[col].values if col in merged.columns else float("nan")
        else:
            for col in ["track_temp_c", "air_temp_c", "humidity", "rainfall"]:
                df[col] = float("nan")
    else:
        for col in ["track_temp_c", "air_temp_c", "humidity", "rainfall"]:
            df[col] = float("nan")

    # Track evolution index: 0 at session start, 1 at session end.
    if "LapStartTime" in df.columns:
        valid = df["LapStartTime"].notna()
        if valid.any():
            min_t = df.loc[valid, "LapStartTime"].min()
            max_t = df.loc[valid, "LapStartTime"].max()
            span = max_t - min_t
            if span > 0:
                df.loc[valid, "track_evolution_index"] = (
                    (df.loc[valid, "LapStartTime"] - min_t) / span
                )
            else:
                df["track_evolution_index"] = 0.0
        else:
            df["track_evolution_index"] = float("nan")
    else:
        df["track_evolution_index"] = float("nan")

    return df


# ---------------------------------------------------------------------------
# Feature Group 6: Traffic / Dirty Air Features
# ---------------------------------------------------------------------------


def _add_dirty_air_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute gap to car ahead and cumulative dirty-air exposure per stint."""
    df = df.copy()
    df["gap_to_car_ahead_s"] = float("nan")
    df["is_dirty_air"] = False
    df["dirty_air_cumulative_laps"] = 0

    if "Position" not in df.columns or "Time" not in df.columns:
        return df

    has_position = df["Position"].notna() & df["Time"].notna()
    usable = df.loc[has_position, ["LapNumber", "Position", "Time", "Driver"]].copy()
    usable["Position"] = usable["Position"].astype(int)

    # Build a lookup: for each (LapNumber, Position) -> Time
    lap_pos_time = usable.set_index(["LapNumber", "Position"])["Time"]
    # Drop duplicates keeping first (shouldn't happen, but guard against it)
    lap_pos_time = lap_pos_time[~lap_pos_time.index.duplicated(keep="first")]

    gaps = []
    for _, row in usable.iterrows():
        lap = row["LapNumber"]
        pos = row["Position"]
        if pos <= 1:
            gaps.append(float("nan"))
            continue
        ahead_key = (lap, pos - 1)
        if ahead_key in lap_pos_time.index:
            gap = abs(row["Time"] - lap_pos_time[ahead_key])
            gaps.append(gap)
        else:
            gaps.append(float("nan"))

    usable["_gap"] = gaps
    df.loc[has_position, "gap_to_car_ahead_s"] = usable["_gap"].values

    df["is_dirty_air"] = df["gap_to_car_ahead_s"] < DIRTY_AIR_THRESHOLD_S
    df["is_dirty_air"] = df["is_dirty_air"].fillna(False)

    df = df.sort_values(["Driver", "Stint", "stint_lap_number"])
    df["dirty_air_cumulative_laps"] = df.groupby(["Driver", "Stint"])[
        "is_dirty_air"
    ].transform("cumsum")

    return df


# ---------------------------------------------------------------------------
# Feature Group 7: Track-Specific Static Features
# ---------------------------------------------------------------------------


def _load_circuit_reference(path: Path) -> pd.DataFrame | None:
    if not path.exists():
        logger.warning("Circuit reference file not found: {}", path)
        return None
    return pd.read_csv(path)


def _add_circuit_features(
    df: pd.DataFrame,
    circuit_ref: pd.DataFrame | None,
    location: str | None,
) -> pd.DataFrame:
    """Join static circuit characteristics from the reference table."""
    df = df.copy()

    circuit_cols = {
        "high_speed_corners": "circuit_high_speed_corners",
        "medium_speed_corners": "circuit_medium_speed_corners",
        "low_speed_corners": "circuit_low_speed_corners",
        "surface_abrasiveness": "circuit_abrasiveness",
        "altitude_m": "circuit_altitude_m",
        "tire_limitation": "circuit_tire_limitation",
        "street_circuit": "circuit_street_circuit",
    }

    if circuit_ref is None or location is None:
        for target_col in circuit_cols.values():
            df[target_col] = float("nan")
        return df

    match = circuit_ref[circuit_ref["location"] == location]
    if match.empty:
        logger.debug("No circuit reference data for location '{}'", location)
        for target_col in circuit_cols.values():
            df[target_col] = float("nan")
        return df

    row = match.iloc[0]
    for src_col, target_col in circuit_cols.items():
        df[target_col] = row[src_col]

    return df


# ---------------------------------------------------------------------------
# Feast Entity Keys
# ---------------------------------------------------------------------------


def _add_feast_keys(
    df: pd.DataFrame,
    season: int,
    round_num: int,
    session_type: str,
    event_date: str | None,
    location: str | None,
) -> pd.DataFrame:
    """Add Feast-compatible entity keys and event_timestamp.

    Keys: session_id, driver_number, stint_number, location, event_timestamp.
    The event_timestamp is synthesized from event_date + a per-session hour
    offset + the LapStartTime offset so that each lap in every session has a
    unique, chronologically ordered timestamp.
    """
    df = df.copy()
    safe_session = session_type.replace(" ", "_")

    df["session_id"] = f"{season}_{round_num}_{safe_session}"
    df["driver_number"] = df["DriverNumber"].astype(str)
    df["stint_number"] = df["Stint"].fillna(0).astype(int)
    df["location"] = location if location else ""

    hour_offset = SESSION_HOUR_OFFSETS.get(safe_session, 0)
    if event_date is not None and "LapStartTime" in df.columns:
        base = pd.Timestamp(event_date) + pd.Timedelta(hours=hour_offset)
        lap_offsets = pd.to_timedelta(df["LapStartTime"].fillna(0), unit="s")
        df["event_timestamp"] = base + lap_offsets
    else:
        df["event_timestamp"] = pd.Timestamp("1970-01-01")

    df["event_timestamp"] = df["event_timestamp"].fillna(pd.Timestamp("1970-01-01"))
    return df


# ---------------------------------------------------------------------------
# Lap Flags (edge-case markers)
# ---------------------------------------------------------------------------


def _add_lap_flags(df: pd.DataFrame) -> pd.DataFrame:
    """Flag inlaps, outlaps, and safety car laps."""
    df = df.copy()

    df["is_inlap"] = df["PitInTime"].notna() if "PitInTime" in df.columns else False
    df["is_outlap"] = df["PitOutTime"].notna() if "PitOutTime" in df.columns else False

    if "TrackStatus" in df.columns:
        df["is_sc_lap"] = df["TrackStatus"].astype(str).apply(
            lambda s: bool(set(str(s)) & SC_TRACK_STATUS_CODES)
        )
    else:
        df["is_sc_lap"] = False

    return df


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------


def engineer_features_for_session(
    season: int,
    round_num: int,
    session_type: str,
    raw_data_dir: Path = RAW_DATA_DIR,
    output_dir: Path = PROCESSED_DATA_DIR,
    circuit_ref_path: Path = CIRCUIT_REF_PATH,
) -> pd.DataFrame | None:
    """Run the full feature engineering pipeline for one session.

    Returns the engineered DataFrame, or None if the raw lap data is missing.
    """
    round_dir = raw_data_dir / str(season) / str(round_num)
    safe_session = session_type.replace(" ", "_")

    # --- Load raw data ---
    lap_path = round_dir / f"laps_{safe_session}.parquet"
    df = _read_parquet_safe(lap_path)
    if df is None:
        logger.warning("No lap data at {} – skipping", lap_path)
        return None

    weather_path = round_dir / f"weather_{safe_session}.parquet"
    weather_df = _read_parquet_safe(weather_path)

    meta_path = round_dir / "session_metadata.parquet"
    meta_df = _read_parquet_safe(meta_path)

    circuit_ref = _load_circuit_reference(circuit_ref_path)

    location: str | None = None
    event_date_str: str | None = None
    if meta_df is not None:
        meta_row = meta_df.iloc[0]
        location = str(meta_row["location"])
        if "event_date" in meta_row.index:
            event_date_str = str(meta_row["event_date"])

    # --- Apply feature groups sequentially ---
    logger.info(
        "Engineering features for {}/{}/{} ({} laps)",
        season,
        round_num,
        session_type,
        len(df),
    )

    df = _add_lap_flags(df)
    df = _add_stint_aggregation(df)
    df = _add_rolling_stats(df)
    df = _add_sector_deltas(df)
    df = _add_tire_age_features(df)
    df = _add_environmental_features(df, weather_df)
    df = _add_dirty_air_features(df)
    df = _add_circuit_features(df, circuit_ref, location)
    df = _add_feast_keys(df, season, round_num, session_type, event_date_str, location)

    # --- Write output ---
    out_path = output_dir / str(season) / str(round_num) / f"features_{safe_session}.parquet"
    _write_parquet(df, out_path)

    return df


def engineer_features_for_round(
    season: int,
    round_num: int,
    raw_data_dir: Path = RAW_DATA_DIR,
    output_dir: Path = PROCESSED_DATA_DIR,
    circuit_ref_path: Path = CIRCUIT_REF_PATH,
) -> dict[str, pd.DataFrame]:
    """Run feature engineering for every session found in a round directory."""
    round_dir = raw_data_dir / str(season) / str(round_num)
    if not round_dir.exists():
        logger.warning("Round directory does not exist: {}", round_dir)
        return {}

    results: dict[str, pd.DataFrame] = {}
    for lap_file in sorted(round_dir.glob("laps_*.parquet")):
        session_type = lap_file.stem.replace("laps_", "").replace("_", " ")
        df = engineer_features_for_session(
            season,
            round_num,
            session_type,
            raw_data_dir=raw_data_dir,
            output_dir=output_dir,
            circuit_ref_path=circuit_ref_path,
        )
        if df is not None:
            results[session_type] = df

    return results


def engineer_features_batch(
    seasons: list[int],
    rounds: list[int] | None = None,
    raw_data_dir: Path = RAW_DATA_DIR,
    output_dir: Path = PROCESSED_DATA_DIR,
    circuit_ref_path: Path = CIRCUIT_REF_PATH,
) -> None:
    """Batch-process multiple seasons/rounds. Discovers rounds from disk."""
    for season in seasons:
        season_dir = raw_data_dir / str(season)
        if not season_dir.exists():
            logger.info("No raw data for season {} – skipping", season)
            continue

        round_dirs = sorted(
            (d for d in season_dir.iterdir() if d.is_dir() and d.name.isdigit()),
            key=lambda d: int(d.name),
        )

        for rd in round_dirs:
            round_num = int(rd.name)
            if rounds is not None and round_num not in rounds:
                continue
            logger.info("Processing season {} round {}", season, round_num)
            engineer_features_for_round(
                season,
                round_num,
                raw_data_dir=raw_data_dir,
                output_dir=output_dir,
                circuit_ref_path=circuit_ref_path,
            )


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Engineer features from raw F1 lap/weather data."
    )
    parser.add_argument(
        "--seasons",
        nargs="+",
        type=int,
        default=[2025],
        help="Season years to process (default: 2025)",
    )
    parser.add_argument(
        "--rounds",
        nargs="*",
        type=int,
        default=None,
        help="Specific round numbers (default: all rounds found on disk)",
    )
    parser.add_argument(
        "--raw-dir",
        type=Path,
        default=RAW_DATA_DIR,
        help="Root directory for raw Parquet files",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROCESSED_DATA_DIR,
        help="Root directory for processed feature files",
    )
    parser.add_argument(
        "--circuit-ref",
        type=Path,
        default=CIRCUIT_REF_PATH,
        help="Path to circuit characteristics CSV",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    engineer_features_batch(
        seasons=args.seasons,
        rounds=args.rounds,
        raw_data_dir=args.raw_dir,
        output_dir=args.output_dir,
        circuit_ref_path=args.circuit_ref,
    )
