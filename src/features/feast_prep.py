"""Consolidate processed features into Feast-ready data sources.

Reads all ``data/processed/**/features_*.parquet`` files, ensures Feast entity
keys are present (adding them from file-path metadata when missing), and writes
a single consolidated Parquet file that can be used as a Feast ``FileSource``.

Also converts the circuit-characteristics CSV into a Feast-compatible Parquet.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import pandas as pd
from loguru import logger

from src.features.constants import (
    CIRCUIT_REF_PATH,
    FEAST_DATA_DIR,
    PROCESSED_DATA_DIR,
    SESSION_HOUR_OFFSETS,
)
from src.ingestion.constants import RAW_DATA_DIR


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_FEATURE_FILE_RE = re.compile(
    r".*/(\d{4})/(\d+)/features_(.+)\.parquet$"
)


def _derive_feast_keys(
    df: pd.DataFrame,
    season: int,
    round_num: int,
    session_slug: str,
    raw_data_dir: Path = RAW_DATA_DIR,
) -> pd.DataFrame:
    """Add Feast entity columns when they are absent from processed data.

    Used for backward compatibility with feature files produced before
    ``_add_feast_keys`` was added to the engineering pipeline.
    """
    df = df.copy()

    df["session_id"] = f"{season}_{round_num}_{session_slug}"
    df["driver_number"] = df["DriverNumber"].astype(str)
    df["stint_number"] = df["Stint"].fillna(0).astype(int)

    # Resolve location and event_date from raw metadata
    meta_path = raw_data_dir / str(season) / str(round_num) / "session_metadata.parquet"
    location = ""
    event_date_str: str | None = None
    if meta_path.exists():
        meta = pd.read_parquet(meta_path, engine="pyarrow")
        if not meta.empty:
            location = str(meta.iloc[0]["location"])
            if "event_date" in meta.columns:
                event_date_str = str(meta.iloc[0]["event_date"])

    df["location"] = location

    hour_offset = SESSION_HOUR_OFFSETS.get(session_slug, 0)
    if event_date_str is not None and "LapStartTime" in df.columns:
        base = pd.Timestamp(event_date_str) + pd.Timedelta(hours=hour_offset)
        lap_offsets = pd.to_timedelta(df["LapStartTime"].fillna(0), unit="s")
        df["event_timestamp"] = base + lap_offsets
    else:
        df["event_timestamp"] = pd.Timestamp("1970-01-01")

    df["event_timestamp"] = df["event_timestamp"].fillna(pd.Timestamp("1970-01-01"))
    return df


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def consolidate_stint_features(
    processed_dir: Path = PROCESSED_DATA_DIR,
    output_path: Path | None = None,
    raw_data_dir: Path = RAW_DATA_DIR,
) -> Path:
    """Merge all processed feature Parquets into a single Feast-ready file."""
    if output_path is None:
        output_path = FEAST_DATA_DIR / "stint_features.parquet"

    output_path.parent.mkdir(parents=True, exist_ok=True)

    frames: list[pd.DataFrame] = []
    for parquet_file in sorted(processed_dir.rglob("features_*.parquet")):
        df = pd.read_parquet(parquet_file, engine="pyarrow")

        # Add feast keys if missing (backward compat)
        if "session_id" not in df.columns:
            posix = parquet_file.as_posix()
            m = _FEATURE_FILE_RE.search(posix)
            if m is None:
                logger.warning("Cannot parse path for feast keys: {}", parquet_file)
                continue
            season, round_num, session_slug = int(m.group(1)), int(m.group(2)), m.group(3)
            df = _derive_feast_keys(df, season, round_num, session_slug, raw_data_dir)

        frames.append(df)

    if not frames:
        logger.warning("No processed feature files found in {}", processed_dir)
        return output_path

    combined = pd.concat(frames, ignore_index=True)

    # Cast boolean columns to int for Feast materialization compatibility
    bool_cols = combined.select_dtypes(include=["bool"]).columns
    for col in bool_cols:
        combined[col] = combined[col].astype(int)

    combined.to_parquet(output_path, index=False, engine="pyarrow")
    logger.info("Consolidated {} rows into {}", len(combined), output_path)
    return output_path


def prepare_circuit_source(
    csv_path: Path = CIRCUIT_REF_PATH,
    output_path: Path | None = None,
) -> Path:
    """Convert circuit characteristics CSV to Feast-ready Parquet."""
    if output_path is None:
        output_path = FEAST_DATA_DIR / "circuit_features.parquet"

    output_path.parent.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(csv_path)

    rename_map = {
        "high_speed_corners": "circuit_high_speed_corners",
        "medium_speed_corners": "circuit_medium_speed_corners",
        "low_speed_corners": "circuit_low_speed_corners",
        "surface_abrasiveness": "circuit_abrasiveness",
        "altitude_m": "circuit_altitude_m",
        "tire_limitation": "circuit_tire_limitation",
        "street_circuit": "circuit_street_circuit",
    }
    df = df.rename(columns=rename_map)

    df["circuit_street_circuit"] = (
        df["circuit_street_circuit"]
        .astype(str)
        .str.lower()
        .map({"true": 1, "false": 0})
        .fillna(0)
        .astype(int)
    )

    df["event_timestamp"] = pd.Timestamp("2020-01-01")

    keep_cols = ["location", "event_timestamp"] + list(rename_map.values())
    df = df[[c for c in keep_cols if c in df.columns]]

    df.to_parquet(output_path, index=False, engine="pyarrow")
    logger.info("Wrote circuit features ({} rows) to {}", len(df), output_path)
    return output_path


def prepare_all(
    processed_dir: Path = PROCESSED_DATA_DIR,
    circuit_csv: Path = CIRCUIT_REF_PATH,
    feast_dir: Path = FEAST_DATA_DIR,
    raw_data_dir: Path = RAW_DATA_DIR,
) -> None:
    """Run all data preparation steps for Feast."""
    consolidate_stint_features(processed_dir, feast_dir / "stint_features.parquet", raw_data_dir)
    prepare_circuit_source(circuit_csv, feast_dir / "circuit_features.parquet")
    logger.info("Feast data preparation complete")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare Feast data sources.")
    parser.add_argument(
        "--processed-dir", type=Path, default=PROCESSED_DATA_DIR,
        help="Root of processed feature files",
    )
    parser.add_argument(
        "--circuit-csv", type=Path, default=CIRCUIT_REF_PATH,
        help="Path to circuit characteristics CSV",
    )
    parser.add_argument(
        "--feast-dir", type=Path, default=FEAST_DATA_DIR,
        help="Output directory for Feast data sources",
    )
    args = parser.parse_args()
    prepare_all(args.processed_dir, args.circuit_csv, args.feast_dir)
