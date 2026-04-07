"""Target variable construction for tire degradation cliff prediction.

Identifies the "cliff lap" in each stint — the first lap where lap time exceeds
the stint average by more than CLIFF_THRESHOLD_S — and computes a forward-looking
``laps_to_cliff`` target for every observation.

Stints where no cliff occurred (driver pitted strategically before degradation)
are marked as censored and clipped to remaining stint length, suitable for
survival-analysis framing.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from loguru import logger

from src.features.constants import (
    CLIFF_THRESHOLD_S,
    FEAST_DATA_DIR,
    MIN_STINT_LAPS,
    TRAINING_DATA_DIR,
)

STINT_GROUP_KEYS = ["session_id", "driver_number", "stint_number"]


# ---------------------------------------------------------------------------
# Cliff detection
# ---------------------------------------------------------------------------


def detect_cliff_lap(stint_df: pd.DataFrame) -> int | None:
    """Return the ``stint_lap_number`` of the first cliff lap, or *None*.

    A cliff lap is the first *clean* lap whose ``fuel_corrected_laptime``
    exceeds the clean-lap stint average by more than ``CLIFF_THRESHOLD_S``.

    "Clean" means the lap is not an in-lap, out-lap, or safety-car lap.
    """
    clean = stint_df[
        (stint_df["is_inlap"] == 0)
        & (stint_df["is_outlap"] == 0)
        & (stint_df["is_sc_lap"] == 0)
    ].copy()

    if len(clean) < MIN_STINT_LAPS:
        return None

    fct = clean["fuel_corrected_laptime"]
    if fct.isna().all():
        return None

    stint_avg = fct.mean()
    threshold = stint_avg + CLIFF_THRESHOLD_S

    cliff_rows = clean.loc[fct > threshold]
    if cliff_rows.empty:
        return None

    return int(cliff_rows.iloc[0]["stint_lap_number"])


# ---------------------------------------------------------------------------
# Per-stint target computation
# ---------------------------------------------------------------------------


def _compute_stint_targets(stint_df: pd.DataFrame) -> pd.DataFrame:
    """Add target columns to a single stint's rows.

    Returns the stint DataFrame augmented with:
    * ``cliff_lap`` — stint_lap_number where the cliff was detected (NaN if censored)
    * ``laps_to_cliff`` — forward-looking laps until the cliff
    * ``is_censored`` — 1 if no cliff was detected (strategic pit)
    * ``laps_remaining_in_stint`` — laps left regardless of cliff
    """
    df = stint_df.copy()

    clean_mask = (
        (df["is_inlap"] == 0)
        & (df["is_outlap"] == 0)
        & (df["is_sc_lap"] == 0)
    )
    n_clean = clean_mask.sum()

    max_stint_lap = int(df["stint_lap_number"].max())
    df["laps_remaining_in_stint"] = max_stint_lap - df["stint_lap_number"]

    if n_clean < MIN_STINT_LAPS:
        df["cliff_lap"] = float("nan")
        df["laps_to_cliff"] = float("nan")
        df["is_censored"] = float("nan")
        return df

    cliff = detect_cliff_lap(df)

    if cliff is not None:
        df["cliff_lap"] = cliff
        df["laps_to_cliff"] = cliff - df["stint_lap_number"]
        df["is_censored"] = 0
    else:
        df["cliff_lap"] = float("nan")
        df["laps_to_cliff"] = max_stint_lap - df["stint_lap_number"]
        df["is_censored"] = 1

    return df


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def compute_targets(df: pd.DataFrame) -> pd.DataFrame:
    """Compute target variables for every row in the dataset.

    Parameters
    ----------
    df:
        DataFrame with at least the columns: ``session_id``, ``driver_number``,
        ``stint_number``, ``stint_lap_number``, ``fuel_corrected_laptime``,
        ``is_inlap``, ``is_outlap``, ``is_sc_lap``.

    Returns
    -------
    pd.DataFrame
        Input DataFrame augmented with ``cliff_lap``, ``laps_to_cliff``,
        ``is_censored``, and ``laps_remaining_in_stint``.  Rows belonging to
        unusable stints (too short) and rows past the cliff (``laps_to_cliff < 0``)
        are dropped.  Out-lap rows are also dropped (they are not meaningful
        prediction points).
    """
    required = [
        *STINT_GROUP_KEYS,
        "stint_lap_number",
        "fuel_corrected_laptime",
        "is_inlap",
        "is_outlap",
        "is_sc_lap",
    ]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    frames: list[pd.DataFrame] = []
    for _keys, stint_df in df.groupby(STINT_GROUP_KEYS, sort=False):
        frames.append(_compute_stint_targets(stint_df))

    if not frames:
        logger.warning("No stints found in input data")
        return df.head(0)

    result = pd.concat(frames, ignore_index=True)

    before = len(result)

    # Drop rows from stints too short to be usable (targets are NaN)
    result = result.dropna(subset=["laps_to_cliff"])

    # Drop post-cliff rows (negative laps_to_cliff)
    result = result[result["laps_to_cliff"] >= 0]

    # Drop out-laps — not meaningful prediction points
    result = result[result["is_outlap"] == 0]

    after = len(result)
    logger.info(
        "Target computation: {} -> {} rows ({} dropped)",
        before,
        after,
        before - after,
    )

    return result.reset_index(drop=True)


# ---------------------------------------------------------------------------
# Standalone pipeline
# ---------------------------------------------------------------------------


def build_targets(
    input_path: Path | None = None,
    output_path: Path | None = None,
) -> pd.DataFrame:
    """Read consolidated features, compute targets, and write to disk.

    This is the standalone entry point; the dataset builder in ``dataset.py``
    calls ``compute_targets`` directly.
    """
    if input_path is None:
        input_path = FEAST_DATA_DIR / "stint_features.parquet"
    if output_path is None:
        output_path = TRAINING_DATA_DIR / "targets.parquet"

    logger.info("Reading features from {}", input_path)
    df = pd.read_parquet(input_path, engine="pyarrow")
    logger.info("Loaded {} rows", len(df))

    df = compute_targets(df)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(output_path, index=False, engine="pyarrow")
    logger.info("Wrote {} target rows to {}", len(df), output_path)

    n_stints = df.groupby(STINT_GROUP_KEYS).ngroups
    censored_pct = df["is_censored"].mean() * 100
    logger.info(
        "Summary: {} stints, {:.1f}% censored, {:.1f} mean laps_to_cliff",
        n_stints,
        censored_pct,
        df["laps_to_cliff"].mean(),
    )

    return df


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Compute target variables (cliff detection) from processed features."
    )
    parser.add_argument(
        "--input", type=Path, default=None,
        help="Path to consolidated stint features parquet (default: data/feast/stint_features.parquet)",
    )
    parser.add_argument(
        "--output", type=Path, default=None,
        help="Output path for targets parquet (default: data/training/targets.parquet)",
    )
    args = parser.parse_args()
    build_targets(args.input, args.output)
