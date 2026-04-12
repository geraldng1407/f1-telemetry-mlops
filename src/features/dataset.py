"""Build a labeled training dataset via Feast point-in-time joins.

Reads consolidated features, computes targets, then routes through Feast's
``get_historical_features`` to produce the final labeled dataset. This ensures
the training pipeline uses the exact same join logic as production inference.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
from loguru import logger

from src.features.constants import (
    ENGINEERED_FEATURE_COLUMNS,
    FEAST_DATA_DIR,
    FEAST_ENTITY_COLUMNS,
    TARGET_COLUMNS,
    TRAINING_DATA_DIR,
)
from src.features.target import STINT_GROUP_KEYS, compute_targets


# ---------------------------------------------------------------------------
# Feast helpers
# ---------------------------------------------------------------------------

FEAST_REPO_PATH = Path(__file__).resolve().parent.parent.parent / "feature_repo"

_TRAINING_FEATURES = [
    "stint_telemetry_features:stint_lap_number",
    "stint_telemetry_features:rolling_mean_laptime_3",
    "stint_telemetry_features:rolling_mean_laptime_5",
    "stint_telemetry_features:rolling_var_laptime_3",
    "stint_telemetry_features:rolling_var_laptime_5",
    "stint_telemetry_features:sector1_delta_from_best",
    "stint_telemetry_features:sector2_delta_from_best",
    "stint_telemetry_features:sector3_delta_from_best",
    "stint_telemetry_features:tire_age_laps",
    "stint_telemetry_features:fuel_corrected_laptime",
    "stint_telemetry_features:gap_to_car_ahead_s",
    "stint_telemetry_features:is_dirty_air",
    "stint_telemetry_features:dirty_air_cumulative_laps",
    "stint_telemetry_features:is_inlap",
    "stint_telemetry_features:is_outlap",
    "stint_telemetry_features:is_sc_lap",
    "weather_features:track_temp_c",
    "weather_features:air_temp_c",
    "weather_features:humidity",
    "weather_features:rainfall",
    "weather_features:track_evolution_index",
    "circuit_features:circuit_high_speed_corners",
    "circuit_features:circuit_medium_speed_corners",
    "circuit_features:circuit_low_speed_corners",
    "circuit_features:circuit_abrasiveness",
    "circuit_features:circuit_altitude_m",
    "circuit_features:circuit_tire_limitation",
    "circuit_features:circuit_street_circuit",
]


def _get_feast_store():
    """Instantiate a Feast FeatureStore pointing at the repo."""
    from feast import FeatureStore

    return FeatureStore(repo_path=str(FEAST_REPO_PATH))


# ---------------------------------------------------------------------------
# Dataset builder
# ---------------------------------------------------------------------------


def build_labeled_dataset(
    input_path: Path | None = None,
    output_path: Path | None = None,
    feast_repo: Path | None = None,
    use_feast: bool = True,
) -> pd.DataFrame:
    """Build the labeled training dataset.

    Parameters
    ----------
    input_path:
        Consolidated stint features parquet. Defaults to
        ``data/feast/stint_features.parquet``.
    output_path:
        Where to write the final labeled dataset. Defaults to
        ``data/training/labeled_dataset.parquet``.
    feast_repo:
        Path to Feast feature repository. Defaults to ``feature_repo/``.
    use_feast:
        If True, pull features through Feast ``get_historical_features``.
        If False, build the dataset directly from the input parquet (useful
        when Feast has not been applied yet).

    Returns
    -------
    pd.DataFrame
        The labeled dataset with features and target columns.
    """
    if input_path is None:
        input_path = FEAST_DATA_DIR / "stint_features.parquet"
    if output_path is None:
        output_path = TRAINING_DATA_DIR / "labeled_dataset.parquet"
    if feast_repo is not None:
        global FEAST_REPO_PATH  # noqa: PLW0603
        FEAST_REPO_PATH = feast_repo

    # ------------------------------------------------------------------
    # 1. Read consolidated features and compute targets
    # ------------------------------------------------------------------
    logger.info("Reading consolidated features from {}", input_path)
    raw_df = pd.read_parquet(input_path, engine="pyarrow")
    logger.info("Loaded {} rows from features parquet", len(raw_df))

    targeted_df = compute_targets(raw_df)
    logger.info("{} rows after target computation", len(targeted_df))

    if targeted_df.empty:
        logger.warning("No rows survived target computation — aborting")
        return targeted_df

    # ------------------------------------------------------------------
    # 2. Feature retrieval
    # ------------------------------------------------------------------
    if use_feast:
        labeled_df = _build_via_feast(targeted_df)
    else:
        labeled_df = _build_direct(targeted_df)

    # ------------------------------------------------------------------
    # 3. Add compound column (useful for stratified analysis)
    # ------------------------------------------------------------------
    if "Compound" in raw_df.columns and "Compound" not in labeled_df.columns:
        compound_map = raw_df.set_index(
            STINT_GROUP_KEYS + ["stint_lap_number"]
        )["Compound"]
        compound_map = compound_map[~compound_map.index.duplicated(keep="first")]
        join_idx = labeled_df.set_index(
            STINT_GROUP_KEYS + ["stint_lap_number"]
        ).index
        labeled_df["compound"] = compound_map.reindex(join_idx).values

    # ------------------------------------------------------------------
    # 3b. Add team column (needed for midfield validation slicing)
    # ------------------------------------------------------------------
    if "Team" in raw_df.columns and "team" not in labeled_df.columns:
        team_map = raw_df.set_index(
            STINT_GROUP_KEYS + ["stint_lap_number"]
        )["Team"]
        team_map = team_map[~team_map.index.duplicated(keep="first")]
        join_idx = labeled_df.set_index(
            STINT_GROUP_KEYS + ["stint_lap_number"]
        ).index
        labeled_df["team"] = team_map.reindex(join_idx).values

    # ------------------------------------------------------------------
    # 4. Write and report
    # ------------------------------------------------------------------
    output_path.parent.mkdir(parents=True, exist_ok=True)
    labeled_df.to_parquet(output_path, index=False, engine="pyarrow")
    logger.info("Wrote labeled dataset ({} rows) to {}", len(labeled_df), output_path)

    _log_summary(labeled_df, output_path)

    return labeled_df


def _build_via_feast(targeted_df: pd.DataFrame) -> pd.DataFrame:
    """Pull features through Feast point-in-time join."""
    logger.info("Retrieving features via Feast historical join")
    store = _get_feast_store()

    entity_df = targeted_df[
        ["session_id", "driver_number", "stint_number", "location", "event_timestamp"]
    ].copy()

    historical = store.get_historical_features(
        entity_df=entity_df,
        features=_TRAINING_FEATURES,
    ).to_df()

    target_cols = ["laps_to_cliff", "is_censored", "cliff_lap", "laps_remaining_in_stint"]
    for col in target_cols:
        if col in targeted_df.columns:
            historical[col] = targeted_df[col].values

    return historical


def _build_direct(targeted_df: pd.DataFrame) -> pd.DataFrame:
    """Build dataset directly from the input DataFrame (no Feast)."""
    logger.info("Building dataset directly (Feast bypass)")

    keep_cols = (
        FEAST_ENTITY_COLUMNS
        + ["event_timestamp", "stint_lap_number"]
        + [c for c in ENGINEERED_FEATURE_COLUMNS if c in targeted_df.columns]
        + [c for c in TARGET_COLUMNS if c in targeted_df.columns]
    )
    seen: set[str] = set()
    deduped: list[str] = []
    for c in keep_cols:
        if c not in seen and c in targeted_df.columns:
            deduped.append(c)
            seen.add(c)

    return targeted_df[deduped].copy()


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------


def _log_summary(df: pd.DataFrame, output_path: Path) -> None:
    """Log and persist a summary of the labeled dataset."""
    n_stints = df.groupby(STINT_GROUP_KEYS).ngroups
    n_sessions = df["session_id"].nunique() if "session_id" in df.columns else 0
    censored_pct = df["is_censored"].mean() * 100 if "is_censored" in df.columns else 0

    seasons = set()
    if "session_id" in df.columns:
        seasons = set(df["session_id"].str.split("_").str[0].unique())

    summary = {
        "total_rows": len(df),
        "total_stints": n_stints,
        "total_sessions": n_sessions,
        "seasons": sorted(seasons),
        "censored_pct": round(censored_pct, 2),
        "mean_laps_to_cliff": round(float(df["laps_to_cliff"].mean()), 2)
        if "laps_to_cliff" in df.columns
        else None,
        "median_laps_to_cliff": round(float(df["laps_to_cliff"].median()), 2)
        if "laps_to_cliff" in df.columns
        else None,
    }

    if "compound" in df.columns:
        summary["compound_distribution"] = (
            df["compound"].value_counts(dropna=False).to_dict()
        )

    logger.info("=== Labeled Dataset Summary ===")
    for k, v in summary.items():
        logger.info("  {}: {}", k, v)

    summary_path = output_path.parent / "dataset_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2, default=str)
    logger.info("Summary written to {}", summary_path)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Build a labeled training dataset with Feast point-in-time join."
    )
    parser.add_argument(
        "--input", type=Path, default=None,
        help="Consolidated stint features parquet (default: data/feast/stint_features.parquet)",
    )
    parser.add_argument(
        "--output", type=Path, default=None,
        help="Output labeled dataset path (default: data/training/labeled_dataset.parquet)",
    )
    parser.add_argument(
        "--feast-repo", type=Path, default=None,
        help="Feast repository path (default: feature_repo/)",
    )
    parser.add_argument(
        "--no-feast", action="store_true",
        help="Skip Feast and build dataset directly from input parquet",
    )
    args = parser.parse_args()
    build_labeled_dataset(
        input_path=args.input,
        output_path=args.output,
        feast_repo=args.feast_repo,
        use_feast=not args.no_feast,
    )
