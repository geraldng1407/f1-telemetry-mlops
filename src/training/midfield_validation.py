"""Williams Midfield Validation — targeted evaluation on midfield constructors.

Filters the test set to midfield teams and evaluates model accuracy across
challenging conditions: dirty-air exposure, non-optimal tire strategies, and
variable weather.  Also selects case-study battles for narrative analysis.

Usage:
    python -m src.training.midfield_validation
    python -m src.training.midfield_validation --model-uri runs:/<id>/model
    python -m src.training.midfield_validation --data-path data/training/labeled_dataset.parquet
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import tempfile
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import mlflow
import numpy as np
import pandas as pd
from loguru import logger

from src.features.target import STINT_GROUP_KEYS
from src.training.base import (
    MLFLOW_EXPERIMENT,
    MLFLOW_TRACKING_URI,
    load_training_data,
    prepare_features,
    split_by_time,
    split_by_time_adaptive,
)
from src.training.metrics import evaluate_all

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MIDFIELD_CONSTRUCTORS_2024: set[str] = {
    "Williams",
    "Haas F1 Team",
    "Alpine",
    "RB",
    "Kick Sauber",
}

DIRTY_AIR_CUMULATIVE_THRESHOLD = 5
RAINFALL_CHANGE_THRESHOLD = 0.1
TRACK_TEMP_RANGE_THRESHOLD = 8.0  # °C swing within a session


# ---------------------------------------------------------------------------
# Result containers
# ---------------------------------------------------------------------------


@dataclass
class SliceResult:
    """Metrics for a single evaluation slice."""

    name: str
    n_rows: int
    n_stints: int
    metrics: dict[str, float] = field(default_factory=dict)


@dataclass
class CaseStudy:
    """Structured summary of a single midfield battle case study."""

    session_id: str
    drivers: list[str]
    teams: list[str]
    compounds_used: dict[str, list[str]]
    actual_pit_laps: dict[str, list[int]]
    model_recommended_pit: dict[str, float | None]
    mae_per_driver: dict[str, float]
    summary: str


# ---------------------------------------------------------------------------
# Team name matching
# ---------------------------------------------------------------------------


def _resolve_team_names(
    available_teams: set[str],
    target_teams: set[str],
) -> set[str]:
    """Match target constructor names against what exists in the data.

    Falls back to case-insensitive substring matching when exact match fails.
    """
    resolved: set[str] = set()
    for target in target_teams:
        if target in available_teams:
            resolved.add(target)
            continue
        target_lower = target.lower()
        for avail in available_teams:
            if target_lower in avail.lower() or avail.lower() in target_lower:
                resolved.add(avail)
                break
    return resolved


# ---------------------------------------------------------------------------
# Filtering functions
# ---------------------------------------------------------------------------


def filter_midfield(
    df: pd.DataFrame,
    constructors: set[str] | None = None,
) -> pd.DataFrame:
    """Filter to rows belonging to midfield constructors.

    If ``team`` column is missing, logs a warning and returns an empty frame.
    """
    if "team" not in df.columns:
        logger.warning("Column 'team' not found — cannot filter by constructor")
        return df.iloc[0:0].copy()

    if constructors is None:
        constructors = MIDFIELD_CONSTRUCTORS_2024

    available = set(df["team"].dropna().unique())
    matched = _resolve_team_names(available, constructors)

    if not matched:
        logger.warning(
            "No midfield teams matched. Available teams: {}",
            sorted(available),
        )
        return df.iloc[0:0].copy()

    logger.info("Midfield filter matched teams: {}", sorted(matched))
    result = df[df["team"].isin(matched)].copy()
    logger.info(
        "Midfield filter: {} -> {} rows ({} stints)",
        len(df),
        len(result),
        result.groupby(STINT_GROUP_KEYS).ngroups if len(result) else 0,
    )
    return result


def filter_dirty_air_stints(
    df: pd.DataFrame,
    threshold: int = DIRTY_AIR_CUMULATIVE_THRESHOLD,
) -> pd.DataFrame:
    """Select stints with significant dirty-air exposure.

    A stint qualifies when its max ``dirty_air_cumulative_laps`` exceeds
    *threshold* (default 5 laps within 1.5 s of the car ahead).
    """
    if "dirty_air_cumulative_laps" not in df.columns:
        logger.warning("dirty_air_cumulative_laps column missing")
        return df.iloc[0:0].copy()

    stint_max = df.groupby(STINT_GROUP_KEYS)["dirty_air_cumulative_laps"].transform("max")
    result = df[stint_max > threshold].copy()
    logger.info(
        "Dirty-air filter (>{} cumulative laps): {} -> {} rows",
        threshold,
        len(df),
        len(result),
    )
    return result


def _get_race_strategy(df: pd.DataFrame) -> pd.Series:
    """For each driver in each session, return the ordered compound sequence as a tuple."""
    strat = (
        df.groupby(["session_id", "driver_number", "stint_number"])["compound"]
        .first()
        .reset_index()
        .sort_values(["session_id", "driver_number", "stint_number"])
    )
    return strat.groupby(["session_id", "driver_number"])["compound"].apply(tuple)


def filter_non_optimal_strategy(df: pd.DataFrame) -> pd.DataFrame:
    """Select stints from drivers who used a non-consensus tire strategy.

    For each race the "consensus" strategy is the most common compound
    sequence.  Drivers on a different sequence are considered non-optimal.
    """
    if "compound" not in df.columns:
        logger.warning("compound column missing — cannot determine strategy")
        return df.iloc[0:0].copy()

    strategies = _get_race_strategy(df)

    non_optimal_keys: set[tuple[str, str]] = set()
    for session_id in strategies.index.get_level_values("session_id").unique():
        session_strats = strategies.loc[session_id]
        counts = Counter(session_strats.values)
        consensus = counts.most_common(1)[0][0]
        for driver, strat in session_strats.items():
            if strat != consensus:
                non_optimal_keys.add((session_id, driver))

    if not non_optimal_keys:
        logger.info("No non-optimal strategies found in dataset")
        return df.iloc[0:0].copy()

    mask = df.apply(
        lambda r: (r["session_id"], r["driver_number"]) in non_optimal_keys, axis=1
    )
    result = df[mask].copy()
    logger.info(
        "Non-optimal strategy filter: {} -> {} rows ({} driver-races)",
        len(df),
        len(result),
        len(non_optimal_keys),
    )
    return result


def filter_variable_weather(df: pd.DataFrame) -> pd.DataFrame:
    """Select races with variable weather conditions.

    A session qualifies when rainfall transitions (any non-zero rainfall
    appearing alongside zero-rainfall laps) or when track temperature range
    exceeds TRACK_TEMP_RANGE_THRESHOLD within the session.
    """
    variable_sessions: set[str] = set()

    for session_id, grp in df.groupby("session_id"):
        if "rainfall" in grp.columns:
            rain = grp["rainfall"].dropna()
            if len(rain) > 0:
                has_dry = (rain <= RAINFALL_CHANGE_THRESHOLD).any()
                has_wet = (rain > RAINFALL_CHANGE_THRESHOLD).any()
                if has_dry and has_wet:
                    variable_sessions.add(session_id)
                    continue

        if "track_temp_c" in grp.columns:
            temps = grp["track_temp_c"].dropna()
            if len(temps) > 1:
                temp_range = temps.max() - temps.min()
                if temp_range > TRACK_TEMP_RANGE_THRESHOLD:
                    variable_sessions.add(session_id)

    if not variable_sessions:
        logger.info("No variable-weather sessions found")
        return df.iloc[0:0].copy()

    result = df[df["session_id"].isin(variable_sessions)].copy()
    logger.info(
        "Variable-weather filter: {} -> {} rows ({} sessions: {})",
        len(df),
        len(result),
        len(variable_sessions),
        sorted(variable_sessions),
    )
    return result


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------


def evaluate_slice(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    df: pd.DataFrame,
    name: str,
) -> SliceResult:
    """Evaluate a single data slice and return structured results."""
    if len(y_true) == 0:
        return SliceResult(name=name, n_rows=0, n_stints=0, metrics={})

    metrics = evaluate_all(y_true, y_pred, df)
    n_stints = df.groupby(STINT_GROUP_KEYS).ngroups if len(df) else 0
    return SliceResult(name=name, n_rows=len(y_true), n_stints=n_stints, metrics=metrics)


def evaluate_midfield_slices(
    model: Any,
    test_df: pd.DataFrame,
    X_test: np.ndarray,
    y_test: np.ndarray,
) -> dict[str, SliceResult]:
    """Run evaluation on all midfield sub-slices.

    Returns a dict mapping slice name to its :class:`SliceResult`.
    """
    y_pred_all = model.predict(X_test)

    results: dict[str, SliceResult] = {}

    results["full_test"] = evaluate_slice(y_test, y_pred_all, test_df, "full_test")

    midfield_df = filter_midfield(test_df)
    if len(midfield_df) > 0:
        midfield_idx = midfield_df.index
        test_idx_map = {idx: i for i, idx in enumerate(test_df.index)}
        positions = [test_idx_map[idx] for idx in midfield_idx if idx in test_idx_map]
        y_true_mid = y_test[positions]
        y_pred_mid = y_pred_all[positions]
        mid_df_aligned = midfield_df.loc[[test_df.index[p] for p in positions]]
        results["midfield_all"] = evaluate_slice(
            y_true_mid, y_pred_mid, mid_df_aligned, "midfield_all"
        )

        slices = {
            "dirty_air": filter_dirty_air_stints(midfield_df),
            "non_optimal_strategy": filter_non_optimal_strategy(midfield_df),
            "variable_weather": filter_variable_weather(midfield_df),
        }

        for slice_name, slice_df in slices.items():
            if len(slice_df) == 0:
                results[slice_name] = SliceResult(
                    name=slice_name, n_rows=0, n_stints=0, metrics={}
                )
                continue
            slice_idx = slice_df.index
            positions_s = [test_idx_map[idx] for idx in slice_idx if idx in test_idx_map]
            y_true_s = y_test[positions_s]
            y_pred_s = y_pred_all[positions_s]
            df_s = slice_df.loc[[test_df.index[p] for p in positions_s]]
            results[slice_name] = evaluate_slice(y_true_s, y_pred_s, df_s, slice_name)
    else:
        for name in ("midfield_all", "dirty_air", "non_optimal_strategy", "variable_weather"):
            results[name] = SliceResult(name=name, n_rows=0, n_stints=0, metrics={})

    return results


# ---------------------------------------------------------------------------
# Case studies
# ---------------------------------------------------------------------------


def _find_pit_laps(df: pd.DataFrame, driver: str) -> list[int]:
    """Return stint_lap_numbers where the driver pitted (in-lap)."""
    drv = df[df["driver_number"] == driver]
    if "is_inlap" not in drv.columns:
        return []
    pit_rows = drv[drv["is_inlap"] == 1]
    return sorted(pit_rows["stint_lap_number"].astype(int).tolist())


def _recommended_pit_lap(
    drv_df: pd.DataFrame,
    y_pred: np.ndarray,
    early_margin: int = 2,
) -> float | None:
    """Earliest lap where the model recommends pitting (cliff imminent)."""
    if len(drv_df) == 0:
        return None
    work = drv_df[["stint_lap_number"]].copy()
    work["pred_cliff"] = work["stint_lap_number"].values + y_pred
    return float(work["pred_cliff"].min())


def run_case_studies(
    model: Any,
    test_df: pd.DataFrame,
    X_test: np.ndarray,
    y_test: np.ndarray,
    n: int = 5,
) -> list[CaseStudy]:
    """Select and analyse the most interesting midfield battles.

    Selection criteria: sessions with multiple midfield drivers finishing
    close together, ideally with different strategies.
    """
    midfield_df = filter_midfield(test_df)
    if len(midfield_df) == 0:
        logger.warning("No midfield data for case studies")
        return []

    y_pred_all = model.predict(X_test)
    test_idx_map = {idx: i for i, idx in enumerate(test_df.index)}

    session_scores: list[tuple[str, float]] = []
    for session_id, grp in midfield_df.groupby("session_id"):
        n_drivers = grp["driver_number"].nunique()
        n_stints = grp.groupby(STINT_GROUP_KEYS).ngroups
        has_dirty = (
            grp["dirty_air_cumulative_laps"].max() > DIRTY_AIR_CUMULATIVE_THRESHOLD
            if "dirty_air_cumulative_laps" in grp.columns
            else False
        )
        score = n_drivers * 2 + n_stints + (5 if has_dirty else 0)
        session_scores.append((session_id, score))

    session_scores.sort(key=lambda x: x[1], reverse=True)
    selected = [s[0] for s in session_scores[:n]]

    case_studies: list[CaseStudy] = []
    for session_id in selected:
        sess = midfield_df[midfield_df["session_id"] == session_id]
        drivers = sorted(sess["driver_number"].unique().tolist())
        teams = sorted(sess["team"].unique().tolist()) if "team" in sess.columns else []

        compounds_used: dict[str, list[str]] = {}
        actual_pit_laps: dict[str, list[int]] = {}
        model_rec: dict[str, float | None] = {}
        mae_per_driver: dict[str, float] = {}

        for drv in drivers:
            drv_df = sess[sess["driver_number"] == drv]
            if "compound" in drv_df.columns:
                compounds_used[drv] = (
                    drv_df.groupby("stint_number")["compound"]
                    .first()
                    .dropna()
                    .tolist()
                )
            actual_pit_laps[drv] = _find_pit_laps(sess, drv)

            drv_positions = [test_idx_map[idx] for idx in drv_df.index if idx in test_idx_map]
            if drv_positions:
                drv_y_true = y_test[drv_positions]
                drv_y_pred = y_pred_all[drv_positions]
                mae_per_driver[drv] = float(np.mean(np.abs(drv_y_true - drv_y_pred)))
                model_rec[drv] = _recommended_pit_lap(drv_df, drv_y_pred)
            else:
                mae_per_driver[drv] = float("nan")
                model_rec[drv] = None

        avg_mae = np.nanmean(list(mae_per_driver.values()))
        summary = (
            f"Session {session_id}: {len(drivers)} midfield drivers "
            f"({', '.join(teams)}), avg MAE={avg_mae:.2f} laps. "
            f"Strategies: {compounds_used}"
        )

        case_studies.append(
            CaseStudy(
                session_id=session_id,
                drivers=drivers,
                teams=teams,
                compounds_used=compounds_used,
                actual_pit_laps=actual_pit_laps,
                model_recommended_pit=model_rec,
                mae_per_driver=mae_per_driver,
                summary=summary,
            )
        )

    return case_studies


# ---------------------------------------------------------------------------
# MLflow logging
# ---------------------------------------------------------------------------


def log_midfield_validation(
    slice_results: dict[str, SliceResult],
    case_studies: list[CaseStudy],
    parent_run_id: str | None = None,
) -> str:
    """Log midfield validation results to MLflow."""
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(MLFLOW_EXPERIMENT)

    tags = {"validation_type": "midfield", "script": "midfield_validation"}
    if parent_run_id:
        tags["parent_run_id"] = parent_run_id

    with mlflow.start_run(tags=tags) as run:
        for slice_name, sr in slice_results.items():
            mlflow.log_metric(f"{slice_name}_n_rows", sr.n_rows)
            mlflow.log_metric(f"{slice_name}_n_stints", sr.n_stints)
            for metric_name, value in sr.metrics.items():
                if isinstance(value, (int, float)) and math.isfinite(value):
                    mlflow.log_metric(f"{slice_name}_{metric_name}", value)

        with tempfile.TemporaryDirectory() as tmpdir:
            report = _build_report(slice_results, case_studies)
            report_path = Path(tmpdir) / "midfield_validation_report.json"
            with open(report_path, "w") as f:
                json.dump(report, f, indent=2, default=str)
            mlflow.log_artifact(str(report_path), artifact_path="midfield_validation")

            md_path = Path(tmpdir) / "midfield_validation_report.md"
            with open(md_path, "w") as f:
                f.write(_build_markdown_report(slice_results, case_studies))
            mlflow.log_artifact(str(md_path), artifact_path="midfield_validation")

        run_id = run.info.run_id
        logger.info("Logged midfield validation to MLflow run {}", run_id)
        return run_id


def _build_report(
    slice_results: dict[str, SliceResult],
    case_studies: list[CaseStudy],
) -> dict[str, Any]:
    """Build a JSON-serialisable report dict."""
    return {
        "slices": {
            name: {
                "n_rows": sr.n_rows,
                "n_stints": sr.n_stints,
                "metrics": sr.metrics,
            }
            for name, sr in slice_results.items()
        },
        "case_studies": [
            {
                "session_id": cs.session_id,
                "drivers": cs.drivers,
                "teams": cs.teams,
                "compounds_used": cs.compounds_used,
                "actual_pit_laps": cs.actual_pit_laps,
                "model_recommended_pit": cs.model_recommended_pit,
                "mae_per_driver": cs.mae_per_driver,
                "summary": cs.summary,
            }
            for cs in case_studies
        ],
    }


def _build_markdown_report(
    slice_results: dict[str, SliceResult],
    case_studies: list[CaseStudy],
) -> str:
    """Generate a Markdown-formatted validation report."""
    lines: list[str] = ["# Williams Midfield Validation Report\n"]

    lines.append("## Evaluation Slices\n")
    lines.append("| Slice | Rows | Stints | MAE | Precision@3 | Strategy Acc. |")
    lines.append("|-------|------|--------|-----|-------------|---------------|")
    for name, sr in slice_results.items():
        mae = sr.metrics.get("mae", float("nan"))
        p3 = sr.metrics.get("precision_at_3", float("nan"))
        sa = sr.metrics.get("strategy_accuracy", float("nan"))
        lines.append(
            f"| {name} | {sr.n_rows} | {sr.n_stints} "
            f"| {mae:.3f} | {p3:.3f} | {sa:.3f} |"
        )

    lines.append("\n## Case Studies\n")
    for i, cs in enumerate(case_studies, 1):
        lines.append(f"### Case Study {i}: {cs.session_id}\n")
        lines.append(f"**Teams:** {', '.join(cs.teams)}\n")
        lines.append(f"**Drivers:** {', '.join(cs.drivers)}\n")
        lines.append(f"**Strategies:** {cs.compounds_used}\n")
        lines.append("| Driver | Actual Pits | Model Rec. Pit | MAE |")
        lines.append("|--------|-------------|----------------|-----|")
        for drv in cs.drivers:
            pits = cs.actual_pit_laps.get(drv, [])
            rec = cs.model_recommended_pit.get(drv)
            mae = cs.mae_per_driver.get(drv, float("nan"))
            rec_str = f"{rec:.1f}" if rec is not None else "N/A"
            lines.append(f"| {drv} | {pits} | {rec_str} | {mae:.2f} |")
        lines.append(f"\n{cs.summary}\n")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Summary printing
# ---------------------------------------------------------------------------


def print_summary(
    slice_results: dict[str, SliceResult],
    case_studies: list[CaseStudy],
) -> None:
    """Print a human-readable summary to stdout."""
    print("\n" + "=" * 70)
    print("WILLIAMS MIDFIELD VALIDATION RESULTS")
    print("=" * 70)

    header = f"{'Slice':<25} {'Rows':>6} {'Stints':>7} {'MAE':>8} {'P@3':>8} {'Strat':>8}"
    print(header)
    print("-" * 70)
    for name, sr in slice_results.items():
        mae = sr.metrics.get("mae", float("nan"))
        p3 = sr.metrics.get("precision_at_3", float("nan"))
        sa = sr.metrics.get("strategy_accuracy", float("nan"))
        print(f"{name:<25} {sr.n_rows:>6} {sr.n_stints:>7} {mae:>8.3f} {p3:>8.3f} {sa:>8.3f}")

    if case_studies:
        print("\n" + "-" * 70)
        print("CASE STUDIES")
        print("-" * 70)
        for i, cs in enumerate(case_studies, 1):
            print(f"\n  [{i}] {cs.summary}")

    print("=" * 70 + "\n")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Williams Midfield Validation — evaluate tire-cliff models on midfield teams.",
    )
    parser.add_argument(
        "--model-uri",
        default=None,
        help=(
            "MLflow model URI (e.g. 'runs:/<run_id>/model' or "
            "'models:/tire-cliff-xgboost@production'). "
            "If omitted, trains a fresh XGBoost baseline."
        ),
    )
    parser.add_argument(
        "--data-path",
        type=Path,
        default=None,
        help="Path to labeled dataset parquet (default: data/training/labeled_dataset.parquet).",
    )
    parser.add_argument(
        "--n-case-studies",
        type=int,
        default=5,
        help="Number of case studies to generate (default: 5).",
    )
    parser.add_argument(
        "--no-mlflow",
        action="store_true",
        help="Skip MLflow logging.",
    )
    args = parser.parse_args(argv)

    df = load_training_data(path=args.data_path)
    train_df, _val_df, test_df = split_by_time(df)

    if train_df.empty and not df.empty:
        train_df, _val_df, test_df = split_by_time_adaptive(df)

    if test_df.empty:
        logger.error("Test split is empty — nothing to validate")
        return 1

    X_train, _y_train = prepare_features(train_df, fit_encoder=True)
    _X_val, _y_val = prepare_features(_val_df)
    X_test, y_test = prepare_features(test_df)

    parent_run_id: str | None = None
    if args.model_uri:
        logger.info("Loading model from {}", args.model_uri)
        model = mlflow.pyfunc.load_model(args.model_uri)
    else:
        logger.info("No --model-uri; training fresh XGBoost baseline")
        from src.training.baseline import DEFAULT_XGB_PARAMS, xgb_model_fn

        model = xgb_model_fn(
            X_train,
            _y_train,
            DEFAULT_XGB_PARAMS,
            eval_set=(X_test, y_test) if len(X_test) else None,
            show_progress=True,
        )

    logger.info("Running midfield validation on {} test rows", len(test_df))
    slice_results = evaluate_midfield_slices(model, test_df, X_test, y_test)

    logger.info("Generating {} case studies", args.n_case_studies)
    case_studies = run_case_studies(model, test_df, X_test, y_test, n=args.n_case_studies)

    print_summary(slice_results, case_studies)

    if not args.no_mlflow:
        run_id = log_midfield_validation(slice_results, case_studies, parent_run_id)
        logger.info("Validation logged to MLflow run {}", run_id)

    return 0


if __name__ == "__main__":
    sys.exit(main())
