"""Data quality checks and feature distribution documentation.

Validates the labeled training dataset, checks value ranges and null rates,
and produces a structured JSON report with per-feature distribution statistics.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
from loguru import logger

from src.features.constants import (
    ENGINEERED_FEATURE_COLUMNS,
    FEAST_ENTITY_COLUMNS,
    TARGET_COLUMNS,
    TRAINING_DATA_DIR,
)


# ---------------------------------------------------------------------------
# Check definitions
# ---------------------------------------------------------------------------

_RANGE_CHECKS: dict[str, tuple[float | None, float | None]] = {
    "tire_age_laps": (0, None),
    "stint_lap_number": (1, None),
    "laps_to_cliff": (0, None),
    "laps_remaining_in_stint": (0, None),
    "is_censored": (0, 1),
    "track_temp_c": (-10, 70),
    "air_temp_c": (-10, 55),
    "humidity": (0, 100),
    "rolling_mean_laptime_3": (40, 200),
    "rolling_mean_laptime_5": (40, 200),
    "fuel_corrected_laptime": (40, 200),
}

_TARGET_REQUIRED_COLS = ["laps_to_cliff", "is_censored"]

_MIN_VOLUME = 50_000


# ---------------------------------------------------------------------------
# Individual checks
# ---------------------------------------------------------------------------


def check_schema(df: pd.DataFrame) -> dict:
    """Verify all expected columns are present."""
    expected_features = [c for c in ENGINEERED_FEATURE_COLUMNS]
    expected_targets = [c for c in TARGET_COLUMNS]
    expected_entities = [c for c in FEAST_ENTITY_COLUMNS]

    missing_features = [c for c in expected_features if c not in df.columns]
    missing_targets = [c for c in expected_targets if c not in df.columns]
    missing_entities = [c for c in expected_entities if c not in df.columns]

    passed = not missing_targets
    return {
        "check": "schema",
        "passed": passed,
        "missing_features": missing_features,
        "missing_targets": missing_targets,
        "missing_entities": missing_entities,
        "total_columns": len(df.columns),
    }


def check_null_rates(df: pd.DataFrame) -> dict:
    """Check null rates across all columns."""
    null_counts = df.isnull().sum()
    null_pcts = (null_counts / len(df) * 100).round(2)

    high_null_features = {
        col: float(pct)
        for col, pct in null_pcts.items()
        if pct > 10 and col not in FEAST_ENTITY_COLUMNS
    }

    target_nulls = {
        col: int(null_counts[col])
        for col in _TARGET_REQUIRED_COLS
        if col in df.columns and null_counts[col] > 0
    }

    passed = len(target_nulls) == 0
    return {
        "check": "null_rates",
        "passed": passed,
        "target_nulls": target_nulls,
        "high_null_features": high_null_features,
        "per_column": {
            col: {"null_count": int(null_counts[col]), "null_pct": float(null_pcts[col])}
            for col in df.columns
        },
    }


def check_value_ranges(df: pd.DataFrame) -> dict:
    """Validate that feature values fall within expected ranges."""
    violations: dict[str, dict] = {}

    for col, (lo, hi) in _RANGE_CHECKS.items():
        if col not in df.columns:
            continue
        valid = df[col].dropna()
        if valid.empty:
            continue

        col_violations = {}
        if lo is not None and valid.min() < lo:
            col_violations["below_min"] = {
                "min_expected": lo,
                "actual_min": float(valid.min()),
                "count": int((valid < lo).sum()),
            }
        if hi is not None and valid.max() > hi:
            col_violations["above_max"] = {
                "max_expected": hi,
                "actual_max": float(valid.max()),
                "count": int((valid > hi).sum()),
            }
        if col_violations:
            violations[col] = col_violations

    return {
        "check": "value_ranges",
        "passed": len(violations) == 0,
        "violations": violations,
    }


def check_volume(df: pd.DataFrame) -> dict:
    """Assert minimum row count for Phase 1 target."""
    return {
        "check": "volume",
        "passed": len(df) >= _MIN_VOLUME,
        "total_rows": len(df),
        "minimum_required": _MIN_VOLUME,
    }


def check_coverage(df: pd.DataFrame) -> dict:
    """Report data coverage across sessions, seasons, circuits, and drivers."""
    seasons = set()
    if "session_id" in df.columns:
        seasons = set(df["session_id"].str.split("_").str[0].unique())

    return {
        "check": "coverage",
        "passed": True,
        "unique_sessions": int(df["session_id"].nunique()) if "session_id" in df.columns else 0,
        "unique_drivers": int(df["driver_number"].nunique()) if "driver_number" in df.columns else 0,
        "unique_circuits": int(df["location"].nunique()) if "location" in df.columns else 0,
        "seasons": sorted(seasons),
        "unique_stints": int(
            df.groupby(["session_id", "driver_number", "stint_number"]).ngroups
        )
        if all(c in df.columns for c in ["session_id", "driver_number", "stint_number"])
        else 0,
    }


def compute_distributions(df: pd.DataFrame) -> dict:
    """Compute per-feature distribution statistics."""
    stats: dict[str, dict] = {}
    numeric_cols = df.select_dtypes(include="number").columns

    for col in numeric_cols:
        s = df[col].dropna()
        if s.empty:
            stats[col] = {"count": 0, "null_count": int(df[col].isna().sum())}
            continue

        stats[col] = {
            "count": int(s.count()),
            "null_count": int(df[col].isna().sum()),
            "null_pct": round(float(df[col].isna().mean() * 100), 2),
            "min": round(float(s.min()), 4),
            "max": round(float(s.max()), 4),
            "mean": round(float(s.mean()), 4),
            "median": round(float(s.median()), 4),
            "std": round(float(s.std()), 4),
            "p5": round(float(s.quantile(0.05)), 4),
            "p25": round(float(s.quantile(0.25)), 4),
            "p75": round(float(s.quantile(0.75)), 4),
            "p95": round(float(s.quantile(0.95)), 4),
        }

    return stats


def compute_target_distribution(df: pd.DataFrame) -> dict:
    """Detailed breakdown of the target variable."""
    result: dict = {}

    if "laps_to_cliff" in df.columns:
        ltc = df["laps_to_cliff"].dropna()
        bins = [0, 3, 6, 10, 15, 20, 30, float("inf")]
        labels = ["0-3", "4-6", "7-10", "11-15", "16-20", "21-30", "30+"]
        hist = pd.cut(ltc, bins=bins, labels=labels, right=True).value_counts().sort_index()
        result["laps_to_cliff_histogram"] = hist.to_dict()

    if "is_censored" in df.columns:
        result["censored_pct"] = round(float(df["is_censored"].mean() * 100), 2)
        result["censored_count"] = int(df["is_censored"].sum())
        result["uncensored_count"] = int((df["is_censored"] == 0).sum())

    if "compound" in df.columns:
        result["per_compound"] = {}
        for compound, group in df.groupby("compound", dropna=False):
            compound_key = str(compound) if pd.notna(compound) else "UNKNOWN"
            entry: dict = {"count": len(group)}
            if "laps_to_cliff" in group.columns:
                entry["mean_laps_to_cliff"] = round(float(group["laps_to_cliff"].mean()), 2)
            if "is_censored" in group.columns:
                entry["censored_pct"] = round(float(group["is_censored"].mean() * 100), 2)
            result["per_compound"][compound_key] = entry

    return result


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def run_quality_checks(
    df: pd.DataFrame,
    report_path: Path | None = None,
) -> dict:
    """Run all data quality checks and write a structured report.

    Returns the full report dict.  Each check has a ``passed`` boolean.
    """
    logger.info("Running data quality checks on {} rows", len(df))

    report: dict = {
        "checks": {},
        "feature_distributions": {},
        "target_distribution": {},
        "overall_passed": True,
    }

    checks = [
        check_schema(df),
        check_null_rates(df),
        check_value_ranges(df),
        check_volume(df),
        check_coverage(df),
    ]

    for chk in checks:
        name = chk["check"]
        report["checks"][name] = chk
        if not chk["passed"]:
            report["overall_passed"] = False
            logger.warning("FAIL: {}", name)
        else:
            logger.info("PASS: {}", name)

    report["feature_distributions"] = compute_distributions(df)
    report["target_distribution"] = compute_target_distribution(df)

    if report_path is not None:
        report_path.parent.mkdir(parents=True, exist_ok=True)
        with open(report_path, "w") as f:
            json.dump(report, f, indent=2, default=str)
        logger.info("Quality report written to {}", report_path)

    return report


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run data quality checks on the labeled training dataset."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=TRAINING_DATA_DIR / "labeled_dataset.parquet",
        help="Path to labeled dataset parquet",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=TRAINING_DATA_DIR / "data_quality_report.json",
        help="Output path for JSON quality report",
    )
    args = parser.parse_args()

    data = pd.read_parquet(args.input, engine="pyarrow")
    result = run_quality_checks(data, args.report)

    if result["overall_passed"]:
        logger.info("All quality checks PASSED")
    else:
        failed = [
            name for name, chk in result["checks"].items() if not chk["passed"]
        ]
        logger.error("Quality checks FAILED: {}", failed)
        raise SystemExit(1)
