"""Tests for target variable construction (Step 1.4).

Unit tests use synthetic DataFrames; the integration test runs the full
pipeline against real processed data (skipped if none exists).
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from src.features.constants import CLIFF_THRESHOLD_S, MIN_STINT_LAPS
from src.features.target import compute_targets, detect_cliff_lap

PROCESSED_DIR = Path("data/processed")
FEAST_DIR = Path("data/feast")
HAS_PROCESSED_DATA = (
    (FEAST_DIR / "stint_features.parquet").exists()
    if FEAST_DIR.exists()
    else False
)


# ---------------------------------------------------------------------------
# Helpers — synthetic stint builders
# ---------------------------------------------------------------------------


def _make_stint(
    lap_times: list[float],
    *,
    session_id: str = "2024_1_Race",
    driver_number: str = "44",
    stint_number: int = 1,
    sc_laps: set[int] | None = None,
    inlap: int | None = None,
    outlap: int | None = None,
) -> pd.DataFrame:
    """Build a single-stint DataFrame from a list of lap times."""
    n = len(lap_times)
    sc_laps = sc_laps or set()

    df = pd.DataFrame(
        {
            "session_id": session_id,
            "driver_number": driver_number,
            "stint_number": stint_number,
            "stint_lap_number": list(range(1, n + 1)),
            "fuel_corrected_laptime": lap_times,
            "is_inlap": [1 if (i + 1) == inlap else 0 for i in range(n)],
            "is_outlap": [1 if (i + 1) == outlap else 0 for i in range(n)],
            "is_sc_lap": [1 if (i + 1) in sc_laps else 0 for i in range(n)],
        }
    )
    return df


# ---------------------------------------------------------------------------
# Unit tests — detect_cliff_lap
# ---------------------------------------------------------------------------


class TestDetectCliffLap:
    def test_clear_cliff(self):
        """A stint where lap 8 is clearly above avg + 1.5s."""
        times = [90.0] * 7 + [92.5, 93.0, 94.0]
        df = _make_stint(times)
        assert detect_cliff_lap(df) == 8

    def test_no_cliff_flat_stint(self):
        """Flat stint with consistent times — no cliff."""
        times = [90.0, 90.1, 89.9, 90.2, 90.0, 89.8, 90.1]
        df = _make_stint(times)
        assert detect_cliff_lap(df) is None

    def test_sc_lap_excluded(self):
        """A safety-car lap with high time should not trigger the cliff."""
        # Lap 5 is SC and slow, but the real cliff is at lap 8
        times = [90.0, 90.1, 89.9, 90.0, 120.0, 90.0, 90.1, 92.5, 93.0]
        df = _make_stint(times, sc_laps={5})
        result = detect_cliff_lap(df)
        assert result != 5
        assert result == 8

    def test_inlap_outlap_excluded(self):
        """In-lap and out-lap should not be considered for cliff detection."""
        times = [95.0, 90.0, 90.1, 89.9, 90.0, 90.1, 89.8]
        df = _make_stint(times, outlap=1)
        # Lap 1 is outlap with high time — should be excluded, no cliff
        assert detect_cliff_lap(df) is None

    def test_short_stint_returns_none(self):
        """Stints with fewer than MIN_STINT_LAPS clean laps return None."""
        times = [90.0, 92.0]
        df = _make_stint(times)
        assert len(times) < MIN_STINT_LAPS
        assert detect_cliff_lap(df) is None

    def test_cliff_at_threshold_boundary(self):
        """A lap clearly above the full-stint average + threshold triggers."""
        # The cliff lap is included in the stint average, so it must be high
        # enough to still exceed (avg_including_itself + threshold).
        # With n=6, base=90: need cliff_time > 90 + 1.5 * 6/5 = 91.8
        times = [90.0] * 5 + [92.0]
        df = _make_stint(times)
        assert detect_cliff_lap(df) == 6

    def test_all_nan_laptimes(self):
        """All-NaN lap times should return None gracefully."""
        df = _make_stint([float("nan")] * 5)
        assert detect_cliff_lap(df) is None


# ---------------------------------------------------------------------------
# Unit tests — compute_targets
# ---------------------------------------------------------------------------


class TestComputeTargets:
    def test_laps_to_cliff_values(self):
        """Verify forward-looking laps_to_cliff is correct."""
        times = [90.0] * 7 + [92.5, 93.0, 94.0]
        df = _make_stint(times)
        result = compute_targets(df)

        cliff_lap = 8
        for _, row in result.iterrows():
            expected = cliff_lap - row["stint_lap_number"]
            if expected >= 0:
                assert row["laps_to_cliff"] == expected

    def test_post_cliff_rows_dropped(self):
        """Rows after the cliff (laps_to_cliff < 0) are excluded."""
        times = [90.0] * 5 + [92.5, 93.0, 94.0]
        df = _make_stint(times)
        result = compute_targets(df)

        assert (result["laps_to_cliff"] >= 0).all()

    def test_censored_stint(self):
        """A flat stint produces is_censored=1 and laps_to_cliff = remaining."""
        times = [90.0, 90.1, 89.9, 90.2, 90.0]
        df = _make_stint(times)
        result = compute_targets(df)

        assert (result["is_censored"] == 1).all()
        max_lap = 5
        for _, row in result.iterrows():
            assert row["laps_to_cliff"] == max_lap - row["stint_lap_number"]

    def test_short_stint_dropped(self):
        """Stints with < MIN_STINT_LAPS clean laps produce no output rows."""
        times = [90.0, 92.0]
        df = _make_stint(times)
        result = compute_targets(df)
        assert len(result) == 0

    def test_outlap_rows_dropped(self):
        """Out-lap rows are excluded from the result."""
        times = [95.0, 90.0, 90.1, 89.9, 90.0, 90.1]
        df = _make_stint(times, outlap=1)
        result = compute_targets(df)
        assert (result["is_outlap"] == 0).all()

    def test_multi_stint_dataset(self):
        """Multiple stints are processed independently."""
        stint1 = _make_stint(
            [90.0] * 5 + [92.5],
            stint_number=1,
        )
        stint2 = _make_stint(
            [91.0, 91.1, 90.9, 91.2, 91.0],
            stint_number=2,
        )
        df = pd.concat([stint1, stint2], ignore_index=True)
        result = compute_targets(df)

        stint1_rows = result[result["stint_number"] == 1]
        stint2_rows = result[result["stint_number"] == 2]

        assert (stint1_rows["is_censored"] == 0).all()
        assert (stint2_rows["is_censored"] == 1).all()

    def test_required_columns_missing_raises(self):
        """Missing required columns raises ValueError."""
        df = pd.DataFrame({"session_id": ["a"], "driver_number": ["1"]})
        with pytest.raises(ValueError, match="Missing required columns"):
            compute_targets(df)

    def test_laps_remaining_in_stint(self):
        """laps_remaining_in_stint is always max_lap - current_lap."""
        times = [90.0] * 6
        df = _make_stint(times)
        result = compute_targets(df)

        for _, row in result.iterrows():
            expected = 6 - row["stint_lap_number"]
            assert row["laps_remaining_in_stint"] == expected


# ---------------------------------------------------------------------------
# Integration test — real data
# ---------------------------------------------------------------------------


@pytest.mark.skipif(
    not HAS_PROCESSED_DATA,
    reason="No processed feature data (run feast_prep first)",
)
class TestBuildLabeledDataset:
    @pytest.fixture(scope="class")
    def labeled_df(self, tmp_path_factory):
        from src.features.dataset import build_labeled_dataset

        tmpdir = tmp_path_factory.mktemp("dataset_test")
        output = tmpdir / "labeled_dataset.parquet"

        df = build_labeled_dataset(
            input_path=FEAST_DIR / "stint_features.parquet",
            output_path=output,
            use_feast=False,
        )
        return df

    def test_has_target_columns(self, labeled_df):
        for col in ["laps_to_cliff", "is_censored"]:
            assert col in labeled_df.columns, f"Missing target column: {col}"

    def test_no_null_targets(self, labeled_df):
        assert labeled_df["laps_to_cliff"].notna().all()
        assert labeled_df["is_censored"].notna().all()

    def test_laps_to_cliff_non_negative(self, labeled_df):
        assert (labeled_df["laps_to_cliff"] >= 0).all()

    def test_is_censored_binary(self, labeled_df):
        assert set(labeled_df["is_censored"].unique()).issubset({0, 1})

    def test_has_feature_columns(self, labeled_df):
        expected = [
            "stint_lap_number",
            "tire_age_laps",
            "fuel_corrected_laptime",
            "rolling_mean_laptime_5",
        ]
        for col in expected:
            assert col in labeled_df.columns, f"Missing feature: {col}"

    def test_nonzero_volume(self, labeled_df):
        assert len(labeled_df) > 0

    def test_quality_checks_pass(self, labeled_df, tmp_path_factory):
        from src.features.quality import run_quality_checks

        tmpdir = tmp_path_factory.mktemp("quality_test")
        report = run_quality_checks(
            labeled_df,
            report_path=tmpdir / "report.json",
        )
        failed = [
            name
            for name, chk in report["checks"].items()
            if not chk["passed"] and name != "volume"
        ]
        assert not failed, f"Quality checks failed: {failed}"
