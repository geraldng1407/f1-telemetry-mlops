"""Tests for midfield validation filtering, evaluation, and reporting."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.training.midfield_validation import (
    DIRTY_AIR_CUMULATIVE_THRESHOLD,
    MIDFIELD_CONSTRUCTORS_2024,
    CaseStudy,
    SliceResult,
    _build_markdown_report,
    _resolve_team_names,
    evaluate_midfield_slices,
    evaluate_slice,
    filter_dirty_air_stints,
    filter_midfield,
    filter_non_optimal_strategy,
    filter_variable_weather,
    run_case_studies,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_RNG = np.random.default_rng(42)


def _make_df(
    n: int = 30,
    teams: list[str] | None = None,
    include_dirty_air: bool = True,
    include_compound: bool = True,
    include_weather: bool = True,
) -> pd.DataFrame:
    """Build a synthetic labeled-dataset-like DataFrame for testing."""
    if teams is None:
        teams = ["Williams", "Haas F1 Team", "Mercedes", "Red Bull Racing"]

    team_col = [teams[i % len(teams)] for i in range(n)]
    session_ids = [f"2024_{13 + (i % 3)}_Race" for i in range(n)]

    df = pd.DataFrame(
        {
            "session_id": session_ids,
            "driver_number": [str(i % 10) for i in range(n)],
            "stint_number": [1 + (i % 3) for i in range(n)],
            "stint_lap_number": list(range(1, n + 1)),
            "laps_to_cliff": _RNG.uniform(1, 20, size=n).astype(np.float32),
            "is_censored": np.zeros(n, dtype=int),
            "cliff_lap": _RNG.integers(15, 30, size=n).astype(np.float32),
            "team": team_col,
        }
    )

    if include_dirty_air:
        df["dirty_air_cumulative_laps"] = _RNG.integers(0, 12, size=n)

    if include_compound:
        compounds = ["SOFT", "MEDIUM", "HARD"]
        df["compound"] = [compounds[i % 3] for i in range(n)]

    if include_weather:
        df["rainfall"] = np.zeros(n)
        df["track_temp_c"] = _RNG.uniform(25, 35, size=n)

    return df


# ---------------------------------------------------------------------------
# Team name resolution
# ---------------------------------------------------------------------------


class TestResolveTeamNames:
    def test_exact_match(self):
        available = {"Williams", "Mercedes", "Red Bull Racing"}
        result = _resolve_team_names(available, {"Williams"})
        assert result == {"Williams"}

    def test_substring_fallback(self):
        available = {"Haas F1 Team"}
        result = _resolve_team_names(available, {"Haas"})
        assert result == {"Haas F1 Team"}

    def test_no_match_returns_empty(self):
        available = {"Mercedes", "Red Bull Racing"}
        result = _resolve_team_names(available, {"Andretti"})
        assert result == set()

    def test_case_insensitive(self):
        available = {"alpine"}
        result = _resolve_team_names(available, {"Alpine"})
        assert result == {"alpine"}


# ---------------------------------------------------------------------------
# filter_midfield
# ---------------------------------------------------------------------------


class TestFilterMidfield:
    def test_filters_to_midfield_teams(self):
        df = _make_df(20, teams=["Williams", "Mercedes", "Red Bull Racing", "Alpine"])
        result = filter_midfield(df, {"Williams", "Alpine"})
        assert set(result["team"].unique()) <= {"Williams", "Alpine"}
        assert len(result) < len(df)

    def test_missing_team_column_returns_empty(self):
        df = _make_df(10)
        df = df.drop(columns=["team"])
        result = filter_midfield(df)
        assert len(result) == 0

    def test_no_matching_teams_returns_empty(self):
        df = _make_df(10, teams=["Mercedes", "Red Bull Racing"])
        result = filter_midfield(df, {"Andretti"})
        assert len(result) == 0


# ---------------------------------------------------------------------------
# filter_dirty_air_stints
# ---------------------------------------------------------------------------


class TestFilterDirtyAir:
    def test_selects_high_dirty_air_stints(self):
        df = _make_df(20)
        df["dirty_air_cumulative_laps"] = 0
        df.loc[0:4, "dirty_air_cumulative_laps"] = 10
        result = filter_dirty_air_stints(df, threshold=5)
        assert len(result) > 0
        stint_max = result.groupby(
            ["session_id", "driver_number", "stint_number"]
        )["dirty_air_cumulative_laps"].max()
        assert (stint_max > 5).all()

    def test_no_dirty_air_returns_empty(self):
        df = _make_df(10)
        df["dirty_air_cumulative_laps"] = 0
        result = filter_dirty_air_stints(df, threshold=5)
        assert len(result) == 0

    def test_missing_column_returns_empty(self):
        df = _make_df(10, include_dirty_air=False)
        result = filter_dirty_air_stints(df)
        assert len(result) == 0


# ---------------------------------------------------------------------------
# filter_non_optimal_strategy
# ---------------------------------------------------------------------------


class TestFilterNonOptimalStrategy:
    def test_detects_non_consensus_strategy(self):
        df = pd.DataFrame(
            {
                "session_id": ["2024_15_Race"] * 8,
                "driver_number": ["1", "1", "2", "2", "3", "3", "4", "4"],
                "stint_number": [1, 2, 1, 2, 1, 2, 1, 2],
                "stint_lap_number": [1, 1, 1, 1, 1, 1, 1, 1],
                "compound": [
                    "MEDIUM", "HARD",  # driver 1: M-H (consensus)
                    "MEDIUM", "HARD",  # driver 2: M-H (consensus)
                    "MEDIUM", "HARD",  # driver 3: M-H (consensus)
                    "HARD", "MEDIUM",  # driver 4: H-M (non-optimal)
                ],
                "laps_to_cliff": [5.0] * 8,
                "is_censored": [0] * 8,
                "cliff_lap": [10] * 8,
            }
        )
        result = filter_non_optimal_strategy(df)
        assert len(result) > 0
        assert set(result["driver_number"].unique()) == {"4"}

    def test_all_same_strategy_returns_empty(self):
        df = pd.DataFrame(
            {
                "session_id": ["2024_15_Race"] * 4,
                "driver_number": ["1", "1", "2", "2"],
                "stint_number": [1, 2, 1, 2],
                "stint_lap_number": [1, 1, 1, 1],
                "compound": ["MEDIUM", "HARD", "MEDIUM", "HARD"],
                "laps_to_cliff": [5.0] * 4,
            }
        )
        result = filter_non_optimal_strategy(df)
        assert len(result) == 0

    def test_missing_compound_returns_empty(self):
        df = _make_df(10, include_compound=False)
        result = filter_non_optimal_strategy(df)
        assert len(result) == 0


# ---------------------------------------------------------------------------
# filter_variable_weather
# ---------------------------------------------------------------------------


class TestFilterVariableWeather:
    def test_detects_rainfall_transition(self):
        df = _make_df(10)
        df["session_id"] = "2024_15_Race"
        df["rainfall"] = [0, 0, 0, 0, 0, 0.5, 0.5, 0.3, 0, 0]
        result = filter_variable_weather(df)
        assert len(result) == 10

    def test_dry_race_excluded(self):
        df = _make_df(10)
        df["session_id"] = "2024_15_Race"
        df["rainfall"] = 0.0
        df["track_temp_c"] = 30.0
        result = filter_variable_weather(df)
        assert len(result) == 0

    def test_large_temp_range_qualifies(self):
        df = _make_df(10)
        df["session_id"] = "2024_15_Race"
        df["rainfall"] = 0.0
        df["track_temp_c"] = np.linspace(20, 35, 10)
        result = filter_variable_weather(df)
        assert len(result) == 10


# ---------------------------------------------------------------------------
# evaluate_slice
# ---------------------------------------------------------------------------


class TestEvaluateSlice:
    def test_returns_slice_result_with_metrics(self):
        df = pd.DataFrame(
            {
                "session_id": ["2024_15_Race"] * 3,
                "driver_number": ["1"] * 3,
                "stint_number": [1] * 3,
                "stint_lap_number": [1, 2, 3],
                "is_censored": [0, 0, 0],
                "cliff_lap": [10, 10, 10],
            }
        )
        y_true = np.array([5.0, 4.0, 3.0])
        y_pred = np.array([6.0, 4.0, 2.0])
        sr = evaluate_slice(y_true, y_pred, df, "test_slice")
        assert sr.name == "test_slice"
        assert sr.n_rows == 3
        assert sr.n_stints == 1
        assert "mae" in sr.metrics
        assert "precision_at_3" in sr.metrics
        assert "strategy_accuracy" in sr.metrics

    def test_empty_input_returns_empty_metrics(self):
        df = pd.DataFrame(columns=["session_id", "driver_number", "stint_number"])
        sr = evaluate_slice(np.array([]), np.array([]), df, "empty")
        assert sr.n_rows == 0
        assert sr.metrics == {}


# ---------------------------------------------------------------------------
# evaluate_midfield_slices
# ---------------------------------------------------------------------------


class TestEvaluateMidfieldSlices:
    def test_returns_all_expected_keys(self):
        xgb = pytest.importorskip("xgboost")
        df = _make_df(40)
        df.index = pd.RangeIndex(len(df))

        X = _RNG.random((len(df), 4)).astype(np.float32)
        y = df["laps_to_cliff"].to_numpy()

        model = xgb.XGBRegressor(n_estimators=5, max_depth=2)
        model.fit(X, y)

        results = evaluate_midfield_slices(model, df, X, y)
        expected_keys = {
            "full_test",
            "midfield_all",
            "dirty_air",
            "non_optimal_strategy",
            "variable_weather",
        }
        assert set(results.keys()) == expected_keys
        assert results["full_test"].n_rows == len(df)


# ---------------------------------------------------------------------------
# run_case_studies
# ---------------------------------------------------------------------------


class TestRunCaseStudies:
    def test_returns_case_study_list(self):
        xgb = pytest.importorskip("xgboost")
        df = _make_df(40)
        df.index = pd.RangeIndex(len(df))

        X = _RNG.random((len(df), 4)).astype(np.float32)
        y = df["laps_to_cliff"].to_numpy()

        model = xgb.XGBRegressor(n_estimators=5, max_depth=2)
        model.fit(X, y)

        studies = run_case_studies(model, df, X, y, n=2)
        assert isinstance(studies, list)
        for cs in studies:
            assert isinstance(cs, CaseStudy)
            assert cs.session_id
            assert isinstance(cs.drivers, list)
            assert isinstance(cs.mae_per_driver, dict)


# ---------------------------------------------------------------------------
# Markdown report
# ---------------------------------------------------------------------------


class TestMarkdownReport:
    def test_report_contains_expected_sections(self):
        slices = {
            "full_test": SliceResult("full_test", 100, 20, {"mae": 2.5, "precision_at_3": 0.7, "strategy_accuracy": 0.6}),
            "midfield_all": SliceResult("midfield_all", 40, 8, {"mae": 3.1, "precision_at_3": 0.5, "strategy_accuracy": 0.4}),
        }
        studies = [
            CaseStudy(
                session_id="2024_15_Race",
                drivers=["1", "2"],
                teams=["Williams", "Alpine"],
                compounds_used={"1": ["MEDIUM", "HARD"], "2": ["HARD", "MEDIUM"]},
                actual_pit_laps={"1": [15], "2": [18]},
                model_recommended_pit={"1": 14.5, "2": 17.0},
                mae_per_driver={"1": 2.0, "2": 3.5},
                summary="Test case study",
            )
        ]
        md = _build_markdown_report(slices, studies)
        assert "# Williams Midfield Validation Report" in md
        assert "## Evaluation Slices" in md
        assert "## Case Studies" in md
        assert "full_test" in md
        assert "midfield_all" in md
        assert "2024_15_Race" in md
