"""Tests for the streaming module (models, session_loader, producer CLI)."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from src.streaming.models import (
    LapCompletedEvent,
    RaceControlEvent,
    RaceControlEventType,
)
from src.streaming.session_loader import (
    _compute_gap_ahead,
    _join_track_temp,
    load_session_timeline,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def sample_laps() -> pd.DataFrame:
    """Minimal lap DataFrame that mirrors the raw laps_Race.parquet schema."""
    return pd.DataFrame({
        "DriverNumber": ["1", "1", "44", "44"],
        "LapNumber": [1, 2, 1, 2],
        "LapTime": [90.5, 91.2, 90.8, 91.5],
        "LapStartTime": [0.0, 90.5, 0.2, 91.0],
        "Time": [90.5, 181.7, 91.0, 182.5],
        "Sector1Time": [28.0, 28.5, 28.2, 28.8],
        "Sector2Time": [32.0, 32.3, 32.1, 32.4],
        "Sector3Time": [30.5, 30.4, 30.5, 30.3],
        "Compound": ["SOFT", "SOFT", "MEDIUM", "MEDIUM"],
        "TyreLife": [1, 2, 1, 2],
        "Stint": [1, 1, 1, 1],
        "Position": [1, 1, 2, 2],
        "TrackStatus": ["1", "1", "1", "1"],
    })


@pytest.fixture()
def sample_weather() -> pd.DataFrame:
    return pd.DataFrame({
        "Time": [0.0, 60.0, 120.0, 180.0],
        "TrackTemp": [45.0, 45.5, 46.0, 46.5],
    })


@pytest.fixture()
def tmp_session(tmp_path: Path, sample_laps: pd.DataFrame, sample_weather: pd.DataFrame):
    """Write sample Parquet files to a temp directory matching the raw layout."""
    round_dir = tmp_path / "2024" / "12"
    round_dir.mkdir(parents=True)
    sample_laps.to_parquet(round_dir / "laps_Race.parquet", index=False, engine="pyarrow")
    sample_weather.to_parquet(round_dir / "weather_Race.parquet", index=False, engine="pyarrow")
    return tmp_path


# ---------------------------------------------------------------------------
# Model tests
# ---------------------------------------------------------------------------

class TestLapCompletedEvent:
    def test_roundtrip_json(self):
        event = LapCompletedEvent(
            session_id="2024_12_Race",
            driver_number="1",
            lap_number=5,
            lap_time=91.234,
            sector_times=[28.1, 32.0, 31.134],
            compound="SOFT",
            tire_age=5,
            track_temp=45.2,
            gap_ahead=1.3,
        )
        json_str = event.model_dump_json()
        restored = LapCompletedEvent.model_validate_json(json_str)
        assert restored == event

    def test_nullable_fields(self):
        event = LapCompletedEvent(
            session_id="2024_12_Race",
            driver_number="44",
            lap_number=1,
        )
        assert event.lap_time is None
        assert event.gap_ahead is None
        assert event.sector_times == [None, None, None]


class TestRaceControlEvent:
    def test_roundtrip_json(self):
        event = RaceControlEvent(
            session_id="2024_12_Race",
            lap_number=10,
            event_type=RaceControlEventType.SAFETY_CAR,
            timestamp=900.0,
        )
        json_str = event.model_dump_json()
        restored = RaceControlEvent.model_validate_json(json_str)
        assert restored == event

    def test_event_types(self):
        for et in RaceControlEventType:
            event = RaceControlEvent(
                session_id="x", lap_number=1, event_type=et, timestamp=0.0,
            )
            assert event.event_type == et


# ---------------------------------------------------------------------------
# Gap computation tests
# ---------------------------------------------------------------------------

class TestGapComputation:
    def test_p1_has_nan_gap(self, sample_laps: pd.DataFrame):
        result = _compute_gap_ahead(sample_laps)
        p1_rows = result[result["Position"] == 1]
        assert p1_rows["gap_ahead"].isna().all()

    def test_p2_has_positive_gap(self, sample_laps: pd.DataFrame):
        result = _compute_gap_ahead(sample_laps)
        p2_rows = result[result["Position"] == 2]
        valid = p2_rows["gap_ahead"].dropna()
        assert len(valid) > 0
        assert (valid > 0).all()

    def test_missing_position_returns_nan(self):
        df = pd.DataFrame({
            "LapNumber": [1],
            "Time": [90.0],
        })
        result = _compute_gap_ahead(df)
        assert result["gap_ahead"].isna().all()


# ---------------------------------------------------------------------------
# Weather join tests
# ---------------------------------------------------------------------------

class TestWeatherJoin:
    def test_joins_nearest_temp(self, sample_laps: pd.DataFrame, sample_weather: pd.DataFrame):
        result = _join_track_temp(sample_laps, sample_weather)
        assert "track_temp" in result.columns
        assert result["track_temp"].notna().all()

    def test_none_weather(self, sample_laps: pd.DataFrame):
        result = _join_track_temp(sample_laps, None)
        assert result["track_temp"].isna().all()


# ---------------------------------------------------------------------------
# Session loader integration test
# ---------------------------------------------------------------------------

class TestLoadSessionTimeline:
    def test_loads_from_parquet(self, tmp_session: Path):
        timeline = load_session_timeline(2024, 12, raw_data_dir=tmp_session)
        assert len(timeline) > 0
        delays, events = zip(*timeline)
        lap_events = [e for e in events if isinstance(e, LapCompletedEvent)]
        assert len(lap_events) == 4
        assert delays[0] == 0.0

    def test_events_have_session_id(self, tmp_session: Path):
        timeline = load_session_timeline(2024, 12, raw_data_dir=tmp_session)
        for _, event in timeline:
            assert event.session_id == "2024_12_Race"

    def test_all_fields_populated(self, tmp_session: Path):
        timeline = load_session_timeline(2024, 12, raw_data_dir=tmp_session)
        for _, event in timeline:
            if isinstance(event, LapCompletedEvent):
                assert event.driver_number in ("1", "44")
                assert event.lap_number >= 1
                assert event.compound in ("SOFT", "MEDIUM")

    def test_missing_session_raises(self, tmp_path: Path):
        with pytest.raises(FileNotFoundError):
            load_session_timeline(9999, 1, raw_data_dir=tmp_path)


# ---------------------------------------------------------------------------
# Speed multiplier test
# ---------------------------------------------------------------------------

class TestSpeedMultiplier:
    def test_delays_divided_by_speed(self, tmp_session: Path):
        timeline = load_session_timeline(2024, 12, raw_data_dir=tmp_session)
        total_real = sum(d for d, _ in timeline)

        speed = 10.0
        adjusted = [d / speed for d, _ in timeline]
        total_adjusted = sum(adjusted)

        assert total_adjusted == pytest.approx(total_real / speed, rel=1e-9)


# ---------------------------------------------------------------------------
# Producer CLI arg parsing
# ---------------------------------------------------------------------------

class TestProducerCLI:
    def test_parse_args(self):
        from src.streaming.producer import _parse_args

        args = _parse_args(["--season", "2024", "--round", "12", "--speed", "10"])
        assert args.season == 2024
        assert args.round == 12
        assert args.speed == 10.0
        assert args.bootstrap_servers == "localhost:9092"
        assert args.lap_topic == "lap_completed"
        assert args.rc_topic == "race_control"

    def test_defaults(self):
        from src.streaming.producer import _parse_args

        args = _parse_args(["--season", "2025", "--round", "3"])
        assert args.speed == 1.0
