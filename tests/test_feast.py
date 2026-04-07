"""Integration tests for the Feast feature store (Step 1.3).

Tests cover:
- Feast data-source preparation (consolidation)
- ``feast apply`` via the Python SDK
- Materialization into the SQLite online store
- Historical (offline) and online feature retrieval
- Schema presence and value-range sanity checks
"""

from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
import pytest

PROCESSED_DIR = Path("data/processed")
HAS_PROCESSED_DATA = (
    any(PROCESSED_DIR.rglob("features_*.parquet")) if PROCESSED_DIR.exists() else False
)

pytestmark = pytest.mark.skipif(
    not HAS_PROCESSED_DATA,
    reason="No processed feature data (run feature engineering first)",
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def feast_env(tmp_path_factory: pytest.TempPathFactory):
    """Prepare consolidated data, stand up a temp Feast store, and apply."""
    from feast import Entity, FeatureService, FeatureView, Field, FileSource
    from feast import FeatureStore, ValueType
    from feast.types import Float64, Int64, String

    from src.features.feast_prep import consolidate_stint_features, prepare_circuit_source

    tmpdir = tmp_path_factory.mktemp("feast_test")
    data_dir = tmpdir / "data"
    data_dir.mkdir()

    # --- Prepare data sources ---
    stint_path = consolidate_stint_features(PROCESSED_DIR, data_dir / "stint_features.parquet")
    circuit_path = prepare_circuit_source(output_path=data_dir / "circuit_features.parquet")

    # --- Write temp feature_store.yaml ---
    repo_dir = tmpdir / "repo"
    repo_dir.mkdir()

    # Use relative paths so Feast doesn't trip on Windows drive-letter URI parsing
    yaml_text = (
        "project: f1_test\n"
        "provider: local\n"
        "registry: registry.db\n"
        "online_store:\n"
        "  type: sqlite\n"
        "  path: online_store.db\n"
        "offline_store:\n"
        "  type: file\n"
        "entity_key_serialization_version: 3\n"
    )
    (repo_dir / "feature_store.yaml").write_text(yaml_text)

    # --- Define Feast objects programmatically ---
    session_entity = Entity(
        name="session", join_keys=["session_id"], value_type=ValueType.STRING,
    )
    driver_entity = Entity(
        name="driver", join_keys=["driver_number"], value_type=ValueType.STRING,
    )
    stint_entity = Entity(
        name="stint", join_keys=["stint_number"], value_type=ValueType.INT64,
    )
    circuit_entity = Entity(
        name="circuit", join_keys=["location"], value_type=ValueType.STRING,
    )

    stint_source = FileSource(path=str(stint_path), timestamp_field="event_timestamp")
    circuit_source = FileSource(path=str(circuit_path), timestamp_field="event_timestamp")

    stint_fv = FeatureView(
        name="stint_telemetry_features",
        entities=[session_entity, driver_entity, stint_entity],
        schema=[
            Field(name="stint_lap_number", dtype=Int64),
            Field(name="rolling_mean_laptime_3", dtype=Float64),
            Field(name="rolling_mean_laptime_5", dtype=Float64),
            Field(name="tire_age_laps", dtype=Float64),
            Field(name="fuel_corrected_laptime", dtype=Float64),
            Field(name="is_dirty_air", dtype=Int64),
            Field(name="dirty_air_cumulative_laps", dtype=Int64),
        ],
        source=stint_source,
        ttl=timedelta(days=365 * 5),
    )
    weather_fv = FeatureView(
        name="weather_features",
        entities=[session_entity, driver_entity, stint_entity],
        schema=[
            Field(name="track_temp_c", dtype=Float64),
            Field(name="air_temp_c", dtype=Float64),
            Field(name="humidity", dtype=Float64),
        ],
        source=stint_source,
        ttl=timedelta(days=365 * 5),
    )
    circuit_fv = FeatureView(
        name="circuit_features",
        entities=[circuit_entity],
        schema=[
            Field(name="circuit_high_speed_corners", dtype=Int64),
            Field(name="circuit_abrasiveness", dtype=Int64),
        ],
        source=circuit_source,
        ttl=timedelta(days=365 * 10),
    )
    training_fs = FeatureService(
        name="training_feature_service",
        features=[stint_fv, weather_fv, circuit_fv],
    )

    store = FeatureStore(repo_path=str(repo_dir))
    store.apply([
        session_entity, driver_entity, stint_entity, circuit_entity,
        stint_fv, weather_fv, circuit_fv,
        training_fs,
    ])

    stint_df = pd.read_parquet(stint_path)

    return {
        "store": store,
        "stint_df": stint_df,
        "stint_path": stint_path,
        "circuit_path": circuit_path,
    }


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestDataSources:
    """Verify consolidated Feast data sources."""

    def test_stint_features_parquet_exists(self, feast_env):
        assert feast_env["stint_path"].exists()

    def test_circuit_features_parquet_exists(self, feast_env):
        assert feast_env["circuit_path"].exists()

    def test_stint_features_has_entity_columns(self, feast_env):
        df = feast_env["stint_df"]
        for col in ("session_id", "driver_number", "stint_number", "event_timestamp"):
            assert col in df.columns, f"Missing entity column: {col}"

    def test_stint_features_has_engineered_columns(self, feast_env):
        df = feast_env["stint_df"]
        required = [
            "stint_lap_number", "rolling_mean_laptime_3",
            "tire_age_laps", "fuel_corrected_laptime",
            "track_temp_c", "air_temp_c",
        ]
        for col in required:
            assert col in df.columns, f"Missing feature column: {col}"


class TestValueRanges:
    """Sanity-check that feature values fall in expected ranges."""

    def test_stint_lap_number_positive(self, feast_env):
        df = feast_env["stint_df"]
        assert df["stint_lap_number"].min() >= 1

    def test_tire_age_non_negative(self, feast_env):
        df = feast_env["stint_df"]
        valid = df["tire_age_laps"].dropna()
        if not valid.empty:
            assert valid.min() >= 0

    def test_track_temp_reasonable(self, feast_env):
        df = feast_env["stint_df"]
        valid = df["track_temp_c"].dropna()
        if not valid.empty:
            assert valid.min() >= -10
            assert valid.max() <= 70

    def test_event_timestamp_not_epoch(self, feast_env):
        df = feast_env["stint_df"]
        epoch = pd.Timestamp("1970-01-01")
        non_epoch = df["event_timestamp"] != epoch
        assert non_epoch.any(), "All event_timestamps are epoch — metadata missing?"


class TestHistoricalRetrieval:
    """Test offline (historical) feature retrieval."""

    def test_historical_features_returned(self, feast_env):
        store = feast_env["store"]
        df = feast_env["stint_df"]

        sample = df.head(5)[
            ["session_id", "driver_number", "stint_number", "event_timestamp"]
        ].copy()

        historical = store.get_historical_features(
            entity_df=sample,
            features=[
                "stint_telemetry_features:stint_lap_number",
                "stint_telemetry_features:tire_age_laps",
                "weather_features:track_temp_c",
            ],
        ).to_df()

        assert len(historical) == len(sample)
        assert "stint_lap_number" in historical.columns
        assert "tire_age_laps" in historical.columns
        assert "track_temp_c" in historical.columns

    def test_historical_values_non_null(self, feast_env):
        store = feast_env["store"]
        df = feast_env["stint_df"]

        sample = df.head(5)[
            ["session_id", "driver_number", "stint_number", "event_timestamp"]
        ].copy()

        historical = store.get_historical_features(
            entity_df=sample,
            features=["stint_telemetry_features:stint_lap_number"],
        ).to_df()

        assert historical["stint_lap_number"].notna().all()


class TestMaterializeAndOnline:
    """Test materialization to SQLite and online retrieval."""

    @pytest.fixture(scope="class")
    def materialized_store(self, feast_env):
        store = feast_env["store"]
        df = feast_env["stint_df"]

        min_ts = df["event_timestamp"].min()
        max_ts = df["event_timestamp"].max()
        store.materialize(
            start_date=min_ts.to_pydatetime(),
            end_date=max_ts.to_pydatetime(),
        )
        return store

    def test_online_features_returned(self, feast_env, materialized_store):
        store = materialized_store
        df = feast_env["stint_df"]
        row = df.iloc[0]

        online = store.get_online_features(
            features=[
                "stint_telemetry_features:stint_lap_number",
                "stint_telemetry_features:tire_age_laps",
            ],
            entity_rows=[{
                "session_id": row["session_id"],
                "driver_number": row["driver_number"],
                "stint_number": int(row["stint_number"]),
            }],
        ).to_dict()

        assert "stint_lap_number" in online
        assert "tire_age_laps" in online
        assert online["stint_lap_number"][0] is not None

    def test_online_weather_features(self, feast_env, materialized_store):
        store = materialized_store
        df = feast_env["stint_df"]
        row = df.iloc[0]

        online = store.get_online_features(
            features=["weather_features:track_temp_c"],
            entity_rows=[{
                "session_id": row["session_id"],
                "driver_number": row["driver_number"],
                "stint_number": int(row["stint_number"]),
            }],
        ).to_dict()

        assert "track_temp_c" in online
