from datetime import timedelta
from pathlib import Path

from feast import FeatureView, Field, FileSource
from feast.types import Float64, Int64, String

from entities import circuit, driver, session, stint

REPO_ROOT = Path(__file__).resolve().parent.parent
FEAST_DATA_DIR = REPO_ROOT / "data" / "feast"

# ---------------------------------------------------------------------------
# Data Sources
# ---------------------------------------------------------------------------

stint_features_source = FileSource(
    path=str(FEAST_DATA_DIR / "stint_features.parquet"),
    timestamp_field="event_timestamp",
)

circuit_features_source = FileSource(
    path=str(FEAST_DATA_DIR / "circuit_features.parquet"),
    timestamp_field="event_timestamp",
)

# ---------------------------------------------------------------------------
# Feature Views
# ---------------------------------------------------------------------------

stint_telemetry_features = FeatureView(
    name="stint_telemetry_features",
    entities=[session, driver, stint],
    schema=[
        Field(name="stint_lap_number", dtype=Int64),
        Field(name="rolling_mean_laptime_3", dtype=Float64),
        Field(name="rolling_mean_laptime_5", dtype=Float64),
        Field(name="rolling_var_laptime_3", dtype=Float64),
        Field(name="rolling_var_laptime_5", dtype=Float64),
        Field(name="sector1_delta_from_best", dtype=Float64),
        Field(name="sector2_delta_from_best", dtype=Float64),
        Field(name="sector3_delta_from_best", dtype=Float64),
        Field(name="tire_age_laps", dtype=Float64),
        Field(name="fuel_corrected_laptime", dtype=Float64),
        Field(name="gap_to_car_ahead_s", dtype=Float64),
        Field(name="is_dirty_air", dtype=Int64),
        Field(name="dirty_air_cumulative_laps", dtype=Int64),
        Field(name="is_inlap", dtype=Int64),
        Field(name="is_outlap", dtype=Int64),
        Field(name="is_sc_lap", dtype=Int64),
    ],
    source=stint_features_source,
    ttl=timedelta(days=365 * 5),
)

weather_features = FeatureView(
    name="weather_features",
    entities=[session, driver, stint],
    schema=[
        Field(name="track_temp_c", dtype=Float64),
        Field(name="air_temp_c", dtype=Float64),
        Field(name="humidity", dtype=Float64),
        Field(name="rainfall", dtype=Float64),
        Field(name="track_evolution_index", dtype=Float64),
    ],
    source=stint_features_source,
    ttl=timedelta(days=365 * 5),
)

circuit_features_view = FeatureView(
    name="circuit_features",
    entities=[circuit],
    schema=[
        Field(name="circuit_high_speed_corners", dtype=Int64),
        Field(name="circuit_medium_speed_corners", dtype=Int64),
        Field(name="circuit_low_speed_corners", dtype=Int64),
        Field(name="circuit_abrasiveness", dtype=Int64),
        Field(name="circuit_altitude_m", dtype=Int64),
        Field(name="circuit_tire_limitation", dtype=String),
        Field(name="circuit_street_circuit", dtype=Int64),
    ],
    source=circuit_features_source,
    ttl=timedelta(days=365 * 10),
)
