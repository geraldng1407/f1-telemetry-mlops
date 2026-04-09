from pathlib import Path

from src.ingestion.constants import RAW_DATA_DIR

PROCESSED_DATA_DIR = Path("data/processed")
CIRCUIT_REF_PATH = Path("data/reference/circuit_characteristics.csv")
FEAST_DATA_DIR = Path("data/feast")

FUEL_CORRECTION_PER_LAP = 0.06  # seconds gained per lap of fuel burn
DIRTY_AIR_THRESHOLD_S = 1.5  # gap in seconds below which dirty air affects car

ROLLING_WINDOWS = [3, 5]

SECTOR_TIME_COLUMNS = ["Sector1Time", "Sector2Time", "Sector3Time"]

# TrackStatus codes that indicate safety car / VSC periods
SC_TRACK_STATUS_CODES = {"4", "5", "6", "7"}

# Hour offsets from event_date to separate sessions within a weekend and
# preserve chronological ordering for Feast point-in-time joins.
SESSION_HOUR_OFFSETS: dict[str, int] = {
    "FP1": -48,
    "FP2": -44,
    "FP3": -24,
    "Sprint_Qualifying": -28,
    "Sprint": -20,
    "Qualifying": -18,
    "Race": 0,
}

TRAINING_DATA_DIR = Path("data/training")

# Target variable construction
CLIFF_THRESHOLD_S = 1.5  # seconds above stint average that defines a tire cliff
MIN_STINT_LAPS = 3  # minimum clean laps for a stint to be usable for training

TARGET_COLUMNS = ["cliff_lap", "laps_to_cliff", "is_censored", "laps_remaining_in_stint"]

FEAST_ENTITY_COLUMNS = ["session_id", "driver_number", "stint_number", "location"]

ENGINEERED_FEATURE_COLUMNS = [
    "stint_lap_number",
    "rolling_mean_laptime_3",
    "rolling_mean_laptime_5",
    "rolling_var_laptime_3",
    "rolling_var_laptime_5",
    "sector1_delta_from_best",
    "sector2_delta_from_best",
    "sector3_delta_from_best",
    "tire_age_laps",
    "fuel_corrected_laptime",
    "track_temp_c",
    "air_temp_c",
    "humidity",
    "rainfall",
    "track_evolution_index",
    "gap_to_car_ahead_s",
    "is_dirty_air",
    "dirty_air_cumulative_laps",
    "is_inlap",
    "is_outlap",
    "is_sc_lap",
    "circuit_high_speed_corners",
    "circuit_medium_speed_corners",
    "circuit_low_speed_corners",
    "circuit_abrasiveness",
    "circuit_altitude_m",
    "circuit_tire_limitation",  # string in source data ("rear"/"front"); encoded in training
    "circuit_street_circuit",
]
