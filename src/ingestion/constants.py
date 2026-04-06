from pathlib import Path

RAW_DATA_DIR = Path("data/raw")
FASTF1_CACHE_DIR = Path("data/fastf1_cache")

SEASONS = list(range(2021, 2026))
MAX_ROUNDS_PER_SEASON = 24

SESSION_TYPES_STANDARD = ["FP1", "FP2", "FP3", "Qualifying", "Race"]
SESSION_TYPES_SPRINT = ["FP1", "Sprint Qualifying", "Sprint", "Qualifying", "Race"]

TIMEDELTA_COLUMNS = [
    "Time",
    "LapTime",
    "LapStartTime",
    "Sector1Time",
    "Sector2Time",
    "Sector3Time",
    "Sector1SessionTime",
    "Sector2SessionTime",
    "Sector3SessionTime",
    "PitInTime",
    "PitOutTime",
]

LAP_COLUMNS = [
    "Time",
    "Driver",
    "DriverNumber",
    "Team",
    "LapNumber",
    "LapTime",
    "LapStartTime",
    "Sector1Time",
    "Sector2Time",
    "Sector3Time",
    "SpeedI1",
    "SpeedI2",
    "SpeedFL",
    "SpeedST",
    "Compound",
    "TyreLife",
    "FreshTyre",
    "Stint",
    "PitInTime",
    "PitOutTime",
    "TrackStatus",
    "Position",
    "IsPersonalBest",
    "IsAccurate",
]

TELEMETRY_COLUMNS = [
    "Speed",
    "RPM",
    "nGear",
    "Throttle",
    "Brake",
    "DRS",
    "Distance",
    "Time",
    "SessionTime",
    "X",
    "Y",
    "Z",
]

WEATHER_COLUMNS = [
    "Time",
    "AirTemp",
    "TrackTemp",
    "Humidity",
    "Pressure",
    "Rainfall",
    "WindSpeed",
    "WindDirection",
]
