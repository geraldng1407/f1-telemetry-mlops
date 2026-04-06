from dagster import Definitions

from src.ingestion.assets import (
    raw_lap_data,
    raw_session_metadata,
    raw_telemetry_data,
    raw_weather_data,
)
from src.ingestion.resources import FastF1Resource
from src.ingestion.schedules import ingestion_job, weekly_ingestion_schedule

defs = Definitions(
    assets=[raw_session_metadata, raw_lap_data, raw_telemetry_data, raw_weather_data],
    jobs=[ingestion_job],
    resources={"fastf1": FastF1Resource()},
    schedules=[weekly_ingestion_schedule],
)
