import datetime

from dagster import (
    AssetSelection,
    DefaultScheduleStatus,
    MultiPartitionKey,
    RunRequest,
    ScheduleEvaluationContext,
    define_asset_job,
    schedule,
)

from src.ingestion.partitions import season_round_partitions

ingestion_job = define_asset_job(
    name="ingest_race_weekend",
    selection=AssetSelection.groups("raw_ingestion"),
    partitions_def=season_round_partitions,
)


@schedule(
    cron_schedule="0 0 * * 1",  # Every Monday at midnight UTC (day after race Sunday)
    target=ingestion_job,
    default_status=DefaultScheduleStatus.STOPPED,
)
def weekly_ingestion_schedule(context: ScheduleEvaluationContext):
    """Trigger ingestion for the most recent race round of the current season.

    Runs every Monday at midnight UTC, targeting the round that just completed.
    Uses a simple heuristic: current calendar week maps roughly to the round
    number. In practice, you would look up the actual calendar from FastF1,
    but for scheduling purposes this provides a reasonable default that can
    be adjusted in the Dagster UI.
    """
    today = datetime.date.today()
    current_season = str(today.year)
    iso_week = today.isocalendar()[1]
    estimated_round = str(min(iso_week // 2, 24))

    target_key = MultiPartitionKey({"season": current_season, "round": estimated_round})

    yield RunRequest(
        partition_key=target_key,
        run_key=f"weekly_{current_season}_R{estimated_round}",
    )
