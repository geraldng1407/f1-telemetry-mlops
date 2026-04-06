from dagster import MultiPartitionsDefinition, StaticPartitionsDefinition

from src.ingestion.constants import MAX_ROUNDS_PER_SEASON, SEASONS

season_round_partitions = MultiPartitionsDefinition(
    {
        "season": StaticPartitionsDefinition([str(s) for s in SEASONS]),
        "round": StaticPartitionsDefinition([str(r) for r in range(1, MAX_ROUNDS_PER_SEASON + 1)]),
    }
)
