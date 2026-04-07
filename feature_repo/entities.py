from feast import Entity, ValueType

session = Entity(
    name="session",
    join_keys=["session_id"],
    value_type=ValueType.STRING,
    description="A race-weekend session, e.g. '2025_1_Race'.",
)

driver = Entity(
    name="driver",
    join_keys=["driver_number"],
    value_type=ValueType.STRING,
    description="Driver identified by their car number.",
)

stint = Entity(
    name="stint",
    join_keys=["stint_number"],
    value_type=ValueType.INT64,
    description="Stint number within a session (1-indexed).",
)

circuit = Entity(
    name="circuit",
    join_keys=["location"],
    value_type=ValueType.STRING,
    description="Racing circuit identified by its location name.",
)
