# Model Card: Tire Cliff Prediction

## Model Details

- **Model Name:** tire-cliff-prediction
- **Version:** 1.0
- **Type:** Regression (laps remaining until tire performance cliff)
- **Architectures:**
  - **XGBoost** — gradient-boosted decision trees (primary baseline)
  - **LSTM** — Long Short-Term Memory recurrent network (sequential variant)
  - **TFT** — Temporal Fusion Transformer via pytorch-forecasting (sequential variant)
  - **Ensemble** — weighted blend of XGBoost + best sequential model
- **Framework:** scikit-learn, XGBoost, PyTorch, pytorch-forecasting
- **Training Infrastructure:** MLflow experiment tracking (SQLite/Postgres + MinIO artifact store)
- **License:** Project-internal (not publicly distributed)
- **Contact:** Project maintainers

## Intended Use

### Primary Use Case

Pit strategy advisory for Formula 1 teams, with a specific focus on midfield constructors
(Williams, Haas, Alpine, RB, Kick Sauber). The model predicts how many laps remain before
a tire performance "cliff" — a nonlinear degradation event where lap times exceed the
stint average by more than 1.5 seconds.

### Intended Users

- F1 race strategy engineers evaluating pit stop windows
- Data analysts performing post-race strategy reviews
- Researchers studying tire degradation modelling

### Out-of-Scope Uses

- **Real-time safety decisions** — this model is advisory only and must not be used for
  safety-critical decisions
- **Tyre manufacturer recommendations** — the model does not account for tire compound
  development changes between seasons
- **Accurate prediction beyond the training data window** — regulation changes
  (e.g. new tire specifications, car design rules) may invalidate learned patterns

## Training Data

| Property | Value |
|----------|-------|
| **Source** | FastF1 API (official F1 timing data) |
| **Seasons** | 2021, 2022, 2023 (training) |
| **Sessions** | Race sessions only for target computation |
| **Granularity** | One row per driver per lap per stint |
| **Target** | `laps_to_cliff` — laps remaining until the cliff lap |
| **Censoring** | Stints where the driver pitted before a cliff are right-censored; `laps_to_cliff` is clipped to remaining stint length |

### Feature Set (30 features)

| Group | Features |
|-------|----------|
| Stint structure | `stint_lap_number` |
| Rolling statistics | `rolling_mean_laptime_3`, `rolling_mean_laptime_5`, `rolling_var_laptime_3`, `rolling_var_laptime_5` |
| Sector deltas | `sector1_delta_from_best`, `sector2_delta_from_best`, `sector3_delta_from_best` |
| Tire/fuel | `tire_age_laps`, `fuel_corrected_laptime`, `compound` |
| Environment | `track_temp_c`, `air_temp_c`, `humidity`, `rainfall`, `track_evolution_index` |
| Traffic | `gap_to_car_ahead_s`, `is_dirty_air`, `dirty_air_cumulative_laps` |
| Lap flags | `is_inlap`, `is_outlap`, `is_sc_lap` |
| Circuit (static) | `circuit_high_speed_corners`, `circuit_medium_speed_corners`, `circuit_low_speed_corners`, `circuit_abrasiveness`, `circuit_altitude_m`, `circuit_tire_limitation`, `circuit_street_circuit` |

## Evaluation Data

| Split | Scope | Purpose |
|-------|-------|---------|
| **Validation** | 2024 season, rounds 1–12 | Hyperparameter tuning, early stopping |
| **Test** | 2024 season, rounds 13+ | Final evaluation and midfield validation |

The temporal split ensures no future data leaks into training.

## Metrics

### Primary Metrics

| Metric | Definition |
|--------|------------|
| **MAE** | Mean absolute error of `laps_to_cliff` (excluding censored stints) |
| **Precision@3** | Fraction of predictions within 3 laps of the actual cliff (excluding censored stints) |
| **Strategy Accuracy** | Fraction of stints where the model's earliest recommended pit lap falls within `[cliff_lap - 2, cliff_lap]` |

### Evaluation Slices (Williams Midfield Validation)

The model is evaluated on progressively harder subsets of the test data:

| Slice | Description |
|-------|-------------|
| **full_test** | All test set rows (2024 rounds 13+) |
| **midfield_all** | Only midfield constructors (Williams, Haas, Alpine, RB, Kick Sauber) |
| **dirty_air** | Midfield stints where max cumulative dirty-air laps > 5 |
| **non_optimal_strategy** | Midfield drivers who used a non-consensus compound sequence |
| **variable_weather** | Midfield rows from races with rainfall transitions or large temperature swings (>8°C) |

*Actual metric values should be populated after running the validation pipeline:*

```
python -m src.training.midfield_validation
```

## Ethical Considerations

- **Not safety-critical.** The model provides advisory recommendations for pit strategy.
  Human race engineers retain full decision authority.
- **No personal data.** The model uses only publicly available timing data from the
  FastF1 API. Driver numbers are used as identifiers, not personal information.
- **Competitive fairness.** All training data is sourced from historical, publicly
  broadcast race data. No proprietary team telemetry is used.

## Caveats and Recommendations

### Known Limitations

1. **Censored stints bias.** A significant fraction of stints are right-censored (the driver
   pitted before any cliff occurred). The model clips predictions to remaining stint length
   for these cases, but this introduces a systematic optimistic bias for well-managed stints.

2. **Dirty-air degradation.** Midfield cars spend substantially more time in dirty air
   than front-runners. The `dirty_air_cumulative_laps` feature captures this partially,
   but the aerodynamic effect varies by circuit type and car design — factors not fully
   modelled.

3. **Weather transitions.** The model uses instantaneous weather readings (track temp,
   rainfall) but does not explicitly model *transitions* (e.g. drying track after rain).
   Variable-weather races show elevated prediction error.

4. **Regulation changes.** The model is trained on 2021–2023 data under specific
   technical regulations. Significant regulation changes (tire specifications, downforce
   levels) may degrade predictive accuracy.

5. **Strategy interaction effects.** The "non-optimal strategy" filter uses a per-race
   consensus definition. Truly evaluating strategy quality requires counterfactual
   analysis that is beyond the scope of this model.

### Recommendations

- **Retrain annually** with the latest season's data to capture regulation and
  tire compound changes.
- **Monitor drift** using Evidently or similar tools on incoming race data
  (planned in Phase 3).
- **Weight midfield evaluation** more heavily than front-runner performance when
  selecting champion models, since midfield conditions are harder and more
  strategically impactful.
- **Use ensemble predictions** when available, as the XGBoost + sequential blend
  tends to smooth out failure modes in individual architectures.

## Quantitative Analysis

### Failure Mode Categories

From the Williams Midfield Validation case studies, the following failure modes
have been identified:

| Category | Description | Impact |
|----------|-------------|--------|
| **Late cliff on Hard tires** | Model underestimates cliff timing on hard compounds in hot conditions | Recommends pitting too early |
| **Dirty-air amplification** | Model over-corrects for dirty air on circuits with long straights where DRS compensates | Predicts cliff earlier than actual |
| **Rain-to-dry transition** | Track evolution after rain is faster than the model expects | Elevated MAE in first dry stint |
| **Short censored stints** | Very short stints (< 8 laps) from early strategic pits produce noisy targets | High variance in predictions |

### References

- Mitchell et al., "Model Cards for Model Reporting" (2019) — https://arxiv.org/abs/1810.03993
- FastF1 Python package — https://docs.fastf1.dev/
- MLflow Model Registry — https://mlflow.org/docs/latest/model-registry.html
