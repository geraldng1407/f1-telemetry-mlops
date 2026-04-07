from feast import FeatureService

from feature_views import circuit_features_view, stint_telemetry_features, weather_features

training_feature_service = FeatureService(
    name="training_feature_service",
    features=[
        stint_telemetry_features,
        weather_features,
        circuit_features_view,
    ],
    description=(
        "Full feature bundle for model training — includes telemetry, "
        "weather, and circuit features."
    ),
)

inference_feature_service = FeatureService(
    name="inference_feature_service",
    features=[
        stint_telemetry_features,
        weather_features,
    ],
    description=(
        "Latency-focused feature bundle for real-time inference — "
        "telemetry and weather only."
    ),
)
