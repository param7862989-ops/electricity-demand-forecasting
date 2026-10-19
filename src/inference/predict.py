from pathlib import Path
import numpy as np
import pandas as pd
import joblib


PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_PATH = PROJECT_ROOT / "reports" / "latest_forecast.csv"
MODEL_PATH = PROJECT_ROOT / "models" / "final_gru_best.keras"
FEATURE_SCALER_PATH = PROJECT_ROOT / "models" / "feature_scaler.joblib"
TARGET_SCALER_PATH = PROJECT_ROOT / "models" / "target_scaler.joblib"
DATA_PATH = PROJECT_ROOT / "data" / "processed" / "hourly_demand.csv"

MODEL_NAME = "GRU-128"
INPUT_WINDOW = 24
HORIZON = 24

FEATURE_COLUMNS = [
    "total_demand",
    "hour_sin",
    "hour_cos",
    "day_of_week_sin",
    "day_of_week_cos",
    "month_sin",
    "month_cos",
    "is_weekend",
]


def load_artifacts():
    # Imported lazily so that importing this module (e.g. in tests or the
    # API startup) does not pay the TensorFlow import cost.
    import tensorflow as tf

    model = tf.keras.models.load_model(MODEL_PATH)
    feature_scaler = joblib.load(FEATURE_SCALER_PATH)
    target_scaler = joblib.load(TARGET_SCALER_PATH)

    return model, feature_scaler, target_scaler


def validate_history(df):
    """Validate an hourly demand history before it is used for inference."""
    if df["timestamp"].duplicated().any():
        raise ValueError("Duplicate timestamps found.")

    if df["total_demand"].isna().any():
        raise ValueError("Missing demand values found.")

    if (df["total_demand"] < 0).any():
        raise ValueError("Negative demand values found.")

    latest = df["timestamp"].tail(INPUT_WINDOW)
    if len(latest) < INPUT_WINDOW:
        raise ValueError(
            f"At least {INPUT_WINDOW} hourly observations are required."
        )

    if not (latest.diff().dropna() == pd.Timedelta(hours=1)).all():
        raise ValueError(
            f"The last {INPUT_WINDOW} observations must be consecutive hours."
        )


def load_data():
    df = pd.read_csv(DATA_PATH, parse_dates=["timestamp"])

    df = df.sort_values("timestamp").reset_index(drop=True)

    validate_history(df)

    return df


def build_features(df):
    features = pd.DataFrame(index=df.index)

    timestamp = df["timestamp"]

    features["total_demand"] = df["total_demand"]

    hour = timestamp.dt.hour
    day_of_week = timestamp.dt.dayofweek
    month = timestamp.dt.month

    features["hour_sin"] = np.sin(2 * np.pi * hour / 24)
    features["hour_cos"] = np.cos(2 * np.pi * hour / 24)

    features["day_of_week_sin"] = np.sin(2 * np.pi * day_of_week / 7)
    features["day_of_week_cos"] = np.cos(2 * np.pi * day_of_week / 7)

    features["month_sin"] = np.sin(2 * np.pi * (month - 1) / 12)
    features["month_cos"] = np.cos(2 * np.pi * (month - 1) / 12)

    features["is_weekend"] = (day_of_week >= 5).astype(int)

    return features[FEATURE_COLUMNS]


def prepare_input(features, feature_scaler):
    latest_features = features.tail(INPUT_WINDOW).copy()

    if len(latest_features) < INPUT_WINDOW:
        raise ValueError("Not enough historical data for prediction.")

    scaled_features = feature_scaler.transform(latest_features.to_numpy())

    model_input = scaled_features.reshape(1, INPUT_WINDOW, features.shape[1])

    return model_input


def generate_forecast(model, target_scaler, model_input, last_timestamp):
    predictions_scaled = model.predict(model_input, verbose=0)

    predictions = target_scaler.inverse_transform(predictions_scaled)[0]

    future_timestamps = pd.date_range(
        start=last_timestamp + pd.Timedelta(hours=1),
        periods=HORIZON,
        freq="1h"
    )

    forecast = pd.DataFrame({
        "timestamp": future_timestamps,
        "predicted_demand": predictions.astype(float)
    })

    return forecast


def forecast_from_history(df, model, feature_scaler, target_scaler):
    """Forecast the next HORIZON hours from an hourly demand history."""
    df = df.sort_values("timestamp").reset_index(drop=True)
    validate_history(df)

    features = build_features(df)
    model_input = prepare_input(features, feature_scaler)

    return generate_forecast(
        model,
        target_scaler,
        model_input,
        df["timestamp"].iloc[-1]
    )


def save_forecast(forecast):
    forecast.to_csv(OUTPUT_PATH, index=False)
    print(f"Forecast saved to: {OUTPUT_PATH}")


if __name__ == "__main__":
    model, feature_scaler, target_scaler = load_artifacts()
    df = load_data()

    forecast = forecast_from_history(
        df,
        model,
        feature_scaler,
        target_scaler
    )

    save_forecast(forecast)
    print(forecast.to_string(index=False))
