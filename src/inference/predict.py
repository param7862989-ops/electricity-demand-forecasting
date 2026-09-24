from pathlib import Path
import numpy as np
import pandas as pd
import joblib
import tensorflow as tf


PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_PATH = PROJECT_ROOT / "reports" / "latest_forecast.csv"
MODEL_PATH = PROJECT_ROOT / "models" / "final_gru_best.keras"
FEATURE_SCALER_PATH = PROJECT_ROOT / "models" / "feature_scaler.joblib"
TARGET_SCALER_PATH = PROJECT_ROOT / "models" / "target_scaler.joblib"


def load_artifacts():
    model = tf.keras.models.load_model(MODEL_PATH)
    feature_scaler = joblib.load(FEATURE_SCALER_PATH)
    target_scaler = joblib.load(TARGET_SCALER_PATH)

    return model, feature_scaler, target_scaler


DATA_PATH = PROJECT_ROOT / "data" / "processed" / "hourly_demand.csv"


def load_data():
    df = pd.read_csv(DATA_PATH, parse_dates=["timestamp"])

    df = df.sort_values("timestamp").reset_index(drop=True)

    if df["timestamp"].duplicated().any():
        raise ValueError("Duplicate timestamps found.")

    if df["total_demand"].isna().any():
        raise ValueError("Missing demand values found.")

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

    return features

INPUT_WINDOW = 24
HORIZON = 24


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
        "predicted_demand": predictions
    })

    return forecast
def save_forecast(forecast):
    forecast.to_csv(OUTPUT_PATH, index=False)
    print(f"Forecast saved to: {OUTPUT_PATH}")
if __name__ == "__main__":
    model, feature_scaler, target_scaler = load_artifacts()
    df = load_data()
    features = build_features(df)
    model_input = prepare_input(features, feature_scaler)

    last_timestamp = df["timestamp"].iloc[-1]

    forecast = generate_forecast(
        model,
        target_scaler,
        model_input,
        last_timestamp
    )

    save_forecast(forecast)
    print(forecast.to_string(index=False))
# if __name__ == "__main__":
#     model, feature_scaler, target_scaler = load_artifacts()

#     print("Model loaded:", MODEL_PATH)
#     print("Feature scaler loaded:", FEATURE_SCALER_PATH)
#     print("Target scaler loaded:", TARGET_SCALER_PATH)