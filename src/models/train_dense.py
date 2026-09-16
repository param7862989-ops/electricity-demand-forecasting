SEED = 42

from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import StandardScaler

np.random.seed(SEED)
tf.random.set_seed(SEED)

PROJECT_ROOT = Path(__file__).resolve().parents[2]

PROCESSED_DATA_PATH = (
    PROJECT_ROOT / "data" / "processed" / "hourly_demand.csv"
)

TRAIN_END = pd.Timestamp("2012-12-31 23:00:00")
VALIDATION_END = pd.Timestamp("2013-12-31 23:00:00")

HORIZON = 24


FEATURE_COLUMNS = [
    "lag_1",
    "lag_24",
    "lag_168",
    "hour",
    "day_of_week",
    "month",
    "day_of_year",
    "is_weekend",
    "rolling_mean_24",
    "rolling_mean_168",
    "hour_sin",
    "hour_cos",
    "day_of_week_sin",
    "day_of_week_cos",
    "month_sin",
    "month_cos",
]

TARGET_COLUMNS = [
    f"target_{step}"
    for step in range(1, HORIZON + 1)
]


def load_data() -> pd.DataFrame:
    """Load processed hourly demand data."""
    return pd.read_csv(
        PROCESSED_DATA_PATH,
        parse_dates=["timestamp"],
        index_col="timestamp",
    )


def build_dataset(
    data: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Create features and 24-hour targets."""

    data = data.copy()

    data["lag_1"] = data["total_demand"].shift(1)
    data["lag_24"] = data["total_demand"].shift(24)
    data["lag_168"] = data["total_demand"].shift(168)

    data["hour"] = data.index.hour
    data["day_of_week"] = data.index.dayofweek
    data["month"] = data.index.month
    data["day_of_year"] = data.index.dayofyear
    data["is_weekend"] = (
        data.index.dayofweek >= 5
    ).astype(int)

    previous_demand = data["total_demand"].shift(1)

    data["rolling_mean_24"] = (
        previous_demand.rolling(24).mean()
    )

    data["rolling_mean_168"] = (
        previous_demand.rolling(168).mean()
    )

    data["hour_sin"] = np.sin(
        2 * np.pi * data["hour"] / 24
    )
    data["hour_cos"] = np.cos(
        2 * np.pi * data["hour"] / 24
    )

    data["day_of_week_sin"] = np.sin(
        2 * np.pi * data["day_of_week"] / 7
    )
    data["day_of_week_cos"] = np.cos(
        2 * np.pi * data["day_of_week"] / 7
    )

    data["month_sin"] = np.sin(
        2 * np.pi * (data["month"] - 1) / 12
    )
    data["month_cos"] = np.cos(
        2 * np.pi * (data["month"] - 1) / 12
    )

    for step in range(1, HORIZON + 1):
        data[f"target_{step}"] = (
            data["total_demand"].shift(-step)
        )

    train = data.loc[
        data.index <= TRAIN_END
    ].dropna(
        subset=FEATURE_COLUMNS + TARGET_COLUMNS
    )

    validation = data.loc[
        (data.index > TRAIN_END)
        & (data.index <= VALIDATION_END)
    ].dropna(
        subset=FEATURE_COLUMNS + TARGET_COLUMNS
    )

    test = data.loc[
        data.index > VALIDATION_END
    ].dropna(
        subset=FEATURE_COLUMNS + TARGET_COLUMNS
    )

    return train, validation, test


def build_dense_model(
    input_size: int,
    output_size: int,
) -> tf.keras.Model:
    """Build the baseline dense forecasting model."""

    model = tf.keras.Sequential(
        [
            tf.keras.layers.Input(
                shape=(input_size,)
            ),
            tf.keras.layers.Dense(
                128,
                activation="relu",
            ),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(
                64,
                activation="relu",
            ),
            tf.keras.layers.Dense(
                output_size
            ),
        ]
    )

    model.compile(
        optimizer=tf.keras.optimizers.Adam(
            learning_rate=0.001
        ),
        loss="mse",
        metrics=["mae"],
    )

    return model

def train_model(
    model: tf.keras.Model,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_validation: np.ndarray,
    y_validation: np.ndarray,
) -> tf.keras.callbacks.History:
    """Train the Dense forecasting model."""

    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=10,
        restore_best_weights=True,
    )

    history = model.fit(
        X_train,
        y_train,
        validation_data=(
            X_validation,
            y_validation,
        ),
        epochs=100,
        batch_size=64,
        callbacks=[early_stopping],
        verbose=1,
    )

    return history

def evaluate_predictions(
    model: tf.keras.Model,
    X_validation_scaled: np.ndarray,
    y_validation: pd.DataFrame,
    target_scaler: StandardScaler,
) -> dict[str, float]:
    """Generate forecasts and evaluate them in original demand units."""

    predictions_scaled = model.predict(
        X_validation_scaled,
        verbose=0,
    )

    predictions = target_scaler.inverse_transform(
        predictions_scaled
    )

    actual = y_validation.to_numpy()

    mae = np.mean(
        np.abs(actual - predictions)
    )

    rmse = np.sqrt(
        np.mean(
            (actual - predictions) ** 2
        )
    )

    mape = np.mean(
        np.abs(
            (actual - predictions) / actual
        )
    ) * 100

    return {
        "MAE": float(mae),
        "RMSE": float(rmse),
        "MAPE": float(mape),
    }

def save_results(
    dense_metrics: dict[str, float],
) -> None:
    """Save baseline and Dense validation results."""

    results = pd.DataFrame(
        [
            {
                "model": "Naive",
                "MAE": 107900.28,
                "RMSE": 134801.28,
                "MAPE": 41.13,
            },
            {
                "model": "Seasonal Naive 24h",
                "MAE": 8105.95,
                "RMSE": 13459.18,
                "MAPE": 3.84,
            },
            {
                "model": "Seasonal Naive 168h",
                "MAE": 12436.73,
                "RMSE": 19947.32,
                "MAPE": 5.24,
            },
            {
                "model": "Dense NN",
                "MAE": dense_metrics["MAE"],
                "RMSE": dense_metrics["RMSE"],
                "MAPE": dense_metrics["MAPE"],
            },
        ]
    )

    output_path = (
        PROJECT_ROOT
        / "reports"
        / "baseline_dense_results.csv"
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    results.to_csv(
        output_path,
        index=False,
    )

    print(
        f"\nResults saved to: {output_path}"
    )

def main() -> None:
    print(f"TensorFlow version: {tf.__version__}")
    print(f"Random seed: {SEED}")
    data = load_data()

    train, validation, test = build_dataset(data)

    X_train = train[FEATURE_COLUMNS]
    y_train = train[TARGET_COLUMNS]

    X_validation = validation[FEATURE_COLUMNS]
    y_validation = validation[TARGET_COLUMNS]

    X_test = test[FEATURE_COLUMNS]
    y_test = test[TARGET_COLUMNS]

    

    print("\nDense model architecture:")
    print("\nDataset shapes:")
    print(
        f"Train:      X={X_train.shape}, "
        f"y={y_train.shape}"
    )
    print(
        f"Validation: X={X_validation.shape}, "
        f"y={y_validation.shape}"
    )
    print(
        f"Test:       X={X_test.shape}, "
        f"y={y_test.shape}"
    )

    feature_scaler = StandardScaler()

    X_train_scaled = feature_scaler.fit_transform(
        X_train
    )

    X_validation_scaled = feature_scaler.transform(
        X_validation
    )

    X_test_scaled = feature_scaler.transform(
        X_test
    )

    print("\nScaled feature shapes:")
    print(f"X_train:      {X_train_scaled.shape}")
    print(
        f"X_validation: {X_validation_scaled.shape}"
    )
    print(f"X_test:       {X_test_scaled.shape}")

    print("\nFeature scaling:")
    print("✓ Scaler fitted using training data only.")
    print("✓ Validation data transformed using train scaler.")
    print("✓ Test data transformed using train scaler.")

    target_scaler = StandardScaler()

    y_train_scaled = target_scaler.fit_transform(
        y_train
    )

    y_validation_scaled = target_scaler.transform(
        y_validation
    )

    y_test_scaled = target_scaler.transform(
        y_test
    )

    print("\nScaled target shapes:")
    print(f"y_train:      {y_train_scaled.shape}")
    print(f"y_validation: {y_validation_scaled.shape}")
    print(f"y_test:       {y_test_scaled.shape}")

    print("\nTarget scaling:")
    print("✓ Target scaler fitted using training data only.")
    print("✓ Validation targets transformed using train scaler.")
    print("✓ Test targets transformed using train scaler.")
    model = build_dense_model(
            input_size=X_train_scaled.shape[1],
            output_size=y_train_scaled.shape[1],
    )
    model.summary()
    print("\nTraining Dense model...")

    history = train_model(
        model,
        X_train_scaled,
        y_train_scaled,
        X_validation_scaled,
        y_validation_scaled,
    )
    dense_metrics = evaluate_predictions(
        model,
        X_validation_scaled,
        y_validation,
        target_scaler,
    )
    save_results(dense_metrics)
    print("\nDense model validation results:")
    print(
        f"MAE:  {dense_metrics['MAE']:,.2f}"
    )
    print(
        f"RMSE: {dense_metrics['RMSE']:,.2f}"
    )
    print(
        f"MAPE: {dense_metrics['MAPE']:.2f}%"
    )
    print("\nTraining complete.")
    print(
        f"Epochs trained: "
        f"{len(history.history['loss'])}"
    )

    print(
        f"Best validation loss: "
        f"{min(history.history['val_loss']):.6f}"
    )

if __name__ == "__main__":
    main()