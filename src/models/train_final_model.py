from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import StandardScaler


PROJECT_ROOT = Path(__file__).resolve().parents[2]

PROCESSED_DATA_PATH = (
    PROJECT_ROOT / "data" / "processed" / "hourly_demand.csv"
)

MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"

TRAIN_END = pd.Timestamp("2012-12-31 23:00:00")
VALIDATION_END = pd.Timestamp("2013-12-31 23:00:00")

INPUT_WINDOW = 24
HORIZON = 24

GRU_UNITS = 128
DROPOUT = 0.2
DENSE_UNITS = 64

SEED = 42

SEQUENCE_FEATURES = [
    "total_demand",
    "hour_sin",
    "hour_cos",
    "day_of_week_sin",
    "day_of_week_cos",
    "month_sin",
    "month_cos",
    "is_weekend",
]


def load_data():
    return pd.read_csv(
        PROCESSED_DATA_PATH,
        parse_dates=["timestamp"],
        index_col="timestamp",
    )


def add_sequence_features(data):
    data = data.copy()

    data["hour"] = data.index.hour
    data["day_of_week"] = data.index.dayofweek
    data["month"] = data.index.month

    data["is_weekend"] = (
        data.index.dayofweek >= 5
    ).astype(float)

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

    return data


def create_sequences(data, start_time, end_time):
    features = data[SEQUENCE_FEATURES].to_numpy(
        dtype=np.float32
    )

    demand = data["total_demand"].to_numpy(
        dtype=np.float32
    )

    timestamps = data.index

    X = []
    y = []
    origins = []

    for end_position in range(
        INPUT_WINDOW - 1,
        len(data) - HORIZON,
    ):
        origin = timestamps[end_position]

        if origin < start_time:
            continue

        if origin > end_time - pd.Timedelta(
            hours=HORIZON
        ):
            continue

        target_start = end_position + 1
        target_end = (
            end_position + 1 + HORIZON
        )

        X.append(
            features[
                end_position - INPUT_WINDOW + 1:
                end_position + 1
            ]
        )

        y.append(
            demand[target_start:target_end]
        )

        origins.append(origin)

    return (
        np.asarray(X),
        np.asarray(y),
        pd.DatetimeIndex(origins),
    )


def build_model():
    model = tf.keras.Sequential(
        [
            tf.keras.layers.Input(
                shape=(
                    INPUT_WINDOW,
                    len(SEQUENCE_FEATURES),
                )
            ),
            tf.keras.layers.GRU(
                GRU_UNITS,
                return_sequences=False,
            ),
            tf.keras.layers.Dropout(DROPOUT),
            tf.keras.layers.Dense(
                DENSE_UNITS,
                activation="relu",
            ),
            tf.keras.layers.Dense(HORIZON),
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


def main():
    tf.keras.utils.set_random_seed(SEED)

    MODELS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    REPORTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    print(f"TensorFlow version: {tf.__version__}")
    print(f"Random seed: {SEED}")

    print("\nFinal model configuration:")
    print(f"Input window: {INPUT_WINDOW} hours")
    print(f"GRU units: {GRU_UNITS}")
    print(f"Dropout: {DROPOUT}")
    print(f"Dense units: {DENSE_UNITS}")
    print(f"Forecast horizon: {HORIZON} hours")

    data = add_sequence_features(
        load_data()
    )

    train_X, train_y, _ = create_sequences(
        data,
        pd.Timestamp("2011-01-01 00:00:00"),
        TRAIN_END,
    )

    validation_X, validation_y, _ = create_sequences(
        data,
        pd.Timestamp("2013-01-01 00:00:00"),
        VALIDATION_END,
    )

    test_X, test_y, test_origins = create_sequences(
        data,
        pd.Timestamp("2014-01-01 00:00:00"),
        pd.Timestamp("2014-12-31 23:00:00"),
    )

    print("\nSequence dataset shapes:")
    print(f"Train:      X={train_X.shape}, y={train_y.shape}")
    print(
        f"Validation: X={validation_X.shape}, "
        f"y={validation_y.shape}"
    )
    print(f"Test:       X={test_X.shape}, y={test_y.shape}")

    feature_scaler = StandardScaler()

    train_shape = train_X.shape

    train_X_scaled = feature_scaler.fit_transform(
        train_X.reshape(-1, train_shape[-1])
    ).reshape(train_shape)

    validation_X_scaled = feature_scaler.transform(
        validation_X.reshape(-1, validation_X.shape[-1])
    ).reshape(validation_X.shape)

    test_X_scaled = feature_scaler.transform(
        test_X.reshape(-1, test_X.shape[-1])
    ).reshape(test_X.shape)

    target_scaler = StandardScaler()

    train_y_scaled = target_scaler.fit_transform(
        train_y
    )

    validation_y_scaled = target_scaler.transform(
        validation_y
    )

    test_y_scaled = target_scaler.transform(
        test_y
    )

    print("\nScaling complete.")
    print("Feature scaler fitted using training data only.")
    print("Target scaler fitted using training data only.")

    model = build_model()

    model.summary()

    checkpoint_path = (
        MODELS_DIR / "final_gru_best.keras"
    )

    checkpoint = tf.keras.callbacks.ModelCheckpoint(
        checkpoint_path,
        monitor="val_loss",
        save_best_only=True,
        mode="min",
        verbose=1,
    )

    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=7,
        mode="min",
        restore_best_weights=True,
        verbose=1,
    )

    print("\nTraining final model...")

    history = model.fit(
        train_X_scaled,
        train_y_scaled,
        validation_data=(
            validation_X_scaled,
            validation_y_scaled,
        ),
        epochs=100,
        batch_size=64,
        callbacks=[
            checkpoint,
            early_stopping,
        ],
        verbose=1,
    )

    best_validation_loss = min(
        history.history["val_loss"]
    )

    print("\nTraining complete.")
    print(
        f"Epochs trained: "
        f"{len(history.history['loss'])}"
    )
    print(
        f"Best validation loss: "
        f"{best_validation_loss:.6f}"
    )

    feature_scaler_path = (
        MODELS_DIR / "feature_scaler.joblib"
    )

    target_scaler_path = (
        MODELS_DIR / "target_scaler.joblib"
    )

    joblib.dump(
        feature_scaler,
        feature_scaler_path,
    )

    joblib.dump(
        target_scaler,
        target_scaler_path,
    )

    print(
        f"\nFeature scaler saved to: "
        f"{feature_scaler_path}"
    )

    print(
        f"Target scaler saved to: "
        f"{target_scaler_path}"
    )

    predictions_scaled = model.predict(
        test_X_scaled,
        verbose=0,
    )

    predictions = target_scaler.inverse_transform(
        predictions_scaled
    )

    actual = test_y

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
            (actual - predictions)
            / np.maximum(
                np.abs(actual),
                1e-8,
            )
        )
    ) * 100

    print("\nFINAL TEST RESULTS")
    print(f"MAE:  {mae:,.2f}")
    print(f"RMSE: {rmse:,.2f}")
    print(f"MAPE: {mape:.2f}%")

    results = pd.DataFrame(
        [
            {
                "model": "Final GRU",
                "input_window": INPUT_WINDOW,
                "gru_units": GRU_UNITS,
                "dropout": DROPOUT,
                "dense_units": DENSE_UNITS,
                "MAE": mae,
                "RMSE": rmse,
                "MAPE": mape,
                "epochs_trained": len(
                    history.history["loss"]
                ),
                "best_validation_loss": (
                    best_validation_loss
                ),
            }
        ]
    )

    results.to_csv(
        REPORTS_DIR / "final_model_results.csv",
        index=False,
    )

    prediction_data = pd.DataFrame(
        {"timestamp": test_origins}
    )

    for step in range(HORIZON):
        prediction_data[
            f"actual_{step + 1}"
        ] = actual[:, step]

        prediction_data[
            f"prediction_{step + 1}"
        ] = predictions[:, step]

    prediction_data.to_csv(
        REPORTS_DIR / "final_model_predictions.csv",
        index=False,
    )

    print(
        "\nFinal model saved to: "
        f"{checkpoint_path}"
    )


if __name__ == "__main__":
    main()