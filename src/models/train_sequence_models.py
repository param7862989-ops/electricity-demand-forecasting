from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import StandardScaler


PROJECT_ROOT = Path(__file__).resolve().parents[2]

PROCESSED_DATA_PATH = (
    PROJECT_ROOT / "data" / "processed" / "hourly_demand.csv"
)

TRAIN_END = pd.Timestamp("2012-12-31 23:00:00")
VALIDATION_END = pd.Timestamp("2013-12-31 23:00:00")

INPUT_WINDOW = 168
HORIZON = 24

SEED = 42

np.random.seed(SEED)
tf.random.set_seed(SEED)

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


def load_data() -> pd.DataFrame:
    """Load processed hourly demand data."""
    return pd.read_csv(
        PROCESSED_DATA_PATH,
        parse_dates=["timestamp"],
        index_col="timestamp",
    )


def add_sequence_features(
    data: pd.DataFrame,
) -> pd.DataFrame:
    """Add calendar features used by sequence models."""

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


def create_sequences(
    data: pd.DataFrame,
    start_time: pd.Timestamp,
    end_time: pd.Timestamp,
) -> tuple[np.ndarray, np.ndarray, pd.DatetimeIndex]:
    """Create 168-hour input sequences and 24-hour targets."""

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
                end_position - INPUT_WINDOW + 1 :
                end_position + 1
            ]
        )

        y.append(
            demand[
                target_start :
                target_end
            ]
        )

        origins.append(origin)

    return (
        np.asarray(X),
        np.asarray(y),
        pd.DatetimeIndex(origins),
    )


def scale_sequence_features(
    X_train: np.ndarray,
    X_validation: np.ndarray,
    X_test: np.ndarray,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    StandardScaler,
]:
    """Scale sequence features using training data only."""

    scaler = StandardScaler()

    train_shape = X_train.shape
    validation_shape = X_validation.shape
    test_shape = X_test.shape

    X_train_scaled = scaler.fit_transform(
        X_train.reshape(-1, train_shape[-1])
    ).reshape(train_shape)

    X_validation_scaled = scaler.transform(
        X_validation.reshape(
            -1,
            validation_shape[-1],
        )
    ).reshape(validation_shape)

    X_test_scaled = scaler.transform(
        X_test.reshape(
            -1,
            test_shape[-1],
        )
    ).reshape(test_shape)

    return (
        X_train_scaled,
        X_validation_scaled,
        X_test_scaled,
        scaler,
    )


def scale_targets(
    y_train: np.ndarray,
    y_validation: np.ndarray,
    y_test: np.ndarray,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    StandardScaler,
]:
    """Scale targets using training data only."""

    scaler = StandardScaler()

    y_train_scaled = scaler.fit_transform(
        y_train
    )

    y_validation_scaled = scaler.transform(
        y_validation
    )

    y_test_scaled = scaler.transform(
        y_test
    )

    return (
        y_train_scaled,
        y_validation_scaled,
        y_test_scaled,
        scaler,
    )

def build_cnn_model() -> tf.keras.Model:
    """Build the 1D CNN forecasting model."""

    model = tf.keras.Sequential(
        [
            tf.keras.layers.Input(
                shape=(INPUT_WINDOW, len(SEQUENCE_FEATURES))
            ),
            tf.keras.layers.Conv1D(
                filters=64,
                kernel_size=3,
                activation="relu",
            ),
            tf.keras.layers.MaxPooling1D(
                pool_size=2
            ),
            tf.keras.layers.Conv1D(
                filters=32,
                kernel_size=3,
                activation="relu",
            ),
            tf.keras.layers.GlobalAveragePooling1D(),
            tf.keras.layers.Dense(
                64,
                activation="relu",
            ),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(
                HORIZON
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
def build_lstm_model() -> tf.keras.Model:
    """Build the LSTM forecasting model."""

    model = tf.keras.Sequential(
        [
            tf.keras.layers.Input(
                shape=(
                    INPUT_WINDOW,
                    len(SEQUENCE_FEATURES),
                )
            ),
            tf.keras.layers.LSTM(
                64,
                return_sequences=False,
            ),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(
                64,
                activation="relu",
            ),
            tf.keras.layers.Dense(
                HORIZON
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
def evaluate_predictions(
    model: tf.keras.Model,
    X_validation: np.ndarray,
    y_validation: np.ndarray,
    target_scaler: StandardScaler,
) -> dict[str, float]:
    """Evaluate forecasts in original demand units."""

    predictions_scaled = model.predict(
        X_validation,
        verbose=0,
    )

    predictions = target_scaler.inverse_transform(
        predictions_scaled
    )

    actual = y_validation

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
def train_model(
    model: tf.keras.Model,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_validation: np.ndarray,
    y_validation: np.ndarray,
) -> tf.keras.callbacks.History:
    """Train a sequence forecasting model."""

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
    X_validation: np.ndarray,
    y_validation: np.ndarray,
    target_scaler: StandardScaler,
) -> dict[str, float]:
    """Evaluate forecasts in original demand units."""

    predictions_scaled = model.predict(
        X_validation,
        verbose=0,
    )

    predictions = target_scaler.inverse_transform(
        predictions_scaled
    )

    actual = y_validation

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

def main() -> None:
    print(f"TensorFlow version: {tf.__version__}")
    print(f"Random seed: {SEED}")

    data = add_sequence_features(
        load_data()
    )

    train_X, train_y, train_origins = create_sequences(
        data,
        pd.Timestamp("2011-01-01 00:00:00"),
        TRAIN_END,
    )

    validation_X, validation_y, validation_origins = (
        create_sequences(
            data,
            pd.Timestamp("2013-01-01 00:00:00"),
            VALIDATION_END,
        )
    )

    test_X, test_y, test_origins = create_sequences(
        data,
        pd.Timestamp("2014-01-01 00:00:00"),
        pd.Timestamp("2014-12-31 23:00:00"),
    )

    print("\nSequence dataset shapes:")
    print(
        f"Train:      X={train_X.shape}, "
        f"y={train_y.shape}"
    )
    print(
        f"Validation: X={validation_X.shape}, "
        f"y={validation_y.shape}"
    )
    print(
        f"Test:       X={test_X.shape}, "
        f"y={test_y.shape}"
    )

    (
        train_X_scaled,
        validation_X_scaled,
        test_X_scaled,
        feature_scaler,
    ) = scale_sequence_features(
        train_X,
        validation_X,
        test_X,
    )

    (
        train_y_scaled,
        validation_y_scaled,
        test_y_scaled,
        target_scaler,
    ) = scale_targets(
        train_y,
        validation_y,
        test_y,
    )

    print("\nScaled sequence shapes:")
    print(
        f"Train X:      {train_X_scaled.shape}"
    )
    print(
        f"Validation X: {validation_X_scaled.shape}"
    )
    print(
        f"Test X:       {test_X_scaled.shape}"
    )

    print("\nScaled target shapes:")
    print(
        f"Train y:      {train_y_scaled.shape}"
    )
    print(
        f"Validation y: {validation_y_scaled.shape}"
    )
    print(
        f"Test y:       {test_y_scaled.shape}"
    )

    print("\nScaling validation:")
    print(
        "✓ Sequence feature scaler fitted "
        "using training data only."
    )
    print(
        "✓ Validation sequence features "
        "transformed using train scaler."
    )
    print(
        "✓ Test sequence features "
        "transformed using train scaler."
    )
    print(
        "✓ Target scaler fitted using "
        "training data only."
    )
    print(
        "✓ Validation targets transformed "
        "using train scaler."
    )
    print(
        "✓ Test targets transformed "
        "using train scaler."
    )
    lstm_model = build_lstm_model()

    print("\nLSTM model architecture:")
    lstm_model.summary()

    print("\nTraining LSTM model...")

    history = train_model(
        lstm_model,
        train_X_scaled,
        train_y_scaled,
        validation_X_scaled,
        validation_y_scaled,
    )
    lstm_metrics = evaluate_predictions(
        lstm_model,
        validation_X_scaled,
        validation_y,
        target_scaler,
    )

    print("\nLSTM validation results:")
    print(
        f"MAE:  {lstm_metrics['MAE']:,.2f}"
    )
    print(
        f"RMSE: {lstm_metrics['RMSE']:,.2f}"
    )
    print(
        f"MAPE: {lstm_metrics['MAPE']:.2f}%"
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