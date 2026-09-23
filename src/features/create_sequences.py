from pathlib import Path
import argparse

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]

PROCESSED_DATA_PATH = (
    PROJECT_ROOT / "data" / "processed" / "hourly_demand.csv"
)

TRAIN_END = pd.Timestamp("2012-12-31 23:00:00")
VALIDATION_END = pd.Timestamp("2013-12-31 23:00:00")

HORIZON = 24

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
    return pd.read_csv(
        PROCESSED_DATA_PATH,
        parse_dates=["timestamp"],
        index_col="timestamp",
    )


def add_sequence_features(
    data: pd.DataFrame,
) -> pd.DataFrame:

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
    input_window: int,
) -> tuple[np.ndarray, np.ndarray, pd.DatetimeIndex]:

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
        input_window - 1,
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
                end_position - input_window + 1 :
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


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--input-window",
        type=int,
        required=True,
        choices=[24, 168, 336],
    )

    args = parser.parse_args()

    input_window = args.input_window

    data = add_sequence_features(
        load_data()
    )

    train_X, train_y, train_origins = create_sequences(
        data,
        pd.Timestamp("2011-01-01 00:00:00"),
        TRAIN_END,
        input_window,
    )

    validation_X, validation_y, validation_origins = (
        create_sequences(
            data,
            pd.Timestamp("2013-01-01 00:00:00"),
            VALIDATION_END,
            input_window,
        )
    )

    test_X, test_y, test_origins = create_sequences(
        data,
        pd.Timestamp("2014-01-01 00:00:00"),
        pd.Timestamp("2014-12-31 23:00:00"),
        input_window,
    )

    print("Sequence dataset shapes:")
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

    print("\nSequence properties:")
    print(
        f"Input window: {input_window} hours"
    )
    print(
        f"Forecast horizon: {HORIZON} hours"
    )
    print(
        f"Sequence features: "
        f"{len(SEQUENCE_FEATURES)}"
    )

    print("\nForecast origins:")
    print(
        f"Train:      "
        f"{train_origins.min()} -> "
        f"{train_origins.max()}"
    )
    print(
        f"Validation: "
        f"{validation_origins.min()} -> "
        f"{validation_origins.max()}"
    )
    print(
        f"Test:       "
        f"{test_origins.min()} -> "
        f"{test_origins.max()}"
    )


if __name__ == "__main__":
    main()