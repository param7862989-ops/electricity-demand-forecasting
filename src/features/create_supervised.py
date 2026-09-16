from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]

PROCESSED_DATA_PATH = (
    PROJECT_ROOT / "data" / "processed" / "hourly_demand.csv"
)

TRAIN_END = pd.Timestamp("2012-12-31 23:00:00")
VALIDATION_END = pd.Timestamp("2013-12-31 23:00:00")

HORIZON = 24


def load_data() -> pd.DataFrame:
    """Load processed hourly demand."""
    return pd.read_csv(
        PROCESSED_DATA_PATH,
        parse_dates=["timestamp"],
        index_col="timestamp",
    )


def add_features(data: pd.DataFrame) -> pd.DataFrame:
    """Create forecasting features."""
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
        previous_demand
        .rolling(24)
        .mean()
    )

    data["rolling_mean_168"] = (
        previous_demand
        .rolling(168)
        .mean()
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

    return data


def split_data(
    data: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Create chronological train, validation, and test datasets."""

    train = data.loc[
        data.index <= TRAIN_END
    ]

    validation = data.loc[
        (data.index > TRAIN_END)
        & (data.index <= VALIDATION_END)
    ]

    test = data.loc[
        data.index > VALIDATION_END
    ]

    return train, validation, test


def main() -> None:
    data = load_data()
    data = add_features(data)

    train, validation, test = split_data(data)

    feature_columns = [
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

    target_columns = [
        f"target_{step}"
        for step in range(1, HORIZON + 1)
    ]

    train = train.dropna(
        subset=feature_columns + target_columns
    )

    validation = validation.dropna(
        subset=feature_columns + target_columns
    )

    test = test.dropna(
        subset=feature_columns + target_columns
    )

    X_train = train[feature_columns]
    y_train = train[target_columns]

    X_validation = validation[feature_columns]
    y_validation = validation[target_columns]

    X_test = test[feature_columns]
    y_test = test[target_columns]

    print("Supervised dataset summary:")
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

    print("\nFeature count:")
    print(len(feature_columns))

    print("\nTarget count:")
    print(len(target_columns))


if __name__ == "__main__":
    main()
    