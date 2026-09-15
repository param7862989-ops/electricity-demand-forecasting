from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]

PROCESSED_DATA_PATH = (
    PROJECT_ROOT / "data" / "processed" / "hourly_demand.csv"
)

TRAIN_END = pd.Timestamp("2012-12-31 23:00:00")
VALIDATION_END = pd.Timestamp("2013-12-31 23:00:00")

FORECAST_HORIZON = 24


def load_processed_data() -> pd.DataFrame:
    """Load the processed hourly demand dataset."""
    data = pd.read_csv(
        PROCESSED_DATA_PATH,
        parse_dates=["timestamp"],
        index_col="timestamp",
    )

    return data


def add_lag_features(data: pd.DataFrame) -> pd.DataFrame:
    """Add historical demand lag features."""
    data = data.copy()

    data["lag_1"] = data["total_demand"].shift(1)
    data["lag_24"] = data["total_demand"].shift(24)
    data["lag_168"] = data["total_demand"].shift(168)

    return data


def add_calendar_features(data: pd.DataFrame) -> pd.DataFrame:
    """Add calendar-based time features."""
    data = data.copy()

    data["hour"] = data.index.hour
    data["day_of_week"] = data.index.dayofweek
    data["month"] = data.index.month
    data["day_of_year"] = data.index.dayofyear
    data["is_weekend"] = (
        data.index.dayofweek >= 5
    ).astype(int)

    return data


def add_rolling_features(data: pd.DataFrame) -> pd.DataFrame:
    """Add leakage-safe rolling demand statistics."""
    data = data.copy()

    previous_demand = data["total_demand"].shift(1)

    data["rolling_mean_24"] = (
        previous_demand
        .rolling(window=24)
        .mean()
    )

    data["rolling_mean_168"] = (
        previous_demand
        .rolling(window=168)
        .mean()
    )

    return data


def add_cyclic_features(data: pd.DataFrame) -> pd.DataFrame:
    """Add cyclic encodings for recurring calendar patterns."""
    data = data.copy()

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


def validate_features(data: pd.DataFrame) -> None:
    """Validate feature columns, ranges, and missing values."""

    required_columns = [
        "total_demand",
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

    missing_columns = [
        column
        for column in required_columns
        if column not in data.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing feature columns: {missing_columns}"
        )

    if not data["hour"].between(0, 23).all():
        raise ValueError(
            "Hour feature is outside 0–23."
        )

    if not data["day_of_week"].between(0, 6).all():
        raise ValueError(
            "Day-of-week feature is outside 0–6."
        )

    if not data["month"].between(1, 12).all():
        raise ValueError(
            "Month feature is outside 1–12."
        )

    if not data["day_of_year"].between(1, 366).all():
        raise ValueError(
            "Day-of-year feature is outside 1–366."
        )

    if not data["is_weekend"].isin([0, 1]).all():
        raise ValueError(
            "Weekend feature contains invalid values."
        )

    cyclic_columns = [
        "hour_sin",
        "hour_cos",
        "day_of_week_sin",
        "day_of_week_cos",
        "month_sin",
        "month_cos",
    ]

    for column in cyclic_columns:
        if not data[column].between(-1, 1).all():
            raise ValueError(
                f"{column} contains values outside [-1, 1]."
            )

    complete_data = data.dropna()

    expected_complete_rows = len(data) - 168

    if len(complete_data) != expected_complete_rows:
        raise ValueError(
            "Unexpected number of complete feature rows. "
            f"Expected {expected_complete_rows:,}, "
            f"found {len(complete_data):,}."
        )

    if not (
        complete_data["rolling_mean_24"]
        .notna()
        .all()
    ):
        raise ValueError(
            "Missing 24-hour rolling features remain."
        )

    if not (
        complete_data["rolling_mean_168"]
        .notna()
        .all()
    ):
        raise ValueError(
            "Missing 168-hour rolling features remain."
        )

    print("\nFeature validation:")
    print("✓ All required feature columns are present.")
    print("✓ Hour values are valid.")
    print("✓ Day-of-week values are valid.")
    print("✓ Month values are valid.")
    print("✓ Day-of-year values are valid.")
    print("✓ Weekend values are valid.")
    print("✓ Cyclic features are within [-1, 1].")
    print("✓ Complete feature-row count is correct.")
    print("✓ Rolling features contain no missing values.")


def split_data(
    data: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split the feature dataset chronologically.

    Historical feature values are calculated before splitting,
    allowing validation/test rows to use genuinely available
    historical information from earlier periods.
    """
    train = data.loc[
        data.index <= TRAIN_END
    ].copy()

    validation = data.loc[
        (data.index > TRAIN_END)
        & (data.index <= VALIDATION_END)
    ].copy()

    test = data.loc[
        data.index > VALIDATION_END
    ].copy()

    return train, validation, test


def add_forecast_targets(
    data: pd.DataFrame,
    horizon: int = FORECAST_HORIZON,
) -> pd.DataFrame:
    """Add future demand targets within one split only."""
    data = data.copy()

    for step in range(1, horizon + 1):
        data[f"target_{step}"] = (
            data["total_demand"].shift(-step)
        )

    return data


def validate_target_rows(
    split_name: str,
    data: pd.DataFrame,
    horizon: int = FORECAST_HORIZON,
) -> None:
    """Validate the number of complete target rows in one split."""

    target_columns = [
        f"target_{step}"
        for step in range(1, horizon + 1)
    ]

    complete_targets = data[target_columns].dropna()

    expected_rows = len(data) - horizon

    if len(complete_targets) != expected_rows:
        raise ValueError(
            f"{split_name} target validation failed. "
            f"Expected {expected_rows:,} complete rows, "
            f"found {len(complete_targets):,}."
        )

    print(
        f"✓ {split_name} complete target rows: "
        f"{len(complete_targets):,}"
    )


def validate_target_alignment(
    data: pd.DataFrame,
    horizon: int = FORECAST_HORIZON,
) -> None:
    """Verify that target columns contain the correct future values."""

    target_columns = [
        f"target_{step}"
        for step in range(1, horizon + 1)
    ]

    complete_data = data.dropna(
        subset=target_columns
    )

    if complete_data.empty:
        raise ValueError(
            "No rows available for target alignment validation."
        )

    for step in range(1, horizon + 1):
        target_column = f"target_{step}"

        expected = data["total_demand"].shift(-step)

        comparison = (
            complete_data[target_column]
            == expected.loc[complete_data.index]
        )

        if not comparison.all():
            raise ValueError(
                f"{target_column} is not aligned correctly."
            )

    print(
        "✓ All target columns align with the correct "
        "future demand observations."
    )


def validate_split_boundaries(
    train: pd.DataFrame,
    validation: pd.DataFrame,
    test: pd.DataFrame,
    horizon: int = FORECAST_HORIZON,
) -> None:
    """
    Validate that targets do not cross train/validation/test boundaries.
    """

    target_columns = [
        f"target_{step}"
        for step in range(1, horizon + 1)
    ]

    for split_name, split_data, split_end in [
        ("Training", train, TRAIN_END),
        ("Validation", validation, VALIDATION_END),
        ("Test", test, test.index.max()),
    ]:
        complete_targets = split_data[target_columns].dropna()

        if complete_targets.empty:
            raise ValueError(
                f"{split_name} has no complete target rows."
            )

        last_target_row = complete_targets.index.max()

        if split_name == "Training":
            expected_end = TRAIN_END
        elif split_name == "Validation":
            expected_end = VALIDATION_END
        else:
            expected_end = test.index.max()

        expected_last_feature_timestamp = (
            expected_end
            - pd.Timedelta(hours=horizon)
        )

        if last_target_row != expected_last_feature_timestamp:
            raise ValueError(
                f"{split_name} target boundary is incorrect. "
                f"Expected last usable feature timestamp "
                f"{expected_last_feature_timestamp}, "
                f"found {last_target_row}."
            )

    print(
        "✓ No 24-hour target crosses a train/validation/test boundary."
    )


def main() -> None:
    print(
        f"Processed dataset: {PROCESSED_DATA_PATH}"
    )

    if not PROCESSED_DATA_PATH.exists():
        raise FileNotFoundError(
            f"Processed dataset not found: "
            f"{PROCESSED_DATA_PATH}"
        )

    data = load_processed_data()

    print(f"\nRows: {len(data):,}")
    print(f"Columns: {list(data.columns)}")
    print(f"Start: {data.index.min()}")
    print(f"End: {data.index.max()}")

    # ---------------------------------------------------------
    # Feature engineering
    # ---------------------------------------------------------

    data = add_lag_features(data)
    data = add_calendar_features(data)
    data = add_rolling_features(data)
    data = add_cyclic_features(data)

    validate_features(data)

    # ---------------------------------------------------------
    # Chronological split
    # ---------------------------------------------------------

    train, validation, test = split_data(data)

    print("\nSplit summary:")
    print(
        f"Train:      {len(train):,} rows | "
        f"{train.index.min()} → {train.index.max()}"
    )
    print(
        f"Validation: {len(validation):,} rows | "
        f"{validation.index.min()} → {validation.index.max()}"
    )
    print(
        f"Test:       {len(test):,} rows | "
        f"{test.index.min()} → {test.index.max()}"
    )

    # ---------------------------------------------------------
    # Boundary-safe 24-hour targets
    # ---------------------------------------------------------

    train = add_forecast_targets(
        train,
        FORECAST_HORIZON,
    )

    validation = add_forecast_targets(
        validation,
        FORECAST_HORIZON,
    )

    test = add_forecast_targets(
        test,
        FORECAST_HORIZON,
    )

    # ---------------------------------------------------------
    # Target validation
    # ---------------------------------------------------------

    print("\nTarget validation:")

    validate_target_rows(
        "Training",
        train,
        FORECAST_HORIZON,
    )

    validate_target_rows(
        "Validation",
        validation,
        FORECAST_HORIZON,
    )

    validate_target_rows(
        "Test",
        test,
        FORECAST_HORIZON,
    )

    # ---------------------------------------------------------
    # Target alignment validation
    # ---------------------------------------------------------

    print("\nTarget alignment validation:")

    validate_target_alignment(
        train,
        FORECAST_HORIZON,
    )

    validate_target_alignment(
        validation,
        FORECAST_HORIZON,
    )

    validate_target_alignment(
        test,
        FORECAST_HORIZON,
    )

    # ---------------------------------------------------------
    # Split-boundary validation
    # ---------------------------------------------------------

    print("\nSplit target-boundary validation:")

    validate_split_boundaries(
        train,
        validation,
        test,
        FORECAST_HORIZON,
    )

    print("\nFeature columns:")
    print(
        list(
            data.columns
        )
    )

    print("\nTarget columns:")
    print(
        [
            column
            for column in train.columns
            if column.startswith("target_")
        ]
    )


if __name__ == "__main__":
    main()