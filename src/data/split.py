from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]

PROCESSED_DATA_PATH = (
    PROJECT_ROOT / "data" / "processed" / "hourly_demand.csv"
)


TRAIN_END = pd.Timestamp("2012-12-31 23:00:00")
VALIDATION_END = pd.Timestamp("2013-12-31 23:00:00")


def load_processed_data() -> pd.DataFrame:
    """Load the processed hourly demand dataset."""
    data = pd.read_csv(
        PROCESSED_DATA_PATH,
        parse_dates=["timestamp"],
        index_col="timestamp",
    )

    return data


def split_data(
    data: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split the dataset chronologically into train, validation, and test."""
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


def validate_splits(
    train: pd.DataFrame,
    validation: pd.DataFrame,
    test: pd.DataFrame,
) -> None:
    """Validate chronological ordering and separation of splits."""

    if len(train) == 0:
        raise ValueError("Training set is empty.")

    if len(validation) == 0:
        raise ValueError("Validation set is empty.")

    if len(test) == 0:
        raise ValueError("Test set is empty.")

    if train.index.max() >= validation.index.min():
        raise ValueError(
            "Training and validation sets overlap or are out of order."
        )

    if validation.index.max() >= test.index.min():
        raise ValueError(
            "Validation and test sets overlap or are out of order."
        )

    if not train.index.is_monotonic_increasing:
        raise ValueError("Training timestamps are not sorted.")

    if not validation.index.is_monotonic_increasing:
        raise ValueError("Validation timestamps are not sorted.")

    if not test.index.is_monotonic_increasing:
        raise ValueError("Test timestamps are not sorted.")

    if train.index.duplicated().any():
        raise ValueError("Duplicate timestamps found in training set.")

    if validation.index.duplicated().any():
        raise ValueError("Duplicate timestamps found in validation set.")

    if test.index.duplicated().any():
        raise ValueError("Duplicate timestamps found in test set.")

    combined_index = train.index.append(
    validation.index
    ).append(
        test.index
    )

    expected_intervals = (
        combined_index.to_series()
        .diff()
        .dropna()
    )

    if not (
        expected_intervals
        == pd.Timedelta(hours=1)
    ).all():
        raise ValueError(
            "Combined dataset contains irregular hourly intervals."
        )

    if not combined_index.equals(
        combined_index.sort_values()
    ):
        raise ValueError(
            "Combined split timestamps are not in chronological order."
        )

    if not combined_index.equals(
        combined_index.unique()
    ):
        raise ValueError(
            "Duplicate timestamps found across splits."
        )
    
    total_rows = len(train) + len(validation) + len(test)

    if total_rows != 35_064:
        raise ValueError(
            f"Split sizes do not sum to 35,064. Found {total_rows:,}."
        )

    print("\n✓ Training set is non-empty.")
    print("✓ Validation set is non-empty.")
    print("✓ Test set is non-empty.")
    print("✓ No train/validation overlap.")
    print("✓ No validation/test overlap.")
    print("✓ All splits are chronologically ordered.")
    print("✓ Split sizes sum to the full dataset.")
    print("✓ No duplicate timestamps within any split.")
    print("✓ Combined splits contain no duplicate timestamps.")
    print("✓ Combined splits preserve chronological order.")
    print("✓ Combined dataset has continuous hourly intervals.")
    
def main() -> None:
    print(f"Processed dataset: {PROCESSED_DATA_PATH}")

    if not PROCESSED_DATA_PATH.exists():
        raise FileNotFoundError(
            f"Processed dataset not found: {PROCESSED_DATA_PATH}"
        )

    data = load_processed_data()

    print(f"\nFull dataset: {len(data):,} rows")
    print(f"Start: {data.index.min()}")
    print(f"End: {data.index.max()}")

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

    validate_splits(train, validation, test)


if __name__ == "__main__":
    main()