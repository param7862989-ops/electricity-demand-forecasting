from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]

RAW_DATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "LD2011_2014.txt"
)

PROCESSED_DATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "hourly_demand.csv"
)

CHUNK_SIZE = 10_000


def aggregate_clients() -> pd.Series:
    """Aggregate all client demand into a single 15-minute series."""

    aggregate_chunks = []

    print("Reading raw dataset in chunks...")

    for chunk_number, chunk in enumerate(
        pd.read_csv(
            RAW_DATA_PATH,
            sep=";",
            decimal=",",
            chunksize=CHUNK_SIZE,
        ),
        start=1,
    ):
        timestamps = pd.to_datetime(chunk.iloc[:, 0])

        client_data = chunk.iloc[:, 1:]

        total_demand = client_data.sum(axis=1)

        aggregate_chunk = pd.Series(
            total_demand.to_numpy(),
            index=timestamps,
            name="total_demand",
        )

        aggregate_chunks.append(aggregate_chunk)

        print(
            f"Processed chunk {chunk_number}: "
            f"{len(chunk):,} rows"
        )

    total_demand_15min = pd.concat(aggregate_chunks)

    return total_demand_15min

def convert_to_hourly(
    total_demand_15min: pd.Series,
) -> pd.Series:
    """Convert 15-minute demand observations to complete hourly values."""

    shifted_demand = total_demand_15min.copy()

    shifted_demand.index = (
        shifted_demand.index
        - pd.Timedelta(minutes=15)
    )

    hourly_resampled = shifted_demand.resample("1h").agg(
        ["mean", "count"]
    )

    hourly_demand = hourly_resampled.loc[
    hourly_resampled["count"] == 4,
    "mean"
    ].rename("total_demand")

    hourly_demand.index.name = "timestamp"

    return hourly_demand

def validate_hourly_data(hourly_demand: pd.Series) -> None:
    """Validate the processed hourly demand series."""

    print("\nValidating hourly dataset...")

    if not hourly_demand.index.is_monotonic_increasing:
        raise ValueError("Hourly timestamps are not sorted.")

    if hourly_demand.index.duplicated().any():
        raise ValueError("Duplicate hourly timestamps detected.")

    if hourly_demand.isna().any():
        raise ValueError("Missing hourly demand values detected.")

    if (hourly_demand < 0).any():
        raise ValueError("Negative demand values detected.")

    intervals = hourly_demand.index.to_series().diff().dropna()

    irregular_intervals = (
        intervals != pd.Timedelta(hours=1)
    ).sum()

    if irregular_intervals > 0:
        raise ValueError(
            f"Found {irregular_intervals} irregular hourly intervals."
        )

    expected_observations = 35_064

    if len(hourly_demand) != expected_observations:
        raise ValueError(
            f"Expected {expected_observations:,} hourly observations, "
            f"but found {len(hourly_demand):,}."
        )

    expected_start = pd.Timestamp("2011-01-01 00:00:00")
    expected_end = pd.Timestamp("2014-12-31 23:00:00")

    if hourly_demand.index.min() != expected_start:
        raise ValueError(
            f"Unexpected start timestamp: {hourly_demand.index.min()}"
        )

    if hourly_demand.index.max() != expected_end:
        raise ValueError(
            f"Unexpected end timestamp: {hourly_demand.index.max()}"
        )

    print("✓ Timestamps are sorted.")
    print("✓ No duplicate timestamps.")
    print("✓ No missing demand values.")
    print("✓ No negative demand values.")
    print("✓ All hourly intervals are exactly 1 hour.")
    print(f"✓ Observation count: {len(hourly_demand):,}")
    print(f"✓ Start: {hourly_demand.index.min()}")
    print(f"✓ End: {hourly_demand.index.max()}")


def save_processed_data(hourly_demand: pd.Series) -> None:
    """Save the processed hourly demand dataset."""

    PROCESSED_DATA_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    output = hourly_demand.rename(
        "total_demand"
    ).to_frame()

    output.index.name = "timestamp"

    output.to_csv(
        PROCESSED_DATA_PATH,
        index=True,
    )

    print("\nProcessed dataset saved.")
    print(f"Path: {PROCESSED_DATA_PATH}")
    print(f"Rows: {len(output):,}")
    print(f"Columns: {output.shape[1]}")


def validate_saved_data() -> None:
    """Validate the processed CSV after saving."""

    print("\nValidating saved processed dataset...")

    saved_data = pd.read_csv(
        PROCESSED_DATA_PATH,
        parse_dates=["timestamp"],
        index_col="timestamp",
    )

    if list(saved_data.columns) != ["total_demand"]:
        raise ValueError(
            "Unexpected processed dataset columns."
        )

    if len(saved_data) != 35_064:
        raise ValueError(
            f"Expected 35,064 rows, found {len(saved_data):,}."
        )

    if saved_data.index.duplicated().any():
        raise ValueError(
            "Duplicate timestamps found in saved dataset."
        )

    if not saved_data.index.is_monotonic_increasing:
        raise ValueError(
            "Saved timestamps are not sorted."
        )

    if saved_data["total_demand"].isna().any():
        raise ValueError(
            "Missing demand values found in saved dataset."
        )

    if (saved_data["total_demand"] < 0).any():
        raise ValueError(
            "Negative demand values found in saved dataset."
        )

    intervals = saved_data.index.to_series().diff().dropna()

    if not (intervals == pd.Timedelta(hours=1)).all():
        raise ValueError(
            "Irregular hourly intervals found in saved dataset."
        )

    print("✓ Correct column structure.")
    print("✓ Correct observation count.")
    print("✓ No duplicate timestamps.")
    print("✓ Timestamps are sorted.")
    print("✓ No missing demand values.")
    print("✓ No negative demand values.")
    print("✓ All timestamps are exactly 1 hour apart.")


def preprocess_data() -> None:
    print(f"Raw dataset: {RAW_DATA_PATH}")
    print(f"Raw dataset exists: {RAW_DATA_PATH.exists()}")

    if not RAW_DATA_PATH.exists():
        raise FileNotFoundError(
            f"Raw dataset not found: {RAW_DATA_PATH}"
        )

    print("\nStarting preprocessing...")

    total_demand_15min = aggregate_clients()
    print("\n15-minute aggregation complete.")
    print(f"15-minute observations: {len(total_demand_15min):,}")
    print(f"Start: {total_demand_15min.index.min()}")
    print(f"End: {total_demand_15min.index.max()}")

    hourly_demand = convert_to_hourly(
        total_demand_15min
    )
    validate_hourly_data(hourly_demand)
    save_processed_data(hourly_demand)
    validate_saved_data()
    print("\nHourly conversion complete.")
    print(f"Hourly observations: {len(hourly_demand):,}")
    print(f"Start: {hourly_demand.index.min()}")
    print(f"End: {hourly_demand.index.max()}")
    print(
        f"Missing hourly values: "
        f"{hourly_demand.isna().sum()}"
    )

    print("\nFirst 5 hourly observations:")
    print(hourly_demand.head())

    print("\nLast 5 hourly observations:")
    print(hourly_demand.tail())

if __name__ == "__main__":
    preprocess_data()