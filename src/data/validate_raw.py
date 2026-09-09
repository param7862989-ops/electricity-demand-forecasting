from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_DATA_PATH = PROJECT_ROOT / "data" / "raw" / "LD2011_2014.txt"

EXPECTED_CLIENT_COUNT = 370
CHUNK_SIZE = 10_000


def validate_raw_data() -> None:
    """Validate the structure and quality of the raw dataset."""

    print(f"Dataset: {RAW_DATA_PATH}")
    print(f"Exists: {RAW_DATA_PATH.exists()}")

    if not RAW_DATA_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found: {RAW_DATA_PATH}"
        )

    total_rows = 0
    total_missing_values = 0
    total_negative_values = 0
    duplicate_timestamps = 0

    first_timestamp = None
    last_timestamp = None
    previous_timestamp = None
    irregular_intervals = 0

    expected_columns = ["Unnamed: 0"] + [
        f"MT_{i:03d}" for i in range(1, EXPECTED_CLIENT_COUNT + 1)
    ]

    print("\nReading dataset in chunks...")

    for chunk_number, chunk in enumerate(
        pd.read_csv(
            RAW_DATA_PATH,
            sep=";",
            decimal=",",
            chunksize=CHUNK_SIZE,
        ),
        start=1,
    ):
        # -------------------------------------------------
        # 1. Validate columns
        # -------------------------------------------------

        if chunk.columns.tolist() != expected_columns:
            raise ValueError(
                "Unexpected column structure detected."
            )

        # -------------------------------------------------
        # 2. Convert timestamp
        # -------------------------------------------------

        timestamps = pd.to_datetime(
            chunk["Unnamed: 0"],
            errors="coerce",
        )

        invalid_timestamps = timestamps.isna().sum()

        if invalid_timestamps > 0:
            raise ValueError(
                f"Found {invalid_timestamps} invalid timestamps "
                f"in chunk {chunk_number}."
            )

        # -------------------------------------------------
        # 3. Track timestamp range
        # -------------------------------------------------

        if first_timestamp is None:
            first_timestamp = timestamps.iloc[0]

        last_timestamp = timestamps.iloc[-1]

        # -------------------------------------------------
        # 4. Check timestamp ordering and intervals
        # -------------------------------------------------

        if previous_timestamp is not None:
            interval = timestamps.iloc[0] - previous_timestamp

            if interval != pd.Timedelta(minutes=15):
                irregular_intervals += 1

        internal_intervals = timestamps.diff().dropna()

        irregular_intervals += (
            internal_intervals != pd.Timedelta(minutes=15)
        ).sum()

        previous_timestamp = timestamps.iloc[-1]

        # -------------------------------------------------
        # 5. Check duplicate timestamps
        # -------------------------------------------------

        duplicate_timestamps += timestamps.duplicated().sum()

        # -------------------------------------------------
        # 6. Check missing values
        # -------------------------------------------------

        total_missing_values += chunk.isna().sum().sum()

        # -------------------------------------------------
        # 7. Check negative electricity values
        # -------------------------------------------------

        client_data = chunk.iloc[:, 1:]

        total_negative_values += (
            client_data < 0
        ).sum().sum()

        # -------------------------------------------------
        # 8. Count rows
        # -------------------------------------------------

        total_rows += len(chunk)

        print(
            f"Processed chunk {chunk_number}: "
            f"{total_rows:,} rows"
        )

    # -----------------------------------------------------
    # Final report
    # -----------------------------------------------------

    print("\n" + "=" * 60)
    print("RAW DATA VALIDATION REPORT")
    print("=" * 60)

    print(f"Rows:                {total_rows:,}")
    print(f"Client columns:      {EXPECTED_CLIENT_COUNT}")
    print(f"Total columns:       {EXPECTED_CLIENT_COUNT + 1}")

    print(f"First timestamp:     {first_timestamp}")
    print(f"Last timestamp:      {last_timestamp}")

    print(f"Missing values:      {total_missing_values:,}")
    print(f"Negative values:     {total_negative_values:,}")
    print(f"Duplicate timestamps:{duplicate_timestamps:,}")
    print(f"Irregular intervals: {irregular_intervals:,}")

    print("=" * 60)

    if total_missing_values == 0:
        print("✓ No missing values detected.")
    else:
        print("⚠ Missing values detected.")

    if total_negative_values == 0:
        print("✓ No negative values detected.")
    else:
        print("⚠ Negative values detected.")

    if duplicate_timestamps == 0:
        print("✓ No duplicate timestamps detected.")
    else:
        print("⚠ Duplicate timestamps detected.")

    if irregular_intervals == 0:
        print("✓ All timestamps follow 15-minute intervals.")
    else:
        print("⚠ Irregular timestamp intervals detected.")


if __name__ == "__main__":
    validate_raw_data()