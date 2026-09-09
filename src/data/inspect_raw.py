from pathlib import Path

import pandas as pd


# Project paths
PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_DATA_PATH = PROJECT_ROOT / "data" / "raw" / "LD2011_2014.txt"


def inspect_raw_data() -> None:
    """Inspect a small sample of the raw electricity dataset."""

    print(f"Dataset path: {RAW_DATA_PATH}")
    print(f"Dataset exists: {RAW_DATA_PATH.exists()}")

    if not RAW_DATA_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found at: {RAW_DATA_PATH}"
        )

    # Read only a small sample.
    sample = pd.read_csv(
        RAW_DATA_PATH,
        sep=";",
        decimal=",",
        nrows=5,
    )

    print("\n--- Shape of sample ---")
    print(sample.shape)

    print("\n--- Columns ---")
    print(sample.columns.tolist())

    print("\n--- First 5 rows ---")
    print(sample.head())

    print("\n--- Data types ---")
    print(sample.dtypes)

    print("\n--- Timestamp range in sample ---")
    print(sample.iloc[:, 0].min())
    print(sample.iloc[:, 0].max())


if __name__ == "__main__":
    inspect_raw_data()