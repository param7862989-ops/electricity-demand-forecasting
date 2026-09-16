from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]

PROCESSED_DATA_PATH = (
    PROJECT_ROOT / "data" / "processed" / "hourly_demand.csv"
)

TRAIN_END = pd.Timestamp("2012-12-31 23:00:00")
VALIDATION_END = pd.Timestamp("2013-12-31 23:00:00")


def load_data() -> pd.DataFrame:
    """Load processed hourly demand data."""
    return pd.read_csv(
        PROCESSED_DATA_PATH,
        parse_dates=["timestamp"],
        index_col="timestamp",
    )


def get_validation_period(
    data: pd.DataFrame,
) -> tuple[pd.Series, pd.Series]:
    """Return training history and validation demand."""
    train = data.loc[
        data.index <= TRAIN_END,
        "total_demand",
    ]

    validation = data.loc[
        (data.index > TRAIN_END)
        & (data.index <= VALIDATION_END),
        "total_demand",
    ]

    return train, validation

def naive_forecast(
    train: pd.Series,
    validation: pd.Series,
) -> pd.Series:
    """Forecast each validation hour using the latest observed demand."""

    predictions = pd.Series(
        train.iloc[-1],
        index=validation.index,
        name="naive_prediction",
    )

    return predictions

def seasonal_naive_168(
    data: pd.DataFrame,
    validation: pd.Series,
) -> pd.Series:
    """Forecast using demand from the same hour one week earlier."""

    predictions = data.loc[
        validation.index - pd.Timedelta(hours=168),
        "total_demand",
    ].copy()

    predictions.index = validation.index
    predictions.name = "seasonal_naive_168_prediction"

    return predictions

def seasonal_naive_24(
    data: pd.DataFrame,
    validation: pd.Series,
) -> pd.Series:
    """Forecast using demand from the same hour on the previous day."""

    predictions = data.loc[
        validation.index - pd.Timedelta(hours=24),
        "total_demand",
    ].copy()

    predictions.index = validation.index
    predictions.name = "seasonal_naive_24_prediction"

    return predictions

import numpy as np


def calculate_metrics(
    actual: pd.Series,
    predicted: pd.Series,
) -> dict[str, float]:
    """Calculate forecasting evaluation metrics."""

    mae = np.mean(
        np.abs(actual - predicted)
    )

    rmse = np.sqrt(
        np.mean(
            (actual - predicted) ** 2
        )
    )

    mape = np.mean(
        np.abs(
            (actual - predicted) / actual
        )
    ) * 100

    return {
        "MAE": float(mae),
        "RMSE": float(rmse),
        "MAPE": float(mape),
    }

def main() -> None:
    print(f"Processed dataset: {PROCESSED_DATA_PATH}")

    if not PROCESSED_DATA_PATH.exists():
        raise FileNotFoundError(
            f"Processed dataset not found: {PROCESSED_DATA_PATH}"
        )

    data = load_data()

    train, validation = get_validation_period(data)

    print(f"\nTraining observations: {len(train):,}")
    print(
        f"Training period: "
        f"{train.index.min()} → {train.index.max()}"
    )

    print(f"\nValidation observations: {len(validation):,}")
    print(
        f"Validation period: "
        f"{validation.index.min()} → {validation.index.max()}"
    )
    predictions = naive_forecast(
        train,
        validation,
    )

    print("\nNaive baseline:")
    print(f"Prediction rows: {len(predictions):,}")
    print(
        f"Prediction start: {predictions.index.min()}"
    )
    print(
        f"Prediction end: {predictions.index.max()}"
    )
    print(
        f"Forecast value: {predictions.iloc[0]:.2f}"
    )
    seasonal_predictions = seasonal_naive_24(
        data,
        validation,
    )

    print("\n24-hour seasonal-naive baseline:")
    print(
        f"Prediction rows: "
        f"{len(seasonal_predictions):,}"
    )
    print(
        f"Prediction start: "
        f"{seasonal_predictions.index.min()}"
    )
    print(
        f"Prediction end: "
        f"{seasonal_predictions.index.max()}"
    )

    print("\nFirst 5 seasonal-naive predictions:")
    print(seasonal_predictions.head())
    
    weekly_predictions = seasonal_naive_168(
        data,
        validation,
    )

    print("\n168-hour seasonal-naive baseline:")
    print(
        f"Prediction rows: "
        f"{len(weekly_predictions):,}"
    )
    print(
        f"Prediction start: "
        f"{weekly_predictions.index.min()}"
    )
    print(
        f"Prediction end: "
        f"{weekly_predictions.index.max()}"
    )

    print("\nFirst 5 weekly seasonal-naive predictions:")
    print(weekly_predictions.head())
    
    naive_metrics = calculate_metrics(
        validation,
        predictions,
    )

    seasonal_24_metrics = calculate_metrics(
        validation,
        seasonal_predictions,
    )

    seasonal_168_metrics = calculate_metrics(
        validation,
        weekly_predictions,
    )

    results = pd.DataFrame(
        [
            naive_metrics,
            seasonal_24_metrics,
            seasonal_168_metrics,
        ],
        index=[
            "Naive",
            "Seasonal Naive 24h",
            "Seasonal Naive 168h",
        ],
    )

    print("\nBaseline evaluation results:")
    print(results.round(2))

if __name__ == "__main__":
    main()