from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[2]

PREDICTIONS_PATH = ROOT / "reports" / "gru_validation_predictions.csv"
ERRORS_PATH = ROOT / "reports" / "gru_horizon_errors.csv"
LARGE_ERRORS_PATH = ROOT / "reports" / "gru_largest_errors.csv"
FIGURES_DIR = ROOT / "reports" / "figures"

FIGURES_DIR.mkdir(parents=True, exist_ok=True)


df = pd.read_csv(PREDICTIONS_PATH)
df["timestamp"] = pd.to_datetime(df["timestamp"])


results = []

for horizon in range(1, 25):
    actual = df[f"actual_{horizon}"].to_numpy()
    prediction = df[f"prediction_{horizon}"].to_numpy()

    error = actual - prediction

    mae = np.mean(np.abs(error))
    rmse = np.sqrt(np.mean(error ** 2))
    mape = np.mean(
        np.abs(error) / np.where(actual == 0, np.nan, np.abs(actual))
    ) * 100

    results.append(
        {
            "forecast_hour": horizon,
            "MAE": mae,
            "RMSE": rmse,
            "MAPE": mape,
        }
    )


horizon_errors = pd.DataFrame(results)
horizon_errors.to_csv(ERRORS_PATH, index=False)


plt.figure(figsize=(10, 5))
plt.plot(
    horizon_errors["forecast_hour"],
    horizon_errors["MAE"],
    marker="o"
)
plt.xlabel("Forecast Hour")
plt.ylabel("MAE")
plt.title("GRU MAE by Forecast Horizon")
plt.xticks(range(1, 25))
plt.tight_layout()
plt.savefig(FIGURES_DIR / "gru_mae_by_horizon.png", dpi=150)
plt.close()


plt.figure(figsize=(10, 5))
plt.plot(
    horizon_errors["forecast_hour"],
    horizon_errors["RMSE"],
    marker="o"
)
plt.xlabel("Forecast Hour")
plt.ylabel("RMSE")
plt.title("GRU RMSE by Forecast Horizon")
plt.xticks(range(1, 25))
plt.tight_layout()
plt.savefig(FIGURES_DIR / "gru_rmse_by_horizon.png", dpi=150)
plt.close()


plt.figure(figsize=(10, 5))
plt.plot(
    horizon_errors["forecast_hour"],
    horizon_errors["MAPE"],
    marker="o"
)
plt.xlabel("Forecast Hour")
plt.ylabel("MAPE (%)")
plt.title("GRU MAPE by Forecast Horizon")
plt.xticks(range(1, 25))
plt.tight_layout()
plt.savefig(FIGURES_DIR / "gru_mape_by_horizon.png", dpi=150)
plt.close()


sample = df.iloc[:168].copy()

actual_values = []
prediction_values = []

for _, row in sample.iterrows():
    for horizon in range(1, 25):
        actual_values.append(row[f"actual_{horizon}"])
        prediction_values.append(row[f"prediction_{horizon}"])

comparison = pd.DataFrame(
    {
        "actual": actual_values,
        "prediction": prediction_values,
    }
)

plt.figure(figsize=(14, 5))
plt.plot(comparison["actual"], label="Actual")
plt.plot(comparison["prediction"], label="GRU Prediction")
plt.xlabel("Forecasted Hour")
plt.ylabel("Electricity Demand")
plt.title("GRU Actual vs Predicted Demand")
plt.legend()
plt.tight_layout()
plt.savefig(FIGURES_DIR / "gru_actual_vs_predicted.png", dpi=150)
plt.close()


errors = []

for _, row in df.iterrows():
    for horizon in range(1, 25):
        actual = row[f"actual_{horizon}"]
        prediction = row[f"prediction_{horizon}"]
        absolute_error = abs(actual - prediction)

        errors.append(
            {
                "forecast_origin": row["timestamp"],
                "forecast_hour": horizon,
                "actual": actual,
                "prediction": prediction,
                "absolute_error": absolute_error,
            }
        )


error_df = pd.DataFrame(errors)

largest_errors = error_df.sort_values(
    "absolute_error",
    ascending=False
).head(20)

largest_errors.to_csv(
    LARGE_ERRORS_PATH,
    index=False
)


plt.figure(figsize=(10, 5))
plt.hist(
    error_df["absolute_error"],
    bins=50
)
plt.xlabel("Absolute Error")
plt.ylabel("Frequency")
plt.title("Distribution of GRU Absolute Errors")
plt.tight_layout()
plt.savefig(FIGURES_DIR / "gru_error_distribution.png", dpi=150)
plt.close()


print("Day 12 error analysis complete.")
print()
print("Horizon-wise errors:")
print(horizon_errors.to_string(index=False))
print()
print("Largest absolute errors:")
print(largest_errors.to_string(index=False))
print()
print(f"Saved: {ERRORS_PATH}")
print(f"Saved: {LARGE_ERRORS_PATH}")
print(f"Figures saved to: {FIGURES_DIR}")