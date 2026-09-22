from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[2]

OUTPUT_PATH = ROOT / "reports" / "model_comparison.csv"
FIGURES_DIR = ROOT / "reports" / "figures"

FIGURES_DIR.mkdir(parents=True, exist_ok=True)


results = [
    {
        "model": "Naive",
        "MAE": 107900.28,
        "RMSE": 134801.28,
        "MAPE": 41.13,
    },
    {
        "model": "Seasonal Naive 24h",
        "MAE": 8105.95,
        "RMSE": 13459.18,
        "MAPE": 3.84,
    },
    {
        "model": "Seasonal Naive 168h",
        "MAE": 12436.73,
        "RMSE": 19947.32,
        "MAPE": 5.24,
    },
    {
        "model": "Dense NN",
        "MAE": 10113.58,
        "RMSE": 13451.73,
        "MAPE": 5.07,
    },
    {
        "model": "1D CNN",
        "MAE": 15649.42,
        "RMSE": 20537.18,
        "MAPE": 7.39,
    },
    {
        "model": "LSTM",
        "MAE": 10808.22,
        "RMSE": 13987.64,
        "MAPE": 5.53,
    },
    {
        "model": "GRU",
        "MAE": 10684.33,
        "RMSE": 14190.67,
        "MAPE": 5.48,
    },
]


comparison = pd.DataFrame(results)

comparison.to_csv(OUTPUT_PATH, index=False)


plt.figure(figsize=(11, 6))
plt.bar(comparison["model"], comparison["MAE"])
plt.xlabel("Model")
plt.ylabel("MAE")
plt.title("Model Comparison — MAE")
plt.xticks(rotation=30, ha="right")
plt.tight_layout()
plt.savefig(FIGURES_DIR / "model_comparison_mae.png", dpi=150)
plt.close()


plt.figure(figsize=(11, 6))
plt.bar(comparison["model"], comparison["RMSE"])
plt.xlabel("Model")
plt.ylabel("RMSE")
plt.title("Model Comparison — RMSE")
plt.xticks(rotation=30, ha="right")
plt.tight_layout()
plt.savefig(FIGURES_DIR / "model_comparison_rmse.png", dpi=150)
plt.close()


plt.figure(figsize=(11, 6))
plt.bar(comparison["model"], comparison["MAPE"])
plt.xlabel("Model")
plt.ylabel("MAPE (%)")
plt.title("Model Comparison — MAPE")
plt.xticks(rotation=30, ha="right")
plt.tight_layout()
plt.savefig(FIGURES_DIR / "model_comparison_mape.png", dpi=150)
plt.close()


print("Model comparison complete.")
print()
print(comparison.to_string(index=False))
print()
print(f"Saved: {OUTPUT_PATH}")
print(f"Figures saved to: {FIGURES_DIR}")