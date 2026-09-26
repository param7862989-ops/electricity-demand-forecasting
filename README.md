# Electricity Demand Forecasting

An end-to-end machine learning project for forecasting the next 24 hours of aggregate electricity demand using historical electricity consumption data, temporal features, TensorFlow sequence models, and a production-ready inference API.

The project covers the complete ML lifecycle:

**Data → EDA → Preprocessing → Feature Engineering → Baselines → Deep Learning → Model Evaluation → Hyperparameter Tuning → Inference → FastAPI → Dashboard → Testing → CI**

---

## Project Overview

Electricity demand varies according to strong temporal patterns such as:

- Hour of the day
- Day of the week
- Month of the year
- Recent electricity demand
- Weekly and daily seasonal patterns
- Calendar-related behavior

The objective of this project is to build a forecasting system that predicts the **next 24 hours of aggregate electricity demand at hourly resolution**.

The project uses the UCI ElectricityLoadDiagrams20112014 dataset, which contains electricity consumption measurements from 370 clients at 15-minute intervals.

---

## Problem Statement

> Develop an end-to-end TensorFlow-based time-series forecasting system that predicts the next 24 hours of aggregate electricity demand at hourly resolution using historical electricity demand and temporal patterns, while evaluating multiple forecasting approaches against appropriate baseline models.

---

## Final Forecasting Configuration

| Component | Configuration |
|---|---|
| Forecast target | Aggregate electricity demand |
| Original resolution | 15 minutes |
| Forecast resolution | 1 hour |
| Input history | Previous 24 hours |
| Forecast horizon | Next 24 hours |
| Number of clients | 370 |
| Final model | GRU |
| GRU units | 128 |
| Dropout | 0.2 |
| Dense units | 64 |
| Output units | 24 |
| Optimizer | Adam |
| Learning rate | 0.001 |
| Loss | Mean Squared Error |
| Random seed | 42 |

---

# Dataset

## Dataset Used

**UCI ElectricityLoadDiagrams20112014**

The dataset contains electricity consumption measurements for 370 clients collected over the period from 2011 to 2014.

The original data is recorded at 15-minute intervals.

Dataset source:

https://archive.ics.uci.edu/dataset/321/electricityloaddiagrams20112014

### Dataset characteristics

- 370 electricity clients
- 15-minute measurements
- 2011–2014 time period
- 140,256 observations
- Aggregate demand created by summing all 370 clients
- Hourly dataset created from complete 15-minute observations

The raw dataset is intentionally excluded from Git using `.gitignore` because of its large size.

---

# Data Processing

The original dataset contains electricity demand for individual clients.

The preprocessing pipeline performs the following operations:

```text
Raw 15-minute electricity data
        ↓
Parse timestamps
        ↓
Aggregate all 370 clients
        ↓
Correct timestamp alignment
        ↓
Convert 15-minute data to hourly demand
        ↓
Keep only complete hourly observations
        ↓
Validate continuity and data quality
        ↓
Save processed hourly dataset
```

The resulting processed dataset contains:

- **35,064 hourly observations**
- Start: `2011-01-01 00:00:00`
- End: `2014-12-31 23:00:00`

The preprocessing pipeline validates:

- Missing values
- Duplicate timestamps
- Negative demand values
- Timestamp ordering
- Hourly continuity
- Expected dataset boundaries

---

# Exploratory Data Analysis

The EDA showed that electricity demand contains strong temporal structure.

## Daily Seasonality

Demand varies significantly by hour of the day.

The average demand was approximately:

- Lowest around **04:00**
- Highest around **18:00**

This demonstrates strong daily seasonality.

## Weekly Seasonality

Demand also varies by day of the week.

Average demand was approximately:

| Day | Average Demand |
|---|---:|
| Sunday | 190,922 |
| Monday | 193,827 |
| Wednesday | 195,684 |
| Tuesday | 196,221 |
| Thursday | 196,554 |
| Saturday | 196,996 |
| Friday | 198,714 |

The weekly effect is weaker than the hour-of-day effect but is still useful for forecasting.

## Annual Seasonality

Average yearly demand:

| Year | Average Demand |
|---|---:|
| 2011 | 122,714.96 |
| 2012 | 212,318.89 |
| 2013 | 222,736.79 |
| 2014 | 224,411.20 |

There is a substantial structural increase between 2011 and 2012.

This is consistent with the staggered activation of clients in the dataset and should not simply be interpreted as a normal linear trend.

## Monthly Seasonality

Average demand varies substantially throughout the year.

Demand generally increases toward the summer months and reaches its highest average around August.

Approximate monthly averages:

| Month | Average Demand |
|---|---:|
| January | 171,540.84 |
| February | 169,133.81 |
| March | 170,037.05 |
| April | 176,761.82 |
| May | 193,302.94 |
| June | 212,098.69 |
| July | 237,175.28 |
| August | 242,072.12 |
| September | 226,265.54 |
| October | 199,258.32 |
| November | 175,287.76 |
| December | 171,669.22 |

---

# Anomaly Analysis

The dataset contains some extreme demand values.

An IQR-based analysis identified approximately 81 potential high outliers, representing only around 0.23% of hourly observations.

However, these observations were not automatically removed.

Several high-demand observations occurred during summer daytime periods and appeared to represent legitimate electricity demand peaks.

There were also unusually low observations around certain late-March dates.

These low values were confirmed to already exist in the original 15-minute dataset rather than being introduced by hourly aggregation.

Therefore, the project treats extreme observations as potentially meaningful time-series behavior instead of blindly removing them.

---

# Feature Engineering

The sequence models use eight input features:

```text
1. total_demand
2. hour_sin
3. hour_cos
4. day_of_week_sin
5. day_of_week_cos
6. month_sin
7. month_cos
8. is_weekend
```

Cyclical encoding is used for time-based variables.

For example, hour-of-day is represented using sine and cosine transformations so that:

```text
23:00
```

and

```text
00:00
```

are treated as temporally close rather than numerically far apart.

---

# Train / Validation / Test Strategy

The project uses chronological splitting rather than random splitting.

This is important for time-series forecasting because randomly mixing future observations into the training data would create data leakage.

The general structure is:

```text
Past
│
├── Training Data
│
├── Validation Data
│
└── Test Data
│
Future
```

Scaling is also fitted only on the training data.

The validation and test sets are transformed using parameters learned from the training set.

The test set is kept untouched until final model evaluation.

---

# Baseline Models

Before training neural networks, simple forecasting baselines were implemented.

## Naive Forecast

The naive forecast predicts the next value using the most recent observed value.

Results:

| Model | MAE | RMSE | MAPE |
|---|---:|---:|---:|
| Naive | 107,900.28 | 134,801.28 | 41.13% |

This provides a basic reference point.

## Seasonal Naive — 24 Hours

The forecast uses demand from the same hour of the previous day.

Results:

| Model | MAE | RMSE | MAPE |
|---|---:|---:|---:|
| Seasonal Naive 24h | 8,105.95 | 13,459.18 | 3.84% |

This proved to be a very strong baseline.

## Seasonal Naive — 168 Hours

The forecast uses demand from the same hour of the previous week.

Results:

| Model | MAE | RMSE | MAPE |
|---|---:|---:|---:|
| Seasonal Naive 168h | 12,436.73 | 19,947.32 | 5.24% |

---

# Deep Learning Models

Several TensorFlow models were evaluated.

The project intentionally compares multiple architectures instead of assuming that a more complex neural network will automatically perform better.

## Dense Neural Network

Architecture:

```text
Input
 ↓
Dense(128)
 ↓
Dropout
 ↓
Dense(64)
 ↓
Dense(24)
```

Validation results:

- MAE: **10,113.58**
- RMSE: **13,451.73**
- MAPE: **5.07%**

---

## 1D CNN

Architecture:

```text
Input
 ↓
Conv1D(64)
 ↓
MaxPooling1D
 ↓
Conv1D(32)
 ↓
GlobalAveragePooling1D
 ↓
Dense(64)
 ↓
Dropout
 ↓
Dense(24)
```

Validation results:

- MAE: **15,649.42**
- RMSE: **20,537.18**
- MAPE: **7.39%**

---

## LSTM

Architecture:

```text
Input
 ↓
LSTM(64)
 ↓
Dropout
 ↓
Dense(64)
 ↓
Dense(24)
```

Validation results:

- MAE: **10,808.22**
- RMSE: **13,987.64**
- MAPE: **5.53%**

---

## GRU

Architecture:

```text
Input
 ↓
GRU(64)
 ↓
Dropout
 ↓
Dense(64)
 ↓
Dense(24)
```

Validation results:

- MAE: **10,684.33**
- RMSE: **14,190.67**
- MAPE: **5.48%**

---

# Model Comparison

The main model comparison was performed on the same forecasting task.

| Model | MAE | RMSE | MAPE |
|---|---:|---:|---:|
| Naive | 107,900.28 | 134,801.28 | 41.13% |
| Seasonal Naive 24h | 8,105.95 | 13,459.18 | 3.84% |
| Seasonal Naive 168h | 12,436.73 | 19,947.32 | 5.24% |
| Dense NN | 10,113.58 | 13,451.73 | 5.07% |
| 1D CNN | 15,649.42 | 20,537.18 | 7.39% |
| LSTM | 10,808.22 | 13,987.64 | 5.53% |
| GRU | 10,684.33 | 14,190.67 | 5.48% |

An important finding is that the **24-hour Seasonal Naive baseline is highly competitive** and achieved lower MAE and MAPE than the neural network models in the reported validation comparison.

This is an important result because it demonstrates why strong baseline models are necessary in time-series forecasting.

---

# Input Window Experiment

Different historical input lengths were tested using the GRU architecture.

The experiments compared:

- Previous 24 hours
- Previous 168 hours
- Previous 336 hours

Results:

| Input Window | MAE | RMSE | MAPE |
|---|---:|---:|---:|
| 24 hours | 10,262.77 | 13,620.21 | 5.19% |
| 168 hours | 10,684.33 | 14,190.67 | 5.48% |
| 336 hours | 10,754.23 | 14,277.85 | 5.47% |

The 24-hour input window produced the strongest result among the tested GRU configurations.

Therefore, the final model uses a **24-hour input window**.

---

# Hyperparameter Tuning

The GRU model was tuned systematically.

The following parameters were evaluated:

- GRU units
- Dropout
- Dense layer size

## GRU Units

| GRU Units | MAE | RMSE | MAPE |
|---:|---:|---:|---:|
| 32 | 11,125.94 | 14,921.66 | 5.60% |
| 64 | 10,262.77 | 13,620.21 | 5.19% |
| 128 | 9,915.37 | 12,883.15 | 5.00% |

## Dropout

| Dropout | MAE | RMSE | MAPE |
|---:|---:|---:|---:|
| 0.0 | 10,260.24 | 13,194.28 | 5.27% |
| 0.2 | 9,915.37 | 12,883.15 | 5.00% |
| 0.3 | 10,050.86 | 12,960.06 | 5.02% |

## Dense Layer Size

| Dense Units | MAE | RMSE | MAPE |
|---:|---:|---:|---:|
| 32 | 10,027.34 | 13,036.21 | 5.04% |
| 64 | 9,915.37 | 12,883.15 | 5.00% |
| 128 | 10,373.74 | 13,306.57 | 5.32% |

The final configuration selected for deployment was:

```text
Input window: 24 hours
GRU units: 128
Dropout: 0.2
Dense units: 64
Forecast horizon: 24 hours
```

---

# Final Model

The final TensorFlow model uses the following architecture:

```text
Input: 24 × 8
      ↓
GRU(128)
      ↓
Dropout(0.2)
      ↓
Dense(64)
      ↓
Dense(24)
      ↓
Next 24 hourly demand predictions
```

Total trainable parameters:

```text
62,808
```

The model is saved as:

```text
models/final_gru_best.keras
```

The preprocessing artifacts are saved as:

```text
models/feature_scaler.joblib
models/target_scaler.joblib
```

---

# Final Test Results

The final model was evaluated on the held-out test set.

| Metric | Result |
|---|---:|
| MAE | **10,072.27** |
| RMSE | **12,887.29** |
| MAPE | **5.11%** |

The test set was not used for model selection or hyperparameter tuning.

This helps provide an unbiased estimate of final model performance.

---

# Error Analysis

The project also performs error analysis instead of evaluating the model only using aggregate metrics.

Analysis includes:

- Error by forecast horizon
- Largest individual prediction errors
- Actual vs predicted demand
- Model comparison
- Error distribution

One notable pattern was increased forecasting error around the **Christmas period**, particularly around December 23–25, 2013.

This indicates that holiday-specific demand behavior is more difficult for the current feature set and model to capture.

These observations were retained rather than being removed as outliers.

---

# Production Inference Pipeline

A reusable inference pipeline was created in:

```text
src/inference/predict.py
```

The pipeline performs:

```text
Load trained model
        ↓
Load feature scaler
        ↓
Load target scaler
        ↓
Load latest hourly demand
        ↓
Build temporal features
        ↓
Select previous 24 hours
        ↓
Scale features
        ↓
Generate 24-hour forecast
        ↓
Inverse transform predictions
        ↓
Return forecast timestamps + demand
```

The generated forecast is saved to:

```text
reports/latest_forecast.csv
```

---

# FastAPI Backend

A REST API was created using FastAPI.

Location:

```text
src/api/main.py
```

## Available Endpoints

### Health Check

```text
GET /health
```

Used to verify that the API is running.

### Forecast

```text
GET /predict
```

Returns the next 24 hours of predicted electricity demand.

Example response structure:

```json
{
  "forecast_horizon": 24,
  "input_window": 24,
  "model": "GRU-128",
  "forecast": [
    {
      "timestamp": "2015-01-01T00:00:00",
      "predicted_demand": 102801.45
    }
  ]
}
```

The API also provides interactive documentation through:

```text
/docs
```

and:

```text
/redoc
```

Model artifacts are loaded lazily and cached so that the model does not need to be loaded from disk for every request.

---

# Forecasting Dashboard

A Streamlit dashboard was created to provide a simple user interface for the forecasting system.

Location:

```text
dashboard/app.py
```

The dashboard provides:

- Forecast generation
- Peak demand
- Minimum demand
- Average demand
- Forecast line chart
- Forecast detail table
- API error handling

The dashboard communicates with the FastAPI backend.

Architecture:

```text
User
 │
 ▼
Streamlit Dashboard
 │
 │ HTTP request
 ▼
FastAPI Backend
 │
 ▼
Inference Pipeline
 │
 ▼
TensorFlow GRU Model
 │
 ▼
24-Hour Forecast
```

---

# Testing

Automated tests are implemented using `pytest`.

The project currently contains tests for:

### Inference

- Feature column generation
- Expected row count
- Missing-value validation

### API

- Health endpoint
- Prediction endpoint
- Prediction response structure

Run the complete test suite with:

```powershell
pytest -v
```

Current local test result:

```text
6 passed
```

The API tests mock the inference artifacts where appropriate so that the tests do not depend on ignored local model binaries being present in the CI environment.

---

# Continuous Integration

GitHub Actions is configured to automatically run the test suite.

Workflow:

```text
.github/workflows/tests.yml
```

The workflow:

1. Checks out the repository
2. Sets up Python 3.13
3. Installs dependencies
4. Runs pytest

The workflow runs on:

- Pushes to `main`
- Pull requests targeting `main`

This provides automated verification of the project after code changes.

---

# Project Structure

```text
electricity-demand-forecasting/
│
├── .github/
│   └── workflows/
│       └── tests.yml
│
├── configs/
│
├── dashboard/
│   └── app.py
│
├── data/
│   ├── raw/
│   │   └── LD2011_2014.txt
│   │
│   └── processed/
│       └── hourly_demand.csv
│
├── models/
│   ├── final_gru_best.keras
│   ├── feature_scaler.joblib
│   └── target_scaler.joblib
│
├── notebooks/
│   └── 01_data_exploration.ipynb
│
├── reports/
│   ├── figures/
│   ├── baseline_dense_results.csv
│   ├── gru_training_history.csv
│   ├── gru_validation_results.csv
│   ├── gru_validation_predictions.csv
│   ├── gru_horizon_errors.csv
│   ├── gru_largest_errors.csv
│   ├── gru_window_comparison.csv
│   ├── model_comparison.csv
│   └── latest_forecast.csv
│
├── src/
│   ├── api/
│   │   └── main.py
│   │
│   ├── data/
│   │   ├── preprocess.py
│   │   └── validate_raw.py
│   │
│   ├── evaluation/
│   │   └── baselines.py
│   │
│   ├── features/
│   │   ├── build_features.py
│   │   ├── create_sequences.py
│   │   └── create_supervised.py
│   │
│   ├── inference/
│   │   └── predict.py
│   │
│   ├── models/
│   │   ├── train_dense.py
│   │   ├── train_sequence_models.py
│   │   └── train_final_model.py
│   │
│   └── training/
│
├── tests/
│   ├── test_api.py
│   └── test_inference.py
│
├── .gitignore
├── pytest.ini
├── README.md
└── requirements.txt
```

---

# Running the Project Locally

## 1. Clone the repository

```powershell
git clone https://github.com/param7862989-ops/electricity-demand-forecasting.git
cd electricity-demand-forecasting
```

## 2. Create the virtual environment

```powershell
python -m venv .venv
```

## 3. Activate the environment

```powershell
.venv\Scripts\Activate.ps1
```

## 4. Install dependencies

```powershell
pip install -r requirements.txt
```

---

# Run Data Processing

The raw dataset should be placed at:

```text
data/raw/LD2011_2014.txt
```

Then run:

```powershell
python src/data/preprocess.py
```

This creates:

```text
data/processed/hourly_demand.csv
```

---

# Run Tests

```powershell
pytest -v
```

---

# Run the FastAPI Backend

From the project root:

```powershell
uvicorn src.api.main:app --reload
```

The API will be available at:

```text
http://127.0.0.1:8000
```

Interactive API documentation:

```text
http://127.0.0.1:8000/docs
```

---

# Run the Dashboard

Keep the FastAPI server running and open another terminal.

Activate the virtual environment:

```powershell
.venv\Scripts\Activate.ps1
```

Then run:

```powershell
streamlit run dashboard/app.py
```

The Streamlit dashboard will open in the browser.

---

# Technologies Used

## Programming

- Python 3.13

## Machine Learning

- TensorFlow
- Keras
- Scikit-learn
- NumPy
- Pandas

## Models

- Naive Forecast
- Seasonal Naive Forecast
- Dense Neural Network
- 1D CNN
- LSTM
- GRU

## Backend

- FastAPI
- Uvicorn
- Pydantic

## Frontend / Dashboard

- Streamlit
- Requests

## Testing

- pytest

## Development

- Jupyter Notebook
- VS Code
- Git
- GitHub
- GitHub Actions

---

# Key Engineering Practices

This project was developed with several practical ML engineering principles.

### Chronological Data Splitting

Future observations are not randomly mixed into training data.

### Training-Only Scaling

Scalers are fitted using training data and then applied to validation and test data.

### Baseline Comparison

Simple forecasting baselines are evaluated before deep learning models.

### Multiple Model Architectures

Dense, CNN, LSTM, and GRU architectures are compared rather than assuming one model is automatically optimal.

### Hyperparameter Experiments

GRU size, dropout, dense layer size, and input history length are evaluated systematically.

### Error Analysis

The project examines where and when the model makes large errors.

### Reproducibility

A fixed random seed is used during experiments.

### Production Inference

The final model is separated from the training workflow and exposed through a reusable inference pipeline.

### API Layer

FastAPI provides a programmatic interface for generating forecasts.

### Dashboard Layer

Streamlit provides a user-facing forecasting interface.

### Automated Testing

Core inference and API behavior are covered by automated tests.

### Continuous Integration

GitHub Actions automatically executes the test suite on repository changes.

---

# Limitations

The current version has several limitations.

### Limited Calendar Features

The model currently uses temporal features such as hour, day of week, month, and weekend status but does not explicitly encode public holidays.

This likely contributes to higher errors during unusual holiday periods.

### No Weather Data

Temperature, humidity, weather conditions, and other external variables are not currently included.

### Aggregate Forecasting

The system forecasts total aggregate demand rather than forecasting individual clients.

### Single Dataset

The model has been evaluated on the UCI ElectricityLoadDiagrams20112014 dataset.

Performance on other regions or electricity systems may differ.

### Strong Seasonal Baseline

The 24-hour Seasonal Naive model performs extremely well on this dataset.

Therefore, the deep learning model should not be considered universally superior simply because it is more complex.

---

# Future Improvements

Potential future extensions include:

- Public holiday features
- Temperature and weather features
- Attention mechanisms
- Transformer-based forecasting
- Probabilistic forecasting
- Prediction intervals
- Multi-step direct vs recursive forecasting comparison
- Automated hyperparameter optimization
- Individual-client forecasting
- Multi-model ensemble forecasting
- Model monitoring
- Data drift detection
- Forecast drift monitoring
- Cloud deployment
- Containerization using Docker
- Database-backed prediction history
- Authentication and authorization
- Production monitoring and logging

---

# Project Status

## Completed

The project currently includes:

- [x] Dataset acquisition and validation
- [x] Exploratory data analysis
- [x] Forecasting problem definition
- [x] Data preprocessing pipeline
- [x] Chronological train/validation/test splitting
- [x] Feature engineering
- [x] Baseline forecasting models
- [x] Dense neural network
- [x] 1D CNN
- [x] LSTM
- [x] GRU
- [x] Model comparison
- [x] Error analysis
- [x] Input-window experiments
- [x] Hyperparameter tuning
- [x] Final model training
- [x] Model serialization
- [x] Production inference pipeline
- [x] FastAPI backend
- [x] Streamlit dashboard
- [x] Automated testing
- [x] GitHub Actions CI
- [x] Project documentation

---

# Author

**Param Thakkar**

GitHub:

https://github.com/param7862989-ops

---

# License

The project code can be used and modified for educational and research purposes.

The electricity consumption dataset is provided by the UCI Machine Learning Repository under its applicable dataset license.

Dataset:

https://archive.ics.uci.edu/dataset/321/electricityloaddiagrams20112014
