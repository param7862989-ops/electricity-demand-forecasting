# Electricity Demand Forecasting

An end-to-end time-series forecasting project using TensorFlow to predict electricity demand for the next 24 hours from historical electricity consumption.

## Project Objective

The objective of this project is to develop and evaluate machine-learning and deep-learning models for short-term electricity demand forecasting.

Given the previous 7 days of hourly electricity demand, the system predicts electricity demand for the following 24 hours.

## Forecasting Problem

This project is formulated as a multi-step time-series forecasting problem.

**Input:**

* Previous 168 hourly observations (7 days × 24 hours)

**Output:**

* Next 24 hourly electricity-demand values

Conceptually:

```text
Previous 7 days
      ↓
168 hourly observations
      ↓
Preprocessing & feature engineering
      ↓
Forecasting model
      ↓
Next 24 hours
```

## Dataset

The project uses the **UCI ElectricityLoadDiagrams20112014** dataset.

The original dataset contains electricity consumption measurements from 370 clients in Portugal at 15-minute intervals between 2011 and 2014.

For the initial version of this project, the 15-minute observations will be aggregated into hourly demand.

The raw dataset will not be committed to this repository. Data will be downloaded and processed locally according to the project's data pipeline.

Dataset source:

https://archive.ics.uci.edu/dataset/321/electricityloaddiagrams20112014

License: CC BY 4.0

## Modeling Approach

Models will be developed incrementally and compared against increasingly strong baselines:

1. Seasonal Naive baseline
2. Dense neural network
3. 1D Convolutional Neural Network
4. LSTM
5. GRU

The final model will be selected based on experimental results rather than model type alone.

## Evaluation

Models will be evaluated using:

* Mean Absolute Error (MAE)
* Root Mean Squared Error (RMSE)
* Mean Absolute Percentage Error (MAPE)

In addition to numerical metrics, predicted and actual demand will be visualized to evaluate forecasting behavior and identify systematic errors.

## Project Structure

```text
electricity-demand-forecasting/
│
├── data/
│   ├── raw/
│   ├── processed/
│   └── external/
│
├── notebooks/
│
├── src/
│   ├── data/
│   ├── features/
│   ├── models/
│   ├── training/
│   └── evaluation/
│
├── models/
│
├── reports/
│   └── figures/
│
├── tests/
│
├── configs/
│
├── .gitignore
├── README.md
└── requirements.txt
```

## Development Roadmap

* [ ] Dataset acquisition
* [ ] Data validation
* [ ] Exploratory data analysis
* [ ] Data preprocessing
* [ ] Time-series feature engineering
* [ ] Seasonal Naive baseline
* [ ] Dense neural network
* [ ] 1D CNN
* [ ] LSTM
* [ ] GRU
* [ ] Model comparison
* [ ] Error analysis
* [ ] Final model selection
* [ ] Model serialization
* [ ] Prediction API
* [ ] Testing
* [ ] CI/CD
* [ ] Documentation and final results

## Status

🚧 **Currently in development**

This repository is being developed incrementally with experiments, evaluation results, and implementation changes tracked through Git.
