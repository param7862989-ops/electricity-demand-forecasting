import pandas as pd

from src.inference.predict import build_features


def test_build_features_columns():
    timestamps = pd.date_range(
        "2014-12-31 00:00:00",
        periods=24,
        freq="h"
    )

    df = pd.DataFrame({
        "timestamp": timestamps,
        "total_demand": range(24)
    })

    features = build_features(df)

    expected_columns = [
        "total_demand",
        "hour_sin",
        "hour_cos",
        "day_of_week_sin",
        "day_of_week_cos",
        "month_sin",
        "month_cos",
        "is_weekend"
    ]

    assert list(features.columns) == expected_columns


def test_build_features_preserves_row_count():
    timestamps = pd.date_range(
        "2014-12-31 00:00:00",
        periods=24,
        freq="h"
    )

    df = pd.DataFrame({
        "timestamp": timestamps,
        "total_demand": range(24)
    })

    features = build_features(df)

    assert len(features) == 24


def test_build_features_contains_no_missing_values():
    timestamps = pd.date_range(
        "2014-12-31 00:00:00",
        periods=24,
        freq="h"
    )

    df = pd.DataFrame({
        "timestamp": timestamps,
        "total_demand": range(24)
    })

    features = build_features(df)

    assert not features.isnull().any().any()

import numpy as np
import pytest

from src.inference.predict import (
    HORIZON,
    INPUT_WINDOW,
    forecast_from_history,
    validate_history,
)


class IdentityScaler:
    def transform(self, values):
        return values

    def inverse_transform(self, values):
        return values


class ConstantModel:
    def predict(self, model_input, verbose=0):
        assert model_input.shape == (1, INPUT_WINDOW, 8)
        return np.full((1, HORIZON), 123.0)


def make_history(periods=48, freq="h"):
    return pd.DataFrame({
        "timestamp": pd.date_range("2014-12-30 00:00:00", periods=periods, freq=freq),
        "total_demand": np.linspace(100000, 200000, periods),
    })


def test_forecast_from_history_returns_next_horizon():
    history = make_history()

    forecast = forecast_from_history(
        history,
        ConstantModel(),
        IdentityScaler(),
        IdentityScaler()
    )

    assert len(forecast) == HORIZON
    assert forecast["timestamp"].iloc[0] == (
        history["timestamp"].iloc[-1] + pd.Timedelta(hours=1)
    )
    assert (forecast["predicted_demand"] == 123.0).all()


def test_validate_history_rejects_short_history():
    with pytest.raises(ValueError, match="At least"):
        validate_history(make_history(periods=INPUT_WINDOW - 1))


def test_validate_history_rejects_gaps():
    with pytest.raises(ValueError, match="consecutive"):
        validate_history(make_history(freq="2h"))


def test_validate_history_rejects_negative_demand():
    history = make_history()
    history.loc[5, "total_demand"] = -1

    with pytest.raises(ValueError, match="Negative"):
        validate_history(history)
