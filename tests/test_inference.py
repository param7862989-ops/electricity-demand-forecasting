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