"""
Unit tests for baseline forecasting models (Phase 2.1).
"""

import pandas as pd
import numpy as np
import pytest

from src.data.synthetic import SyntheticDemandGenerator
from src.forecasting.baselines import (
    NaiveForecaster,
    SeasonalNaiveForecaster,
    MovingAverageForecaster,
    ExponentialSmoothingForecaster,
    SARIMAForecaster,
)


@pytest.fixture
def sample_demand_data():
    gen = SyntheticDemandGenerator(start_date="2024-01-01", end_date="2024-03-31", random_state=42)
    return gen.generate()


def test_naive_forecaster(sample_demand_data):
    model = NaiveForecaster()
    model.fit(sample_demand_data)
    fc_df = model.predict(horizon=14)

    assert len(fc_df) == 5 * 14
    assert set(fc_df.columns) == {"date", "product_id", "forecast"}

    # Assert forecast equals last historical value per product
    for p_id in ["P1", "P2", "P3", "P4", "P5"]:
        last_val = sample_demand_data[sample_demand_data["product_id"] == p_id]["demand"].iloc[-1]
        p_fc = fc_df[fc_df["product_id"] == p_id]["forecast"].values
        np.testing.assert_allclose(p_fc, last_val)


def test_seasonal_naive_forecaster(sample_demand_data):
    model = SeasonalNaiveForecaster(seasonal_period=7)
    model.fit(sample_demand_data)
    fc_df = model.predict(horizon=14)

    assert len(fc_df) == 5 * 14
    # Assert repeating 7-day pattern
    p1_fc = fc_df[fc_df["product_id"] == "P1"]["forecast"].values
    np.testing.assert_allclose(p1_fc[:7], p1_fc[7:14])


def test_moving_average_forecaster(sample_demand_data):
    window = 7
    model = MovingAverageForecaster(window=window)
    model.fit(sample_demand_data)
    fc_df = model.predict(horizon=14)

    assert len(fc_df) == 5 * 14
    p1_hist = sample_demand_data[sample_demand_data["product_id"] == "P1"]["demand"].tail(window).values
    expected_ma = float(np.mean(p1_hist))

    p1_fc = fc_df[fc_df["product_id"] == "P1"]["forecast"].values
    np.testing.assert_allclose(p1_fc, expected_ma)


def test_exponential_smoothing_forecaster(sample_demand_data):
    model = ExponentialSmoothingForecaster(seasonal_periods=7)
    model.fit(sample_demand_data)
    fc_df = model.predict(horizon=14)

    assert len(fc_df) == 5 * 14
    assert fc_df["forecast"].isna().sum() == 0
    assert (fc_df["forecast"] < 0).sum() == 0


def test_sarima_forecaster(sample_demand_data):
    model = SARIMAForecaster(order=(1, 1, 1), seasonal_order=(1, 1, 1, 7))
    model.fit(sample_demand_data)
    fc_df = model.predict(horizon=14)

    assert len(fc_df) == 5 * 14
    assert fc_df["forecast"].isna().sum() == 0
    assert (fc_df["forecast"] < 0).sum() == 0


def test_all_baselines_unified_interface(sample_demand_data):
    models = [
        NaiveForecaster(),
        SeasonalNaiveForecaster(seasonal_period=7),
        MovingAverageForecaster(window=7),
        MovingAverageForecaster(window=14),
        ExponentialSmoothingForecaster(seasonal_periods=7),
        SARIMAForecaster(order=(1, 1, 1), seasonal_order=(1, 1, 1, 7)),
    ]

    for model in models:
        model.fit(sample_demand_data)
        fc_df = model.predict(horizon=14)

        assert isinstance(fc_df, pd.DataFrame)
        assert len(fc_df) == 70  # 5 products * 14 days
        assert fc_df["forecast"].isna().sum() == 0
        assert (fc_df["forecast"] < 0).sum() == 0
