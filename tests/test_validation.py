"""
Unit tests for TemporalSplitter and RollingOriginValidator (Phase 2.2).
"""

import pandas as pd
import numpy as np
import pytest

from src.data.synthetic import SyntheticDemandGenerator
from src.forecasting.baselines import SeasonalNaiveForecaster, NaiveForecaster
from src.evaluation.validation import TemporalSplitter, RollingOriginValidator


@pytest.fixture
def sample_demand_data():
    gen = SyntheticDemandGenerator(start_date="2024-01-01", end_date="2024-06-30", random_state=42)
    return gen.generate()


def test_temporal_splitter_ratios(sample_demand_data):
    splitter = TemporalSplitter()
    train_df, val_df, test_df = splitter.split_ratios(sample_demand_data, 0.70, 0.15, 0.15)

    assert len(train_df) + len(val_df) + len(test_df) == len(sample_demand_data)

    max_train_date = train_df["date"].max()
    min_val_date = val_df["date"].min()
    max_val_date = val_df["date"].max()
    min_test_date = test_df["date"].min()

    # Strict temporal ordering assertions
    assert max_train_date < min_val_date, f"Train max date ({max_train_date}) >= Val min date ({min_val_date})"
    assert max_val_date < min_test_date, f"Val max date ({max_val_date}) >= Test min date ({min_test_date})"


def test_temporal_splitter_dates(sample_demand_data):
    splitter = TemporalSplitter()
    train_df, val_df, test_df = splitter.split_by_dates(
        sample_demand_data, train_end="2024-03-31", val_end="2024-04-30"
    )

    assert train_df["date"].max() <= pd.to_datetime("2024-03-31")
    assert val_df["date"].min() > pd.to_datetime("2024-03-31")
    assert val_df["date"].max() <= pd.to_datetime("2024-04-30")
    assert test_df["date"].min() > pd.to_datetime("2024-04-30")


def test_rolling_origin_validator(sample_demand_data):
    validator = RollingOriginValidator()
    forecaster = SeasonalNaiveForecaster(seasonal_period=7)

    results_df = validator.evaluate(
        df=sample_demand_data,
        forecaster=forecaster,
        horizon=14,
        step_size=14,
        num_origins=3,
        min_train_days=60,
    )

    assert isinstance(results_df, pd.DataFrame)
    assert set(results_df.columns) == {
        "origin_id", "origin_date", "date", "product_id", "horizon_step", "actual", "forecast"
    }

    # 3 origins * 14 horizon steps * 5 products = 210 rows
    assert len(results_df) == 3 * 14 * 5
    assert results_df["actual"].isna().sum() == 0
    assert results_df["forecast"].isna().sum() == 0
    assert (results_df["forecast"] < 0).sum() == 0


def test_rolling_origin_anti_leakage(sample_demand_data):
    """
    Assert that during rolling origin evaluation, forecaster.fit() is called strictly
    with data dates <= origin_date.
    """
    class LeakageDetectorForecaster(NaiveForecaster):
        def __init__(self):
            super().__init__()
            self.max_fit_dates = []

        def fit(self, df: pd.DataFrame):
            super().fit(df)
            self.max_fit_dates.append(pd.to_datetime(df["date"]).max())
            return self

    validator = RollingOriginValidator()
    detector = LeakageDetectorForecaster()

    results_df = validator.evaluate(
        df=sample_demand_data,
        forecaster=detector,
        horizon=14,
        step_size=14,
        num_origins=3,
        min_train_days=60,
    )

    origin_dates = sorted(results_df["origin_date"].unique())
    assert len(detector.max_fit_dates) == len(origin_dates)

    for fit_max_date, orig_date in zip(detector.max_fit_dates, origin_dates):
        assert fit_max_date <= orig_date, f"Leakage detected: fit max date ({fit_max_date}) > origin date ({orig_date})"
