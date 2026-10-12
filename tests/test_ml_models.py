"""
Unit tests for Machine Learning Forecasting Models (Phase 3.2).
"""

import pandas as pd
import numpy as np
import pytest

from src.data.synthetic import SyntheticDemandGenerator
from src.forecasting.ml_models import (
    RandomForestForecaster,
    XGBoostForecaster,
    LightGBMForecaster,
)
from src.evaluation.validation import RollingOriginValidator


@pytest.fixture
def sample_demand_data():
    gen = SyntheticDemandGenerator(start_date="2024-01-01", end_date="2024-06-30", random_state=42)
    return gen.generate()


def test_random_forest_forecaster(sample_demand_data):
    model = RandomForestForecaster(n_estimators=10, max_depth=4)
    model.fit(sample_demand_data)
    fc_df = model.predict(horizon=14)

    assert len(fc_df) == 5 * 14
    assert set(fc_df.columns) == {"date", "product_id", "forecast"}
    assert fc_df["forecast"].isna().sum() == 0
    assert (fc_df["forecast"] < 0).sum() == 0


def test_xgboost_forecaster(sample_demand_data):
    model = XGBoostForecaster(n_estimators=10, max_depth=3)
    model.fit(sample_demand_data)
    fc_df = model.predict(horizon=14)

    assert len(fc_df) == 5 * 14
    assert fc_df["forecast"].isna().sum() == 0
    assert (fc_df["forecast"] < 0).sum() == 0


def test_lightgbm_forecaster(sample_demand_data):
    model = LightGBMForecaster(n_estimators=10, max_depth=3)
    model.fit(sample_demand_data)
    fc_df = model.predict(horizon=14)

    assert len(fc_df) == 5 * 14
    assert fc_df["forecast"].isna().sum() == 0
    assert (fc_df["forecast"] < 0).sum() == 0


def test_feature_importances(sample_demand_data):
    model = XGBoostForecaster(n_estimators=10, max_depth=3)
    model.fit(sample_demand_data)

    imp_df = model.get_feature_importances()
    assert isinstance(imp_df, pd.DataFrame)
    assert set(imp_df.columns) == {"feature", "importance"}
    assert len(imp_df) > 0
    assert imp_df["importance"].sum() > 0.0


def test_ml_models_rolling_validation_integration(sample_demand_data):
    validator = RollingOriginValidator()
    model = LightGBMForecaster(n_estimators=10, max_depth=3)

    results_df = validator.evaluate(
        df=sample_demand_data,
        forecaster=model,
        horizon=14,
        step_size=14,
        num_origins=2,
        min_train_days=60,
    )

    assert isinstance(results_df, pd.DataFrame)
    assert len(results_df) == 2 * 14 * 5
    assert results_df["actual"].isna().sum() == 0
    assert results_df["forecast"].isna().sum() == 0
