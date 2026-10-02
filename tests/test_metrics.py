"""
Unit tests for forecasting metrics and ForecastEvaluator (Phase 2.3).
"""

import pandas as pd
import numpy as np
import pytest

from src.data.synthetic import SyntheticDemandGenerator
from src.forecasting.baselines import SeasonalNaiveForecaster
from src.evaluation.validation import RollingOriginValidator
from src.evaluation.metrics import (
    mae,
    rmse,
    smape,
    wape,
    mase,
    ForecastEvaluator,
)


def test_individual_metrics_analytical():
    y_true = np.array([10.0, 20.0, 30.0])
    y_pred = np.array([12.0, 18.0, 33.0])

    # MAE = (2 + 2 + 3) / 3 = 2.3333
    assert np.isclose(mae(y_true, y_pred), 7.0 / 3.0)

    # RMSE = sqrt((4 + 4 + 9) / 3) = sqrt(17/3)
    assert np.isclose(rmse(y_true, y_pred), np.sqrt(17.0 / 3.0))

    # WAPE = (2 + 2 + 3) / (10 + 20 + 30) * 100% = 7 / 60 * 100% = 11.6667%
    assert np.isclose(wape(y_true, y_pred), (7.0 / 60.0) * 100.0)


def test_perfect_forecast():
    y_true = np.array([10.0, 20.0, 30.0])
    y_pred = np.array([10.0, 20.0, 30.0])

    assert mae(y_true, y_pred) == 0.0
    assert rmse(y_true, y_pred) == 0.0
    assert smape(y_true, y_pred) == 0.0
    assert wape(y_true, y_pred) == 0.0


def test_zero_division_resilience():
    y_true = np.array([0.0, 0.0])
    y_pred = np.array([0.0, 0.0])

    assert not np.isnan(smape(y_true, y_pred))
    assert not np.isnan(wape(y_true, y_pred))
    assert not np.isnan(mase(y_true, y_pred))


def test_mase_scaling():
    y_true = np.array([10.0, 20.0, 30.0, 40.0])
    y_pred_good = np.array([10.5, 20.5, 30.5, 40.5])  # Low error (MAE = 0.5)
    in_sample = np.array([10.0, 30.0, 25.0, 15.0, 10.0, 40.0, 20.0, 35.0])  # Non-zero seasonal diff (mean scale ~ 15)

    mase_val = mase(y_true, y_pred_good, in_sample_history=in_sample, seasonal_period=2)
    assert mase_val < 0.5, f"MASE should be small for good forecast (got {mase_val})"


def test_forecast_evaluator_aggregations():
    gen = SyntheticDemandGenerator(start_date="2024-01-01", end_date="2024-04-30", random_state=42)
    df = gen.generate()

    validator = RollingOriginValidator()
    forecaster = SeasonalNaiveForecaster(seasonal_period=7)

    eval_df = validator.evaluate(
        df=df,
        forecaster=forecaster,
        horizon=14,
        step_size=14,
        num_origins=2,
        min_train_days=60,
    )

    evaluator = ForecastEvaluator()

    # 1. Overall metrics
    overall = evaluator.evaluate_overall(eval_df, in_sample_df=df)
    assert set(overall.keys()) == {"MAE", "RMSE", "sMAPE", "WAPE", "MASE"}
    assert overall["WAPE"] > 0.0

    # 2. Per-product metrics
    prod_metrics = evaluator.evaluate_by_product(eval_df, in_sample_df=df)
    assert len(prod_metrics) == 5
    assert set(prod_metrics.columns) == {"product_id", "WAPE", "MAE", "RMSE", "sMAPE", "MASE"}

    # 3. Per-horizon metrics
    horizon_metrics = evaluator.evaluate_by_horizon(eval_df)
    assert len(horizon_metrics) == 14
    assert set(horizon_metrics.columns) == {"horizon_step", "WAPE", "MAE", "RMSE", "sMAPE", "MASE"}
