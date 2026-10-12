"""
Unit tests for TimeSeriesTuner (Phase 3.3).
"""

from pathlib import Path
import pandas as pd
import pytest

from src.data.synthetic import SyntheticDemandGenerator
from src.forecasting.ml_models import RandomForestForecaster
from src.forecasting.tuning import TimeSeriesTuner


@pytest.fixture
def sample_demand_data():
    gen = SyntheticDemandGenerator(start_date="2024-01-01", end_date="2024-05-31", random_state=42)
    return gen.generate()


def test_time_series_tuner_execution(sample_demand_data, tmp_path):
    tuner = TimeSeriesTuner(output_dir=tmp_path, target_metric="WAPE")

    param_grid = {
        "n_estimators": [5, 10],
        "max_depth": [3],
    }

    best_params, best_score, trials_df, best_model = tuner.tune(
        df=sample_demand_data,
        forecaster_cls=RandomForestForecaster,
        param_grid=param_grid,
        n_splits=2,
        horizon=14,
        step_size=14,
        min_train_days=60,
    )

    assert isinstance(best_params, dict)
    assert "n_estimators" in best_params
    assert best_score > 0.0
    assert isinstance(trials_df, pd.DataFrame)
    assert len(trials_df) == 2
    assert "WAPE" in trials_df.columns
    assert "duration_sec" in trials_df.columns

    # Verify best model is runnable
    assert isinstance(best_model, RandomForestForecaster)
    best_model.fit(sample_demand_data)
    fc_df = best_model.predict(horizon=14)
    assert len(fc_df) == 5 * 14

    # Verify file persistence in output_dir
    log_csv = tmp_path / "tuning_experiments.csv"
    best_yaml = tmp_path / "best_ml_params.yaml"
    assert log_csv.exists()
    assert best_yaml.exists()
