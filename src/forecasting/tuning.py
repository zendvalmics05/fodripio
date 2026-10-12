"""
Hyperparameter Tuning Module for Fodripio.

Executes hyperparameter optimization exclusively using rolling-origin temporal
cross-validation (strictly no random splits or data shuffling). Logs all experiment
trials to experiments/ and persists the best model configuration to experiments/best_ml_params.yaml.
"""

import itertools
import logging
from pathlib import Path
import time
from typing import Any, Dict, List, Optional, Tuple, Type, Union
import yaml

import numpy as np
import pandas as pd

from src.evaluation.validation import RollingOriginValidator
from src.evaluation.metrics import ForecastEvaluator
from src.forecasting.baselines import BaseForecaster

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


class TimeSeriesTuner:
    """
    Hyperparameter tuning engine based on temporal rolling-origin cross-validation.
    """

    def __init__(
        self,
        output_dir: Union[str, Path] = "experiments",
        target_metric: str = "WAPE",
    ):
        self.output_dir = Path(output_dir)
        self.target_metric = target_metric
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.validator = RollingOriginValidator()
        self.evaluator = ForecastEvaluator()

    def tune(
        self,
        df: pd.DataFrame,
        forecaster_cls: Type[BaseForecaster],
        param_grid: Dict[str, List[Any]],
        n_splits: int = 3,
        horizon: int = 14,
        step_size: int = 14,
        min_train_days: int = 60,
    ) -> Tuple[Dict[str, Any], float, pd.DataFrame, BaseForecaster]:
        """
        Executes grid search over rolling-origin temporal folds.

        Args:
            df (pd.DataFrame): Panel demand history DataFrame.
            forecaster_cls: Forecaster class to instantiate and tune.
            param_grid: Dictionary of parameter names and candidate lists.
            n_splits (int): Number of rolling temporal folds.
            horizon (int): Forecast horizon H.
            step_size (int): Days between origin steps.
            min_train_days (int): Minimum historical days before first origin.

        Returns:
            Tuple[Dict[str, Any], float, pd.DataFrame, BaseForecaster]:
                (best_params, best_score, trials_df, best_forecaster)
        """
        keys = list(param_grid.keys())
        combinations = list(itertools.product(*param_grid.values()))
        total_trials = len(combinations)

        logger.info(f"Starting tuning for {forecaster_cls.__name__}: {total_trials} candidate configurations, {n_splits} rolling folds.")

        trials = []

        for idx, values in enumerate(combinations, 1):
            params = dict(zip(keys, values))
            logger.info(f"Trial {idx}/{total_trials} evaluating params: {params}")

            start_t = time.time()
            model = forecaster_cls(**params)

            try:
                eval_df = self.validator.evaluate(
                    df=df,
                    forecaster=model,
                    horizon=horizon,
                    step_size=step_size,
                    num_origins=n_splits,
                    min_train_days=min_train_days,
                )

                metrics = self.evaluator.evaluate_overall(eval_df, in_sample_df=df)
                elapsed_sec = round(time.time() - start_t, 2)

                trial_record = {
                    "trial_id": idx,
                    "model_class": forecaster_cls.__name__,
                    "duration_sec": elapsed_sec,
                    **params,
                    **metrics,
                }
                trials.append(trial_record)
                logger.info(f"Trial {idx} result: {self.target_metric} = {metrics.get(self.target_metric):.2f}% ({elapsed_sec}s)")

            except Exception as e:
                logger.error(f"Trial {idx} failed with error: {e}")
                trial_record = {
                    "trial_id": idx,
                    "model_class": forecaster_cls.__name__,
                    "duration_sec": round(time.time() - start_t, 2),
                    **params,
                    "WAPE": 999.0,
                    "MAE": 999.0,
                    "RMSE": 999.0,
                    "sMAPE": 999.0,
                    "MASE": 999.0,
                }
                trials.append(trial_record)

        trials_df = pd.DataFrame(trials)
        trials_df = trials_df.sort_values(by=self.target_metric, ascending=True).reset_index(drop=True)

        best_trial = trials_df.iloc[0]
        best_params = {k: best_trial[k] for k in keys}
        best_score = float(best_trial[self.target_metric])

        logger.info(f"Tuning complete. Best {self.target_metric}: {best_score:.2f}% with params: {best_params}")

        # 1. Save trials log to CSV
        log_file = self.output_dir / "tuning_experiments.csv"
        if log_file.exists():
            trials_df.to_csv(log_file, mode="a", header=False, index=False)
        else:
            trials_df.to_csv(log_file, index=False)
        logger.info(f"Saved trials log to {log_file}")

        # Clean best_params into native python types (int, float, str)
        clean_best_params = {}
        for k, v in best_params.items():
            if isinstance(v, (np.integer, int)):
                clean_best_params[k] = int(v)
            elif isinstance(v, (np.floating, float)):
                clean_best_params[k] = float(v)
            else:
                clean_best_params[k] = v

        # 2. Save best configuration to YAML
        best_config = {
            "model_class": forecaster_cls.__name__,
            "target_metric": self.target_metric,
            "best_score": best_score,
            "best_params": clean_best_params,
            "horizon": horizon,
            "n_splits": n_splits,
        }
        yaml_path = self.output_dir / "best_ml_params.yaml"
        with open(yaml_path, "w") as f:
            yaml.dump(best_config, f, default_flow_style=False)
        logger.info(f"Saved best configuration to {yaml_path}")

        best_forecaster = forecaster_cls(**clean_best_params)
        return clean_best_params, best_score, trials_df, best_forecaster
