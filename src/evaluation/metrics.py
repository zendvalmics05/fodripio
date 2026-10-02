"""
Forecasting Evaluation Metrics for Fodripio.

Provides statistical and operational forecasting error metrics:
- MAE (Mean Absolute Error)
- RMSE (Root Mean Squared Error)
- sMAPE (Symmetric Mean Absolute Percentage Error)
- WAPE (Weighted Absolute Percentage Error / Volume-weighted MAPE)
- MASE (Mean Absolute Scaled Error using in-sample seasonal naive baseline)
- ForecastEvaluator: Aggregator class for overall, per-product, and per-horizon metrics.
"""

from typing import Dict, List, Optional, Union
import numpy as np
import pandas as pd


def mae(y_true: Union[np.ndarray, pd.Series], y_pred: Union[np.ndarray, pd.Series]) -> float:
    """
    Computes Mean Absolute Error (MAE).
    """
    y_true_arr = np.asarray(y_true, dtype=float)
    y_pred_arr = np.asarray(y_pred, dtype=float)
    return float(np.mean(np.abs(y_true_arr - y_pred_arr)))


def rmse(y_true: Union[np.ndarray, pd.Series], y_pred: Union[np.ndarray, pd.Series]) -> float:
    """
    Computes Root Mean Squared Error (RMSE).
    """
    y_true_arr = np.asarray(y_true, dtype=float)
    y_pred_arr = np.asarray(y_pred, dtype=float)
    return float(np.sqrt(np.mean((y_true_arr - y_pred_arr) ** 2)))


def smape(y_true: Union[np.ndarray, pd.Series], y_pred: Union[np.ndarray, pd.Series], eps: float = 1e-8) -> float:
    """
    Computes Symmetric Mean Absolute Percentage Error (sMAPE) in percent (0-100%).
    """
    y_true_arr = np.asarray(y_true, dtype=float)
    y_pred_arr = np.asarray(y_pred, dtype=float)

    denominator = (np.abs(y_true_arr) + np.abs(y_pred_arr)) / 2.0 + eps
    numerator = np.abs(y_true_arr - y_pred_arr)

    return float(np.mean(numerator / denominator) * 100.0)


def wape(y_true: Union[np.ndarray, pd.Series], y_pred: Union[np.ndarray, pd.Series], eps: float = 1e-8) -> float:
    """
    Computes Weighted Absolute Percentage Error (WAPE / volume-weighted MAPE) in percent.
    WAPE = (sum |y_true - y_pred|) / (sum y_true + eps) * 100%
    """
    y_true_arr = np.asarray(y_true, dtype=float)
    y_pred_arr = np.asarray(y_pred, dtype=float)

    sum_abs_err = np.sum(np.abs(y_true_arr - y_pred_arr))
    sum_actual = np.sum(y_true_arr) + eps

    return float((sum_abs_err / sum_actual) * 100.0)


def mase(
    y_true: Union[np.ndarray, pd.Series],
    y_pred: Union[np.ndarray, pd.Series],
    in_sample_history: Optional[Union[np.ndarray, pd.Series]] = None,
    seasonal_period: int = 7,
    eps: float = 1e-8,
) -> float:
    """
    Computes Mean Absolute Scaled Error (MASE).
    If in_sample_history is provided, scales by in-sample seasonal naive MAE:
        scale = mean(|y_t - y_{t-m}|) for t = m+1..N
    If in_sample_history is None, uses in-sample 1-step non-seasonal naive scaling on y_true.
    """
    y_true_arr = np.asarray(y_true, dtype=float)
    y_pred_arr = np.asarray(y_pred, dtype=float)
    forecast_mae = mae(y_true_arr, y_pred_arr)

    if in_sample_history is not None and len(in_sample_history) > seasonal_period:
        history_arr = np.asarray(in_sample_history, dtype=float)
        m = seasonal_period
        scale = np.mean(np.abs(history_arr[m:] - history_arr[:-m]))
    elif len(y_true_arr) > 1:
        scale = np.mean(np.abs(y_true_arr[1:] - y_true_arr[:-1]))
    else:
        scale = 1.0

    if scale < eps:
        scale = eps

    return float(forecast_mae / scale)


class ForecastEvaluator:
    """
    Evaluator class for aggregating metrics across rolling origins, products, and horizon steps.
    """

    def __init__(
        self,
        actual_col: str = "actual",
        forecast_col: str = "forecast",
        product_col: str = "product_id",
        horizon_col: str = "horizon_step",
    ):
        self.actual_col = actual_col
        self.forecast_col = forecast_col
        self.product_col = product_col
        self.horizon_col = horizon_col

    def compute_metrics(
        self,
        y_true: Union[np.ndarray, pd.Series],
        y_pred: Union[np.ndarray, pd.Series],
        in_sample_history: Optional[Union[np.ndarray, pd.Series]] = None,
        seasonal_period: int = 7,
    ) -> Dict[str, float]:
        """
        Computes dictionary of all core forecasting metrics.
        """
        return {
            "MAE": round(mae(y_true, y_pred), 4),
            "RMSE": round(rmse(y_true, y_pred), 4),
            "sMAPE": round(smape(y_true, y_pred), 2),
            "WAPE": round(wape(y_true, y_pred), 2),
            "MASE": round(mase(y_true, y_pred, in_sample_history, seasonal_period), 4),
        }

    def evaluate_overall(
        self,
        eval_df: pd.DataFrame,
        in_sample_df: Optional[pd.DataFrame] = None,
        seasonal_period: int = 7,
    ) -> Dict[str, float]:
        """
        Computes overall metrics across all records in eval_df.
        """
        y_true = eval_df[self.actual_col]
        y_pred = eval_df[self.forecast_col]

        in_sample = in_sample_df["demand"] if in_sample_df is not None and "demand" in in_sample_df.columns else None
        return self.compute_metrics(y_true, y_pred, in_sample, seasonal_period)

    def evaluate_by_product(
        self,
        eval_df: pd.DataFrame,
        in_sample_df: Optional[pd.DataFrame] = None,
        seasonal_period: int = 7,
    ) -> pd.DataFrame:
        """
        Computes metric summary breakdown per product_id.
        """
        records = []
        products = sorted(eval_df[self.product_col].unique())

        for p_id in products:
            sub_df = eval_df[eval_df[self.product_col] == p_id]
            y_true = sub_df[self.actual_col]
            y_pred = sub_df[self.forecast_col]

            in_sample = None
            if in_sample_df is not None and self.product_col in in_sample_df.columns:
                p_in_sample = in_sample_df[in_sample_df[self.product_col] == p_id]
                if "demand" in p_in_sample.columns:
                    in_sample = p_in_sample["demand"].values

            m_dict = self.compute_metrics(y_true, y_pred, in_sample, seasonal_period)
            m_dict[self.product_col] = p_id
            records.append(m_dict)

        res_df = pd.DataFrame(records)
        cols_order = [self.product_col, "WAPE", "MAE", "RMSE", "sMAPE", "MASE"]
        return res_df[cols_order]

    def evaluate_by_horizon(
        self,
        eval_df: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Computes metric summary breakdown per horizon_step (1..H).
        """
        records = []
        steps = sorted(eval_df[self.horizon_col].unique())

        for step in steps:
            sub_df = eval_df[eval_df[self.horizon_col] == step]
            y_true = sub_df[self.actual_col]
            y_pred = sub_df[self.forecast_col]

            m_dict = self.compute_metrics(y_true, y_pred)
            m_dict[self.horizon_col] = step
            records.append(m_dict)

        res_df = pd.DataFrame(records)
        cols_order = [self.horizon_col, "WAPE", "MAE", "RMSE", "sMAPE", "MASE"]
        return res_df[cols_order]
