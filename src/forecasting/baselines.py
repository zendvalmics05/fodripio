"""
Baseline Forecasting Models for Fodripio.

Implements classical statistical and baseline forecasting models:
1. NaiveForecaster (Last value)
2. SeasonalNaiveForecaster (Lag-7 persistence)
3. MovingAverageForecaster (7, 14, 28-day window average)
4. ExponentialSmoothingForecaster (Holt-Winters / ETS)
5. SARIMAForecaster (Seasonal ARIMA)

All forecasters conform to a unified interface (fit, predict).
"""

from abc import ABC, abstractmethod
import warnings
from typing import Dict, List, Optional, Union, Tuple
import numpy as np
import pandas as pd
from statsmodels.tsa.api import ExponentialSmoothing
from statsmodels.tsa.statespace.sarimax import SARIMAX


class BaseForecaster(ABC):
    """
    Abstract Base Class for all forecasting models.
    """

    def __init__(
        self,
        target_col: str = "demand",
        date_col: str = "date",
        group_col: str = "product_id",
    ):
        self.target_col = target_col
        self.date_col = date_col
        self.group_col = group_col
        self.history_df: Optional[pd.DataFrame] = None
        self.products: List[str] = []

    @abstractmethod
    def fit(self, df: pd.DataFrame) -> "BaseForecaster":
        """
        Fits the model using historical demand data.
        """
        pass

    @abstractmethod
    def predict(self, horizon: int = 14) -> pd.DataFrame:
        """
        Generates forecasts for the specified horizon H.

        Returns:
            pd.DataFrame with columns: date, product_id, forecast
        """
        pass

    def _prepare_history(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Prepares and validates historical demand dataframe.
        """
        history = df.copy()
        history[self.date_col] = pd.to_datetime(history[self.date_col])
        history = history.sort_values(by=[self.group_col, self.date_col]).reset_index(drop=True)
        self.history_df = history
        self.products = sorted(history[self.group_col].unique())
        return history

    def _generate_future_dates(self, last_date: pd.Timestamp, horizon: int) -> pd.DatetimeIndex:
        """
        Generates future daily dates for the forecast horizon.
        """
        return pd.date_range(start=last_date + pd.Timedelta(days=1), periods=horizon, freq="D")


class NaiveForecaster(BaseForecaster):
    """
    Last-value persistence forecaster: y_{t+h} = y_t.
    """

    def fit(self, df: pd.DataFrame) -> "NaiveForecaster":
        self._prepare_history(df)
        return self

    def predict(self, horizon: int = 14) -> pd.DataFrame:
        if self.history_df is None:
            raise ValueError("Model must be fitted before predicting.")

        records = []
        for p_id in self.products:
            p_df = self.history_df[self.history_df[self.group_col] == p_id]
            last_val = p_df[self.target_col].iloc[-1]
            last_date = p_df[self.date_col].iloc[-1]
            future_dates = self._generate_future_dates(last_date, horizon)

            for d in future_dates:
                records.append({
                    self.date_col: d,
                    self.group_col: p_id,
                    "forecast": max(0.0, float(last_val)),
                })

        return pd.DataFrame(records)


class SeasonalNaiveForecaster(BaseForecaster):
    """
    Seasonal naïve forecaster: y_{t+h} = y_{t+h-m}.
    Default seasonal period m = 7 (weekly periodicity).
    """

    def __init__(self, seasonal_period: int = 7, **kwargs):
        super().__init__(**kwargs)
        self.seasonal_period = seasonal_period

    def fit(self, df: pd.DataFrame) -> "SeasonalNaiveForecaster":
        self._prepare_history(df)
        return self

    def predict(self, horizon: int = 14) -> pd.DataFrame:
        if self.history_df is None:
            raise ValueError("Model must be fitted before predicting.")

        records = []
        for p_id in self.products:
            p_df = self.history_df[self.history_df[self.group_col] == p_id].copy()
            last_date = p_df[self.date_col].iloc[-1]
            future_dates = self._generate_future_dates(last_date, horizon)

            history_vals = p_df[self.target_col].values
            n_hist = len(history_vals)

            for step in range(1, horizon + 1):
                # Calculate lag index mapping back into historical series
                lag_idx = (step - 1) % self.seasonal_period
                val = history_vals[n_hist - self.seasonal_period + lag_idx]
                records.append({
                    self.date_col: future_dates[step - 1],
                    self.group_col: p_id,
                    "forecast": max(0.0, float(val)),
                })

        return pd.DataFrame(records)


class MovingAverageForecaster(BaseForecaster):
    """
    Moving average forecaster over window W: y_{t+h} = mean(y_{t-W+1:t}).
    """

    def __init__(self, window: int = 7, **kwargs):
        super().__init__(**kwargs)
        self.window = window

    def fit(self, df: pd.DataFrame) -> "MovingAverageForecaster":
        self._prepare_history(df)
        return self

    def predict(self, horizon: int = 14) -> pd.DataFrame:
        if self.history_df is None:
            raise ValueError("Model must be fitted before predicting.")

        records = []
        for p_id in self.products:
            p_df = self.history_df[self.history_df[self.group_col] == p_id]
            last_date = p_df[self.date_col].iloc[-1]
            future_dates = self._generate_future_dates(last_date, horizon)

            recent_vals = p_df[self.target_col].tail(self.window).values
            ma_val = float(np.mean(recent_vals))

            for d in future_dates:
                records.append({
                    self.date_col: d,
                    self.group_col: p_id,
                    "forecast": max(0.0, ma_val),
                })

        return pd.DataFrame(records)


class ExponentialSmoothingForecaster(BaseForecaster):
    """
    Holt-Winters Exponential Smoothing (ETS) forecaster via statsmodels.
    Supports trend and seasonal components (default seasonal period = 7).
    """

    def __init__(self, trend: Optional[str] = "add", seasonal: Optional[str] = "add", seasonal_periods: int = 7, **kwargs):
        super().__init__(**kwargs)
        self.trend = trend
        self.seasonal = seasonal
        self.seasonal_periods = seasonal_periods
        self.fitted_models: Dict[str, ExponentialSmoothing] = {}

    def fit(self, df: pd.DataFrame) -> "ExponentialSmoothingForecaster":
        self._prepare_history(df)

        for p_id in self.products:
            p_series = self.history_df[self.history_df[self.group_col] == p_id][self.target_col].values

            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    model = ExponentialSmoothing(
                        p_series,
                        trend=self.trend,
                        seasonal=self.seasonal,
                        seasonal_periods=self.seasonal_periods,
                        initialization_method="estimated",
                    )
                    fitted = model.fit()
                    self.fitted_models[p_id] = fitted
            except Exception as e:
                # Fallback to Simple Exponential Smoothing if Holt-Winters fails
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    model = ExponentialSmoothing(p_series, trend=None, seasonal=None)
                    fitted = model.fit()
                    self.fitted_models[p_id] = fitted

        return self

    def predict(self, horizon: int = 14) -> pd.DataFrame:
        if self.history_df is None or not self.fitted_models:
            raise ValueError("Model must be fitted before predicting.")

        records = []
        for p_id in self.products:
            p_df = self.history_df[self.history_df[self.group_col] == p_id]
            last_date = p_df[self.date_col].iloc[-1]
            future_dates = self._generate_future_dates(last_date, horizon)

            fitted_model = self.fitted_models[p_id]
            raw_forecasts = fitted_model.forecast(horizon)

            for idx, d in enumerate(future_dates):
                fc = float(raw_forecasts[idx]) if idx < len(raw_forecasts) else float(raw_forecasts[-1])
                records.append({
                    self.date_col: d,
                    self.group_col: p_id,
                    "forecast": max(0.0, fc),
                })

        return pd.DataFrame(records)


class SARIMAForecaster(BaseForecaster):
    """
    Seasonal ARIMA (SARIMAX) forecaster via statsmodels.
    Default order (1,1,1) x (1,1,1)_7.
    """

    def __init__(
        self,
        order: Tuple[int, int, int] = (1, 1, 1),
        seasonal_order: Tuple[int, int, int, int] = (1, 1, 1, 7),
        **kwargs
    ):
        super().__init__(**kwargs)
        self.order = order
        self.seasonal_order = seasonal_order
        self.fitted_models: Dict[str, SARIMAX] = {}
        self.fallback_forecasters: Dict[str, SeasonalNaiveForecaster] = {}

    def fit(self, df: pd.DataFrame) -> "SARIMAForecaster":
        self._prepare_history(df)

        for p_id in self.products:
            p_series = self.history_df[self.history_df[self.group_col] == p_id][self.target_col].values

            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    model = SARIMAX(
                        p_series,
                        order=self.order,
                        seasonal_order=self.seasonal_order,
                        enforce_stationarity=False,
                        enforce_invertibility=False,
                    )
                    fitted = model.fit(disp=False, maxiter=50)
                    self.fitted_models[p_id] = fitted
            except Exception:
                # Fallback to Seasonal Naïve if SARIMAX optimization fails
                fallback = SeasonalNaiveForecaster(seasonal_period=self.seasonal_order[3])
                p_sub_df = self.history_df[self.history_df[self.group_col] == p_id]
                fallback.fit(p_sub_df)
                self.fallback_forecasters[p_id] = fallback

        return self

    def predict(self, horizon: int = 14) -> pd.DataFrame:
        if self.history_df is None:
            raise ValueError("Model must be fitted before predicting.")

        records = []
        for p_id in self.products:
            p_df = self.history_df[self.history_df[self.group_col] == p_id]
            last_date = p_df[self.date_col].iloc[-1]
            future_dates = self._generate_future_dates(last_date, horizon)

            if p_id in self.fitted_models:
                fitted_model = self.fitted_models[p_id]
                raw_forecasts = fitted_model.forecast(steps=horizon)
                for idx, d in enumerate(future_dates):
                    fc = float(raw_forecasts[idx]) if idx < len(raw_forecasts) else float(raw_forecasts[-1])
                    records.append({
                        self.date_col: d,
                        self.group_col: p_id,
                        "forecast": max(0.0, fc),
                    })
            else:
                fallback = self.fallback_forecasters[p_id]
                fallback_df = fallback.predict(horizon)
                records.extend(fallback_df.to_dict("records"))

        return pd.DataFrame(records)
