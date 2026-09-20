"""
Forecasting module package.
Contains baseline and machine learning forecasting models.
"""

from src.forecasting.baselines import (
    BaseForecaster,
    NaiveForecaster,
    SeasonalNaiveForecaster,
    MovingAverageForecaster,
    ExponentialSmoothingForecaster,
    SARIMAForecaster,
)

__all__ = [
    "BaseForecaster",
    "NaiveForecaster",
    "SeasonalNaiveForecaster",
    "MovingAverageForecaster",
    "ExponentialSmoothingForecaster",
    "SARIMAForecaster",
]
