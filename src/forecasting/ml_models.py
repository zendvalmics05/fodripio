"""
Machine Learning Forecasting Models for Fodripio.

Implements tree-based machine learning forecasters:
1. RandomForestForecaster (scikit-learn RandomForestRegressor)
2. XGBoostForecaster (XGBoost XGBRegressor with early stopping)
3. LightGBMForecaster (LightGBM LGBMRegressor with early stopping)

All models inherit from BaseForecaster interface (fit, predict) and utilize
SupervisedDatasetBuilder for direct multi-step forecasting across horizon H.
"""

from abc import ABC, abstractmethod
import warnings
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb
import lightgbm as lgb

from src.forecasting.baselines import BaseForecaster
from src.data.features import SupervisedDatasetBuilder, FeatureEngineer


class BaseMLForecaster(BaseForecaster):
    """
    Base class for supervised machine learning forecasters using direct multi-horizon strategy.
    For horizon H, fits H separate models (one for target y_{t+h} for h = 1..H).
    """

    def __init__(
        self,
        builder: Optional[SupervisedDatasetBuilder] = None,
        random_state: int = 42,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.builder = builder or SupervisedDatasetBuilder(
            target_col=self.target_col, date_col=self.date_col, group_col=self.group_col
        )
        self.random_state = random_state
        self.models: Dict[int, object] = {}
        self.feature_names: List[str] = []

    @abstractmethod
    def _create_base_model(self) -> object:
        """Instantiates the underlying ML regressor for a single horizon step."""
        pass

    def fit(self, df: pd.DataFrame, val_df: Optional[pd.DataFrame] = None) -> "BaseMLForecaster":
        """
        Fits H separate models for direct forecasting steps h = 1..14 (or up to horizon).
        """
        self._prepare_history(df)
        horizon = 14

        self.models = {}
        for h in range(1, horizon + 1):
            X_train, y_train, _ = self.builder.build_dataset(self.history_df, horizon_step=h, mode="direct")
            self.feature_names = list(X_train.columns)

            model = self._create_base_model()

            if val_df is not None:
                X_val, y_val, _ = self.builder.build_dataset(val_df, horizon_step=h, mode="direct")
                # Fit with validation evaluation if model supports it
                if hasattr(model, "fit") and "eval_set" in model.fit.__code__.co_varnames:
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
                else:
                    model.fit(X_train, y_train)
            else:
                model.fit(X_train, y_train)

            self.models[h] = model

        return self

    def predict(self, horizon: int = 14) -> pd.DataFrame:
        """
        Generates H-step forecasts using the direct trained models.
        """
        if self.history_df is None or not self.models:
            raise ValueError("Model must be fitted before predicting.")

        # 1. Compute features for the full historical DataFrame
        df_feat = self.builder.feature_engineer.create_features(self.history_df)

        # 2. Get the latest feature row available for each product at historical cutoff T_last
        latest_records = []
        for p_id in self.products:
            p_df = df_feat[df_feat[self.group_col] == p_id].sort_values(self.date_col)
            last_row = p_df.iloc[-1:].copy()
            latest_records.append(last_row)

        df_latest = pd.concat(latest_records, ignore_index=True)

        feature_cols = self.builder.feature_engineer.feature_cols
        X_latest = df_latest[feature_cols].copy()

        if self.builder.encode_categories:
            product_dummies = pd.get_dummies(df_latest[self.group_col], prefix="product", dtype=float)
            # Reindex columns to match trained feature_names
            X_latest = pd.concat([X_latest, product_dummies], axis=1)

        # Align columns exactly with training feature_names
        for col in self.feature_names:
            if col not in X_latest.columns:
                X_latest[col] = 0.0
        X_latest = X_latest[self.feature_names]

        # 3. Predict for each horizon step h
        records = []

        for p_idx, p_id in enumerate(self.products):
            p_df = self.history_df[self.history_df[self.group_col] == p_id]
            last_date = p_df[self.date_col].iloc[-1]
            future_dates = self._generate_future_dates(last_date, horizon)

            x_prod = X_latest.iloc[p_idx: p_idx + 1]

            for h in range(1, horizon + 1):
                model_h = self.models.get(h, self.models[1])
                pred_val = float(model_h.predict(x_prod)[0])
                pred_val = max(0.0, pred_val)

                records.append({
                    self.date_col: future_dates[h - 1],
                    self.group_col: p_id,
                    "forecast": pred_val,
                })

        return pd.DataFrame(records)

    def get_feature_importances(self) -> pd.DataFrame:
        """
        Returns average feature importances across all horizon step models.
        """
        if not self.models or not self.feature_names:
            raise ValueError("Model must be fitted before extracting feature importances.")

        importance_list = []
        for h, model in self.models.items():
            if hasattr(model, "feature_importances_"):
                imp = model.feature_importances_
                importance_list.append(imp)

        if not importance_list:
            return pd.DataFrame()

        avg_imp = np.mean(importance_list, axis=0)
        imp_df = pd.DataFrame({
            "feature": self.feature_names,
            "importance": avg_imp,
        }).sort_values(by="importance", ascending=False).reset_index(drop=True)

        return imp_df


class RandomForestForecaster(BaseMLForecaster):
    """
    Random Forest Regressor Forecaster using Scikit-Learn.
    """

    def __init__(self, n_estimators: int = 100, max_depth: int = 12, **kwargs):
        super().__init__(**kwargs)
        self.n_estimators = n_estimators
        self.max_depth = max_depth

    def _create_base_model(self) -> RandomForestRegressor:
        return RandomForestRegressor(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            random_state=self.random_state,
            n_jobs=-1,
        )


class XGBoostForecaster(BaseMLForecaster):
    """
    XGBoost Regressor Forecaster with optional early stopping support.
    """

    def __init__(
        self,
        n_estimators: int = 150,
        learning_rate: float = 0.05,
        max_depth: int = 6,
        subsample: float = 0.8,
        colsample_bytree: float = 0.8,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.subsample = subsample
        self.colsample_bytree = colsample_bytree

    def _create_base_model(self) -> xgb.XGBRegressor:
        return xgb.XGBRegressor(
            n_estimators=self.n_estimators,
            learning_rate=self.learning_rate,
            max_depth=self.max_depth,
            subsample=self.subsample,
            colsample_bytree=self.colsample_bytree,
            random_state=self.random_state,
            n_jobs=-1,
        )


class LightGBMForecaster(BaseMLForecaster):
    """
    LightGBM Regressor Forecaster with optional early stopping support.
    """

    def __init__(
        self,
        n_estimators: int = 150,
        learning_rate: float = 0.05,
        max_depth: int = 6,
        num_leaves: int = 31,
        subsample: float = 0.8,
        colsample_bytree: float = 0.8,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.num_leaves = num_leaves
        self.subsample = subsample
        self.colsample_bytree = colsample_bytree

    def _create_base_model(self) -> lgb.LGBMRegressor:
        return lgb.LGBMRegressor(
            n_estimators=self.n_estimators,
            learning_rate=self.learning_rate,
            max_depth=self.max_depth,
            num_leaves=self.num_leaves,
            subsample=self.subsample,
            colsample_bytree=self.colsample_bytree,
            random_state=self.random_state,
            verbose=-1,
            n_jobs=-1,
        )
