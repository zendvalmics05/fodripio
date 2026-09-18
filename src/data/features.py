"""
Feature Engineering Module for Fodripio.

Constructs temporal and exogenous features for demand forecasting while enforcing
a strict "no future information leakage" rule. All rolling statistics and lag features
are computed strictly on data available up to time t-1 when predicting for time t.
"""

from typing import List, Optional, Tuple, Dict, Union
import numpy as np
import pandas as pd


class FeatureEngineer:
    """
    Feature engineering pipeline for multi-product panel time-series data.
    """

    def __init__(
        self,
        lags: List[int] = [1, 2, 7, 14, 28],
        rolling_windows: List[int] = [7, 14, 28],
        ewma_alphas: List[float] = [0.1, 0.3],
        target_col: str = "demand",
        date_col: str = "date",
        group_col: str = "product_id",
    ):
        self.lags = lags
        self.rolling_windows = rolling_windows
        self.ewma_alphas = ewma_alphas
        self.target_col = target_col
        self.date_col = date_col
        self.group_col = group_col

    def create_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Creates all features for the given DataFrame.

        Args:
            df (pd.DataFrame): Panel demand DataFrame containing date, product_id, demand, etc.

        Returns:
            pd.DataFrame: Augmented DataFrame containing original columns and generated features.
        """
        df_feat = df.copy()
        df_feat[self.date_col] = pd.to_datetime(df_feat[self.date_col])
        df_feat = df_feat.sort_values(by=[self.group_col, self.date_col]).reset_index(drop=True)

        feature_cols = []

        # 1. Calendar Features (Known at time t)
        df_feat["day_of_week"] = df_feat[self.date_col].dt.dayofweek
        df_feat["month"] = df_feat[self.date_col].dt.month
        df_feat["day_of_month"] = df_feat[self.date_col].dt.day
        df_feat["day_of_year"] = df_feat[self.date_col].dt.dayofyear
        df_feat["is_weekend"] = (df_feat["day_of_week"] >= 5).astype(int)

        # Cyclical encodings
        df_feat["dow_sin"] = np.sin(2 * np.pi * df_feat["day_of_week"] / 7.0)
        df_feat["dow_cos"] = np.cos(2 * np.pi * df_feat["day_of_week"] / 7.0)
        df_feat["month_sin"] = np.sin(2 * np.pi * df_feat["month"] / 12.0)
        df_feat["month_cos"] = np.cos(2 * np.pi * df_feat["month"] / 12.0)

        calendar_cols = [
            "day_of_week", "month", "day_of_month", "day_of_year",
            "is_weekend", "dow_sin", "dow_cos", "month_sin", "month_cos"
        ]
        feature_cols.extend(calendar_cols)

        # 2. Exogenous Features & Interactions (Known at time t)
        if "price" in df_feat.columns:
            # Price ratio relative to per-product average price
            mean_price = df_feat.groupby(self.group_col)["price"].transform("mean")
            df_feat["price_ratio"] = df_feat["price"] / (mean_price + 1e-8)
            feature_cols.append("price_ratio")

        if "promotion" in df_feat.columns and "price" in df_feat.columns:
            df_feat["promo_price_interaction"] = df_feat["promotion"] * df_feat.get("price_ratio", df_feat["price"])
            feature_cols.append("promo_price_interaction")

        # 3. Lag Features (Target demand shifted by lag k >= 1)
        for lag in self.lags:
            col_name = f"demand_lag_{lag}"
            df_feat[col_name] = df_feat.groupby(self.group_col)[self.target_col].shift(lag)
            feature_cols.append(col_name)

        # 4. Rolling Statistics (Shifted by 1 first to prevent current-day target leakage)
        # Shift target by 1 day per product so rolling window strictly sees data <= t-1
        shifted_target = df_feat.groupby(self.group_col)[self.target_col].shift(1)

        for window in self.rolling_windows:
            mean_col = f"demand_roll_mean_{window}"
            std_col = f"demand_roll_std_{window}"
            min_col = f"demand_roll_min_{window}"
            max_col = f"demand_roll_max_{window}"

            grouped_shifted = shifted_target.groupby(df_feat[self.group_col])
            df_feat[mean_col] = grouped_shifted.transform(lambda x: x.rolling(window, min_periods=1).mean())
            df_feat[std_col] = grouped_shifted.transform(lambda x: x.rolling(window, min_periods=1).std()).fillna(0)
            df_feat[min_col] = grouped_shifted.transform(lambda x: x.rolling(window, min_periods=1).min())
            df_feat[max_col] = grouped_shifted.transform(lambda x: x.rolling(window, min_periods=1).max())

            feature_cols.extend([mean_col, std_col, min_col, max_col])

        # 5. EWMA Features (Exponentially Weighted Moving Average on shifted_target)
        for alpha in self.ewma_alphas:
            alpha_str = str(alpha).replace(".", "_")
            ewma_col = f"demand_ewma_a{alpha_str}"
            grouped_shifted = shifted_target.groupby(df_feat[self.group_col])
            df_feat[ewma_col] = grouped_shifted.transform(lambda x: x.ewm(alpha=alpha, adjust=False).mean())
            feature_cols.append(ewma_col)

        self.feature_cols = feature_cols
        return df_feat

    def audit_leakage(self, original_df: pd.DataFrame, cutoff_date: str) -> bool:
        """
        Audit test ensuring features generated at time t <= cutoff_date are 100% invariant
        to any changes in demand data at time t > cutoff_date.

        Args:
            original_df (pd.DataFrame): Panel demand DataFrame.
            cutoff_date (str): Cutoff date string (YYYY-MM-DD).

        Returns:
            bool: True if zero leakage detected, False otherwise.
        """
        cutoff_dt = pd.to_datetime(cutoff_date)

        # Baseline feature creation
        feat_base = self.create_features(original_df)
        base_subset = feat_base[feat_base[self.date_col] <= cutoff_dt][self.feature_cols].copy()

        # Mutate demand strictly AFTER cutoff date using scale-invariant relative + additive shift
        mutated_df = original_df.copy()
        mutated_df[self.date_col] = pd.to_datetime(mutated_df[self.date_col])
        future_mask = mutated_df[self.date_col] > cutoff_dt
        mutated_df.loc[future_mask, self.target_col] = (mutated_df.loc[future_mask, self.target_col] + 1000.0) * 10.0

        # Feature creation on mutated dataset
        feat_mutated = self.create_features(mutated_df)
        mutated_subset = feat_mutated[feat_mutated[self.date_col] <= cutoff_dt][self.feature_cols].copy()

        # Check exact equality
        try:
            pd.testing.assert_frame_equal(base_subset, mutated_subset)
            return True
        except AssertionError as e:
            print("LEAKAGE AUDIT FAILED:", e)
            return False
