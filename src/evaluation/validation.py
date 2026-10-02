"""
Time-Series Validation Framework for Fodripio.

Provides:
1. TemporalSplitter: Chronological train / validation / test partitioning (no random splits).
2. RollingOriginValidator: Multi-origin receding horizon evaluation framework.
"""

from typing import List, Tuple, Optional, Union, Dict
import numpy as np
import pandas as pd


class TemporalSplitter:
    """
    Handles strict chronological partitioning of panel demand time-series data.
    """

    def __init__(
        self,
        date_col: str = "date",
        group_col: str = "product_id",
        target_col: str = "demand",
    ):
        self.date_col = date_col
        self.group_col = group_col
        self.target_col = target_col

    def split_ratios(
        self,
        df: pd.DataFrame,
        train_ratio: float = 0.70,
        val_ratio: float = 0.15,
        test_ratio: float = 0.15,
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Splits dataset chronologically based on target date ratios.

        Returns:
            Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]: (train_df, val_df, test_df)
        """
        if not np.isclose(train_ratio + val_ratio + test_ratio, 1.0):
            raise ValueError(f"Ratios must sum to 1.0 (got {train_ratio + val_ratio + test_ratio})")

        df_sorted = df.copy()
        df_sorted[self.date_col] = pd.to_datetime(df_sorted[self.date_col])
        df_sorted = df_sorted.sort_values(by=[self.date_col, self.group_col]).reset_index(drop=True)

        unique_dates = sorted(df_sorted[self.date_col].unique())
        n_dates = len(unique_dates)

        n_train = int(np.floor(n_dates * train_ratio))
        n_val = int(np.floor(n_dates * val_ratio))

        train_dates = unique_dates[:n_train]
        val_dates = unique_dates[n_train : n_train + n_val]
        test_dates = unique_dates[n_train + n_val :]

        # Enforce strict temporal ordering assertion
        max_train = max(train_dates)
        min_val = min(val_dates)
        max_val = max(val_dates)
        min_test = min(test_dates)

        if not (max_train < min_val and max_val < min_test):
            raise ValueError("Temporal order violation detected in split_ratios!")

        train_df = df_sorted[df_sorted[self.date_col].isin(train_dates)].copy()
        val_df = df_sorted[df_sorted[self.date_col].isin(val_dates)].copy()
        test_df = df_sorted[df_sorted[self.date_col].isin(test_dates)].copy()

        return train_df, val_df, test_df

    def split_by_dates(
        self,
        df: pd.DataFrame,
        train_end: str,
        val_end: str,
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Splits dataset chronologically based on explicit train_end and val_end dates.

        Returns:
            Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]: (train_df, val_df, test_df)
        """
        df_sorted = df.copy()
        df_sorted[self.date_col] = pd.to_datetime(df_sorted[self.date_col])

        train_end_dt = pd.to_datetime(train_end)
        val_end_dt = pd.to_datetime(val_end)

        if not (train_end_dt < val_end_dt):
            raise ValueError(f"train_end ({train_end}) must be strictly earlier than val_end ({val_end})")

        train_df = df_sorted[df_sorted[self.date_col] <= train_end_dt].copy()
        val_df = df_sorted[
            (df_sorted[self.date_col] > train_end_dt) & (df_sorted[self.date_col] <= val_end_dt)
        ].copy()
        test_df = df_sorted[df_sorted[self.date_col] > val_end_dt].copy()

        return train_df, val_df, test_df


class RollingOriginValidator:
    """
    Performs multi-origin receding horizon cross-validation.
    """

    def __init__(
        self,
        date_col: str = "date",
        group_col: str = "product_id",
        target_col: str = "demand",
    ):
        self.date_col = date_col
        self.group_col = group_col
        self.target_col = target_col

    def evaluate(
        self,
        df: pd.DataFrame,
        forecaster,
        horizon: int = 14,
        step_size: int = 14,
        num_origins: Optional[int] = None,
        min_train_days: int = 60,
    ) -> pd.DataFrame:
        """
        Executes rolling-origin validation on the given forecaster.

        Args:
            df (pd.DataFrame): Panel demand dataset.
            forecaster: Object conforming to BaseForecaster interface (fit, predict).
            horizon (int): Forecast horizon H in days (default 14).
            step_size (int): Days between origin shifts (default 14).
            num_origins (Optional[int]): Number of origin folds to evaluate (default all possible).
            min_train_days (int): Minimum required historical days before first origin.

        Returns:
            pd.DataFrame: Evaluation DataFrame with columns:
                origin_id, origin_date, date, product_id, horizon_step, actual, forecast
        """
        df_proc = df.copy()
        df_proc[self.date_col] = pd.to_datetime(df_proc[self.date_col])
        df_proc = df_proc.sort_values(by=[self.date_col, self.group_col]).reset_index(drop=True)

        unique_dates = sorted(df_proc[self.date_col].unique())
        n_dates = len(unique_dates)

        if n_dates < min_train_days + horizon:
            raise ValueError(
                f"Dataset length ({n_dates} days) insufficient for min_train_days ({min_train_days}) + horizon ({horizon})"
            )

        # Generate candidate origin indices
        origin_indices = []
        curr_idx = min_train_days - 1
        while curr_idx + horizon < n_dates:
            origin_indices.append(curr_idx)
            curr_idx += step_size

        if num_origins is not None and num_origins > 0:
            origin_indices = origin_indices[-num_origins:]

        eval_records = []

        for origin_idx_count, orig_idx in enumerate(origin_indices):
            origin_date = unique_dates[orig_idx]
            origin_id = f"origin_{origin_idx_count + 1}"

            # 1. Train dataset (strictly <= origin_date)
            train_sub = df_proc[df_proc[self.date_col] <= origin_date].copy()

            # 2. Test ground-truth dataset (origin_date < date <= origin_date + horizon)
            future_dates = unique_dates[orig_idx + 1 : orig_idx + 1 + horizon]
            test_sub = df_proc[df_proc[self.date_col].isin(future_dates)].copy()

            # 3. Fit forecaster on train_sub
            forecaster.fit(train_sub)

            # 4. Predict H steps ahead
            pred_df = forecaster.predict(horizon=horizon)
            pred_df[self.date_col] = pd.to_datetime(pred_df[self.date_col])

            # 5. Merge predictions with actual ground truth
            merged = pd.merge(
                test_sub[[self.date_col, self.group_col, self.target_col]],
                pred_df[[self.date_col, self.group_col, "forecast"]],
                on=[self.date_col, self.group_col],
                how="inner",
            ).rename(columns={self.target_col: "actual"})

            # Map horizon step (1..H) per date relative to origin_date
            date_to_step = {d: step + 1 for step, d in enumerate(future_dates)}
            merged["horizon_step"] = merged[self.date_col].map(date_to_step)
            merged["origin_id"] = origin_id
            merged["origin_date"] = origin_date

            eval_records.append(merged)

        results_df = pd.concat(eval_records, ignore_index=True)
        cols_order = ["origin_id", "origin_date", self.date_col, self.group_col, "horizon_step", "actual", "forecast"]
        return results_df[cols_order]
