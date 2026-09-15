"""
Data Ingestion and Preprocessing Module for Fodripio.

Loads raw or synthetic demand data, performs validation, re-indexes to complete
date ranges, flags outliers (without deleting them), and outputs clean processed data.
"""

import logging
from pathlib import Path
from typing import Dict, Optional, Tuple, Union

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


class DataLoader:
    """
    Handles ingestion, validation, cleaning, and outlier flagging of demand data.
    """

    def __init__(self, filepath: Union[str, Path]):
        self.filepath = Path(filepath)
        self.raw_df: Optional[pd.DataFrame] = None
        self.processed_df: Optional[pd.DataFrame] = None

    def load_raw_data(self) -> pd.DataFrame:
        """
        Loads raw data file from CSV.
        """
        if not self.filepath.exists():
            raise FileNotFoundError(f"Data file not found at: {self.filepath}")

        logger.info(f"Loading raw demand data from {self.filepath}")
        self.raw_df = pd.read_csv(self.filepath)

        # Validate minimum required columns
        required_cols = {"date", "product_id", "demand"}
        if not required_cols.issubset(set(self.raw_df.columns)):
            raise ValueError(f"Raw data missing required columns: {required_cols - set(self.raw_df.columns)}")

        self.raw_df["date"] = pd.to_datetime(self.raw_df["date"])
        logger.info(f"Loaded {len(self.raw_df)} rows spanning {self.raw_df['date'].min()} to {self.raw_df['date'].max()}")
        return self.raw_df

    def process(self, outlier_iqr_threshold: float = 2.5) -> pd.DataFrame:
        """
        Performs raw -> processed cleaning:
        1. Deduplication
        2. Re-indexing complete daily date grid per product
        3. Imputation / default filling of missing dates (demand = 0.0 or forward fill features)
        4. Outlier flagging using rolling/group IQR (keeps raw data intact)

        Returns:
            pd.DataFrame: Processed demand dataframe.
        """
        if self.raw_df is None:
            self.load_raw_data()

        df = self.raw_df.copy()

        # 1. Deduplication
        initial_count = len(df)
        df = df.drop_duplicates(subset=["date", "product_id"], keep="last")
        dedup_count = len(df)
        if initial_count != dedup_count:
            logger.warning(f"Removed {initial_count - dedup_count} duplicate rows.")

        # 2. Re-index date ranges per product
        products = df["product_id"].unique()
        min_date = df["date"].min()
        max_date = df["date"].max()
        full_date_range = pd.date_range(start=min_date, end=max_date, freq="D")

        reindexed_dfs = []
        for p_id in products:
            p_df = df[df["product_id"] == p_id].copy()
            p_df = p_df.set_index("date").reindex(full_date_range)
            p_df["product_id"] = p_id

            # Fill missing demand with 0.0 (or appropriate strategy)
            p_df["demand"] = p_df["demand"].fillna(0.0)

            # Forward fill optional exogenous features if present
            for col in ["price", "promotion", "holiday", "shock"]:
                if col in p_df.columns:
                    p_df[col] = p_df[col].ffill().bfill().fillna(0)

            reindexed_dfs.append(p_df.reset_index().rename(columns={"index": "date"}))

        df_clean = pd.concat(reindexed_dfs, ignore_index=True)
        df_clean = df_clean.sort_values(by=["date", "product_id"]).reset_index(drop=True)

        # 3. Outlier Flagging per product (IQR method on non-zero demand)
        df_clean["is_outlier"] = False
        outlier_counts = {}

        for p_id in products:
            p_mask = df_clean["product_id"] == p_id
            p_demand = df_clean.loc[p_mask, "demand"]

            q25 = np.percentile(p_demand, 25)
            q75 = np.percentile(p_demand, 75)
            iqr = q75 - q25

            lower_bound = q25 - outlier_iqr_threshold * iqr
            upper_bound = q75 + outlier_iqr_threshold * iqr

            is_out = (p_demand < lower_bound) | (p_demand > upper_bound)
            df_clean.loc[p_mask, "is_outlier"] = is_out
            outlier_counts[p_id] = is_out.sum()

        logger.info(f"Outlier flagging complete. Total outliers flagged: {df_clean['is_outlier'].sum()}")
        for p_id, count in outlier_counts.items():
            logger.info(f"  Product {p_id}: {count} outliers ({count / len(full_date_range):.1%})")

        self.processed_df = df_clean
        return self.processed_df

    def save_processed(self, output_path: Union[str, Path]) -> None:
        """
        Saves the processed dataframe to CSV.
        """
        if self.processed_df is None:
            raise ValueError("Processed dataframe is empty. Call process() first.")

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        self.processed_df.to_csv(output_path, index=False)
        logger.info(f"Saved processed data to {output_path}")

    def summarize(self) -> Dict:
        """
        Returns a summary dictionary of raw and processed metrics.
        """
        if self.processed_df is None:
            raise ValueError("Processed data not available.")

        summary = {
            "num_rows": len(self.processed_df),
            "num_products": self.processed_df["product_id"].nunique(),
            "min_date": str(self.processed_df["date"].min().date()),
            "max_date": str(self.processed_df["date"].max().date()),
            "total_demand": float(self.processed_df["demand"].sum()),
            "avg_daily_demand": float(self.processed_df["demand"].mean()),
            "total_outliers": int(self.processed_df["is_outlier"].sum()),
        }
        return summary
