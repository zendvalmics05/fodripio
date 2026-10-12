"""
Unit tests and leakage audit tests for FeatureEngineer and SupervisedDatasetBuilder modules.
"""

import pandas as pd
import numpy as np
import pytest
from src.data.synthetic import SyntheticDemandGenerator
from src.data.features import FeatureEngineer, SupervisedDatasetBuilder


@pytest.fixture
def sample_demand_data():
    gen = SyntheticDemandGenerator(start_date="2024-01-01", end_date="2024-03-31", random_state=42)
    return gen.generate()


def test_feature_engineer_creation(sample_demand_data):
    fe = FeatureEngineer()
    df_feat = fe.create_features(sample_demand_data)

    assert len(df_feat) == len(sample_demand_data)
    assert "demand_lag_1" in df_feat.columns
    assert "demand_lag_7" in df_feat.columns
    assert "demand_roll_mean_7" in df_feat.columns
    assert "demand_ewma_a0_1" in df_feat.columns
    assert "dow_sin" in df_feat.columns


def test_feature_engineer_no_leakage_audit(sample_demand_data):
    """
    Assert that mutating future demand at t > cutoff does NOT alter features at t <= cutoff.
    """
    fe = FeatureEngineer()
    cutoff_date = "2024-02-15"
    is_leak_free = fe.audit_leakage(sample_demand_data, cutoff_date)

    assert is_leak_free, "FeatureEngineer failed strict non-leakage audit!"


def test_lag_values_correctness(sample_demand_data):
    fe = FeatureEngineer(lags=[1])
    df_feat = fe.create_features(sample_demand_data)

    p1_df = df_feat[df_feat["product_id"] == "P1"].sort_values("date").reset_index(drop=True)

    # For day index i >= 1, lag_1 at i should equal demand at i-1
    for i in range(1, 10):
        actual_lag_1 = p1_df.loc[i, "demand_lag_1"]
        expected_demand = p1_df.loc[i - 1, "demand"]
        assert actual_lag_1 == expected_demand, f"Lag 1 mismatch at index {i}: {actual_lag_1} != {expected_demand}"


def test_supervised_dataset_builder_direct(sample_demand_data):
    builder = SupervisedDatasetBuilder()
    X, y, meta_df = builder.build_dataset(sample_demand_data, horizon_step=7, mode="direct")

    assert isinstance(X, pd.DataFrame)
    assert isinstance(y, pd.Series)
    assert isinstance(meta_df, pd.DataFrame)

    assert len(X) == len(y) == len(meta_df)
    assert X.isna().sum().sum() == 0
    assert y.isna().sum() == 0

    # One-hot encoding check
    assert "product_P1" in X.columns

    # Alignment check
    is_aligned = builder.audit_supervised_alignment(sample_demand_data, X, y, meta_df)
    assert is_aligned, "SupervisedDatasetBuilder failed temporal alignment audit!"


def test_supervised_dataset_builder_recursive(sample_demand_data):
    builder = SupervisedDatasetBuilder()
    X, y, meta_df = builder.build_dataset(sample_demand_data, mode="recursive")

    assert len(X) == len(y) == len(meta_df)
    assert (meta_df["horizon_step"] == 1).all()
    assert X.isna().sum().sum() == 0
