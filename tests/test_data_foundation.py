"""
Unit tests for Phase 0 Data Foundation.
"""

from pathlib import Path
import yaml
import pandas as pd
import pytest

from src.data.synthetic import SyntheticDemandGenerator
from src.data.load import DataLoader


def test_synthetic_reproducibility():
    """
    Test that two SyntheticDemandGenerator runs with the same seed output identical DataFrames.
    """
    gen1 = SyntheticDemandGenerator(start_date="2024-01-01", end_date="2024-03-31", random_state=42)
    df1 = gen1.generate()

    gen2 = SyntheticDemandGenerator(start_date="2024-01-01", end_date="2024-03-31", random_state=42)
    df2 = gen2.generate()

    pd.testing.assert_frame_equal(df1, df2)


def test_synthetic_different_seeds():
    """
    Test that different seeds produce different outputs.
    """
    gen1 = SyntheticDemandGenerator(start_date="2024-01-01", end_date="2024-03-31", random_state=42)
    df1 = gen1.generate()

    gen2 = SyntheticDemandGenerator(start_date="2024-01-01", end_date="2024-03-31", random_state=99)
    df2 = gen2.generate()

    assert not df1["demand"].equals(df2["demand"])


def test_data_loader_processing(tmp_path):
    """
    Test DataLoader ingestion, date range re-indexing, and outlier flagging.
    """
    gen = SyntheticDemandGenerator(start_date="2024-01-01", end_date="2024-01-31", random_state=42)
    df = gen.generate()

    raw_file = tmp_path / "raw_demand.csv"
    proc_file = tmp_path / "processed_demand.csv"
    df.to_csv(raw_file, index=False)

    loader = DataLoader(raw_file)
    raw_df = loader.load_raw_data()
    assert len(raw_df) == len(df)

    proc_df = loader.process(outlier_iqr_threshold=1.5)
    assert "is_outlier" in proc_df.columns
    assert proc_df["demand"].isna().sum() == 0

    loader.save_processed(proc_file)
    assert proc_file.exists()

    summary = loader.summarize()
    assert summary["num_rows"] == len(proc_df)
    assert summary["num_products"] == df["product_id"].nunique()


def test_config_structure():
    """
    Test that configs/experiment.yaml exists and contains required keys.
    """
    config_path = Path("configs/experiment.yaml")
    assert config_path.exists(), "configs/experiment.yaml is missing"

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    assert "experiment" in config
    assert "seed" in config["experiment"]
    assert "horizon" in config["experiment"]
    assert "data" in config
    assert "products" in config
    assert "costs" in config
