"""
Helper script to initialize raw and processed datasets for Fodripio.
"""

from pathlib import Path
import yaml

from src.data.synthetic import SyntheticDemandGenerator
from src.data.load import DataLoader


def initialize_datasets(config_path: str = "configs/experiment.yaml"):
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    data_cfg = config["data"]
    seed = config["experiment"]["seed"]
    products = config.get("products", None)

    raw_path = Path(data_cfg["raw_path"])
    synthetic_path = Path(data_cfg["synthetic_path"])
    processed_path = Path(data_cfg["processed_path"])

    # Ensure parent directories exist
    raw_path.parent.mkdir(parents=True, exist_ok=True)
    synthetic_path.parent.mkdir(parents=True, exist_ok=True)
    processed_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Generating synthetic demand data with seed {seed}...")
    generator = SyntheticDemandGenerator(
        start_date=data_cfg["start_date"],
        end_date=data_cfg["end_date"],
        products=products,
        random_state=seed,
    )
    df_synthetic = generator.generate()

    # Save to synthetic and raw paths
    df_synthetic.to_csv(synthetic_path, index=False)
    df_synthetic.to_csv(raw_path, index=False)
    print(f"Saved raw data to {raw_path} and {synthetic_path}")

    # Process and save processed dataset
    loader = DataLoader(raw_path)
    loader.load_raw_data()
    loader.process()
    loader.save_processed(processed_path)

    summary = loader.summarize()
    print("Dataset initialization complete. Summary:")
    for k, v in summary.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    initialize_datasets()
