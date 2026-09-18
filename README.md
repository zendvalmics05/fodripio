# Fodripio — Forecast-Driven Production Planning & Inventory Optimization Under Uncertain Demand

[![Phase 1 Completed](https://img.shields.io/badge/Phase_1-Demand_Understanding_Complete-brightgreen)](#)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue)](#)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Build Status](https://img.shields.io/badge/tests-7%20passed-success)](#)

---

## 📌 Executive Summary

**Fodripio** is an industrial production engineering and operations research system that bridges the gap between **time-series demand forecasting** and **operational factory decision-making**.

Instead of evaluating demand forecasting models purely on statistical accuracy (e.g., RMSE, MAE), **Fodripio** evaluates the complete decision-making chain:

### 🎯 Core Engineering Question
> *"Can better demand forecasts lead to better production and inventory decisions, and under what conditions does additional forecasting sophistication actually produce economic value?"*

---

## 🏗️ System Architecture

```text
       Historical Operational Data
                    │
                    ▼
          Data Preprocessing Pipeline (src/data/)
                    │
                    ▼
          Time-Series Analysis & Features (src/data/features.py)
                    │
                    ▼
          Demand Forecasting Engine (src/forecasting/)
         ┌──────────┴──────────┐
         ▼                     ▼
    Point Forecasts     Probabilistic Forecasts (q10, q50, q90)
         └──────────┬──────────┘
                    ▼
     Production & Inventory Optimizer (src/optimization/)
         (OR-Tools CP-SAT / MIP under Capacity & Lead Time Constraints)
                    │
                    ▼
        Receding Horizon Controller (src/integration/)
                    │
                    ▼
       Discrete-Event Factory Simulator (src/simulation/)
         (SimPy Engine: Machine Breakdowns, Stochastic Execution)
                    │
                    ▼
    Operational Metric Suite (Total Cost, Service Level, Stockouts)
```

---

## ✨ Planned Features & Roadmap

The project is being developed across 10 structured phases detailed in [`Implementation Plan.md`](file:///c:/zendvalmics/project/fodripio/Implementation%20Plan.md):

| Phase | Module / Feature Set | Key Capabilities & Deliverables | Status |
| :---: | :--- | :--- | :---: |
| **0** | **Data Foundation & Setup** | Reproducible synthetic demand generator, data ingestion (`DataLoader`), clean repo structure, unit tests. | **Completed** |
| **1** | **Demand Data Understanding** | Per-product distribution analysis, stationarity tests (ADF/KPSS), seasonality decomposition, non-leaking feature builder (`FeatureEngineer`). | **Completed** |
| **2** | **Forecasting Baselines** | Naïve, seasonal naïve, moving average, Exponential Smoothing (ETS), SARIMA pipelines with strict rolling-origin validation. | *Next Phase* |
| **3** | **Machine Learning Forecasting** | XGBoost, LightGBM, Random Forest forecasters with rolling temporal cross-validation. | *Planned* |
| **4** | **Production Optimization Model** | Mathematical programming model in **OR-Tools CP-SAT** solving production quantities, inventory, backlog, setup times, and machine capacity. | *Planned* |
| **5** | **Forecast → Optimization Integration** | Closed-loop pipeline linking demand forecasts directly into the production optimizer. | *Planned* |
| **6** | **Rolling-Horizon System** | Dynamic receding-horizon planning loop re-optimizing daily decisions over horizon $H=14$. | *Planned* |
| **7** | **SimPy Factory Simulation** | Stochastic discrete-event simulation evaluating policies under operational noise and machine execution delays. | *Planned* |
| **8** | **Probabilistic Planning** | Quantile regression forecasts and uncertainty-aware stochastic planning. | *Planned* |
| **9** | **Value-of-Forecasting Study** | Comprehensive operational cost sensitivity analysis and final research report. | *Planned* |

---

## 🛠️ Current State & Quickstart

In its current state (**Phase 0 Complete**), Fodripio includes a complete runnable data foundation pipeline, a multi-product synthetic demand generator, automated data cleaning with outlier flagging, global configuration management, interactive Jupyter exploratory tools, and unit tests.

### 1. Prerequisites
- **Python 3.10+** installed on your system.
- Git (for repository cloning).

### 2. Environment Setup
Clone the repository and install the required dependencies:

```bash
# Clone the repository
git clone https://github.com/your-username/fodripio.git
cd fodripio

# Install dependencies
pip install -r requirements.txt
```

### 3. Initialize Datasets
Run the dataset initialization pipeline to generate multi-product synthetic demand and clean processed data:

```bash
python -m src.data.init_dataset
```

This will produce:
- `data/raw/demand_raw.csv`: Raw multi-product time-series data.
- `data/synthetic/demand_synthetic.csv`: Reproducible synthetic data store.
- `data/processed/demand_processed.csv`: Cleaned data with re-indexed date grids and IQR outlier flags (`is_outlier`).

### 4. Run Automated Unit Tests
Verify that the data generator and ingestion pipelines pass all verification criteria:

```bash
pytest tests/
```

### 5. Explore Demand Data
Launch the exploratory data analysis notebook to inspect demand trajectories, seasonality, and outlier distributions:

```bash
jupyter notebook notebooks/01_data_exploration.ipynb
```

---

## 📁 Repository Structure

```text
fodripio/
├── data/
│   ├── raw/                  # Raw daily demand CSV files
│   ├── processed/            # Cleaned demand datasets with outlier flags
│   └── synthetic/            # Seeded synthetic demand data store
├── notebooks/
│   └── 01_data_exploration.ipynb # Interactive EDA & visual analytics notebook
├── src/
│   ├── data/                 # Data loading, cleaning, & synthetic generation
│   │   ├── load.py           # DataLoader pipeline (re-indexing & outlier flagging)
│   │   ├── synthetic.py      # SyntheticDemandGenerator (multi-product demand engine)
│   │   └── init_dataset.py   # Dataset build script
│   ├── forecasting/          # Baseline & ML forecasting modules (Phases 2–3)
│   ├── optimization/         # OR-Tools CP-SAT production planner (Phase 4)
│   ├── simulation/           # SimPy discrete-event factory simulator (Phase 7)
│   ├── evaluation/           # Forecast & operational cost metric suite
│   └── visualization/        # Time-series & operational plotting utilities
├── configs/
│   └── experiment.yaml       # Global experiment parameters, costs, & horizons
├── experiments/              # Experiment logs, metrics, & model checkpoints
├── reports/
│   └── progress_log.md       # Phase completion progress tracking log
├── tests/
│   └── test_data_foundation.py # Automated pytest test suite
├── Master Plan.md            # Comprehensive project master specification
├── Implementation Plan.md    # Phase-by-phase development roadmap
├── requirements.txt          # Python dependency manifest
└── README.md                 # Project overview and quickstart guide
```

---

## 📐 Methodological Principles

Every component in Fodripio strictly adheres to six operational research rules:

1. **No Future Information Leakage**: Forecasting features use strictly historical information available at prediction time $t$.
2. **Chronological Validation**: No random train/test splits; all evaluation uses temporal rolling-origin validation.
3. **Strong Baselines First**: Complex ML models must beat naïve and statistical baselines (ETS/SARIMA) on operational costs to justify selection.
4. **Separated Evaluation**: Forecasting accuracy (WAPE, RMSE) and operational decision performance (Total Operational Cost) are evaluated separately.
5. **Hidden Future Demand**: The production optimizer observes forecasts only; actual demand is revealed strictly during factory execution.
6. **Stochastic Simulation**: Policy evaluation is conducted over multiple random seeds to establish statistically rigorous confidence intervals.

---

## 📜 License

This project is licensed under the **MIT License** — see the [LICENSE](LICENSE) file for details.
