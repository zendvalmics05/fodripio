# Project Progress Log

## Phase 0: Project Setup & Data Foundation
**Date Completed:** 2026-09-15  
**Status:** COMPLETED  

### Summary of Completed Work
- **Directory Structure:** Established full repository directory tree (`data/raw/`, `data/processed/`, `data/synthetic/`, `notebooks/`, `src/`, `configs/`, `experiments/`, `reports/`, `tests/`).
- **Dependencies:** Created `requirements.txt` defining Python dependencies for forecasting, OR-Tools optimization, SimPy simulation, and unit testing.
- **Global Configuration:** Implemented `configs/experiment.yaml` storing global settings (seed=42, horizon H=14 days, products P1–P5, placeholder operational cost parameters).
- **Synthetic Demand Generator:** Implemented `SyntheticDemandGenerator` in `src/data/synthetic.py` supporting baseline demand, trend, weekly/annual seasonality, promotions, prices, holiday indicators, and demand shocks with 100% seeded reproducibility.
- **Data Ingestion & Cleaning:** Implemented `DataLoader` in `src/data/load.py` for date range re-indexing, missing date filling, and IQR-based outlier flagging without deleting raw demand data.
- **Unit Tests:** Created `tests/test_data_foundation.py` verifying generator reproducibility, data loader pipelines, and configuration integrity.

---

## Phase 1: Demand Data Understanding
**Date Completed:** 2026-09-18  
**Status:** COMPLETED  

### Summary of Completed Work
- **Distribution & Statistical Analysis:** Completed per-product statistical profiling (mean, std, median, skewness, kurtosis, IQR) across all 5 SKUs.
- **Stationarity & Autocorrelation:** Conducted Augmented Dickey-Fuller (ADF) and KPSS stationarity tests alongside ACF/PACF analysis. Identified strong 7-day weekly periodicity and non-stationary/trend-stationary structures.
- **Exogenous Variables Analysis:** Modeled price elasticity ($-0.32$ to $-1.51$) and promotional demand lift ($+14.5\%$ to $+51.1\%$).
- **Cross-Product Correlation Matrix:** Computed demand correlation matrix showing positive inter-SKU correlation ($r \approx 0.63 - 0.68$).
- **Feature Engineering Pipeline:** Developed `FeatureEngineer` in `src/data/features.py` with strict temporal cutoff ($t-1$) for lag features, shifted rolling statistics (windows 7, 14, 28), EWMA, calendar encodings, and price-promo interactions.
- **Data Leakage Audit:** Implemented automated non-leakage audit method `FeatureEngineer.audit_leakage()`.
- **Demand Characterization Memo:** Authored technical memo in `reports/demand_characterization_memo.md`.

---

## Phase 2: Forecasting Baselines
**Date Completed:** 2026-10-02  
**Status:** COMPLETED  

### Summary of Completed Work
- **Baseline Forecasting Engine (`src/forecasting/baselines.py`):** Built unified OOP forecaster hierarchy (`Naive`, `SeasonalNaive`, `MovingAverage_7`, `MovingAverage_14`, `ExponentialSmoothing`, `SARIMAX`).
- **Validation Framework (`src/evaluation/validation.py`):** Implemented `TemporalSplitter` (70/15/15 chronological split) and `RollingOriginValidator` (multi-origin receding horizon cross-validation).
- **Metric Evaluation Suite (`src/evaluation/metrics.py`):** Implemented `mae`, `rmse`, `smape`, `wape`, `mase`, and `ForecastEvaluator` (overall, per-product, per-horizon breakdowns).
- **Baseline Evaluation Notebook (`notebooks/03_baseline_forecasting.ipynb`):** Executed 4-origin rolling evaluation ($H=14$) across all 6 baseline models.

---

## Phase 3: Machine-Learning Forecasting
**Date Completed:** 2026-10-12  
**Status:** COMPLETED  

### Summary of Completed Work
- **Supervised Dataset Engine (`src/data/features.py`):** Implemented `SupervisedDatasetBuilder` converting panel demand into leak-free tabular feature matrices ($X$) and targets ($y$) with categorical product encodings and temporal alignment audits.
- **Machine Learning Forecasters (`src/forecasting/ml_models.py`):** Implemented `RandomForestForecaster`, `XGBoostForecaster`, and `LightGBMForecaster` under direct multi-horizon architecture with early stopping support.
- **Hyperparameter Tuning Engine (`src/forecasting/tuning.py`):** Implemented `TimeSeriesTuner` optimizing tree depth, estimators, and learning rates strictly via rolling-origin temporal validation folds without data leakage. Logged trials to `experiments/tuning_experiments.csv` and optimal parameters to `experiments/best_ml_params.yaml`.
- **ML vs. Baseline Benchmark Comparison (`notebooks/04_ml_forecasting.ipynb`):** Executed comprehensive 6-model comparison across 4 rolling origins ($H=14$ days).

### Final Benchmark Leaderboard (4 Origins, Horizon H=14)

| Rank | Model | WAPE (%) | MAE | RMSE | sMAPE (%) | MASE | Status |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| 🥇 1 | **Tuned LightGBM** | **15.97%** | **12.03** | **18.09** | **16.47%** | **0.1604** | **Selected Global Forecaster** |
| 🥈 2 | **XGBoost** | **16.12%** | 12.14 | 18.00 | 16.67% | 0.1619 | Strong ML Model |
| 🥉 3 | **SARIMA** | **16.50%** | 12.43 | 19.19 | 16.41% | 0.1656 | Statistical Baseline |
| 4 | **Random Forest** | 16.70% | 12.58 | 18.78 | 16.95% | 0.1677 | ML Ensemble |
| 5 | **Exponential Smoothing (ETS)** | 16.74% | 12.60 | 18.96 | 17.62% | 0.1680 | Classical Baseline |
| 6 | **Seasonal Naïve (Lag-7)** | 18.96% | 14.28 | 21.01 | 19.65% | 0.1904 | Simple Benchmark |

### Per-Product WAPE (%) Comparison Table

| SKU | Description | Seasonal Naïve | Holt-Winters ETS | SARIMA | Tuned LightGBM | Best Model |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: |
| **P1** | Standard Bracket A | 17.42% | 16.18% | 16.60% | **15.88%** | **LightGBM / XGBoost** |
| **P2** | Premium Housing B | 16.34% | 10.72% | **10.67%** | 11.93% | **SARIMA / ETS** |
| **P3** | Compact Gear C | 18.83% | 18.70% | 17.91% | **16.27%** | **Tuned LightGBM** |
| **P4** | Heavy Shaft D | 24.70% | 20.08% | 19.55% | **19.43%** | **Tuned LightGBM** |
| **P5** | Custom Valve E | 22.21% | **16.94%** | 17.56% | 19.02% | **Holt-Winters ETS** |

### Exit Criteria Verification
- [x] ML model beats Seasonal Naïve on WAPE for the majority of products (Beats for 5/5 products).
- [x] ML model beats ETS overall (15.97% vs 16.74% WAPE).
- [x] Feature importance summary produced (`demand_lag_1`, `day_of_week`, `demand_lag_7`, `roll_mean_28` dominate).
- [x] Retraining pipeline is deterministic given seed 42.
