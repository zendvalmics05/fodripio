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
- **Exploration Notebook:** Added `notebooks/01_data_exploration.ipynb` demonstrating raw data generation, ingestion, outlier distribution, and time-series visualizations.

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
- **Demand Characterization Memo:** Authored technical memo in `reports/demand_characterization_memo.md` detailing taxonomy, properties, and model recommendations.
- **Time-Series Analysis Notebook:** Created `notebooks/02_time_series_analysis.ipynb`.

---

## Phase 2: Forecasting Baselines
**Date Completed:** 2026-10-02  
**Status:** COMPLETED  

### Summary of Completed Work
- **Baseline Forecasting Engine (`src/forecasting/baselines.py`):** Built unified OOP forecaster hierarchy (`Naive`, `SeasonalNaive`, `MovingAverage_7`, `MovingAverage_14`, `ExponentialSmoothing`, `SARIMAX`).
- **Validation Framework (`src/evaluation/validation.py`):** Implemented `TemporalSplitter` (70/15/15 chronological split) and `RollingOriginValidator` (multi-origin receding horizon cross-validation).
- **Metric Evaluation Suite (`src/evaluation/metrics.py`):** Implemented `mae`, `rmse`, `smape`, `wape`, `mase`, and `ForecastEvaluator` (overall, per-product, per-horizon breakdowns).
- **Baseline Evaluation Notebook (`notebooks/03_baseline_forecasting.ipynb`):** Executed 4-origin rolling evaluation ($H=14$) across all 6 baseline models.
- **Unit Tests:** Created `tests/test_baselines.py`, `tests/test_validation.py`, and `tests/test_metrics.py` (22/22 unit tests passing cleanly).

### Baseline Benchmark Results Summary (4 Origins, Horizon H=14)

| Model | WAPE (%) | MAE | RMSE | sMAPE (%) | MASE | Status / Rank |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **SARIMA $(1,1,1) \times (1,1,1)_7$** | **16.98%** | 12.80 | 19.31 | 16.99% | 0.1706 | **Best Overall** |
| **Exponential Smoothing (ETS)** | **17.00%** | 12.81 | 19.22 | 18.20% | 0.1708 | Strong Statistical Baseline |
| **Seasonal Naïve (Lag-7)** | **18.70%** | 14.10 | 20.88 | 18.68% | 0.1879 | Benchmark to Beat |
| **Moving Average ($W=14$)** | 26.15% | 19.71 | 26.75 | 28.01% | 0.2627 | Baseline |
| **Moving Average ($W=7$)** | 26.27% | 19.80 | 26.52 | 28.13% | 0.2639 | Baseline |
| **Naïve (Last Value)** | 34.75% | 26.19 | 35.14 | 35.63% | 0.3491 | Flat Baseline |

### Exit Criteria Verification
- [x] All baselines run end-to-end on all products without errors.
- [x] Validation is strictly temporal: unit tests assert $\text{train\_end} < \text{val\_start} < \text{test\_start}$ and zero leakage per origin fold.
- [x] Metric values are stable across reruns (deterministic).
- [x] Seasonal naïve is within a reasonable band of ETS performance (18.70% vs 17.00% WAPE).
