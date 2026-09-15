# Implementation Plan
## Forecast-Driven Production Planning and Inventory Optimization Under Uncertain Demand

**Version:** 1.0
**Date:** 2026-09-15
**Source:** Master Plan (Project 1)
**Target completion level:** Level 5 (Forecast + Optimization + SimPy Factory), with optional Level 6–7 extensions

---

## 0. Guiding Principles

Every phase must respect these methodological rules (from Master Plan §22):

1. **No future information leakage** — features contain only data available at prediction time.
2. **No random train/test splits** — chronological or rolling-origin validation only.
3. **Strong baselines first** — a complex model must beat simple baselines to justify its existence.
4. **Separate forecasting evaluation from decision evaluation** — better RMSE ≠ lower cost.
5. **Actual future demand stays hidden during planning** — the optimizer sees forecasts only.
6. **Explicit operational assumptions** — synthetic capacities/costs are labeled as assumptions.
7. **Stochastic simulation** — evaluate across multiple seeds/replications, never one lucky run.

---

## 1. Phase Overview

| Phase | Name | Goal (Deliverable) | Depends On | Est. Effort |
|---|---|---|---|---|
| 0 | Project Setup & Data Foundation | Repo, environment, raw data characterized | — | 1 wk |
| 1 | Demand Data Understanding | Characterized demand-generating process | Phase 0 | 1–2 wks |
| 2 | Forecasting Baselines | Reliable baseline forecasting pipeline | Phase 1 | 2 wks |
| 3 | ML Forecasting | Strong ML forecaster with defensible validation | Phase 2 | 2–3 wks |
| 4 | Production Optimization Model | Functioning production/inventory optimizer | Phase 1 (parallel w/ 2–3) | 2–3 wks |
| 5 | Forecast → Optimization Integration | Evidence of how forecast quality affects decisions | Phases 3, 4 | 2 wks |
| 6 | Rolling-Horizon Planning System | Dynamic forecast-driven planning loop | Phase 5 | 2 wks |
| 7 | SimPy Factory Simulation | Robust policy comparison under uncertainty | Phase 6 | 2–3 wks |
| 8 | Probabilistic Forecasting & Uncertainty-Aware Planning | Uncertainty-aware planning system (Level 6) | Phase 7 | 2–3 wks |
| 9 | Experimental Analysis & Reporting | Value-of-forecasting study + final report (Level 7) | Phase 8 | 2 wks |

**Total estimated effort:** ~16–22 weeks for full Level 7; **Level 5 (core portfolio piece) achievable in ~10–12 weeks.**

---

## 2. Detailed Phase Plans

Each phase is divided into: **Build** (implementation steps), **Use** (how to operate it), and **Test** (verification before moving on).

---

### PHASE 0 — Project Setup & Data Foundation
**Goal:** A runnable repository with data loaded and characterized at a raw level.

**Build**
- 0.1 Create repository structure per Master Plan §21 (`data/`, `notebooks/`, `src/`, `configs/`, `experiments/`, `reports/`, `tests/`).
- 0.2 Set up Python environment (`requirements.txt`): pandas, NumPy, matplotlib, statsmodels, scikit-learn, XGBoost/LightGBM, OR-Tools, SimPy, Jupyter, pytest.
- 0.3 Choose data strategy (Master Plan §6):
  - **Primary:** Public real-world demand dataset (e.g., multi-SKU retail/sales data).
  - **Fallback/Complement:** Synthetic generator with trend + weekly/annual seasonality + promotions + correlated products + shocks (seeded, reproducible).
- 0.4 Write data ingestion module (`src/data/load.py`) with raw → processed cleaning: missing values, duplicates, timezone alignment, outlier flagging (keep raw, don't delete).
- 0.5 Write `configs/experiment.yaml` for global settings (horizon H=14, seeds, cost parameters placeholder).

**Use**
- Run `01_data_exploration.ipynb` to load raw data and produce first visualizations.

**Test (Exit Criteria)**
- [ ] Repo clones and environment installs cleanly on a fresh machine.
- [ ] Raw data loads without errors; row counts and date ranges logged.
- [ ] Synthetic generator (if used) reproduces identical output with same seed.

---

### PHASE 1 — Demand Data Understanding
**Goal:** A clear characterization of the demand-generating process (Master Plan §19 Phase 1).

**Build**
- 1.1 Complete `notebooks/01_data_exploration.ipynb`:
  - Cleaning decisions documented (imputation rules, outlier treatment).
  - Per-product demand distributions (histograms, summary stats).
- 1.2 Complete `notebooks/02_time_series_analysis.ipynb`:
  - Trend analysis (rolling means, decomposition).
  - Seasonality analysis (weekly, annual; ACF/PACF plots).
  - Stationarity tests (ADF, KPSS).
  - Cross-correlation between products (demand correlation matrix).
  - Exogenous variables: price, promotion flags, holidays.
- 1.3 Write `src/data/features.py` — reusable feature builders (lags, rolling stats, calendar features) with a strict "availability timestamp" rule to prevent leakage.
- 1.4 Produce a written demand characterization memo in `reports/` (2–3 pages): trend/seasonality/stationarity per product, variance structure, recommended model families.

**Use**
- The memo becomes the reference document for choosing models in Phases 2–3.

**Test (Exit Criteria)**
- [ ] Every product has documented: trend (Y/N + direction), seasonality (periods), stationarity verdict, outlier rate.
- [ ] Feature builder unit-tested: for a given prediction time `t`, no feature uses data after `t`.
- [ ] Leakage test passes: shifting features forward in time must not change values at `t`.

---

### PHASE 2 — Forecasting Baselines
**Goal:** Reliable baseline forecasting pipeline (Master Plan §19 Phase 2).

**Build**
- 2.1 Implement in `src/forecasting/baselines.py`:
  - Last-value (naïve) forecast.
  - Seasonal naïve (lag-7).
  - Moving average (window = 7, 14, 28).
  - Simple exponential smoothing.
  - Holt-Winters / ETS (statsmodels).
  - ARIMA/SARIMA (auto-order selection via AIC grid or `pmdarima` if allowed).
- 2.2 Implement rolling-origin validation framework in `src/evaluation/validation.py`:
  - Chronological split: train → validation → test (e.g., 70/15/15).
  - Rolling-origin: multiple origins, forecast H=14 days ahead from each.
  - **No random shuffling anywhere.**
- 2.3 Implement metric suite in `src/evaluation/metrics.py`: MAE, RMSE, sMAPE, WAPE, MASE (MAPE only where demand >> 0).
- 2.4 Complete `notebooks/03_baseline_forecasting.ipynb` with results tables + plots (forecast vs actual for a few origins).

**Use**
- Baseline results become the benchmark that ML models (Phase 3) must beat.

**Test (Exit Criteria)**
- [ ] All baselines run end-to-end on all products without errors.
- [ ] Validation is strictly temporal: unit test asserts train end < validation start < test start.
- [ ] Metric values are stable across reruns (deterministic given seed).
- [ ] Seasonal naïve is within a reasonable band of ETS performance (sanity check).

---

### PHASE 3 — Machine-Learning Forecasting
**Goal:** A strong ML forecaster with defensible evaluation (Master Plan §19 Phase 3).

**Build**
- 3.1 Build supervised learning dataset via `src/data/features.py`: lags (1,2,7,14,28), rolling mean/std/min/max (7,14,28), EWMA, day-of-week, month, holiday, price, promotion.
- 3.2 Implement models in `src/forecasting/ml_models.py`:
  - Random Forest (scikit-learn).
  - XGBoost and/or LightGBM with early stopping on temporal validation fold.
- 3.3 Hyperparameter tuning: time-series cross-validation only (e.g., 3 rolling folds); log experiments in `experiments/`.
- 3.4 Compare ML vs. baselines in `notebooks/04_ml_forecasting.ipynb`:
  - Same origins, same horizons, same metrics.
  - Per-product and aggregate results.
  - Error analysis: where does ML win/lose vs. ETS?
- 3.5 (Optional stretch) Level-4 neural models (LSTM/GRU/TCN) — only if tree models clearly underperform on specific products.

**Use**
- Select the best forecaster per product (or globally) for Phase 5 integration. Record selection rationale.

**Test (Exit Criteria)**
- [ ] ML model beats seasonal naïve on WAPE for the majority of products.
- [ ] ML model beats or matches ETS within a justified margin (if not, document why — that itself is a finding).
- [ ] Feature importance / SHAP summary produced to confirm features are sensible (e.g., lag-7 dominates for weekly seasonality).
- [ ] Retraining pipeline is reproducible: same seed → same model → same metrics.

---

### PHASE 4 — Production Optimization Model
**Goal:** A functioning production/inventory planning optimizer (Master Plan §19 Phase 4). *Can be built in parallel with Phases 2–3.*

**Build**
- 4.1 Define factory configuration in `configs/factory.yaml` (Master Plan §5):
  - 5–10 products, 2–5 machines, processing times `a_{p,m}`, material consumption, machine capacities `C_{m,t}`, setup times, lead times, cost parameters (production, holding `c^I`, shortage `c^B`, setup `c^S`, overtime).
  - **Label all synthetic parameters as ASSUMPTIONS in config comments.**
- 4.2 Implement optimizer in `src/optimization/planner.py` using OR-Tools CP-SAT (or Pyomo+HiGHS):
  - Decision variables: `Q[p,t]` production, `I[p,t]` inventory, `B[p,t]` backlog, `X[p,m,t]` machine allocation, setup binaries.
  - Objective: minimize total cost (production + holding + shortage + setup + overtime).
  - Constraints: inventory balance, machine capacity, material availability, warehouse capacity, min batch sizes, overtime limits.
- 4.3 Implement a **deterministic test mode**: optimize against *known* demand to verify the model finds the obvious optimum (e.g., zero demand → zero production; huge shortage cost → produce to meet all demand).
- 4.4 Write unit tests in `tests/test_planner.py`.

**Use**
- The planner takes a demand vector/forecast as input and returns a production plan for horizon H.

**Test (Exit Criteria)**
- [ ] Deterministic sanity tests pass (zero-demand → zero cost; single-product single-machine cases solved to known optima).
- [ ] Constraint checks: no machine overloaded; inventory balance holds for every period; non-negative production.
- [ ] Solve time for H=14, 10 products, 5 machines is under ~30 s (scalability check).
- [ ] Infeasibility handling: when demand exceeds capacity, model returns backlog solution rather than crashing.

---

### PHASE 5 — Forecast → Optimization Integration
**Goal:** Evidence of how forecasting quality affects production decisions (Master Plan §19 Phase 5).

**Build**
- 5.1 Implement the closed loop in `src/integration/pipeline.py`:
  ```
  historical demand → forecast H days → planner → production plan for day t+1
  ```
- 5.2 Implement **Strategy A (fixed policy)** baseline: produce historical average demand, no optimization.
- 5.3 Run offline comparison (no simulation yet — perfect execution, forecast as demand proxy):
  - Strategy B: naïve forecast → optimizer.
  - Strategy C: ETS/SARIMA forecast → optimizer.
  - Strategy D: XGBoost forecast → optimizer.
  - Feed each forecast into the same optimizer, same cost structure.
- 5.4 Evaluate operational metrics (Master Plan §16): total cost, production cost, holding cost, shortage cost, stockout frequency, service level, fill rate, average inventory, utilization.
- 5.5 Produce comparison table + plots in `notebooks/05_results_analysis.ipynb`.

**Use**
- This is the first "money chart": forecast quality (MAE/WAPE) on x-axis vs. total operational cost on y-axis.

**Test (Exit Criteria)**
- [ ] No leakage: unit test asserts optimizer input contains only forecasts generated from data ≤ decision time.
- [ ] Fixed policy (A) is strictly worse than any forecast+optimization strategy on total cost (sanity).
- [ ] Results are reproducible across 3+ runs with fixed seeds.

---

### PHASE 6 — Rolling-Horizon Planning System
**Goal:** A dynamic forecast-driven production-planning system (Master Plan §19 Phase 6).

**Build**
- 6.1 Implement the rolling-horizon controller in `src/integration/rolling.py`:
  ```
  for each day t in evaluation window:
      1. observe demand up to t
      2. retrain/refresh forecaster (or use pre-trained with recalibration)
      3. forecast t+1 … t+H
      4. solve optimization over H days
      5. commit only Q[t+1]
      6. record planned vs committed
  ```
- 6.2 Add "execute only first decision" logic — future plan is discarded and re-optimized next day.
- 6.3 Support multiple forecast sources plugged into the same controller (strategy pattern).
- 6.4 Compare rolling vs. static planning (single optimization at t=0, no re-optimization) — addresses Research Question 9.

**Use**
- Run a 30-day simulated planning exercise on historical data (still deterministic execution; simulation comes in Phase 7).

**Test (Exit Criteria)**
- [ ] Rolling re-optimization outperforms static planning on total cost (expected; if not, investigate horizon/cost parameters).
- [ ] Inventory balance holds at every step of the roll.
- [ ] Controller runs without manual intervention for the full evaluation window.
- [ ] Horizon sensitivity: H ∈ {7, 14, 30} compared and documented (Research Question 5).

---

### PHASE 7 — SimPy Factory Simulation
**Goal:** Robust comparison of planning policies under uncertainty (Master Plan §19 Phase 7).

**Build**
- 7.1 Build SimPy environment in `src/simulation/factory.py`:
  - Entities: demand process (stochastic, seeded), machines (capacity, breakdowns optional), queues, raw material arrivals, finished-goods inventory, order fulfillment.
  - Events: daily demand arrival, production execution with delays, material consumption, inventory update, shortage/backlog or lost-sales.
- 7.2 Implement policy adapters: the controller from Phase 6 runs *inside* the simulation as the decision agent.
- 7.3 Run Monte Carlo experiments: ≥30 replications per strategy × 5 strategies (A–E from Master Plan §15), each with different demand seeds.
- 7.4 Statistical comparison: mean ± confidence intervals of total cost, service level, inventory, stockouts, utilization, changeovers.

**Use**
- This answers the core question: *does better forecasting actually lower cost when reality is noisy?*

**Test (Exit Criteria)**
- [ ] Simulation is seeded and reproducible: same seed → identical trajectory.
- [ ] Warm-up period handled (discard initial transient from statistics).
- [ ] All 5 strategies complete all replications without crashes.
- [ ] Results include confidence intervals, not just point estimates.

---

### PHASE 8 — Probabilistic Forecasting & Uncertainty-Aware Planning
**Goal:** An uncertainty-aware production planning system (Master Plan §19 Phase 8, Level 6).

**Build**
- 8.1 Generate probabilistic forecasts:
  - Quantile regression / quantile gradient boosting (predict q10, q50, q90).
  - Conformal prediction wrappers for calibrated intervals.
  - Evaluate interval coverage and calibration (Master Plan §16).
- 8.2 Implement uncertainty-aware planner:
  - Scenario-based stochastic optimization (sample demand scenarios from quantiles), or
  - Robust optimization (optimize against q90 demand), or
  - Chance-constrained service level (e.g., P(stockout) ≤ 5%).
- 8.3 Strategy E: probabilistic forecast + uncertainty-aware optimizer, integrated into the rolling controller and simulation.

**Use**
- Compare Strategy E vs. Strategy D (point forecast) in the same SimPy environment.

**Test (Exit Criteria)**
- [ ] Prediction intervals achieve nominal coverage (e.g., 80% interval covers ~80% of actuals) within ±5%.
- [ ] Strategy E achieves target service level (e.g., 95%) at lower total cost than Strategy D, or documents the trade-off explicitly (Research Question 4).

---

### PHASE 9 — Experimental Analysis & Reporting
**Goal:** Value-of-forecasting study and final deliverables (Level 7).

**Build**
- 9.1 Cost-of-forecast-error analysis (Master Plan §17): decompose errors into over- vs. under-forecast and map to operational costs.
- 9.2 Sensitivity studies:
  - Capacity tightness (Research Question 7).
  - Planning horizon (RQ5).
  - Service-level target vs. inventory cost (RQ8).
- 9.3 Write final report in `reports/`:
  - Methodology, factory assumptions, results, policy comparison table (Master Plan §25 format), conclusions.
- 9.4 Build final demonstration: 30-day rolling run visualization (dashboard or animated notebook) showing forecast → plan → actual → outcome per Master Plan §25.

**Use**
- Portfolio-ready artifact: report + reproducible code + demo.

**Test (Exit Criteria)**
- [ ] All numbers in the report trace back to experiment outputs (no fabricated values).
- [ ] A fresh user can reproduce headline results by running `README.md` instructions.
- [ ] Report explicitly answers Research Questions 1–2 (minimum) with evidence.

---

## 3. Testing & Quality Assurance Strategy

| Layer | What | When |
|---|---|---|
| Unit tests | Feature leakage, metric correctness, planner constraint satisfaction | Every phase |
| Integration tests | Forecast → planner → rolling loop end-to-end | Phases 5, 6 |
| Deterministic sanity tests | Known-demand optima, zero-demand trivial cases | Phase 4 |
| Reproducibility tests | Same seed → same results across reruns | All phases |
| Monte Carlo validation | 30+ replications, CIs on all operational metrics | Phase 7+ |
| Leakage audits | Automated check that optimizer input timestamps ≤ decision time | Phases 5–9 |

---

## 4. Risk Register & Mitigations

| Risk | Likelihood | Mitigation |
|---|---|---|
| ML models fail to beat ETS | Medium | Document as finding (Master Plan §7 explicitly allows this); rely on ETS for integration |
| OR-Tools model too slow at scale | Low–Medium | Start with CP-SAT; fall back to LP relaxation or heuristics; reduce horizon |
| Public dataset lacks manufacturing constraints | High (expected) | Clearly separate "observed data" from "assumed parameters" (Master Plan §6) |
| Simulation results noisy / inconclusive | Medium | Increase replications; use common random numbers across strategies; report CIs |
| Scope creep (deep learning too early) | Medium | Gate Level-4 models behind a "beat ETS by X%" criterion |
| Time overrun | Medium | Level 5 is a valid stopping point; Phases 8–9 are extensions |

---

## 5. Definition of Done (per Phase)

A phase is complete only when:
1. All **Build** items are implemented and committed to Git.
2. All **Test (Exit Criteria)** checkboxes pass.
3. A short phase summary is added to `reports/progress_log.md`.
4. Notebooks run top-to-bottom without errors (`Kernel → Restart & Run All`).

---

## 6. Milestone Summary

| Milestone | Phases | Output |
|---|---|---|
| M1 — Data & Baselines Ready | 0–2 | Demand characterization memo + baseline pipeline |
| M2 — Strong Forecaster | 3 | ML forecaster beating baselines |
| M3 — Optimizer Working | 4 | Validated production/inventory optimizer |
| M4 — Closed Loop (Level 4) | 5–6 | Rolling forecast→optimize→execute system |
| M5 — Simulation Validated (Level 5) | 7 | Policy comparison under uncertainty with CIs |
| M6 — Uncertainty-Aware (Level 6) | 8 | Probabilistic planning system |
| M7 — Research Complete (Level 7) | 9 | Final report + demonstration |
