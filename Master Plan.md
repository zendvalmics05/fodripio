# Project 1 — Forecast-Driven Production Planning and Inventory Optimization Under Uncertain Demand

## 1. Project concept

The project is an industrial/production-engineering project combining **time-series forecasting, machine learning, operations research, inventory management, production planning, and discrete-event simulation**.

The central idea is:

> Build a decision-support system that forecasts future product demand and uses those forecasts to determine how much a factory should produce and how much inventory it should maintain, while respecting realistic production, material, capacity, and inventory constraints.

The project should NOT be treated as a generic “demand forecasting with machine learning” project. The forecasting model is only one component. The real objective is to demonstrate the complete chain:

**historical operational data → demand forecast → forecast uncertainty → production/inventory optimization → simulated factory operation → performance evaluation**

The central engineering question is:

> **Can better demand forecasts lead to better production and inventory decisions, and under what conditions does the additional forecasting sophistication actually produce economic value?**

The project should therefore evaluate the entire decision-making system rather than optimizing forecasting accuracy in isolation.

---

## 2. Why this project is being developed

The project is intended to build a portfolio piece at the intersection of:

- Production Engineering
- Industrial Engineering
- Operations Research
- Supply-chain/operations management
- Time-series analysis
- Machine learning
- Optimization
- Discrete-event simulation
- Industry 4.0 / intelligent manufacturing

The project is particularly useful because it connects prediction to an actual industrial decision.

A weak version of the project would say:

> “I trained ARIMA, XGBoost and LSTM models to forecast product demand.”

A much stronger version would say:

> “I developed a forecast-driven production planning system that predicts uncertain product demand and dynamically optimizes production and inventory decisions under capacity, material, lead-time and service-level constraints, then evaluated the resulting policies in a simulated manufacturing environment.”

The second formulation demonstrates substantially more engineering understanding.

---

## 3. Core problem statement

Consider a manufacturing plant producing multiple products.

For each product \(p\), demand varies over time:

\[
D_{p,t}
\]

The factory observes historical demand and possibly other explanatory variables and produces a forecast:

\[
\hat D_{p,t+1:t+H}
\]

for a future planning horizon \(H\), such as 7, 14, or 30 days.

The factory must then determine production quantities:

\[
Q_{p,t}
\]

and inventory levels:

\[
I_{p,t}
\]

while satisfying operational constraints.

A basic inventory balance is:

\[
I_{p,t}
=
I_{p,t-1}
+
Q_{p,t}
-
D_{p,t}
\]

with appropriate extensions for lead times, backorders, safety stock, or lost sales.

The production plan should minimize an economic objective such as:

\[
\text{Total Cost}
=
\text{Production Cost}
+
\text{Inventory Holding Cost}
+
\text{Shortage Cost}
+
\text{Setup Cost}
+
\text{Overtime Cost}
+
\text{Other Operational Costs}
\]

subject to realistic manufacturing constraints.

The fundamental question is therefore not merely:

> “What will demand be?”

but:

> **“Given uncertain future demand and finite manufacturing resources, what should the factory do?”**

---

## 4. Overall system architecture

The intended architecture is:

```text
Historical demand and operational data
                ↓
        Data preprocessing
                ↓
      Time-series analysis
                ↓
        Forecasting models
                ↓
     Point / probabilistic forecast
                ↓
       Forecast uncertainty
                ↓
 Production + inventory optimization
                ↓
       Production decisions
                ↓
       Factory simulation
                ↓
 Actual demand realization
                ↓
 Inventory / shortage / production results
                ↓
       Forecast is updated
                ↓
       Reoptimization occurs
```

The system should ideally operate in a **rolling-horizon/receding-horizon manner**.

At each decision period:

1. Observe newly available historical information.
2. Update/retrain or refresh the forecasting model.
3. Forecast demand for the next \(H\) periods.
4. Feed the forecast into the optimization model.
5. Generate a production/inventory plan.
6. Execute only the immediate portion of that plan.
7. Simulate/observe the next period.
8. Update the system with the newly observed demand.
9. Repeat.

This creates a simplified intelligent production-planning system rather than a static optimization exercise.

---

# 5. Factory model

The factory should be sufficiently realistic to demonstrate Production Engineering concepts without becoming unnecessarily complicated.

A reasonable initial factory could contain:

- 5–10 products
- 2–5 machines/workstations
- finite machine capacity
- several production routes
- raw-material constraints
- product-specific processing times
- setup/changeover times
- inventory
- production lead time
- customer demand
- shortage/backorder or lost-sales mechanism
- optional overtime
- production costs
- inventory holding costs
- shortage penalties

Example:

```text
Raw Materials
      ↓
   Factory
 ┌────┼────┐
 ↓    ↓    ↓
M1   M2   M3
 └────┼────┘
      ↓
 Finished Goods Inventory
      ↓
 Customer Demand
```

Products should consume different amounts of machine time and raw materials.

For example:

| Product | Machine 1 | Machine 2 | Machine 3 | Material A | Material B |
|---|---:|---:|---:|---:|---:|
| P1 | 0.20 h | 0.10 h | — | 2 kg | 1 kg |
| P2 | — | 0.30 h | 0.15 h | 1 kg | 3 kg |
| P3 | 0.15 h | — | 0.25 h | 3 kg | 2 kg |

These numbers are illustrative and should eventually be replaced with a coherent synthetic factory or a public industrial dataset where possible.

---

# 6. Demand data

There are two viable approaches.

## Approach A — Public real-world dataset

Use an appropriate public demand/sales dataset containing historical observations for multiple products or SKUs.

Advantages:

- real-world temporal patterns
- credible data
- less need to invent demand behavior
- easier demonstration of forecasting techniques

Disadvantages:

- the dataset may not contain manufacturing constraints
- production/inventory parameters may need to be constructed separately
- the resulting factory becomes partly synthetic

This is acceptable as long as the project clearly distinguishes **observed data** from **assumed operational parameters**.

## Approach B — Synthetic industrial dataset

Construct a realistic simulated demand environment.

Demand can contain:

- trend
- weekly seasonality
- annual/longer-term seasonality
- random variation
- promotions
- price effects
- correlated product demand
- occasional demand shocks
- changing variance

For example:

\[
D_t
=
\text{Trend}_t
+
\text{Seasonality}_t
+
\text{Exogenous Effects}_t
+
\epsilon_t
\]

The advantage is that the relationship between demand and the underlying process can be controlled.

A strong project could eventually use a public dataset for forecasting while using a synthetic factory for the optimization/simulation layer.

---

# 7. Forecasting component

The forecasting component should be developed progressively.

The goal is not to jump immediately to deep learning. The project should establish strong baselines and understand why more complicated models are or are not useful.

## Level 1 — Naïve baselines

Examples:

### Last-value forecast

\[
\hat D_{t+1}=D_t
\]

### Seasonal naïve

For weekly seasonality:

\[
\hat D_t=D_{t-7}
\]

These are important because sophisticated models should beat simple baselines to justify their complexity.

---

## Level 2 — Classical statistical models

Potential models:

- Moving average
- Weighted moving average
- Simple exponential smoothing
- Holt's method
- Holt-Winters / ETS
- ARIMA
- SARIMA
- potentially dynamic regression/state-space models

The exact models should depend on the characteristics discovered during exploratory analysis.

---

## Level 3 — Machine-learning forecasting

Transform the time series into supervised learning data using lagged and rolling features.

Potential features:

- \(D_{t-1}\)
- \(D_{t-2}\)
- \(D_{t-7}\)
- \(D_{t-14}\)
- \(D_{t-28}\)
- rolling mean
- rolling standard deviation
- rolling minimum/maximum
- exponentially weighted averages
- day of week
- month
- holiday indicator
- price
- promotion
- production information
- inventory information, where appropriate and causally valid

Potential models:

- Random Forest
- Gradient Boosting
- XGBoost
- LightGBM

Tree-based models are particularly useful because they provide a strong non-neural baseline and work well with heterogeneous temporal features.

---

## Level 4 — Neural/modern temporal models

Only after establishing strong statistical and tree-based baselines should neural models be considered.

Potential models:

- LSTM
- GRU
- Temporal Convolutional Network
- Transformer-style temporal model

The project should not use deep learning merely because it appears more sophisticated.

If XGBoost or another simpler model performs equally well while being easier to train, interpret, and deploy, that itself is an important engineering conclusion.

---

# 8. Time-series validation

Ordinary random train/test splitting must NOT be used for the forecasting problem.

The project should respect temporal ordering.

For example:

```text
Training              Validation       Test
|--------------------|----------------|--------->
       past                future
```

Better still, use rolling-origin evaluation:

```text
Train ───────→ Validate
Train ───────────→ Validate
Train ───────────────→ Validate
Train ──────────────────→ Validate
```

This reflects how the model would actually operate in production.

Metrics may include:

- MAE
- RMSE
- MAPE where appropriate
- sMAPE
- WAPE
- MASE

Metric choice should depend on the demand distribution. MAPE should not be blindly used when demand can approach zero.

The project should compare models not only on average forecasting accuracy but also on their effect on downstream operational decisions.

---

# 9. Probabilistic forecasting

This is an important potential extension.

A point forecast gives:

\[
\hat D_{t+1}=1,240.
\]

But production planning needs to understand uncertainty.

A probabilistic forecast might instead provide:

\[
P(D_{t+1})
\]

or forecast quantiles:

\[
Q_{0.10}, Q_{0.50}, Q_{0.90}.
\]

For example:

```text
10th percentile: 1,050
50th percentile: 1,220
90th percentile: 1,430
```

This information can be used to determine safety stock and robust production plans.

Potential approaches include:

- prediction intervals from classical models
- quantile regression
- quantile gradient boosting
- conformal prediction
- probabilistic state-space models
- other appropriate probabilistic forecasting methods

The exact methodology should be chosen after the basic project is working.

---

# 10. Production planning optimization

The optimization model is the core Operations Research component.

Decision variables may include:

\[
Q_{p,t}
\]

= quantity of product \(p\) produced in period \(t\)

\[
I_{p,t}
\]

= inventory of product \(p\)

\[
B_{p,t}
\]

= backlog/shortage

\[
X_{p,m,t}
\]

= machine allocation or processing quantity, if necessary.

The model can minimize:

\[
\min
\left[
\sum_{p,t} c^P_pQ_{p,t}
+
\sum_{p,t} c^I_pI_{p,t}
+
\sum_{p,t} c^B_pB_{p,t}
+
\sum_{p,t} c^S_pS_{p,t}
\right]
\]

where:

- \(c^P\) = production cost
- \(c^I\) = inventory holding cost
- \(c^B\) = shortage/backlog cost
- \(c^S\) = setup cost

Subject to inventory balance:

\[
I_{p,t}
=
I_{p,t-1}
+
Q_{p,t}
-
D_{p,t}
\]

or an appropriate forecast-based equivalent during planning.

---

# 11. Capacity constraints

For machine \(m\):

\[
\sum_p a_{p,m}Q_{p,t}
\leq C_{m,t}
\]

where:

- \(a_{p,m}\) = processing time required for product \(p\) on machine \(m\)
- \(C_{m,t}\) = available capacity.

Additional constraints can include:

- labor capacity
- raw-material availability
- warehouse capacity
- machine availability
- maximum production quantities
- minimum batch sizes
- production lead times
- setup/changeover times
- overtime limits

This makes the optimization meaningfully industrial.

---

# 12. Forecast and optimization interaction

This is one of the most important conceptual components.

The optimization model should not simply use actual future demand because that would create unrealistic information leakage.

At decision time \(t\), the optimizer only has access to:

\[
\hat D_{t+1:t+H}
\]

and any uncertainty information available at that time.

Then the actual demand:

\[
D_{t+1}
\]

is revealed only after the decision has been made.

This distinction must be preserved throughout the simulation.

The correct structure is:

```text
Past demand
    ↓
Forecast future demand
    ↓
Optimize using forecast
    ↓
Make production decision
    ↓
Actual demand occurs
    ↓
Measure consequences
    ↓
Update forecast
```

This is essential to avoid unrealistic performance results.

---

# 13. Rolling-horizon optimization

A useful operating framework is Model Predictive Control-like rolling optimization.

Suppose the planning horizon is 14 days.

At day \(t\):

1. Forecast demand for days \(t+1\) through \(t+14\).
2. Optimize production over those 14 days.
3. Execute today's production decision.
4. Move to day \(t+1\).
5. Obtain the new demand observation.
6. Reforecast.
7. Reoptimize.

Only the first decision is committed; future decisions remain flexible.

This allows the system to adapt to forecasting errors.

The architecture is:

\[
\boxed{
\text{Forecast}
\rightarrow
\text{Optimize}
\rightarrow
\text{Execute}
\rightarrow
\text{Observe}
\rightarrow
\text{Reforecast}
}
\]

---

# 14. Simulation component

A discrete-event simulation can be implemented using SimPy.

The simulated factory can contain:

- customer demand process
- production processes
- machines
- machine capacity
- queues
- inventory
- material arrivals
- production delays
- breakdowns if desired
- order fulfillment
- scheduling decisions

The simulation provides a realistic environment in which the forecasting/optimization policy can be tested.

This is important because an optimizer can appear excellent on paper while performing poorly once demand and operational uncertainty are introduced.

---

# 15. Experimental comparison

The project should compare several decision-making strategies.

For example:

### Strategy A — Fixed/simple production policy

Production based on historical average or fixed quantities.

### Strategy B — Naïve forecasting + optimization

Use a naïve or seasonal-naïve forecast as the input to the production optimizer.

### Strategy C — Statistical forecasting + optimization

Use ETS/SARIMA or similar forecasting methods.

### Strategy D — ML forecasting + optimization

Use XGBoost/LightGBM or another strong ML model.

### Strategy E — Probabilistic forecasting + uncertainty-aware optimization

Use forecast distributions/quantiles and an uncertainty-aware planning strategy.

The important question is:

> Does better forecast accuracy actually result in lower operational cost?

It is entirely possible that a forecasting model with slightly better MAE produces almost no improvement in total factory cost.

That is an important finding rather than a failure.

---

# 16. Evaluation metrics

Forecasting metrics:

- MAE
- RMSE
- sMAPE/WAPE
- MASE
- prediction interval coverage, if probabilistic forecasting is used
- calibration metrics where appropriate

Operational metrics:

- total cost
- production cost
- inventory holding cost
- shortage/backorder cost
- stockout frequency
- service level
- fill rate
- average inventory
- inventory variance
- capacity utilization
- machine utilization
- overtime
- production stability
- number of changeovers

The most important comparison is:

\[
\text{Forecast quality}
\quad\leftrightarrow\quad
\text{Operational performance}.
\]

The project should investigate whether improvements in forecast quality translate into measurable business/engineering improvements.

---

# 17. Cost of forecasting errors

A particularly valuable extension is to distinguish between different types of forecasting errors.

Overforecasting can cause:

- excess inventory
- unnecessary production
- capital tied up in stock
- warehouse requirements

Underforecasting can cause:

- stockouts
- lost sales
- backlog
- emergency production
- overtime

Therefore:

\[
\text{Forecast Error} \neq \text{Operational Consequence}
\]

in a simple one-to-one sense.

The economic cost of a +100-unit error can be very different from the cost of a −100-unit error.

This motivates asymmetric loss functions and probabilistic forecasting.

---

# 18. Possible research questions

The project can support several meaningful questions.

### Research Question 1

> Does improved demand forecasting necessarily produce improved production-planning performance?

### Research Question 2

> How much operational value is obtained by moving from classical statistical forecasting to machine-learning forecasting?

### Research Question 3

> How does demand uncertainty affect optimal inventory levels?

### Research Question 4

> Does probabilistic forecasting outperform point forecasting when used for production planning?

### Research Question 5

> How does the planning horizon affect production cost, inventory, and service level?

### Research Question 6

> How robust is the production plan to forecast errors?

### Research Question 7

> How does production capacity influence the value of forecasting accuracy?

### Research Question 8

> What is the trade-off between inventory cost and service level under uncertain demand?

### Research Question 9

> Does rolling-horizon reoptimization outperform static production planning?

These questions give the project academic depth if it later develops into a term paper or larger research project.

---

# 19. Possible project progression

The project should be developed incrementally rather than attempting the full system immediately.

## Phase 1 — Understand the demand data

Perform:

- data cleaning
- visualization
- trend analysis
- seasonality analysis
- autocorrelation
- stationarity analysis
- outlier analysis
- demand distribution analysis

Deliverable:

> A clear characterization of the demand-generating process.

---

## Phase 2 — Build forecasting baselines

Implement:

- naïve
- seasonal naïve
- moving average
- exponential smoothing
- Holt-Winters
- ARIMA/SARIMA where appropriate

Use proper time-series validation.

Deliverable:

> Reliable baseline forecasting pipeline.

---

## Phase 3 — Machine-learning forecasting

Build lag/rolling/exogenous features.

Try:

- Random Forest
- XGBoost/LightGBM

Compare against statistical baselines.

Deliverable:

> A strong forecasting model with a defensible evaluation procedure.

---

## Phase 4 — Production optimization

Construct the factory model.

Implement an OR-Tools/CP-SAT or suitable optimization model containing:

- production decisions
- inventory balance
- machine capacity
- material constraints
- shortage costs
- production costs
- inventory costs

Deliverable:

> A functioning production/inventory planning optimizer.

---

## Phase 5 — Connect forecasting to optimization

Feed the forecasts into the optimizer.

Compare:

```text
Naïve forecast → optimizer
Statistical forecast → optimizer
ML forecast → optimizer
```

Measure downstream operational performance.

Deliverable:

> Evidence of how forecasting quality affects production decisions.

---

## Phase 6 — Rolling-horizon planning

Implement:

```text
forecast
→ optimize
→ execute
→ observe
→ forecast again
→ optimize again
```

Deliverable:

> A dynamic forecast-driven production-planning system.

---

## Phase 7 — Simulation

Build a SimPy factory environment.

Introduce:

- stochastic demand
- capacity limitations
- production delays
- inventory
- possibly machine downtime

Evaluate the competing policies over many simulation runs.

Deliverable:

> Robust comparison of planning policies under uncertainty.

---

## Phase 8 — Probabilistic forecasting

Add:

- prediction intervals
- quantile forecasts
- uncertainty-aware planning

Compare against point forecasting.

Deliverable:

> An uncertainty-aware production planning system.

---

# 20. Technology stack

A reasonable stack is:

### Python

Primary language.

### pandas

Data manipulation and time-series feature engineering.

### NumPy

Numerical operations.

### matplotlib

Visualization.

### statsmodels

Classical statistical forecasting and time-series analysis.

### scikit-learn

ML models, preprocessing and evaluation utilities.

### XGBoost or LightGBM

Strong tree-based forecasting models.

### OR-Tools

Production/inventory optimization and potentially scheduling.

### SimPy

Discrete-event factory simulation.

### Jupyter

Exploration and experimentation.

### Git

Version control.

A database is not necessary initially. The project should prioritize the analytical/engineering problem rather than building unnecessary software infrastructure.

---

# 21. Recommended project structure

A possible repository structure is:

```text
forecast-production-planning/
│
├── data/
│   ├── raw/
│   ├── processed/
│   └── synthetic/
│
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_time_series_analysis.ipynb
│   ├── 03_baseline_forecasting.ipynb
│   ├── 04_ml_forecasting.ipynb
│   └── 05_results_analysis.ipynb
│
├── src/
│   ├── data/
│   ├── forecasting/
│   ├── optimization/
│   ├── simulation/
│   ├── evaluation/
│   └── visualization/
│
├── configs/
│   └── experiment.yaml
│
├── experiments/
│
├── reports/
│
├── tests/
│
├── requirements.txt
│
└── README.md
```

The exact structure can be simplified if necessary.

The goal is to keep the project understandable rather than creating software architecture for its own sake.

---

# 22. Important methodological principles

The project should follow several rules.

### No future information leakage

Forecasting features must contain only information available at prediction time.

### No random train/test split for temporal prediction

Use chronological or rolling validation.

### Strong baselines first

A complicated model should demonstrate a reason for existing.

### Separate forecasting evaluation from decision evaluation

A model can have better RMSE but produce worse production decisions.

Both should be measured.

### Actual future demand must remain hidden during planning

The optimizer should operate on forecasts, not actual future observations.

### Operational assumptions must be explicit

If machine capacities, costs, lead times, or setup times are synthetic, clearly label them as assumptions.

### Simulation should contain stochasticity

The system should be evaluated across multiple demand realizations/seeds rather than one lucky simulation.

---

# 23. Potential final project title

Possible titles include:

**Forecast-Driven Production Planning and Inventory Optimization Under Demand Uncertainty**

or:

**Demand Forecasting–Driven Production Planning for Manufacturing Systems Under Uncertainty**

or, for a more technical/academic framing:

**Integrating Time-Series Forecasting and Operations Research for Dynamic Production and Inventory Planning Under Uncertain Demand**

The third title best communicates the project's multidisciplinary nature.

---

# 24. What the project should demonstrate on a CV

The project should ultimately demonstrate five capabilities:

1. **Time-series forecasting**
   - statistical and ML forecasting
   - temporal validation
   - uncertainty estimation

2. **Production engineering**
   - capacity
   - inventory
   - production planning
   - resource constraints

3. **Operations research**
   - mathematical optimization
   - constrained production planning
   - cost minimization

4. **Simulation**
   - stochastic manufacturing environment
   - rolling-horizon decision making
   - policy comparison

5. **AI-to-decision integration**
   - forecast → optimization → operational outcome

The project should therefore be presented as an **intelligent decision-support system for manufacturing**, not simply as a forecasting model.

---

# 25. Possible final demonstration

A strong final demonstration could show a simulated 30-day factory run.

For each day, the system displays:

```text
Observed demand
↓
Forecast for next 14 days
↓
Forecast uncertainty
↓
Recommended production quantities
↓
Expected inventory
↓
Actual demand
↓
Resulting inventory/shortage
↓
Next-day reoptimization
```

A dashboard or report could compare different policies on:

- total cost
- service level
- inventory
- stockouts
- machine utilization
- production volume

For example:

```text
                     Total Cost    Service Level    Avg Inventory
Fixed policy             X              Y%               Z
Naïve + OR               X              Y%               Z
SARIMA + OR              X              Y%               Z
XGBoost + OR             X              Y%               Z
Probabilistic + OR       X              Y%               Z
```

The numbers should be generated by the actual experiment and never fabricated beforehand.

---

# 26. What would make this project genuinely impressive

The following progression represents increasing project quality:

```text
Level 1
Demand forecasting
        ↓
Level 2
Multiple forecasting models
        ↓
Level 3
Forecasting + production optimization
        ↓
Level 4
Rolling-horizon forecast + optimization
        ↓
Level 5
Forecast + optimization + SimPy factory
        ↓
Level 6
Probabilistic forecasting + uncertainty-aware optimization
        ↓
Level 7
Experimental analysis of the value of forecasting
```

The project does NOT need to reach Level 7 immediately.

A well-executed Level 4–5 project is already substantial.

Level 6–7 would make it much more appropriate for an academic/research-oriented project.

---

# 27. The most important conceptual takeaway

The project should be built around the distinction between **prediction and decision**.

Forecasting answers:

\[
\boxed{\text{What is likely to happen?}}
\]

Optimization answers:

\[
\boxed{\text{What should we do about it?}}
\]

Simulation answers:

\[
\boxed{\text{What happens when we actually implement that decision under uncertainty?}}
\]

The complete project therefore becomes:

\[
\boxed{
\text{Forecast}
\rightarrow
\text{Decision}
\rightarrow
\text{Action}
\rightarrow
\text{Outcome}
\rightarrow
\text{Feedback}
}
\]

This is the central design philosophy of the project.

The eventual goal is not to prove that one forecasting algorithm is superior to another. It is to investigate whether **better information about future demand can be converted into better manufacturing decisions**, while accounting for uncertainty, finite resources, inventory costs, and operational constraints.

That framing should be preserved when extending the project with other AI systems, optimization techniques, or simulation methods.