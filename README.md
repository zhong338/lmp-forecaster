# Day-Ahead LMP Price Forecaster
**MISO Illinois Hub | XGBoost | 2021–2024 data (trained on 2022–2023)**

## Project Summary
Forecasts day-ahead Locational Marginal Prices (LMP) for MISO's 
Illinois Hub using XGBoost with 33 engineered features across 
temporal, autoregressive, rolling statistics, and market signal groups.
Trained on 2022–2023 and evaluated on a held-out 2024, with a
2022–2024 walk-forward backtest.

## Results
| Metric | Value |
|--------|-------|
| 2024 holdout MAE | $5.88/MWh |
| 2024 holdout RMSE | $10.59/MWh |
| Walk-forward MAE (2022–2024, 8 folds) | $8.82 ± $4.16/MWh \* |
| Skill vs seasonal-naive (same hour last week) | +40.2% |
| Skill vs 7-day rolling mean | +40.4% |

\* The walk-forward spans the volatile **2022 energy-crisis regime** (gas-driven
price highs) with thin early-fold training, which dominates the average. On the
2024-only folds the model holds ~$5–7/MWh — consistent with the 2024 holdout.
The figure reflects a tougher multi-regime test, not a worse model.

## Key Findings
- `lmp_lag_24h` is the dominant feature (autoregressive signal; SHAP, notebook 06)
- Most accurate on recent normal periods (2024 H2 folds ~$5/MWh)
- **More data isn't automatically better:** the 2021 regime (Winter Storm Uri,
  post-COVID gas) *degrades* a 2024 forecast, so it is excluded from training —
  a fixed-test-set ablation confirmed 2022–2023 beats 2021–2023
- 2022 energy-crisis prices are the hardest to forecast (walk-forward $12–16/MWh)
- Expected underperformance during extreme weather (polar vortex Jan 2024)

## Model Comparison & Robustness (notebooks 07–09)
XGBoost was benchmarked against Multiple Linear Regression, Bagging, Random Forest,
and a DNN (`MLPRegressor`) on the same data and 2024 holdout:

- **Accuracy depends on the metric.** On **MAE**, XGBoost leads ($5.88), with Bagging,
  Multiple Linear Regression, and Random Forest all within $0.18/MWh; the DNN is weakest
  ($7.42). On **MSE/RMSE** the ranking *flips* — **Multiple Linear Regression is best**
  (MSE 98.1 vs XGBoost 112.2), because squared error punishes the extreme spike misses that
  the tree models can't extrapolate to.
- **Speed.** Multiple Linear Regression trains in ~0.02s and XGBoost in <1s; the DNN is
  ~40× slower to train.
- **Extreme conditions (08).** Every model's error blows up ~10–15× on spike hours. XGBoost
  is best in normal conditions but among the worst in the spikes (trees can't extrapolate
  beyond the training range); the linear model degrades least there.
- **Missing data (09).** Tree models (incl. XGBoost) predict through NaNs natively; Multiple
  Linear Regression and the DNN need a `SimpleImputer`. Counter-intuitively, when trained on
  clean data, XGBoost degrades *worst* as gaps grow while Random Forest/Bagging stay robust —
  "native NaN support" depends on having seen missingness during training.

## Project Structure
- `notebooks/` — Step-by-step analysis notebooks:
  - `01` data exploration · `02` feature engineering · `03` model training ·
    `04` walk-forward backtesting · `05` evaluation vs baselines ·
    `06` feature importance (SHAP) · `07` model comparison ·
    `08` normal vs extreme conditions · `09` missing-data robustness
- `src/`       — Reusable Python modules
- `data/`      — Raw and processed MISO data
- `results/`   — Charts, predictions, backtest results
- `models/`    — Saved XGBoost model (.json)

## Stack
Python · XGBoost · SHAP · pandas · gridstatus · gridstatusio · scikit-learn

## Data Source
2021–2024 MISO Day-Ahead Hourly LMP (`ILLINOIS.HUB`) via the
[gridstatus.io](https://www.gridstatus.io/) hosted API
(`miso_lmp_day_ahead_hourly`), normalized to fixed EST.

> MISO's free public daily market-report files (used by the `gridstatus`
> library) only retain ~2023 onward, so the hosted API is required to reach
> 2021–2022. Pull + feature build is automated in `src/data_pipeline.py`
> (needs a `GRIDSTATUS_API_KEY` env var):
>
> ```
> python src/data_pipeline.py --start 2021-01-01 --end 2025-01-01
> ```

## Roadmap — Planned Enhancements

| Enhancement | Effort | Impact |
|-------------|--------|--------|
| Add weather features (Open-Meteo API) | Medium | High |
| Separate models per LMP component | High | High |
| Quantile regression (predict price range) | Medium | High |
| Gas price features (EIA API) | Low | Medium |
| More nodes (Minnesota, Michigan Hub) | Low | Medium |
| Recency-weighted training to safely reuse 2021 | Low | Medium |