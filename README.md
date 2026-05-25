# Day-Ahead LMP Price Forecaster
**MISO Illinois Hub | XGBoost | 2023–2024**

## Project Summary
Forecasts day-ahead Locational Marginal Prices (LMP) for MISO's 
Illinois Hub using XGBoost with 33 engineered features across 
temporal, autoregressive, rolling statistics, and market signal groups.

## Results
| Metric | Value |
|--------|-------|
| Test MAE | $5.99/MWh |
| Test RMSE | $11.80/MWh |
| Walk-forward MAE (8 folds) | $5.64 ± $1.21/MWh |
| Skill vs Seasonal Baseline | +39.1% |
| Skill vs Rolling Mean | +39.3% |

## Key Findings
- `lmp_lag_24h` is the dominant feature ($5.69/MWh avg SHAP impact)
- Model performs well in normal conditions (Fold 3 MAE: $4.54/MWh)
- Expected underperformance during extreme weather (polar vortex Jan 2024)
- Non-linear price momentum detected above $50/MWh threshold via SHAP

## Project Structure
- `notebooks/` — Step-by-step analysis notebooks
- `src/`       — Reusable Python modules
- `data/`      — Raw and processed MISO data
- `results/`   — Charts, predictions, backtest results
- `models/`    — Saved XGBoost model (.json)

## Stack
Python · XGBoost · SHAP · pandas · gridstatus · scikit-learn

## Data Source
MISO Market Reports via gridstatus library

## Roadmap — Planned Enhancements

| Enhancement | Effort | Impact |
|-------------|--------|--------|
| Add weather features (Open-Meteo API) | Medium | High |
| Separate models per LMP component | High | High |
| Quantile regression (predict price range) | Medium | High |
| Gas price features (EIA API) | Low | Medium |
| More nodes (Minnesota, Michigan Hub) | Low | Medium |