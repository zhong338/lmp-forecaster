"""
Feature engineering for the Day-Ahead LMP forecaster.

Ports the logic from notebooks/02_feature_engineering.ipynb into a reusable
function so the feature matrix can be rebuilt for any date range / location.

Input  : raw DA LMP dataframe with a tz-aware 'Time' column (or index) and
         columns LMP, Energy, Congestion, Loss.
Output : feature matrix indexed by Time, with the 33 model features plus the
         target (LMP) and the three raw components, matching feature_matrix_v1.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import holidays

# The four raw columns we carry through (target + components used for lags)
RAW_COLS = ["LMP", "Energy", "Congestion", "Loss"]


def build_features(df_raw: pd.DataFrame) -> pd.DataFrame:
    """Turn raw hourly DA LMP into the model-ready feature matrix.

    Parameters
    ----------
    df_raw : DataFrame
        Must contain LMP, Energy, Congestion, Loss and a tz-aware 'Time'
        column or a tz-aware DatetimeIndex.

    Returns
    -------
    DataFrame indexed by Time (sorted, NaN warmup rows dropped).
    """
    df = df_raw.copy()

    # --- Normalise the time index ---
    if "Time" in df.columns:
        df = df.set_index("Time")
    # Parse via UTC first so a CSV with mixed offsets (e.g. a DST seam) is
    # accepted, then express in fixed EST to match MISO market convention.
    idx = pd.to_datetime(df.index, utc=True).tz_convert("EST")
    df.index = idx
    df = df.sort_index()
    df = df[RAW_COLS]

    # --- GROUP A: TEMPORAL FEATURES ---
    df["hour"]        = df.index.hour
    df["day_of_week"] = df.index.dayofweek          # 0=Monday, 6=Sunday
    df["month"]       = df.index.month
    df["quarter"]     = df.index.quarter

    df["is_weekend"]  = df["day_of_week"].isin([5, 6]).astype(int)
    df["is_monday"]   = (df["day_of_week"] == 0).astype(int)

    df["is_summer"]   = df["month"].isin([6, 7, 8]).astype(int)
    df["is_winter"]   = df["month"].isin([12, 1, 2]).astype(int)

    df["is_peak_hour"]    = df["hour"].isin(range(16, 21)).astype(int)
    df["is_offpeak_hour"] = df["hour"].isin(range(0, 6)).astype(int)

    # Circular encoding (hour 23 and hour 0 are adjacent, not 23 apart)
    df["hour_sin"]  = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"]  = np.cos(2 * np.pi * df["hour"] / 24)
    df["dow_sin"]   = np.sin(2 * np.pi * df["day_of_week"] / 7)
    df["dow_cos"]   = np.cos(2 * np.pi * df["day_of_week"] / 7)
    df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)

    # --- GROUP B: LAGGED LMP FEATURES (min lag 24h to avoid leakage) ---
    df["lmp_lag_24h"]  = df["LMP"].shift(24)
    df["lmp_lag_48h"]  = df["LMP"].shift(48)
    df["lmp_lag_72h"]  = df["LMP"].shift(72)
    df["lmp_lag_168h"] = df["LMP"].shift(168)       # same hour last week

    df["congestion_lag_24h"]  = df["Congestion"].shift(24)
    df["congestion_lag_168h"] = df["Congestion"].shift(168)

    # Was there a price spike yesterday? (threshold from full-sample stats,
    # matching the original notebook).
    spike_threshold = df["LMP"].mean() + 2 * df["LMP"].std()
    df["spike_lag_24h"] = (df["lmp_lag_24h"] > spike_threshold).astype(int)

    # --- GROUP C: ROLLING STATISTICS (rolled on lagged LMP, no leakage) ---
    lmp_lagged = df["LMP"].shift(24)
    df["lmp_roll_24h_mean"] = lmp_lagged.rolling(24).mean()
    df["lmp_roll_48h_mean"] = lmp_lagged.rolling(48).mean()
    df["lmp_roll_7d_mean"]  = lmp_lagged.rolling(168).mean()
    df["lmp_roll_24h_std"]  = lmp_lagged.rolling(24).std()
    df["lmp_roll_7d_std"]   = lmp_lagged.rolling(168).std()
    df["lmp_roll_24h_max"]  = lmp_lagged.rolling(24).max()
    df["lmp_roll_24h_min"]  = lmp_lagged.rolling(24).min()
    df["lmp_vs_7d_avg"]     = df["lmp_lag_24h"] / df["lmp_roll_7d_mean"]

    # --- GROUP D: HOLIDAY FEATURES ---
    years = sorted(df.index.year.unique().tolist())
    us_holidays = holidays.US(years=years)
    dates_no_tz = df.index.tz_localize(None).normalize()
    df["is_holiday"] = dates_no_tz.isin(us_holidays).astype(int)
    df["is_day_before_holiday"] = dates_no_tz.isin(
        [d - pd.Timedelta(days=1) for d in us_holidays.keys()]
    ).astype(int)

    # Drop warmup rows (first ~168h have incomplete rolling history)
    df_features = df.dropna()
    return df_features


def feature_columns(df_features: pd.DataFrame) -> list[str]:
    """Columns the model trains on (everything except target + components)."""
    return [c for c in df_features.columns if c not in RAW_COLS]
