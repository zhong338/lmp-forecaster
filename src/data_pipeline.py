r"""
Data pipeline for the Day-Ahead LMP forecaster.

Pulls MISO Day-Ahead Hourly LMP from the gridstatus.io hosted API (which
retains deeper history than MISO's free daily market reports — those only go
back to 2023). Normalises the result to the project's raw-CSV schema, then
rebuilds the feature matrix.

Why the hosted API:
    MISO's public `da_expost_lmp` daily files 404 for 2021-2022, so the free
    `gridstatus.MISO().get_lmp()` path cannot reach those years. The hosted
    dataset can.

Usage:
    # one-time: get a free key at https://www.gridstatus.io/ and set it
    setx GRIDSTATUS_API_KEY "your-key-here"          # Windows (new shells)
    $env:GRIDSTATUS_API_KEY = "your-key-here"        # current PowerShell

    # pull 4 years and rebuild features
    python src/data_pipeline.py --start 2021-01-01 --end 2025-01-01

This writes:
    data/raw/da_lmp_illinois_hub_2021_2024.csv
    data/features/feature_matrix_v2.csv
"""

from __future__ import annotations

import argparse
import os
import sys

import pandas as pd

# Local import works whether run as `python src/data_pipeline.py` or imported
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from features import build_features, RAW_COLS  # noqa: E402

# gridstatus.io dataset id for MISO day-ahead hourly LMP.
# Verify against your account with client.list_datasets() if this 404s.
DATASET = "miso_lmp_day_ahead_hourly"

# Market timezone for the 'Time' index (drives hour-of-day features).
# MISO settles in fixed EST (no daylight-saving shifts), matching the original
# 2023-2024 raw file which is -05:00 year-round. Using "EST" keeps a single
# UTC offset, so the saved CSV has no DST seam.
MARKET_TZ = "EST"


def _get_client(api_key: str | None = None):
    """Build a GridStatusClient. Key comes from arg or GRIDSTATUS_API_KEY."""
    from gridstatusio import GridStatusClient

    api_key = api_key or os.environ.get("GRIDSTATUS_API_KEY")
    if not api_key:
        raise RuntimeError(
            "No API key. Set GRIDSTATUS_API_KEY or pass --api-key. "
            "Get a free key at https://www.gridstatus.io/"
        )
    return GridStatusClient(api_key=api_key)


def _normalize(df: pd.DataFrame) -> pd.DataFrame:
    """Map hosted-API columns onto the project's raw-CSV schema.

    Hosted columns are snake_case (lmp, energy, interval_start_local, ...).
    We produce: Time, Interval Start, Interval End, Market, Location,
    Location Type, LMP, Energy, Congestion, Loss.
    """
    lower = {c.lower().strip(): c for c in df.columns}

    def pick(*candidates):
        for cand in candidates:
            if cand in lower:
                return lower[cand]
        return None

    # Time index: prefer a local interval-start column; else convert UTC.
    t_local = pick("interval_start_local", "interval_start")
    t_utc = pick("interval_start_utc")
    if t_local is not None:
        time = pd.to_datetime(df[t_local])
        if time.dt.tz is None:
            time = time.dt.tz_localize(MARKET_TZ)
    elif t_utc is not None:
        time = pd.to_datetime(df[t_utc], utc=True).dt.tz_convert(MARKET_TZ)
    else:
        raise KeyError(f"No interval-start column found in {list(df.columns)}")

    end_local = pick("interval_end_local", "interval_end")
    end_utc = pick("interval_end_utc")
    if end_local is not None:
        iend = pd.to_datetime(df[end_local])
        if iend.dt.tz is None:
            iend = iend.dt.tz_localize(MARKET_TZ)
    elif end_utc is not None:
        iend = pd.to_datetime(df[end_utc], utc=True).dt.tz_convert(MARKET_TZ)
    else:
        iend = time + pd.Timedelta(hours=1)

    out = pd.DataFrame({
        "Time":          time,
        "Interval Start": time,
        "Interval End":   iend,
        "Market":        df[pick("market")] if pick("market") else "DAY_AHEAD_HOURLY",
        "Location":      df[pick("location")] if pick("location") else None,
        "Location Type": df[pick("location_type")] if pick("location_type") else None,
        "LMP":           df[pick("lmp")],
        "Energy":        df[pick("energy")],
        "Congestion":    df[pick("congestion")],
        "Loss":          df[pick("loss")],
    })
    return out.sort_values("Time").reset_index(drop=True)


def fetch_da_lmp(
    start: str,
    end: str,
    locations: list[str] | str = "ILLINOIS.HUB",
    api_key: str | None = None,
    dataset: str = DATASET,
) -> pd.DataFrame:
    """Fetch DA hourly LMP for one or more locations over [start, end).

    Returns a dataframe in the project's raw-CSV schema.
    """
    client = _get_client(api_key)
    if isinstance(locations, str):
        locations = [locations]

    frames = []
    for loc in locations:
        raw = client.get_dataset(
            dataset=dataset,
            start=start,
            end=end,
            filter_column="location",
            filter_value=loc,
            timezone=MARKET_TZ,
        )
        if raw is None or len(raw) == 0:
            raise RuntimeError(
                f"No rows returned for {loc} in [{start}, {end}). "
                "Check the date range is within your plan's history window."
            )
        frames.append(_normalize(raw))

    return pd.concat(frames, ignore_index=True).sort_values(
        ["Location", "Time"]
    ).reset_index(drop=True)


def build_raw_dataset(start, end, location, out_csv, api_key=None,
                      dataset=DATASET) -> pd.DataFrame:
    df = fetch_da_lmp(start, end, location, api_key=api_key, dataset=dataset)
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    df.to_csv(out_csv, index=False)
    print(f"Raw saved: {out_csv}  ({len(df):,} rows, "
          f"{df['Time'].min()} -> {df['Time'].max()})")
    return df


def rebuild_feature_matrix(raw_csv, out_csv) -> pd.DataFrame:
    raw = pd.read_csv(raw_csv, parse_dates=["Time"])
    feats = build_features(raw)
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    feats.to_csv(out_csv)
    print(f"Features saved: {out_csv}  ({len(feats):,} rows, "
          f"{len(feats.columns)} cols incl. target)")
    return feats


def main():
    p = argparse.ArgumentParser(description="Pull MISO DA LMP and build features.")
    p.add_argument("--start", default="2021-01-01")
    p.add_argument("--end", default="2025-01-01", help="exclusive upper bound")
    p.add_argument("--location", default="ILLINOIS.HUB")
    p.add_argument("--raw-out", default="data/raw/da_lmp_illinois_hub_2021_2024.csv")
    p.add_argument("--features-out", default="data/features/feature_matrix_v2.csv")
    p.add_argument("--api-key", default=None)
    p.add_argument("--dataset", default=DATASET)
    args = p.parse_args()

    build_raw_dataset(args.start, args.end, args.location, args.raw_out,
                      args.api_key, dataset=args.dataset)
    rebuild_feature_matrix(args.raw_out, args.features_out)


if __name__ == "__main__":
    main()
