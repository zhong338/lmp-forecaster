r"""
One-off probe: confirm the gridstatus.io dataset id, column names, and whether
your plan can reach 2021 history. Reads GRIDSTATUS_API_KEY from the environment
so your key never has to be pasted anywhere.

Run (PowerShell):
    $env:GRIDSTATUS_API_KEY = "your-key-here"
    venv\Scripts\python.exe scripts\explore_gridstatusio.py
"""

import os
from gridstatusio import GridStatusClient

key = os.environ.get("GRIDSTATUS_API_KEY")
if not key:
    raise SystemExit("Set GRIDSTATUS_API_KEY first (see docstring).")

client = GridStatusClient(api_key=key)

# 1) Which MISO LMP datasets exist on your account?
print("=== MISO LMP datasets ===")
try:
    ds = client.list_datasets()
    df = ds if hasattr(ds, "itertuples") else None
    if df is not None:
        ids = [c for c in df.columns if "id" in c.lower() or "name" in c.lower()]
        col = ids[0] if ids else df.columns[0]
        for v in df[col]:
            if "miso" in str(v).lower() and "lmp" in str(v).lower():
                print("  ", v)
    else:
        print("  (list_datasets returned non-tabular; inspect manually)")
except Exception as e:
    print("  list_datasets failed:", e)

DATASET = "miso_lmp_day_ahead_hourly"

# 2) Tiny pulls: does 2021 work? does 2023 work? what columns come back?
for label, start, end in [("2021", "2021-01-04", "2021-01-05"),
                          ("2023", "2023-01-04", "2023-01-05")]:
    print(f"\n=== {DATASET} {label} probe ===")
    try:
        d = client.get_dataset(
            dataset=DATASET,
            start=start,
            end=end,
            filter_column="location",
            filter_value="ILLINOIS.HUB",
            tz="America/New_York",
        )
        print(f"  rows={len(d)}  cols={list(d.columns)}")
        if len(d):
            print(d.head(2).to_string())
    except Exception as e:
        print(f"  {label} FAILED:", str(e)[:200])
