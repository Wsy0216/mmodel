# preprocess.py
import pandas as pd, numpy as np
from pathlib import Path

SRC = Path("clean_data/tidy_emr_ae.xlsx")
DST = Path("clean_data/emr_ae_30s.parquet")

dfs = []
for sh in ["EMR","AE"]:
    t = pd.read_excel(SRC, sheet_name=sh, parse_dates=["time"])
    t["signal_type"] = sh
    dfs.append(t[["time","value","class","signal_type"]])

df = pd.concat(dfs).sort_values("time")
pivot = (df.pivot_table(index="time", columns="signal_type",
                        values="value")
           .asfreq("30s"))

pivot = (pivot.interpolate(limit=20, limit_direction="both")
               .fillna(method="ffill").fillna(method="bfill"))
pivot.to_parquet(DST)
print("saved ->", DST)
