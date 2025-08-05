# -----------------------------
# common header
# -----------------------------
import matplotlib
matplotlib.use("Agg")            # ≤ Win/远程最稳
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from pathlib import Path
import os

os.makedirs("../figs", exist_ok=True)

sns.set_style("whitegrid")
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": "Arial",   # Win 可换成 "微软雅黑"
    "figure.dpi": 110
})

# ---------------- READ & TAG ----------------
DATA = Path("../clean_data/tidy_emr_ae.xlsx")

# 一次性把 EMR、AE 两个 sheet 拼在一起 → df(time,value,sensor)
dfs = []
for sensor in ["EMR", "AE"]:
    if sensor in pd.ExcelFile(DATA).sheet_names:
        tmp = (pd.read_excel(DATA, sheet_name=sensor, parse_dates=["time"])
                 .loc[:, ["time", "value"]]          # 只要两列
                 .assign(sensor=sensor))
        dfs.append(tmp)

df = (pd.concat(dfs, ignore_index=True)
        .sort_values("time")
        .reset_index(drop=True))

# 顺手做 z-score，后面画图直接用
df["value_z"] = df.groupby("sensor")["value"].transform(
                    lambda s: (s - s.mean()) / s.std())

# 把 label 信息（class）并进来，只需要 EMR sheet 的 time/class
lbl = pd.read_excel(DATA, sheet_name="EMR",
                    usecols=["time", "class"], parse_dates=["time"])
df = df.merge(lbl, on="time", how="left")

WIN = pd.Timedelta(minutes=2)      # 画信号片段用（±2 min）

# ------------- A-1 事件 vs 正常 信号片段 -------------
# ① 事件窗口：选第一个 class == 'C'
idx_evt = df.index[df["class"] == "C"][0]
t0_evt  = df.loc[idx_evt, "time"]
seg_evt = df[(df["time"] >= t0_evt - WIN) & (df["time"] <= t0_evt + WIN)]

# ② 正常窗口：离 t0_evt 3*WIN 之外且无 C
idx_norm = df.index[(df["class"] != "C") &
                    (abs(df["time"] - t0_evt) > 3*WIN)].min()
t0_norm  = df.loc[idx_norm, "time"]
seg_norm = df[(df["time"] >= t0_norm - WIN) & (df["time"] <= t0_norm + WIN)]

# ③ 画双列对比图
fig, axes = plt.subplots(1, 2, figsize=(10, 3), sharey=True)

for ax, seg, title, t0 in zip(
        axes,
        [seg_evt, seg_norm],
        ["Event window (C)", "Normal window"],
        [t0_evt, t0_norm]):

    (seg.pivot(index="time", columns="sensor", values="value")
         .reindex(columns=["EMR", "AE"])
         .plot(ax=ax, linewidth=1.2))

    ax.set_title(title)
    ax.set_xlabel("time"); ax.set_ylabel("raw value")
    ax.axvline(t0, c="red", ls="--", lw=.8,
               label="event point" if title.startswith("Event") else "")
    ax.legend(loc="upper right")

plt.tight_layout()
plt.savefig("figs/A1_event_vs_normal.png", dpi=300)
plt.close()

# ------------- A-2  z-score 分布 ----------------------
fig, axes = plt.subplots(1, 2, figsize=(10, 3))

for ax, sensor in zip(axes, ["EMR", "AE"]):
    sns.histplot(df.loc[df["sensor"] == sensor, "value_z"],
                 kde=True, bins=80, ax=ax,
                 color="steelblue", edgecolor="none")
    ax.set_title(f"{sensor} value_z distribution")
    ax.set_xlabel("z-score"); ax.set_ylabel("count")

plt.tight_layout()
plt.savefig("figs/A2_valuez_hist_kde.png", dpi=300)
plt.close()

# ------------- A-3  每日缺失率热图 ---------------------
df_day = (df.groupby([df["time"].dt.date, "sensor"])
             ["value"].apply(lambda x: x.isna().mean())
             .unstack())                             # 行 = 日期；列 = 传感器

plt.figure(figsize=(6, 5))
sns.heatmap(df_day.T, cmap="YlOrRd", vmin=0, vmax=1,
            cbar_kws={"label": "missing rate"})
plt.xlabel("Date"); plt.ylabel("Sensor")
plt.title("Daily missing-rate matrix")
plt.tight_layout()
plt.savefig("figs/A3_missing_heatmap.png", dpi=300)
plt.close()

print("✅  A-1, A-2, A-3 三张图已生成并保存在 figs/ 目录。")
