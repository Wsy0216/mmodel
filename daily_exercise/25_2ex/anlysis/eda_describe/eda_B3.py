# ---------- eda_B3.py ----------
import matplotlib
matplotlib.use("TKAgg")         # 后端
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd, numpy as np
from pathlib import Path
import os
sns.set_style("whitegrid")
plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.rcParams["figure.dpi"] = 110
os.makedirs("../figs", exist_ok=True)

DATA_RAW   = Path("../clean_data/tidy_emr_ae.xlsx")
DATA_FEATS = Path("../clean_data/feats_emr.parquet")   # ← 训练阶段保存的特征表

# ---------- 1. 读取 ----------
df_raw   = pd.read_excel(DATA_RAW, sheet_name="EMR",  # 只看 EMR
                         parse_dates=["time"])[["time", "class"]]

feats    = pd.read_parquet(DATA_FEATS)               # time 索引 + 特征列
feats.reset_index(inplace=True)

# 合并标签
df = feats.merge(df_raw, on="time", how="left")
df["y"] = (df["class"] == "C").astype(int)           # 1=C, 0=其他

# ---------- 2. 选特征 ----------
top6 = [
    "rms_EMR", "max_change_EMR", "rms_diff_EMR",
    "absdiff_sum_EMR", "kurt_EMR", "min_EMR"
]   # ← 你可以用 B-2 的排序结果

# ---------- 3. 画图 ----------
fig, axes = plt.subplots(2, 3, figsize=(13, 6))
axes = axes.ravel()

for ax, col in zip(axes, top6):
    sns.violinplot(x="y", y=col, data=df,
                   inner=None, palette=["#9ecae1", "#fc9272"],
                   ax=ax)
    sns.boxplot(x="y", y=col, data=df,
                width=.25, showcaps=False, fliersize=1.5,
                boxprops={"facecolor":"white"}, ax=ax)
    ax.set_xlabel("")                       # 去掉 x label
    ax.set_xticklabels(["non-C", "C"])
    ax.set_title(col.replace("_", " "))
    ax.grid(axis="x", visible=False)

fig.suptitle("B-3  Top-features distribution   (C  vs  non-C)", fontsize=15, y=1.02)
plt.tight_layout()
fig.savefig("figs/B3_feature_box_violin.png", dpi=300)
plt.show()
# --------------------------------
