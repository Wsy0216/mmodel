# --------------------------- common header ---------------------------
import matplotlib
matplotlib.use("TKAgg")        # 无 GUI 后端（PyCharm 里避免 backend 报错）
import matplotlib.pyplot as plt

import os
from pathlib import Path
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

sns.set_style("whitegrid")
plt.rcParams["font.family"] = ["sans-serif"]
plt.rcParams["font.sans-serif"] = ["Arial"]   # Win 可换 “微软雅黑”
plt.rcParams["figure.dpi"] = 110
os.makedirs("../figs", exist_ok=True)

# --------------------------- I/O ---------------------------
DATA_RAW = Path("../clean_data/tidy_emr_ae.xlsx")
IDX_CSV  = Path("../models/index_emr_train.csv")
PRED_NPY = Path("../models/p_emr_train.npy")

idx = (
    pd.read_csv(IDX_CSV, usecols=[0])   # 默认 header=0，把首行当列名
      .iloc[:, 0]                       # 取第一列 Series
      .pipe(pd.to_datetime, format="%Y-%m-%d %H:%M:%S")  # 如有固定格式最好显式写
)

p_pred = np.load(PRED_NPY)

# ------------- 生成 1-min 粒度真实标签（与训练脚本一致） -------------
WIN  = pd.Timedelta("2min")
STEP = pd.Timedelta("60s")

df = (pd.read_excel(DATA_RAW, sheet_name="EMR", parse_dates=["time"])
        .loc[:, ["time", "class"]])
df["y"] = df["class"].map({"C": 1, "A": 0, "B": 0})

y30   = df.set_index("time")["y"]
y_win = (y30.rolling(WIN, min_periods=1).max()
               .resample(STEP).last())
y_true = y_win.reindex(idx).astype(int).values
# --------------------------------------------------------------------
# ---------------------- D-1  Confusion Matrix -----------------------
thr = 0.98            # 来自 C-2 F1 峰值
y_pred = (p_pred >= thr).astype(int)

cm  = confusion_matrix(y_true, y_pred, labels=[1, 0])
disp = ConfusionMatrixDisplay(
            confusion_matrix=cm,
            display_labels=["C (pos)", "non-C (neg)"])

fig, ax = plt.subplots(figsize=(4.5, 4.5))
disp.plot(ax=ax, cmap="Blues", colorbar=False, values_format="d")
ax.set_title(f"D-1  Confusion matrix  (threshold={thr:.2f})")
plt.tight_layout()
plt.savefig("figs/D1_confusion_matrix.png", dpi=300)
plt.show()
# ------------------------ D-2  score hist/CDF -----------------------
fig, axes = plt.subplots(1, 2, figsize=(10, 4))

# 直方 + KDE
sns.histplot(p_pred[y_true == 1], ax=axes[0],
             color="tomato", label="C", bins=80, stat="density", kde=True)
sns.histplot(p_pred[y_true == 0], ax=axes[0],
             color="steelblue", label="non-C", bins=80, stat="density", kde=True,
             alpha=.6)
axes[0].axvline(thr, ls="--", lw=1, c="black")
axes[0].set_title("D-2  Score distribution")
axes[0].set_xlabel("Predicted probability"); axes[0].set_ylabel("Density")
axes[0].legend()

# 经验累积分布
def ecdf(arr):
    arr = np.sort(arr)
    y   = np.arange(1, len(arr)+1) / len(arr)
    return arr, y

for cls, col in [(1, "tomato"), (0, "steelblue")]:
    x, y = ecdf(p_pred[y_true == cls])
    axes[1].plot(x, y, color=col, label="C" if cls==1 else "non-C")
axes[1].axvline(thr, ls="--", lw=1, c="black")
axes[1].set_title("Empirical CDF"); axes[1].set_xlabel("Predicted probability")
axes[1].set_ylabel("Cumulative share"); axes[1].legend()

plt.tight_layout()
plt.savefig("figs/D2_score_hist_ecdf.png", dpi=300)
plt.show()
