# -----------------------------  C-2：阈值-指标折线  -----------------------------
# 1. 读取上一步训练保存的时间索引 & 预测概率
# 2. 重新生成 1-min 粒度真实标签 y_true（和训练脚本保持一致）
# 3. 计算不同阈值下的 Precision / Recall / F1
# 4. 绘图保存
# ------------------------------------------------------------------------------

import os
from pathlib import Path

import matplotlib
matplotlib.use("TKAgg")              # 头部强制无 GUI 后端
import matplotlib.pyplot as plt

import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_curve

# ---------- 路径 ----------
DATA_RAW = Path("../clean_data/tidy_emr_ae.xlsx")          # 原始清洗后数据
IDX_CSV  = Path("../models/index_emr_train.csv")           # 训练阶段保存的索引
PRED_NPY = Path("../models/p_emr_train.npy")               # 预测概率
FIG_OUT  = Path("../figs/C2_threshold_metrics.png")

os.makedirs(FIG_OUT.parent, exist_ok=True)

# ---------- STEP 0：读概率 & 索引 ----------
idx   = pd.read_csv(IDX_CSV)["time"]            # csv 第一列就是 time 字段
idx   = pd.to_datetime(idx)                     # → DatetimeIndex
p_pred = np.load(PRED_NPY)                      # numpy array (N,)

# ---------- STEP 1：1-min 级真实标签 ----------
df = (pd.read_excel(DATA_RAW, sheet_name="EMR", parse_dates=["time"])
        .loc[:, ["time", "class"]])

df["y"] = df["class"].map({"C": 1, "A": 0, "B": 0})

WIN  = pd.Timedelta("2min")     # 与训练脚本保持一致
STEP = pd.Timedelta("60s")

y30   = df.set_index("time")["y"]
y_win = (y30.rolling(WIN, min_periods=1).max()
              .resample(STEP).last())

y_true = y_win.reindex(idx).values.astype(int)

# ---------- STEP 2：计算阈值-指标 ----------
prec, rec, thr = precision_recall_curve(y_true, p_pred)
# precision_recall_curve 的 precision / recall 比 thresholds 长 1
prec  = prec[:-1]
rec   = rec[:-1]
f1    = 2 * prec * rec / (prec + rec + 1e-12)   # F1 每个阈值

# ---------- STEP 3：绘图 ----------
plt.figure(figsize=(7, 4))
plt.plot(thr, prec,  label="Precision", lw=1.6)
plt.plot(thr, rec,   label="Recall",    lw=1.6)
plt.plot(thr, f1,    label="F1-score",  lw=1.6)

plt.gca().invert_xaxis()    # 阈值从高→低更直观
plt.xlabel("Threshold")
plt.ylabel("Metric value")
plt.title("C-2  Precision / Recall / F1  vs.  Threshold  (training set)")
plt.legend()
plt.grid(alpha=.3)
plt.tight_layout()
plt.savefig(FIG_OUT, dpi=300)
plt.show()
