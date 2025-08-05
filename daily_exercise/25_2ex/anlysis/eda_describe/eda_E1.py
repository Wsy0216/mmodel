# ---------------------------  E-1  Pred-prob vs. Time  ---------------------------
# 读训练集时间索引、预测概率 → 计算 1-min 真值 → 画长条时间线
# -------------------------------------------------------------------------------
import os
from pathlib import Path

import matplotlib
matplotlib.use("TKAgg")          # 无 GUI 后端，和之前一致
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# ---------- 文件路径 ----------
DATA_RAW = Path("../clean_data/tidy_emr_ae.xlsx")     # 原始 EMR sheet
IDX_CSV  = Path("../models/index_emr_train.csv")      # 训练阶段保存的索引
PRED_NPY = Path("../models/p_emr_train.npy")          # 训练阶段保存的概率
FIG_OUT  = Path("../figs/E1_prob_timeline.png")
os.makedirs(FIG_OUT.parent, exist_ok=True)

# ---------- STEP 0：载入数据 ----------
idx = (pd.read_csv(IDX_CSV, header=None)
         .iloc[:, 0]                # 取第一列
         .iloc[1:]                  # 若首行是 "time" 字符串就跳过；否则可删掉这行
         .pipe(pd.to_datetime)
         .rename("time"))
p_pred  = np.load(PRED_NPY)                    # (N,) 预测概率

# ---------- STEP 1：计算 1-min 真值 ----------
WIN  = pd.Timedelta("2min")   # ±1 min rolling 窗
STEP = pd.Timedelta("60s")    # 与训练一致

df_raw = (pd.read_excel(DATA_RAW, sheet_name="EMR", parse_dates=["time"])
            .loc[:, ["time", "class"]])
df_raw["y"] = df_raw["class"].map({"C": 1, "A": 0, "B": 0})

y30   = df_raw.set_index("time")["y"]
y_win = (y30.rolling(WIN, min_periods=1).max()
              .resample(STEP).last())
y_true = y_win.reindex(idx).fillna(0).astype(int).values  # 同长度 0/1 数组

# ---------- STEP 2：绘图 ----------
fig, ax1 = plt.subplots(figsize=(12, 3))

# (a) 预测概率曲线
ax1.plot(idx, p_pred, lw=0.6, color="#0057b7", label="Pred-prob")

# (b) 真实 C 事件用半透明色块标注
ax1.fill_between(idx, 0, 1,
                 where=y_true == 1,
                 color="orange", alpha=0.25, step="post",
                 label="true C")

ax1.set_title("E-1  Predicted probability vs. Time  (training set)")
ax1.set_xlabel("Time"); ax1.set_ylabel("Probability")
ax1.set_ylim(0, 1.02); ax1.legend(loc="upper right")
ax1.margins(x=0)                       # 去掉左右空白

plt.tight_layout()
plt.savefig(FIG_OUT, dpi=300)
plt.show()
