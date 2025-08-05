# -----------------------------  C-3：ROC 曲线  -----------------------------
# 1. 读取训练阶段保存的时间索引  idx  和预测概率 p_pred
# 2. 与原始标签表重建 y_true（保持 1-min 采样、±1 min rolling 逻辑）
# 3. 绘制 ROC 并保存                                    —— 2025/07/26
# --------------------------------------------------------------------------

import os
from pathlib import Path

import matplotlib
matplotlib.use("TKAgg")              # 兼容无 GUI 环境
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import RocCurveDisplay, roc_auc_score

# ---------- 路径 ----------
DATA_RAW = Path("../clean_data/tidy_emr_ae.xlsx")      # 清洗后数据（EMR sheet 有 class）
IDX_CSV  = Path("../models/index_emr_train.csv")       # 训练时保存的索引
PRED_NPY = Path("../models/p_emr_train.npy")           # 训练时保存的概率
FIG_OUT  = Path("../figs/C3_roc_curve.png")
os.makedirs(FIG_OUT.parent, exist_ok=True)

# ---------- STEP 0：读取概率 & 索引 ----------
idx = pd.read_csv(IDX_CSV, header=None)[0]          # 一列字符串
idx = idx[idx != "time"]                            # 训练脚本首行是列名，剔掉
idx = pd.to_datetime(idx)                           # → DatetimeIndex

p_pred = np.load(PRED_NPY)                          # numpy (N,)

# ---------- STEP 1：重建 1-min 级 y_true ----------
WIN  = pd.Timedelta("2min")     # ±1 min rolling；要与训练脚本一致
STEP = pd.Timedelta("60s")      # 1-min 采样

df = (pd.read_excel(DATA_RAW, sheet_name="EMR", parse_dates=["time"])
        .loc[:, ["time", "class"]])
df["y"] = df["class"].map({"C": 1, "A": 0, "B": 0})

y30 = df.set_index("time")["y"]                          # 原始 30 s 标签
y_win = (y30.rolling(WIN, min_periods=1).max()           # 窗内出现 C→1
              .resample(STEP).last())                    # 1-min 重采样

y_true = y_win.reindex(idx).astype(int).values           # 与 idx 对齐

# ---------- STEP 2：画 ROC ----------
auc = roc_auc_score(y_true, p_pred)
disp = RocCurveDisplay.from_predictions(
    y_true, p_pred, name=f"EMR-only model  (AUC = {auc:.3f})"
)

plt.title("C-3  ROC curve  (training set)")
plt.tight_layout()
plt.savefig(FIG_OUT, dpi=300)
plt.show()

# 若要在终端打印 AUC，可取消下一行
print(f"ROC-AUC = {auc:.3f}")
