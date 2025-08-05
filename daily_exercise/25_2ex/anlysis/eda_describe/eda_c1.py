# ----------------------------- C-1：PR 曲线 -----------------------------
# 1. 读取训练阶段保存的索引与预测概率
# 2. 按 “1-min 采样” 粒度重新生成真实标签 y_true
# 3. 绘制 Precision-Recall 曲线并保存
# ----------------------------------------------------------------------

import os
from pathlib import Path
import matplotlib
matplotlib.use("TKAgg")          # 没装 Tk 的话用 "Agg"
import matplotlib.pyplot as plt

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import PrecisionRecallDisplay

# ---------- 路径配置 ----------
DATA_RAW   = Path("../clean_data/tidy_emr_ae.xlsx")       # 原始（清洗后）数据
IDX_CSV    = Path("../models/index_emr_train.csv")        # 训练阶段保存的时间索引
PRED_NPY   = Path("../models/p_emr_train.npy")            # 训练阶段保存的预测概率
FIG_OUT    = Path("../figs/C1_pr_curve.png")              # 输出图片

os.makedirs(FIG_OUT.parent, exist_ok=True)             # figs 目录

# ---------- STEP 0：读取概率 & 索引 ----------
idx = pd.read_csv(IDX_CSV).iloc[:, 0]          # 自动把第一行当 header
idx = pd.to_datetime(idx, format="%Y-%m-%d %H:%M:%S")   # 明确格式更快更稳                     # → DatetimeIndex
p_pred   = np.load(PRED_NPY)                           # numpy array(N,)

# ---------- STEP 1：重新生成 1-min 级真实标签 ----------
df = (pd.read_excel(DATA_RAW, sheet_name="EMR", parse_dates=["time"])
        .loc[:, ["time", "class"]])

df["y"] = df["class"].map({"C": 1, "A": 0, "B": 0})

# 若训练索引是整分钟，这里先对齐
df["time"] = df["time"].dt.floor("min")

WIN  = pd.Timedelta("2min")      # ±1 min
STEP = pd.Timedelta("60s")       # 1-min

y30 = df.set_index("time")["y"]
y_win = (y30.rolling(WIN, min_periods=1).max()
                 .resample(STEP).last())

# 用 fill_value=0 防掉 NaN，再转 int
y_true = (
    y_win.reindex(idx, fill_value=0)   # ← 关键改动
         .values.astype(int)
)

# ---------- STEP 2：绘制 PR-Curve ----------
disp = PrecisionRecallDisplay.from_predictions(
            y_true, p_pred, name="EMR-only model")

plt.title("C-1  Precision–Recall curve  (training set)")
plt.tight_layout()
plt.savefig(FIG_OUT, dpi=300)
plt.show()

# 若想在控制台打印 PR-AUC 数字，可取消下一行注释
# print(f"PR-AUC = {disp.average_precision:.3f}")
