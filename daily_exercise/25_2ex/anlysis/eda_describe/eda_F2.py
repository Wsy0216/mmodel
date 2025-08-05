# ----------------------------- F-2  Learning Curve -----------------------------
import os, warnings
from pathlib import Path

import matplotlib
matplotlib.use("TKAgg")              # 无 GUI 环境也能出图
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import lightgbm as lgb
from lightgbm import LGBMClassifier
from sklearn.model_selection import TimeSeriesSplit

from feature_utils2 import slide_features, WIN, STEP     # ← 你的工具
warnings.filterwarnings("ignore", category=UserWarning)

# ---------- 路径 ----------
DATA = Path("../clean_data/tidy_emr_ae.xlsx")
FIG  = Path("../figs/F2_learning_curve.png")
os.makedirs(FIG.parent, exist_ok=True)

# ---------- 数据准备 ----------
df = (pd.read_excel(DATA, sheet_name="EMR", parse_dates=["time"])
        .loc[:, ["time", "value", "value_z", "class"]])
df["y"] = df["class"].map({"C": 1, "A": 0, "B": 0})

X = slide_features(df).add_suffix("_EMR")
y30 = df.set_index("time")["y"]
y   = (y30.rolling(WIN, min_periods=1).max()      # 窗口内出现 C→1
           .resample(STEP).last()
           .reindex(X.index)
           .fillna(0)              # 直接把 NaN 当非事件；若想删除 NaN → .dropna()
           .astype(int))

# 丢掉坏列
bad = [c for c in X if X[c].isna().all() or X[c].std() == 0]
X   = X.drop(columns=bad).fillna(X.median())

# ---------- 交叉验证 ----------
tscv = TimeSeriesSplit(n_splits=5)

params = dict(
    objective="binary",
    n_estimators=5000,
    learning_rate=0.03,
    num_leaves=255,
    min_data_in_leaf=20,
    subsample=0.7,
    colsample_bytree=0.8,
    scale_pos_weight=(y == 0).sum() / (y == 1).sum(),
    random_state=42,
    verbose=-1,
)

evals_all  = []
best_iters = []

for fold, (tr, te) in enumerate(tscv.split(X), 1):
    model = LGBMClassifier(**params)

    model.fit(
        X.iloc[tr], y.iloc[tr],
        eval_set=[(X.iloc[tr], y.iloc[tr]), (X.iloc[te], y.iloc[te])],
        eval_metric="binary_logloss",
        callbacks=[lgb.early_stopping(100, verbose=False)],
    )

    best_iters.append(model.best_iteration_)

    # ---- 新增：直接拿模型自带的评估结果 ----
    res = model.evals_result_

    train_key, valid_key = list(res.keys())[:2]
    metric_name = next(iter(res[train_key]))  # 通常是 'binary_logloss' 或 'auc'

    train_loss = res[train_key][metric_name][: model.best_iteration_]
    valid_loss = res[valid_key][metric_name][: model.best_iteration_]

    evals_all.append(
        pd.DataFrame({
            "iter": np.arange(1, model.best_iteration_ + 1),
            "train": train_loss,
            "valid": valid_loss,
            "fold": fold,
        })
    )

# ---------- 聚合（对每个迭代轮取中位数） ----------
df_lr = (pd.concat(evals_all)
           .set_index(["fold", "iter"])
           .groupby("iter")
           .median()
           .reset_index())

median_best = int(np.median(best_iters))

# ---------- 绘图 ----------
sns.set_style("whitegrid")
plt.figure(figsize=(6, 4))
plt.plot(df_lr["iter"], df_lr["train"], label="Train log-loss")
plt.plot(df_lr["iter"], df_lr["valid"], label="Valid log-loss")
plt.axvline(median_best, ls="--", lw=1, c="red",
            label=f"median best_iter = {median_best}")
plt.title("F-2  Learning curve  (median across 5 folds)")
plt.xlabel("Iteration")
plt.ylabel("Binary log-loss")
plt.legend()
plt.tight_layout()
plt.savefig(FIG, dpi=300)
plt.show()

print("best_iteration_ per fold:", best_iters,
      "\nmedian =", median_best, " | mean =", round(np.mean(best_iters), 1))
