# ----------------------------- F-1  best_iteration_ 分布 ----------------------
import os
from pathlib import Path

import matplotlib
matplotlib.use("TKAgg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import lightgbm as lgb
from lightgbm import LGBMClassifier
from sklearn.model_selection import TimeSeriesSplit

from feature_utils2 import slide_features, WIN, STEP   # 保持你的工具文件路径

# ---------- 路径 ----------
DATA = Path("../clean_data/tidy_emr_ae.xlsx")
FIG  = Path("../figs/F1_best_iter_box.png")
os.makedirs(FIG.parent, exist_ok=True)

# ---------- 数据准备 ----------
df = (pd.read_excel(DATA, sheet_name="EMR", parse_dates=["time"])
        .loc[:, ["time", "value", "value_z", "class"]])
df["y"] = df["class"].map({"C": 1, "A": 0, "B": 0})

X = slide_features(df).add_suffix("_EMR")

y30 = df.set_index("time")["y"]
y = (y30.rolling(WIN, min_periods=1).max()      # 窗内出现 C 记为 1
         .resample(STEP).last()
         .reindex(X.index)
         .fillna(0)                              # 缺失当 0
         .astype(int))

X = X.loc[y.index]                               # 保证对齐
bad = [c for c in X if X[c].isna().all() or X[c].std() == 0]
X = X.drop(columns=bad).fillna(X.median())

# ---------- 自定义 test_size：确保每折 train 有 C ----------
n_splits = 5
first_pos_idx = np.where(y.values == 1)[0][0]
min_train_len = first_pos_idx + 1                # 必须 ≥ 这个长度

# 把余下的数据均分成 n_splits 份；向下取整
fold_len = (len(X) - min_train_len) // n_splits
# 有时正样本非常靠前，(len - min_train_len) 可能比 n_splits 还小
# 再兜个底：至少给 test 每折留 1 样本
fold_len = max(1, fold_len)

test_size = fold_len         # 这才是 TimeSeriesSplit 要的
tscv = TimeSeriesSplit(n_splits=n_splits, test_size=test_size)

# ---------- 模型 & 交叉验证 ----------
model = LGBMClassifier(
    objective="binary",
    n_estimators=5000,
    learning_rate=0.03,
    num_leaves=255,
    min_data_in_leaf=20,
    subsample=0.7,
    colsample_bytree=0.8,
    scale_pos_weight=(y == 0).sum() / (y == 1).sum(),
    verbose=-1,
    random_state=42,
)

best_iters = []
for tr, te in tscv.split(X):
    model.fit(
        X.iloc[tr], y.iloc[tr],
        eval_set=[(X.iloc[te], y.iloc[te])],
        eval_metric="auc",
        callbacks=[lgb.early_stopping(100, verbose=False)],
    )
    best_iters.append(model.best_iteration_)

# ---------- 画箱线 + 散点 ----------
sns.set_style("whitegrid")
plt.figure(figsize=(5, 5))
sns.boxplot(y=best_iters, width=.4, color="skyblue")
sns.stripplot(y=best_iters, color="darkblue", size=6, jitter=.15)
median_val = int(np.median(best_iters))
plt.axhline(median_val, ls="--", lw=1, c="red", label=f"median = {median_val}")
plt.title("F-1  Distribution of best_iteration_  (5-fold CV)")
plt.ylabel("best_iteration_"); plt.xlabel(""); plt.legend()
plt.tight_layout(); plt.savefig(FIG, dpi=300); plt.show()

print("5 折 best_iteration_：", best_iters,
      "\nmedian =", median_val, "| mean =", np.mean(best_iters).round(1))
