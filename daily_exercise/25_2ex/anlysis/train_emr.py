# train_emr.py
import pandas as pd, numpy as np
import lightgbm as lgb
import matplotlib.pyplot as plt

from pathlib import Path
from lightgbm import LGBMClassifier
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import average_precision_score, f1_score
from feature_utils2 import slide_features, WIN, STEP

DATA = Path("clean_data/tidy_emr_ae.xlsx")     # ★训练集
SHEET = "EMR"                                  # 只用 EMR

def main():
    df = pd.read_excel(DATA, sheet_name=SHEET, parse_dates=["time"])
    df = df[["time", "value", "value_z","class"]]
    label_map = {"C":1, "A":0, "B":0}
    df["y"] = df["class"].map(label_map)

    # —— 1 滑窗特征
    feats = slide_features(df) \
              .add_suffix("_EMR")

    # —— 2 标签：窗口内出现 C→1
    y30 = df.set_index("time")["y"]
    y_win = (y30.rolling(WIN,min_periods=1).max()
                   .resample(STEP).last())
    X = feats
    y = y_win.reindex(X.index)

    mask = ~y.isna()
    X, y = X[mask], y[mask].astype(int)


    # 缺失列处理
    bad = [c for c in X if X[c].isna().all() or X[c].std()==0]
    X = X.drop(columns=bad).fillna(X.median())

    print("\n=== 特征体检 ===")
    print("方差最小前 10 列：")
    print(X.var().sort_values().head(10))

    print("\n每列缺失率 >0.5 的：")
    print(X.isna().mean()[X.isna().mean() > 0.5])

    print("训练样本:", len(X), "正样本:", y.sum(),
          "特征维:", X.shape[1])

    # —— 3 LightGBM
    # ---------- 交叉验证分层：保证每一折 train 都至少含 1 ----------
    from sklearn.model_selection import TimeSeriesSplit
    import numpy as np

    n_splits = 5  # 想要的折数
    first_pos_idx = np.where(y.values == 1)[0][0]  # 第一个正样本位置
    min_train_len = first_pos_idx + 1  # 第 1 折 train 必须覆盖它
    desired_len = len(X) // 6  # 设想的每折长度

    # 公式允许的最大 test_size
    max_ts = (len(X) - min_train_len) // n_splits

    # 取较小值
    test_size = min(desired_len, max_ts)

    if test_size < 1:
        test_size = 1
        n_splits = max(2, len(X) - min_train_len)  # 至少还能分 2 折

    # 最终的时间序列交叉验证器
    tscv = TimeSeriesSplit(n_splits=n_splits, test_size=test_size)
    # -----------------------------------------------------------------

    model = LGBMClassifier(
        objective="binary",
        n_estimators=5000,
        learning_rate=0.03,
        num_leaves=255,
        min_data_in_leaf=20,
        subsample=0.7,
        colsample_bytree=0.8,
        scale_pos_weight=(y == 0).sum() / (y == 1).sum(),
        random_state=42
    )

    best_iters = []  # 用来收每折的 best_iteration_

    ap, f1 = [], []  # 交叉验证指标

    for tr, te in tscv.split(X):
        model.fit(
            X.iloc[tr], y.iloc[tr],
            eval_set=[(X.iloc[te], y.iloc[te])],
            eval_metric="auc",
            callbacks=[lgb.early_stopping(100)],
        )

        # —— 验证集预测
        p = model.predict_proba(
            X.iloc[te],
            num_iteration=model.best_iteration_
        )[:, 1]

        ap.append(average_precision_score(y.iloc[te], p))
        f1.append(f1_score(y.iloc[te], (p > 0.5).astype(int)))

        best_iters.append(model.best_iteration_)  # 记录下来

    print(f"5-fold PR-AUC={np.mean(ap):.3f} | F1={np.mean(f1):.3f}")

    # 把 5 个 best_iteration_ 取中位数（or 均值）
    best_iter = int(np.median(best_iters))

    #  全量数据再训练一次
    model_final = LGBMClassifier(**model.get_params())
    model_final.set_params(n_estimators=best_iter)
    model_final.fit(X, y)
    feats.to_parquet("clean_data/feats_emr.parquet")

    # —— 特征重要性示例
    importances = pd.Series(model_final.feature_importances_, index=X.columns)
    print("\n=== Top-20 feature_importances_ ===")
    print(importances.sort_values(ascending=False).head(20))


if __name__ == "__main__":
    main()
