# train_stack.py
import pandas as pd, numpy as np
from pathlib import Path
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, f1_score
from feature_utils import slide_features, WIN, STEP

DATA = Path("clean_data/tidy_emr_ae.xlsx")

def main():
    # ① 读取训练集 + AE 特征
    df_ae = pd.read_excel(DATA, sheet_name="AE", parse_dates=["time"])
    ae_feat = slide_features(df_ae[["time","value"]]) \
                .add_suffix("_AE")[["max_AE","p95_AE","zcr_AE"]]
    ae_feat = ae_feat.fillna(0.0)

    # ② 载入 Stage-1 输出
    idx = pd.read_csv("models/index_emr_train.csv", squeeze=True,
                      parse_dates=["time"])
    p_emr = np.load("models/p_emr_train.npy")
    p_series = pd.Series(p_emr, index=idx, name="p_emr")

    # ③ 拼接特征
    X2 = pd.concat([p_series, ae_feat], axis=1).fillna(0.0)

    # 标签
    df_lbl = pd.read_excel(DATA, sheet_name="EMR", parse_dates=["time"])
    y30 = df_lbl.set_index("time")["class"].map({"C":1,"A":0,"B":0})
    y_win = (y30.rolling(WIN,1).max()
                   .resample(STEP).last())
    y = y_win.reindex(X2.index).fillna(0).astype(int)

    # ④ Logistic 回归
    clf = LogisticRegression(max_iter=500, solver="liblinear")
    clf.fit(X2, y)

    p_final = clf.predict_proba(X2)[:,1]
    ap  = average_precision_score(y, p_final)
    f1  = f1_score(y, (p_final>0.5).astype(int))
    print(f"stack train  PR-AUC={ap:.3f} | F1={f1:.3f}")

    joblib.dump(clf, "models/stack_lr.pkl")

if __name__ == "__main__":
    main()
