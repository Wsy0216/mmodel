# 01_train_interf.py -------------------------------------------------------
import pandas as pd, numpy as np
from pathlib import Path
from lightgbm import LGBMClassifier
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import average_precision_score, f1_score
from feature_utils import slide_features, WIN, STEP

DATA = Path("../clean_data/tidy_emr_ae.xlsx")     # ★训练集
SHEETS = ["EMR", "AE"]

# ---------- 1 读数据 & 预处理 --------------------------------------------
dfs = []
for sh in SHEETS:
    tdf = pd.read_excel(DATA, sheet_name=sh,
                        parse_dates=["time"])
    tdf["signal_type"] = sh                 # 万一列里格式不一
    tdf["signal_type"] = tdf["signal_type"].str.upper()
    dfs.append(tdf)
df = pd.concat(dfs, ignore_index=True)

# 统一 class → y_interf (C=1, A/B=0)
label_map = {"C": 1, "A": 0, "B": 0}
df["y_interf"] = df["class"].map(label_map)
df = df.dropna(subset=["y_interf"])          # 丢 D/E

# ---------- 2 滑窗特征 ----------------------------------------------------
fe_emr = slide_features(df.query("signal_type=='EMR'")).add_suffix("_EMR")
fe_ae  = slide_features(df.query("signal_type=='AE'")).add_suffix("_AE")

X = fe_emr.join(fe_ae, how="outer")          # outer 允许 AE 缺失
# ---------- 3 标签：窗口内若出现 C →1--------------------------------------
label_map = {"C": 1, "A": 0, "B": 0}

lab_emr = (df.query("signal_type=='EMR'")
             .set_index("time")["class"].map(label_map))
lab_ae  = (df.query("signal_type=='AE'")
             .set_index("time")["class"].map(label_map))

# 两条信号取逻辑或：窗口内只要出现 C 就算正例
y30 = lab_emr.combine(lab_ae, func=lambda a,b: max(a or 0, b or 0))

y30 = (df.query("signal_type=='EMR'")
         .set_index("time")["y_interf"])
y_win = (y30.rolling(WIN, min_periods=1).max()
               .resample(STEP).last())
y = y_win.reindex(X.index)

mask = ~y.isna()
X, y = X[mask], y[mask].astype(int)

# —— 1. 删掉全缺失或常数列 ——————————————
bad_cols = [c for c in X.columns
            if X[c].isna().all() or (X[c].std(skipna=True) == 0)]
X = X.drop(columns=bad_cols)
print("删除无信息列:", bad_cols)

# —— 2. 用列中位数填剩余 NaN（LightGBM 可识别，但太多 NaN 会抑制增益）—
X = X.fillna(X.median())

# —— 3. 再做一次 sanity check ——————————
print("特征维度 ->", X.shape[1], "; 仍含 NaN? ", X.isna().any().any())

print("训练样本", len(X), "正样本", y.sum())

# ---------- 4 LightGBM 时序交叉验证 --------------------------------------
n_splits = min(5, max(2, len(X)//200))  # 每折≈≥200 行
tscv = TimeSeriesSplit(n_splits=n_splits)

pos_w = (y == 0).sum() / (y == 1).sum()
model = LGBMClassifier(
    objective="binary",
    n_estimators=600,
    learning_rate=0.03,
    num_leaves=127,
    subsample=0.8,
    colsample_bytree=0.8,
    scale_pos_weight=pos_w,   # ★ 新增
    random_state=42
)
# ==== 快速特征体检 ======================================================
print("\n=== FEATURE QUICK CHECK ===")
print("缺失率最高的 10 列：")
na_ratio = X.isna().mean().sort_values(ascending=False)
print(na_ratio.head(10))

print("\n常数列（std==0）的数量：", (X.std() == 0).sum())
if (X.std() == 0).sum():
    print(X.columns[X.std() == 0].tolist()[:10])

print("\n正负样本均值差（abs(mean1-mean0)）Top10：")
means = X.copy()
means["y"] = y
mean0 = means[means.y==0].drop("y",axis=1).mean()
mean1 = means[means.y==1].drop("y",axis=1).mean()
delta = (mean1 - mean0).abs().sort_values(ascending=False)
print(delta.head(10))

print("\n每列唯一值个数 nunique：")
print(X.nunique().sort_values().head(10))
print("=== END CHECK ===\n")
# =======================================================================

ap, f1 = [], []
for tr, te in tscv.split(X):
    model.fit(X.iloc[tr], y.iloc[tr])
    p = model.predict_proba(X.iloc[te])[:, 1]
    ap.append(average_precision_score(y.iloc[te], p))
    f1.append(f1_score(y.iloc[te], (p > 0.5).astype(int)))

print(f"{n_splits}-fold  PR-AUC={np.mean(ap):.3f} | F1={np.mean(f1):.3f}")

model.booster_.save_model("interf_model.txt")
print("模型已保存 → interf_model.txt")
