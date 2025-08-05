# 00_feature.py -------------------------------------------------------------
import pandas as pd

WIN  = "10min"          # 滑窗长度
STEP = "5min"           # 输出步长

def slide_features(df_sig: pd.DataFrame) -> pd.DataFrame:
    """
    单一路径信号 → 10 min 滑窗 → 5 min 栅格特征
    返回索引 = 窗口末端时间戳
    """
    s = df_sig.set_index("time")["value"]

    stat = (s.rolling(WIN)
              .agg(["mean", "std", "min", "max", "skew", "kurt"])
              .resample(STEP).last())

    stat["p95"] = (s.rolling(WIN).quantile(0.95)
                     .resample(STEP).last())
    return stat

# 01_train_interf.py -------------------------------------------------------
import pandas as pd, numpy as np
from pathlib import Path
from lightgbm import LGBMClassifier
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import average_precision_score, f1_score
from feature_utils import slide_features, WIN, STEP

DATA = Path("data/tidy_emr_ae.xlsx")     # ★训练集
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
y30 = (df.query("signal_type=='EMR'")
         .set_index("time")["y_interf"])
y_win = (y30.rolling(WIN, min_periods=1).max()
               .resample(STEP).last())
y = y_win.reindex(X.index)

mask = ~y.isna()
X, y = X[mask], y[mask].astype(int)

print("训练样本", len(X), "正样本", y.sum())

# ---------- 4 LightGBM 时序交叉验证 --------------------------------------
n_splits = min(5, max(2, len(X)//200))  # 每折≈≥200 行
tscv = TimeSeriesSplit(n_splits=n_splits)

model = LGBMClassifier(
    objective="binary",
    is_unbalance=True,
    n_estimators=400,
    learning_rate=0.05,
    num_leaves=63,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42
)

ap, f1 = [], []
for tr, te in tscv.split(X):
    model.fit(X.iloc[tr], y.iloc[tr])
    p = model.predict_proba(X.iloc[te])[:, 1]
    ap.append(average_precision_score(y.iloc[te], p))
    f1.append(f1_score(y.iloc[te], (p > 0.5).astype(int)))

print(f"{n_splits}-fold  PR-AUC={np.mean(ap):.3f} | F1={np.mean(f1):.3f}")

model.booster_.save_model("interf_model.txt")
print("模型已保存 → interf_model.txt")
