# analysis/eda_B1_feat_corr.py  修正版
# ------------------------------
import matplotlib
matplotlib.use("TKAgg")
import matplotlib.pyplot as plt
import os; os.makedirs("../figs", exist_ok=True)

import pandas as pd, numpy as np, seaborn as sns
from pathlib import Path
from feature_utils2 import slide_features, WIN, STEP   # 你的特征函数

DATA  = Path("../clean_data/tidy_emr_ae.xlsx")
SHEET = "EMR"          # 只看主体特征

# 1) 读取 EMR，记得带 value_z！
df = (pd.read_excel(DATA, sheet_name=SHEET, parse_dates=["time"])
        .loc[:, ["time", "value", "value_z"]])

# 2) 滑窗特征
feats = (slide_features(df)
           .add_suffix("_EMR"))

# 3) 相关系数
X     = feats.fillna(feats.median())
corr  = X.corr()

# 4) 画热图
mask = np.triu(np.ones_like(corr, dtype=bool))
plt.figure(figsize=(8, 6))
sns.heatmap(corr, mask=mask, cmap="RdBu_r", center=0,
            vmax=1, vmin=-1, linewidths=.3, cbar_kws={"label": "Pearson r"})
plt.title("B-1  Feature correlation heatmap (EMR)")
plt.tight_layout()
plt.savefig("figs/B1_feat_corr.png", dpi=300)
plt.show()
