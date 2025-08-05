# eda_B2.py  ——  Feature importance barplot (EMR)

import matplotlib
matplotlib.use("TKAgg")
import matplotlib.pyplot as plt
import seaborn as sns, pandas as pd, numpy as np
from pathlib import Path
from lightgbm import Booster

sns.set_style("whitegrid")
plt.rcParams["figure.dpi"] = 110
plt.rcParams["font.family"] = ["sans-serif"]
plt.rcParams["font.sans-serif"] = ["Arial"]

# ----------- 读取已训练好的 EMR 模型 & 特征名 -----------
MODEL_TXT = Path("../models/emr_model.txt")           # 训练脚本保存的模型
INDEX_CSV = Path("../models/index_emr_train.csv")     # 训练脚本保存的特征顺序
FEATS_NPY = Path("../models/p_emr_train.npy")         # 非必须，只验证文件在

bst   = Booster(model_file=str(MODEL_TXT))
f_imp = pd.Series(bst.feature_importance(importance_type="gain"),
                  index=bst.feature_name())
f_imp = (f_imp / f_imp.sum()).sort_values(ascending=False)   # 归一化

topN  = 20                 # 只画前 N 个
plt.figure(figsize=(6, 4))
sns.barplot(y=f_imp.head(topN).index,
            x=f_imp.head(topN).values,
            palette="Blues_r")
plt.xlabel("Normalized gain importance")
plt.ylabel("Feature")
plt.title("B-2  Top-20 feature importances (EMR)")
plt.tight_layout()
plt.savefig("figs/B2_feat_importance_bar.png", dpi=300)
plt.show()
