import matplotlib
matplotlib.use("Agg")         # 服务器/IDE 中不弹窗
import pandas as pd, numpy as np
import matplotlib.pyplot as plt, seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# ---------------- ① 路径 ----------------
TRAIN_PATH = r"D:\软件数据\25数模集训\2025-hs-1\048就业状态分析与预测\附件\gap_with(2).xlsx"
PRED_PATH  = r"D:\软件数据\25数模集训\2025-hs-1\048就业状态分析与预测\预测集.xlsx"

train = pd.read_excel(TRAIN_PATH, engine="openpyxl")
pred  = pd.read_excel(PRED_PATH,  engine="openpyxl")

train.columns = train.columns.str.strip()
pred.columns  = pred.columns.str.strip()

# 预测集若带占位列 `employed`，删除
pred = pred.drop(columns=[c for c in pred.columns if c.lower() == "employed"])

# ---------- ② 衍生统一特征 ----------
EDU_MAP = {10:"小学及以下", 20:"初中", 30:"高中/中专",
           40:"大专", 50:"本科", 60:"硕博"}

def add_features(df: pd.DataFrame) -> pd.DataFrame:
    df["age"] = pd.to_numeric(df["age"], errors="coerce")

    df["age_bin"] = pd.cut(
        df["age"],
        bins=[0, 24, 35, 50, 65, np.inf],
        labels=["≤24", "25–35", "36–50", "51–65", "65+"],
        right=False
    )

    df["edu_cat"]  = pd.to_numeric(df["edu_level"], errors="coerce")\
                        .map(EDU_MAP).fillna("未知")
    df["prof_cat"] = df["profession"].astype(str).str[:2].fillna("未知")
    df["industry_cat"] = df["c_aac009"].astype(str).str[0].fillna("未知")  # 可改列名
    return df

train = add_features(train)
pred  = add_features(pred)

# ---------- ③ 处理标签列 ----------
train["employed"] = pd.to_numeric(train["employed"], errors="coerce")
train = train.dropna(subset=["employed"])
train["employed"] = train["employed"].astype(int)

# ---------- ④ 特征列 ----------
cat_cols = ["age_bin", "sex", "edu_cat", "prof_cat", "industry_cat"]
num_cols = ["age"]

# ---------- ⑤ 预处理 + 模型 ----------
preproc = ColumnTransformer([
    ("cat", Pipeline([
        ("impute", SimpleImputer(strategy="most_frequent")),
        ("oh", OneHotEncoder(handle_unknown="ignore"))
    ]), cat_cols),

    ("num", Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("sc",     StandardScaler())
    ]), num_cols)
])

clf = RandomForestClassifier(
        n_estimators=300,
        class_weight="balanced",
        random_state=42)

pipe = Pipeline([("prep", preproc), ("clf", clf)])

# ---------- ⑥ 训练-验证 ----------
X = train[cat_cols + num_cols]
y = train["employed"]

X_tr, X_val, y_tr, y_val = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42)

pipe.fit(X_tr, y_tr)
y_pred = pipe.predict(X_val)

metrics = pd.DataFrame([{
    "模型": "RandomForest",
    "准确率":  accuracy_score(y_val, y_pred),
    "查准率":  precision_score(y_val, y_pred),
    "召回率":  recall_score(y_val, y_pred),
    "F1":      f1_score(y_val, y_pred)
}]).round(3)
metrics.to_excel("示例表2_评估指标.xlsx", index=False)
print("评估指标：\n", metrics)

# ---------- ⑦ 特征重要性 ----------
sns.set_theme(style="whitegrid", font="SimHei")
feature_names = (pipe.named_steps["prep"]
                    .named_transformers_["cat"]
                    .named_steps["oh"]
                    .get_feature_names_out(cat_cols).tolist() + num_cols)

imp = pd.Series(pipe.named_steps["clf"].feature_importances_,
                index=feature_names).sort_values(ascending=False).head(15)

plt.figure(figsize=(6, 4))
sns.barplot(x=imp.values, y=imp.index, palette="viridis")
plt.xlabel("Gini Importance"); plt.ylabel("特征")
plt.title("特征重要性 TOP15")
plt.tight_layout()
plt.savefig("feature_importance.png", dpi=300)
plt.close()

# ---------- ⑧ 预测集推断 ----------
X_pred = pred[cat_cols + num_cols]
pred["predict"] = pipe.predict(X_pred).astype(int)
pred.to_excel("prediction_output.xlsx", index=False)

print("\n✅ 运行完毕，已生成：")
print(" • 示例表2_评估指标.xlsx")
print(" • feature_importance.png")
print(" • prediction_output.xlsx")
