# run_pipeline.py ==============================================
# ① 训练 + 5 折 CV + GridSearch
# ② 输出评估表 / 混淆矩阵 / ROC / PR / 特征重要度
# ③ 对预测集生成 0/1 并保存

# ---------- 0. IMPORTS ----------
import os, pathlib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt, seaborn as sns
import pandas as pd, numpy as np
from pathlib import Path

from sklearn.model_selection import (StratifiedKFold, cross_validate,
                                     train_test_split, GridSearchCV)
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, ConfusionMatrixDisplay,
                             RocCurveDisplay, PrecisionRecallDisplay)

# ---------- 1. fix Joblib temp folder (中文路径bug) ----------
tmp_dir = pathlib.Path.cwd() / "joblib_tmp"
tmp_dir.mkdir(exist_ok=True)
os.environ["JOBLIB_TEMP_FOLDER"] = str(tmp_dir)

# ---------- 2. CONFIG ----------
TRAIN_PATH = r"D:\软件数据\25数模集训\2025-hs-1\048就业状态分析与预测\附件\gap_with(2).xlsx"
PRED_PATH  = r"D:\软件数据\25数模集训\2025-hs-1\048就业状态分析与预测\预测集.xlsx"
OUT_DIR    = Path("..")          # 输出文件夹

# ---------- 3. 读取 ----------
train = pd.read_excel(TRAIN_PATH, engine="openpyxl")
pred  = pd.read_excel(PRED_PATH,  engine="openpyxl")
train.columns = train.columns.str.strip();  pred.columns = pred.columns.str.strip()
pred = pred.drop(columns=[c for c in pred.columns if c.lower()=="employed"])

# ---------- 4. 衍生特征 ----------
EDU_MAP = {10:"小学及以下", 20:"初中", 30:"高中/中专",
           40:"大专", 50:"本科", 60:"硕博"}

def add_features(df):
    df["age"] = pd.to_numeric(df["age"], errors="coerce")
    df["age_bin"] = pd.cut(df["age"], bins=[0,24,35,50,65,np.inf],
                           labels=["≤24","25–35","36–50","51–65","65+"],
                           right=False)
    df["edu_cat"]  = pd.to_numeric(df["edu_level"], errors="coerce")\
                        .map(EDU_MAP).fillna("未知")
    df["prof_cat"] = df["profession"].astype(str).str[:2].fillna("未知")
    df["industry_cat"] = df["c_aac009"].astype(str).str[0].fillna("未知")
    return df

train = add_features(train);  pred = add_features(pred)

# ---------- 5. 处理标签 ----------
train["employed"] = pd.to_numeric(train["employed"], errors="coerce")
train = train.dropna(subset=["employed"])
train["employed"] = train["employed"].astype(int)

cat_cols = ["age_bin","sex","edu_cat","prof_cat","industry_cat"]
num_cols = ["age"]
X, y = train[cat_cols + num_cols], train["employed"]
X_pred = pred[cat_cols + num_cols]

# ---------- 6. 建立管道 ----------
preproc = ColumnTransformer([
    ("cat", Pipeline([
        ("imp", SimpleImputer(strategy="most_frequent")),
        ("oh",  OneHotEncoder(handle_unknown="ignore"))
    ]), cat_cols),
    ("num", Pipeline([
        ("imp", SimpleImputer(strategy="median")),
        ("sc",  StandardScaler())
    ]), num_cols)
])
base_rf = RandomForestClassifier(
    n_estimators=300, class_weight="balanced", random_state=42)
pipe = Pipeline([("prep", preproc), ("clf", base_rf)])

# ---------- 7. 5-折交叉验证 ----------
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
scores = cross_validate(pipe, X, y, cv=cv,
                        scoring=["accuracy","precision","recall","f1"],
                        n_jobs=-1)
cv_mean = pd.DataFrame(scores).mean().round(3)
print("\n--- 5 折 CV 平均 ---")
print(cv_mean[["test_accuracy","test_precision","test_recall","test_f1"]])

# ---------- 8. GridSearch ----------
param_grid = {"clf__n_estimators":[300,600],
              "clf__max_depth":[None,15,30],
              "clf__min_samples_leaf":[1,3,5]}
gs = GridSearchCV(pipe, param_grid, cv=3,
                  scoring="f1", n_jobs=-1, verbose=0)
gs.fit(X, y)
best_pipe = gs.best_estimator_
print("\n最佳参数:", gs.best_params_,
      "  F1=", round(gs.best_score_,3))

# ---------- 9. 留出法评估 ----------
X_tr, X_val, y_tr, y_val = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42)
best_pipe.fit(X_tr, y_tr)
y_val_pred = best_pipe.predict(X_val)

eval_df = pd.DataFrame([{
    "模型": "BestRF",
    "准确率":  accuracy_score(y_val, y_val_pred),
    "查准率":  precision_score(y_val, y_val_pred),
    "召回率":  recall_score(y_val, y_val_pred),
    "F1":      f1_score(y_val, y_val_pred)
}]).round(3)
eval_df.to_excel(OUT_DIR/"示例表2_评估指标.xlsx", index=False)
print("\n留出法 20% 指标：\n", eval_df)

# ---------- 10. 评估图 ----------
fig, ax = plt.subplots(figsize=(4,4))
ConfusionMatrixDisplay.from_estimator(best_pipe, X_val, y_val, ax=ax)
plt.tight_layout(); plt.savefig(OUT_DIR/"confusion_matrix.png", dpi=300); plt.close()

RocCurveDisplay.from_estimator(best_pipe, X_val, y_val)
plt.tight_layout(); plt.savefig(OUT_DIR/"roc_curve.png", dpi=300); plt.close()

PrecisionRecallDisplay.from_estimator(best_pipe, X_val, y_val)
plt.tight_layout(); plt.savefig(OUT_DIR/"pr_curve.png", dpi=300); plt.close()

feat_names = (best_pipe.named_steps["prep"]
                 .named_transformers_["cat"]
                 .named_steps["oh"]
                 .get_feature_names_out(cat_cols).tolist() + num_cols)
imp = pd.Series(best_pipe.named_steps["clf"].feature_importances_,
                index=feat_names).sort_values(ascending=False).head(15)
sns.set_theme(style="whitegrid", font="SimHei")
plt.figure(figsize=(6,4))
sns.barplot(x=imp.values, y=imp.index, palette="viridis")
plt.xlabel("Gini Importance"); plt.ylabel("特征")
plt.title("特征重要性 TOP15")
plt.tight_layout(); plt.savefig(OUT_DIR/"feature_importance.png", dpi=300); plt.close()

# ---------- 11. 全量重训 + 预测 ----------
best_pipe.fit(X, y)
pred["predict"] = best_pipe.predict(X_pred).astype(int)
pred.to_excel(OUT_DIR/"prediction_output.xlsx", index=False)

print("\n✅ 执行完毕，已生成：")
for f in ["示例表2_评估指标.xlsx", "confusion_matrix.png",
          "roc_curve.png", "pr_curve.png",
          "feature_importance.png", "prediction_output.xlsx"]:
    print(" •", f)
