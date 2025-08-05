import pandas as pd, numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import matplotlib
matplotlib.use("Agg")
# 读原始 Excel
raw = pd.read_excel(r"D:\软件数据\gap_with(2).xlsx", engine="openpyxl")

# ① 训练集 vs 预测集 —— 用“预测”列是否为空来区分
train_df = raw[raw["预测"].isna()].copy()       # 有真值
pred_df  = raw[~raw["预测"].isna()].copy()      # 需要预测

# ② 目标 y & 自变量 X
y = train_df["就业"] = train_df["是否就业"].astype(int)
X = train_df.drop(columns=["是否就业","就业","预测"])

# 把预测集也保留同样特征列
X_pred = pred_df[X.columns]

# ③ 列类型
cat_cols = ["年龄段","性别","学历","专业大类","行业大类"]
num_cols = ["年龄"]           # 连续变量可再补充

# ④ 预处理：One-Hot + 连续特征保持
preprocess = ColumnTransformer(
    transformers=[
        ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
        ("num", "passthrough", num_cols)],
    remainder="drop")
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

# two models for comparison
models = {
    "Logit": LogisticRegression(max_iter=1000, class_weight="balanced"),
    "RF":    RandomForestClassifier(n_estimators=300, class_weight="balanced",
                                   random_state=42)
}

metrics = []
for name, clf in models.items():
    pipe = Pipeline([("prep", preprocess), ("clf", clf)])
    X_tr, X_val, y_tr, y_val = train_test_split(X, y, test_size=0.2, stratify=y,
                                                random_state=42)
    pipe.fit(X_tr, y_tr)
    y_pred = pipe.predict(X_val)
    metrics.append({
        "模型": name,
        "准确率":  accuracy_score(y_val, y_pred),
        "查准率":  precision_score(y_val, y_pred),
        "召回率":  recall_score(y_val, y_pred),
        "F1":      f1_score(y_val, y_pred),
        "pipe":    pipe                     # 留下模型对象
    })

metric_df = (pd.DataFrame(metrics)
             .drop(columns="pipe")
             .set_index("模型")
             .round(3))
metric_df.to_excel("示例表2_评估指标.xlsx")
print(metric_df)
import matplotlib.pyplot as plt, seaborn as sns
sns.set_theme(style="whitegrid")

# 拿到 one-hot 后的列名
feature_names = best_pipe.named_steps["prep"]\
    .named_transformers_["cat"].get_feature_names_out(cat_cols).tolist() + num_cols

importances = best_pipe.named_steps["clf"].feature_importances_
imp_df = (pd.Series(importances, index=feature_names)
          .sort_values(ascending=False).head(15))

plt.figure(figsize=(6,4))
sns.barplot(x=imp_df.values, y=imp_df.index, palette="viridis")
plt.xlabel("Gini Importance"); plt.ylabel("特征")
plt.title("特征重要性 TOP15")
plt.tight_layout()
plt.savefig("feature_importance.png", dpi=300)
# 用最佳模型对预测集推断
pred_df["预测"] = best_pipe.predict(X_pred)

# 小计
job_cnt   = int(pred_df["预测"].sum())
unjob_cnt = int((1 - pred_df["预测"]).sum())

out_cols = ["T1", "T2", "...", "T20"]          # 或直接用索引 / ID
result = pred_df[out_cols]
result.loc["就业小计"] = [""]*(len(out_cols)) + [job_cnt]
result.loc["失业小计"] = [""]*(len(out_cols)) + [unjob_cnt]

result.to_excel("示例表3_预测结果.xlsx", index=False)
print("✅ 预测完成，表格已保存。")
