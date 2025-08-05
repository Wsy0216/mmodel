import matplotlib
matplotlib.use('TkAgg')         # GUI 环境
# 若服务器/无桌面，可用 'Agg'（纯文件输出）:
# matplotlib.use('Agg')

import matplotlib.pyplot as plt

import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import OrdinalEncoder
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, roc_auc_score, roc_curve
import matplotlib.pyplot as plt
import numpy as np

# 1. 读表并预处理 -------------------------------------------------
file = r"D:\软件数据\25数模集训\员工离职预测模型.xlsx"
df   = pd.read_excel(file)

# 把 '离职' 转成 0/1，'工资' 转成序数 0/1/2
df['离职'] = df['离职'].astype(int)
df['工资'] = df['工资'].map({'低':0, '中':1, '高':2})

X = df.drop(columns='离职')
y = df['离职']

# 2. Train/Test 划分（70/30，分层抽样） ---------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, stratify=y, random_state=42
)

# 3. 训练完整树 ----------------------------------------------------
full_tree = DecisionTreeClassifier(
    criterion='gini',
    min_samples_leaf=1,
    random_state=42
).fit(X_train, y_train)

# 4. 自动剪枝（Cost-Complexity 升序枚举，取交叉验证误差最小） -------
path = full_tree.cost_complexity_pruning_path(X_train, y_train)
ccp_alphas = path.ccp_alphas

cv_err = []
kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

for alpha in ccp_alphas:
    fold_err = []
    for train_idx, val_idx in kf.split(X_train, y_train):
        dt = DecisionTreeClassifier(random_state=42, ccp_alpha=alpha)
        dt.fit(X_train.iloc[train_idx], y_train.iloc[train_idx])
        fold_err.append(1 - dt.score(X_train.iloc[val_idx], y_train.iloc[val_idx]))
    cv_err.append(np.mean(fold_err))

best_alpha = ccp_alphas[np.argmin(cv_err)]
tree_pruned = DecisionTreeClassifier(random_state=42, ccp_alpha=best_alpha)
tree_pruned.fit(X_train, y_train)

# 5. 评估 ----------------------------------------------------------
y_pred   = tree_pruned.predict(X_test)
cm       = confusion_matrix(y_test, y_pred)
ConfusionMatrixDisplay(cm).plot()
plt.title("Pruned Tree — Confusion Matrix")
plt.show()

# ROC / AUC
y_prob = tree_pruned.predict_proba(X_test)[:,1]
auc    = roc_auc_score(y_test, y_prob)
fpr, tpr, _ = roc_curve(y_test, y_prob)
plt.plot(fpr, tpr); plt.xlabel("FPR"); plt.ylabel("TPR")
plt.title(f"ROC curve (AUC = {auc:0.3f})"); plt.show()

# 6. 可视化树结构（小数据集效果好） -------------------------------
plt.figure(figsize=(12,6))
plot_tree(tree_pruned,
          feature_names=X.columns,
          class_names=['Stay','Leave'],
          max_depth=3, filled=True)
plt.show()
