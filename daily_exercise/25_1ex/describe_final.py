# =============================================
# ① 全局设置
# =============================================
import matplotlib, pandas as pd, seaborn as sns, matplotlib.pyplot as plt, numpy as np, textwrap
matplotlib.use("Agg")
import statsmodels.api as sm
sns.set_theme(style="whitegrid", palette="Set2")
plt.rcParams["font.sans-serif"] = ["SimHei"];  plt.rcParams["axes.unicode_minus"] = False

# =============================================
# ② 读数据 & 衍生列
# =============================================
df = pd.read_excel(r"D:\软件数据\gap_with(2).xlsx", engine="openpyxl")
df["就业"] = df["是否就业"].astype(int)

# 衍生分类
df["年龄段"] = pd.cut(df["年龄"], bins=[0,24,35,50,65,np.inf],
                     labels=["≤24","25–35","36–50","51–65","65+"], right=False)
df["学历"]   = df["教育程度"].map({10:"小学及以下",20:"初中",30:"高中/中专"})
df["性别"]   = df["性别"].map({1:"男", 2:"女"})
df["专业大类"] = df["所学专业名称"].str[:2]
df["行业大类"] = df["行业代码"].str[0]

features = ["年龄段","性别","学历","专业大类","行业大类"]

# =============================================
# ③ 设计矩阵：one-hot → 去零方差/完全重复 → 连续变量标准化
# =============================================
# ... ③ 构造 X, y, 加截距 ------------
X = pd.get_dummies(df[features], drop_first=True).astype(float)

# 删零方差
X = X.loc[:, X.nunique()>1]

# 删完全重复列
dup_mask = X.T.duplicated();  X = X.loc[:, ~dup_mask]

# 连续变量（年龄）标准化后追加回矩阵
from sklearn.preprocessing import StandardScaler
X["年龄_z"] = StandardScaler().fit_transform(df[["年龄"]])

# y & 截距
y = df["就业"].loc[X.index].astype(int)
X = sm.add_constant(X)
model_pen = sm.Logit(y, X).fit_regularized(method="l1", alpha=1.0, disp=0)

# =============================================
# ④ Logistic 拟合（先带 L1 保证稳，再常规拟合提 CI）
# =============================================
# === ④ 直接用正则化系数（无 CI） ===
# === ④ 由惩罚模型直接提 OR（不算 CI） ===
or_df = (np.exp(model_pen.params)           # e^β
         .rename("OR")
         .reset_index()
         .rename(columns={"index":"变量"})
         .sort_values("OR", ascending=False))

# 如想过滤只看极端效应，可取消下一行注释
# or_df = or_df.query("OR > 1.05 | OR < 0.95")

or_df.to_csv("logit_OR_table.csv", index=False, encoding="utf8")

# === ⑤ 单色条形图 ===
fig, ax = plt.subplots(figsize=(8, 0.4*len(or_df)+2))
sns.barplot(data=or_df, x="OR", y="变量", color="#4caf50", ax=ax)
for p in ax.patches:
    ax.text(p.get_width()+0.02, p.get_y()+p.get_height()/2,
            f"{p.get_width():.2f}", va="center")
ax.axvline(1, ls="--", c="grey"); ax.set_xscale("log")
ax.set_title("Logit（L1 惩罚）— 变量优势比 OR")
plt.tight_layout(); fig.savefig("logit_or.png", dpi=300); plt.close()

# === ⑥ 保存简要文本（无 CI） ===
with open("logit_summary.txt", "w", encoding="utf8") as f:
    f.write(model_pen.summary().as_text())
print("✅ 生成：logit_or.png  logit_summary.txt  logit_OR_table.csv")
