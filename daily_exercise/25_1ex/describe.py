import pandas as pd
from pathlib import Path

import matplotlib
matplotlib.use("TkAgg")

import seaborn as sns
import matplotlib.pyplot as plt

path = Path(r"D:\软件数据\gap_with(2).xlsx")
df = (pd.read_excel(path, sheet_name=0, engine="openpyxl"))
print(df.shape)
print(df.columns.tolist())
# ---- 2.1 类型转换 ----------------------------------------------------------
numeric_cols = ['年龄', '户口所在地区代码', '毕业年份']  # 可按需补充
df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, errors='coerce')

# 失业/就业字段转布尔（1/0）
df['是否就业'] = df['是否就业'].map({1: True, 0: False})

# ---- 2.2 文本规范 ----------------------------------------------------------
df['户籍地址'] = df['户籍地址'].str.strip()
df['行业代码'] = df['行业代码'].str.zfill(4)  # 若需统一宽度

# ---- 2.3 缺失值检查 --------------------------------------------------------
missing_pct = df.isna().mean().sort_values(ascending=False)
print(missing_pct.head(10))       # 先看缺得最多的列

# ---- 2.4 可能的派生变量 -----------------------------------------------------
bins = [0, 18, 35, 50, 65, 150]
labels = ['≤18', '19-35', '36-50', '51-65', '65+']
df['年龄段'] = pd.cut(df['年龄'], bins=bins, labels=labels, right=False)

df['是否弱势群体'] = (
    df['是否残疾人'].eq(1) | df['是否老年人'].eq(1) | df['是否青少年'].eq(1)
)

# 是否外地人（示例：地区代码前两位与某本地码不符）
local_prefix = '32'   # 江苏举例
df['是否外地'] = ~df['户口所在地区代码'].astype(str).str.startswith(local_prefix)
# 3.1 数值型总体描述
print(df[['年龄']].describe())

# 3.2 分类变量频数
for col in ['性别', '婚姻状态', '教育程度', '行业代码', '是否就业']:
    print(df[col].value_counts(dropna=False).head())

# 3.3 交叉透视：年龄段 × 是否就业
pivot_age_job = pd.crosstab(df['年龄段'], df['是否就业'], normalize='index') * 100
print(pivot_age_job.round(1))

# 3.4 失业原因 Top 10
print(df['失业原因'].value_counts().head(10))


# 4.1 年龄直方图
sns.histplot(df['年龄'].dropna(), bins=20)
plt.xlabel('Age'); plt.title('Age Distribution'); plt.show()

# 4.2 不同学历的失业率
edu_job = df.pivot_table(index='教育程度', values='是否就业', aggfunc=lambda s: 1 - s.mean())
edu_job = edu_job.sort_values('是否就业', ascending=False)
sns.barplot(data=edu_job.reset_index(), x='教育程度', y='是否就业')
plt.ylabel('Unemployment Rate')
plt.xlabel('schooling')
plt.show()
