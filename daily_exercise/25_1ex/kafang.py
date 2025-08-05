# kafang.py  ── 一键批量卡方检验示例
import pandas as pd
from scipy.stats import chi2_contingency
import numpy as np

# ---------- ① 读取原始数据 ----------
df = pd.read_excel(r"D:\软件数据\gap_with(2).xlsx", engine="openpyxl")

# ---------- ② 衍生或映射必要字段 ----------


df["就业"] = df["是否就业"].astype(int)

df["年龄段"] = pd.cut(df["年龄"],
                     bins=[0,24,35,50,65,np.inf],
                     labels=["≤24","25–35","36–50","51–65","65+"],
                     right=False)
df["学历"]   = df["教育程度"].map({10:"小学及以下",20:"初中",30:"高中/中专"})
df["性别"]   = df["性别"].map({1:"男", 2:"女"})
df["专业大类"] = df["所学专业名称"].str[:2]
df["行业大类"] = df["行业代码"].str[0]
freq = df['专业大类'].value_counts()
rare = freq[freq < 30].index            # 出现次数 <30 的类别
df.loc[df['专业大类'].isin(rare), '专业大类'] = '其他'

cat_cols = ["年龄段","性别","学历","专业大类","行业大类"]

# ---------- ③ 批量卡方检验 ----------
results = []
for col in cat_cols:
    ct = pd.crosstab(df[col], df["就业"])
    chi2, p, dof, _ = chi2_contingency(ct)
    n = ct.values.sum()
    V = np.sqrt(chi2 / (n * (min(ct.shape)-1)))      # Cramér's V
    results.append({"变量": col, "卡方": chi2, "自由度": dof,
                    "p值": p, "CramérV": V,
                    "行数": ct.shape[0]})

res_df = (pd.DataFrame(results)
          .sort_values("p值")
          .reset_index(drop=True))

# ---------- ④ 保存 & 打印 ----------
res_df.to_excel("chi2_results.xlsx", index=False)
print(res_df)
