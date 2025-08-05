import pandas as pd, seaborn as sns, matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")
sns.set_theme(style="whitegrid", palette="Set2")
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# ---------- 读取数据 ----------
df = pd.read_excel(r"D:\软件数据\gap_with(2).xlsx", engine="openpyxl")
df["就业"] = df["是否就业"].astype(int)

# 若之前没生成分类列，请加上： -------------------
df["行业大类"] = df["行业代码"].str[0]        # 取行业首字母
df["专业大类"] = df["所学专业名称"].str[:2]    # 专业前两字
# ------------------------------------------------
def plot_stacked(category, topN=None):
    ct = (pd.crosstab(df[category], df["就业"], normalize="index")
            .mul(100)
            .rename(columns={0: "失业率", 1: "就业率"}))

    if topN:
        ct = ct.sort_values("失业率", ascending=False).head(topN)

    # 堆叠柱
    ct[["失业率","就业率"]].plot(kind="bar", stacked=True,
                                 figsize=(8,4), edgecolor='none')
    plt.ylabel("%")
    plt.title(f"{category} × 就业状态 (堆叠百分比)")
    plt.legend(loc="upper right")
    plt.tight_layout()
    plt.savefig(f"{category}_stacked.png", dpi=300)
    plt.close()

    # 失业率排序柱
    (ct["失业率"].sort_values(ascending=False)
        .plot(kind="bar", color="#f94144", figsize=(8,4)))
    plt.ylabel("失业率 (%)")
    plt.title(f"{category} 失业率排序")
    plt.tight_layout()
    plt.savefig(f"{category}_失业率排序.png", dpi=300)
    plt.close()

# ---------- 执行 ----------
plot_stacked("行业大类", topN=10)   # 行业：前 10 类
plot_stacked("专业大类", topN=20)   # 专业：失业率最高 20 类

print("✅ 已生成：行业大类_*  与  专业大类_*  两套 PNG")
