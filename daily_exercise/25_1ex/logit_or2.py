import matplotlib
matplotlib.use("Agg")
import pandas as pd, seaborn as sns, matplotlib.pyplot as plt, numpy as np
sns.set_theme(style="whitegrid")
plt.rcParams["font.sans-serif"] = ["SimHei"]
plt.rcParams["axes.unicode_minus"] = False

# ---------- 1 读取 OR 表 ----------
or_df = pd.read_csv("logit_OR_table.csv")
or_df = or_df.query("OR != 1")                # 去掉被 L1 收缩为 1 的
or_df["影响强度"] = np.abs(np.log(or_df["OR"]))

# ---------- 2 选 Top±10 ----------
top_pos = or_df[or_df["OR"] > 1].nlargest(10, "影响强度")
top_neg = or_df[or_df["OR"] < 1].nlargest(10, "影响强度")
plot_df = pd.concat([top_pos, top_neg]).sort_values("OR")

# ---------- 3 颜色：正向↑ 橙色，负向↓ 蓝色 ----------
colors = plot_df["OR"].apply(lambda x: "#ef8a62" if x > 1 else "#67a9cf")

# ---------- 4 绘图 ----------
fig, ax = plt.subplots(figsize=(10, 0.45*len(plot_df)+2))
bars = ax.barh(plot_df["变量"], plot_df["OR"], color=colors, edgecolor="none")

# 条内标注
for bar, or_val in zip(bars, plot_df["OR"]):
    ax.text(or_val*1.05 if or_val > 1 else or_val*0.95,  # 右/左侧偏移
            bar.get_y() + bar.get_height()/2,
            f"{or_val:.2f}",
            va="center",
            ha="left" if or_val > 1 else "right",
            fontsize=9, color="black")

# 美化轴
ax.axvline(1, ls="--", c="grey")
ax.set_xscale("log")
ax.set_xlabel("优势比 OR (log 轴)")
ax.set_title("Logit 模型显著变量 — 正向 / 负向影响 Top 10")
ax.grid(axis="x", linestyle="--", alpha=0.4)
sns.despine(left=True, bottom=True)
plt.tight_layout()
fig.savefig("logit_or_beauty.png", dpi=300)
print("✅ 已生成 logit_or_beauty.png")
