import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path
import matplotlib;
matplotlib.use("TkAgg")

# —— 让 Seaborn 统一配色 / 字体 ——
sns.set_theme(style="whitegrid", palette="Set2")      # 你也可以换 pastel, deep, colorblind…

plt.rcParams['font.sans-serif'] = ['SimHei']          # 解决中文乱码（Windows 自带黑体即可）
plt.rcParams['axes.unicode_minus'] = False            # 负号正常显示
path = Path(r"D:\软件数据\gap_with(2).xlsx")
df = (pd.read_excel(path, engine="openpyxl"))

# —— 把数字代码映射成人类可读文本 ——
sex_map  = {1: '男', 2: '女'}
marry_map = {10: '未婚', 20: '已婚', 30: '丧偶', 40: '离婚'}
edu_map   = {10: '小学及以下', 20: '初中', 30: '高中/中专'}

df['性别']   = df['性别'].map(sex_map)
df['婚姻状态'] = df['婚姻状态'].map(marry_map)
df['教育程度'] = df['教育程度'].map(edu_map)
fig, axes = plt.subplots(2, 2, figsize=(12, 8))
(ax1, ax2), (ax3, ax4) = axes

# ① 年龄直方图
sns.histplot(df['年龄'].dropna(), bins=20, ax=ax1, kde=True)
ax1.set_title('年龄分布')

# ② 性别饼图
sex_count = df['性别'].value_counts()
ax2.pie(sex_count, labels=sex_count.index, autopct='%1.1f%%', startangle=90)
ax2.set_title('性别比例')

# ③ 婚姻状态条形图（按人数）
sns.countplot(data=df, x='婚姻状态', order=marry_map.values(), ax=ax3)
ax3.set_ylabel('人数'); ax3.set_title('婚姻状态分布')

# ④ 教育程度失业率（条形 + 百分比标注）
edu_unemp = df.pivot_table(index='教育程度',
                           values='是否就业',
                           aggfunc=lambda s: 1 - s.mean()).reset_index()
sns.barplot(data=edu_unemp, x='教育程度', y='是否就业', ax=ax4)
for p in ax4.patches:                                    # 在柱顶写百分比
    ax4.text(p.get_x()+p.get_width()/2, p.get_height()+.005,
             f'{p.get_height():.1%}', ha='center', va='bottom')
ax4.set_ylabel('失业率'); ax4.set_title('不同学历层失业率')

plt.tight_layout()
plt.show()
topN = (df['行业代码']
        .value_counts()
        .head(10)
        .sort_values())           # 小→大方便横向

plt.figure(figsize=(8, 5))
sns.barplot(x=topN.values, y=topN.index, palette="Blues_r")
plt.xlabel('人数'); plt.ylabel('行业代码')
plt.title('行业前 10 就业人数')
for x, y in zip(topN.values, topN.index):
    plt.text(x + 5, y, x, va='center')
plt.tight_layout(); plt.show()
import plotly.express as px

fig = px.histogram(df, x='年龄', nbins=20, title='年龄分布 (Interactive)',
                   marginal='rug', hover_data=df.columns)
fig.show()

# Plotly 会自动打开浏览器窗口，条目可 hover / zoom / download PNG
