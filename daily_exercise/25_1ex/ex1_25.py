import pandas as pd
import numpy as np

# ① 读表
file_path = "D:\软件数据\附件1 数据.xlsx"        # ← 换成你的文件名
df = pd.read_excel(file_path)

# ② 解析日期 ── 两列格式可能不一致，要分别处理
# ── 就业时间列：有些单元格是“2021/03”→默认当月 1 号
df["就业时间"] = (
    pd.to_datetime(df["就业时间"], format="%Y/%m", errors="coerce")      # 先按 yyyy/mm 尝试
       .fillna(pd.to_datetime(df["就业时间"], errors="coerce"))          # 剩下的让 pandas 自己识别
)

# ── 失业时间列：有完整日期，也有 '\N'，统统让 pandas 解析；解析不了的会变 NaT
df["失业时间"] = pd.to_datetime(df["失业时间"], errors="coerce")

# ③ 计算差值
# ◆ 按天数
df["就业时长_天"] = (df["就业时间"] - df["失业时间"]).dt.days

# ◆ 按月数（更常用）：年份差 ×12 + 月份差
df["就业时长_月"] = (
    (df["就业时间"].dt.year - df["失业时间"].dt.year) * 12
    + (df["就业时间"].dt.month - df["失业时间"].dt.month)
)

# 如果只想在代码里单独拿这个 Series 当“变量”
就业时长_月 = df["就业时长_月"]

# ④ 保存结果
df.to_excel("gap_with.xlsx", index=False)
print("✓ 处理完成，已输出 gap_with.xlsx")
