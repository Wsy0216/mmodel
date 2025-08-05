import re
import pandas as pd

# === 0. 参数 ===
file_path = r"D:\软件数据\25数模集训\2025-hs-2\1\附件2 (Attachment 2)(1).xlsx"     # Excel 文件
sheet_emr = 'EMR'              # 电磁辐射工作表名
sheet_ae  = 'AE'               # 声发射工作表名

# === 1. 读取两行表头，得到 MultiIndex 列 ===
def read_sheet(sheet_name, signal_type):
    """把两行表头的宽表拆成 tidy 长表"""
    wide = pd.read_excel(
        file_path,
        sheet_name=sheet_name,
        header=[0, 1],          # ← 关键：把第 1、2 行一起读成多级列
        engine='openpyxl'
    )

    # wide.columns 形如 MultiIndex([
    #   ('时间段1 (time period 1)', '电磁辐射 (EMR)'),
    #   ('时间段1 (time period 1)', '时间 (time)'),
    #   ('时间段2 (time period 2)', '电磁辐射 (EMR)'), ...])

    tidy_frames = []
    for period in wide.columns.levels[0]:
        block = wide[period].copy()                  # 取该时间段两列
        block.columns = ['value', 'time']            # 改列名统一
        block['period_id']   = period                # 保存时间段标签
        block['signal_type'] = signal_type           # EMR / AE
        tidy_frames.append(block)

    df_long = pd.concat(tidy_frames, ignore_index=True)
    df_long['time'] = pd.to_datetime(df_long['time'])  # 字符串 → 日期
    return df_long

emr_long = read_sheet(sheet_emr, 'EMR')
ae_long  = read_sheet(sheet_ae,  'AE')

# === 2. 合并两种信号，得到总表 ===
data_all = pd.concat([emr_long, ae_long], ignore_index=True)

# === 2.5 删掉空行（value 或 time 是 NaN 的都去掉） ===
data_all = data_all.dropna(subset=['value', 'time'])

# 可选：把 period_id 简化成数字 1/2/3...
data_all['period_id'] = (data_all['period_id']
                         .str.extract(r'(\d+)')
                         .astype(int))

# === 3. 保存 & 演示 ===
# data_all.to_parquet('tidy_emr_ae.parquet')   # 建议存 parquet，效率高
out_file = 'anlysis/clean_data/tidy2.xlsx'

with pd.ExcelWriter(out_file, engine='openpyxl') as writer:
    # 写 EMR
    data_all.query("signal_type == 'EMR'") \
        .to_excel(writer, sheet_name='EMR', index=False)

    # 写 AE
    data_all.query("signal_type == 'AE'") \
        .to_excel(writer, sheet_name='AE', index=False)

print(f'Saved to {out_file} （EMR+AE 两工作表）')

print(data_all.head())
