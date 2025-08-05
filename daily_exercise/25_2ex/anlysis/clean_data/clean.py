import pandas as pd

# === 0. 路径与参数 ===
file_path = r"D:\软件数据\25数模集训\2025-hs-2\1\附件--预测煤矿深层开采时可能发生的冲击地压.xlsx"
out_file  = 'tidy_emr_ae.xlsx'

# === 1. 读取 & 处理 EMR ===
df_emr = pd.read_excel(file_path, sheet_name='EMR', engine='openpyxl')

df_emr['time']  = pd.to_datetime(df_emr['时间 (time)'])
df_emr['value'] = pd.to_numeric(df_emr['电磁辐射 (EMR)'], errors='coerce')
df_emr['class'] = df_emr['类别 (class)'].str.strip().str.upper()
df_emr = df_emr.dropna(subset=['value', 'time'])
df_emr = df_emr.sort_values('time')

df_emr['dt'] = df_emr['time'].diff().dt.total_seconds()
gap_rows_emr = df_emr[df_emr['dt'] > 90]

df_emr_resamp = (
    df_emr.set_index('time')
          .resample('30s')
          .agg({'value': 'mean', 'class': 'first'})
          .ffill()
)

train_emr = df_emr_resamp.query("`class` in ['A','B','C']").copy()
train_emr['is_noise'] = (train_emr['class'] == 'C').astype(int)
train_emr = train_emr[train_emr['value'].between(0, 150)]

mu_emr  = train_emr['value'].mean()
std_emr = train_emr['value'].std()
train_emr['value_z'] = (train_emr['value'] - mu_emr) / std_emr

emr_long = train_emr.reset_index().copy()
emr_long['signal_type'] = 'EMR'
emr_long['period_id'] = 1

# === 2. 读取 & 处理 AE ===
df_ae = pd.read_excel(file_path, sheet_name='AE', engine='openpyxl')

df_ae['time']  = pd.to_datetime(df_ae['时间 (time)'])
df_ae['value'] = pd.to_numeric(df_ae['声波强度 (AE)'], errors='coerce')
df_ae['class'] = df_ae['类别 (class)'].str.strip().str.upper()
df_ae = df_ae.dropna(subset=['value', 'time'])
df_ae = df_ae.sort_values('time')

df_ae['dt'] = df_ae['time'].diff().dt.total_seconds()
gap_rows_ae = df_ae[df_ae['dt'] > 90]

df_ae_resamp = (
    df_ae.set_index('time')
         .resample('30s')
         .agg({'value': 'mean', 'class': 'first'})
         .ffill()
)

train_ae = df_ae_resamp.query("`class` in ['A','B','C']").copy()
train_ae['is_noise'] = (train_ae['class'] == 'C').astype(int)
train_ae = train_ae[train_ae['value'].between(0, 150)]

mu_ae  = train_ae['value'].mean()
std_ae = train_ae['value'].std()
train_ae['value_z'] = (train_ae['value'] - mu_ae) / std_ae

ae_long = train_ae.reset_index().copy()
ae_long['signal_type'] = 'AE'
ae_long['period_id'] = 1

# === 3. 合并并保存 ===
data_all = pd.concat([emr_long, ae_long], ignore_index=True)
data_all = data_all.dropna(subset=['value', 'time'])
data_all['period_id'] = data_all['period_id'].astype(int)

with pd.ExcelWriter(out_file, engine='openpyxl',
                    datetime_format='yyyy-mm-dd hh:mm:ss') as writer:
    for sig in ('EMR', 'AE'):
        (data_all.query("signal_type == @sig")
                 .to_excel(writer, sheet_name=sig, index=False))

print("✅ 处理完成，文件已保存为：", out_file)
