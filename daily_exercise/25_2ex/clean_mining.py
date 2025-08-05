# ── qc_clean_mining_signals.py ──────────────────────────────────────────────
import pandas as pd
import numpy as np
from pathlib import Path

# ────────── 参数区 ──────────
INPUT_FILES = [
    r"D:\pythonProject\pythonProject1\daily_exercise\25_2ex\tidy2.xlsx",  # <—— 请替换成你的第 1 个文件
    r"D:\pythonProject\pythonProject1\daily_exercise\25_2ex\tidy3.xlsx",  # <—— 请替换成你的第 2 个文件
]
OUT_DIR = Path("anlysis/clean_data")  # 输出目录
FREQ = "30s"  # 采样间隔
SHORT_GAP_LIMIT = 20  # <10 min 认为可插值 (=20×30 s)
MAD_K = 3  # 3×MAD 判极端值


# ──────────────────────────


def read_one_file(path: str) -> pd.DataFrame:
    """读一个 Excel，拼接所有工作表，保证 sheet 名在 signal_type 列中"""
    xls = pd.ExcelFile(path)
    frames = []
    for sh in xls.sheet_names:  # 通常是 ['EMR','AE']
        df = pd.read_excel(xls, sheet_name=sh)
        if 'signal_type' not in df.columns:
            df['signal_type'] = sh
        frames.append(df)
    df_all = pd.concat(frames, ignore_index=True)
    return df_all


def qc_clean(df_raw: pd.DataFrame) -> pd.DataFrame:
    """核心质控与清洗流程，返回长表"""
    # 0. 基础整理 ------------------------------------------------------------
    df = (df_raw.copy()
          .rename(columns=str.lower)
          .assign(time=lambda d: pd.to_datetime(d["time"]))
          .sort_values("time"))

    # 1. 去重复 --------------------------------------------------------------
    df = (df.groupby(["time", "signal_type", "period_id"], as_index=False)
          .agg(value=("value", "mean")))

    # 2. 透视 → 统一 30 s 时间轴 --------------------------------------------
    pivot = df.pivot_table(index="time", columns="signal_type", values="value")
    full_idx = pd.date_range(pivot.index.min(), pivot.index.max(), freq=FREQ)
    pivot = pivot.reindex(full_idx)

    # 3. 缺测处理 ------------------------------------------------------------
    #   3.1 记录连续缺测段长度
    mask_nan = pivot.isna().any(axis=1)  # True 行表示该时刻有至少一个 NaN
    grp = mask_nan.ne(mask_nan.shift()).cumsum()
    run_length = mask_nan.groupby(grp).transform("sum")

    #   3.2 长空洞(≥SHORT_GAP_LIMIT) 保留 NaN 并标 class='D'
    long_gap = (mask_nan & (run_length >= SHORT_GAP_LIMIT))
    pivot["class"] = np.where(long_gap, "D", "N")  # N=normal 先占位

    #   3.3 其余短空洞插值
    pivot.loc[~long_gap, :] = (
        pivot.loc[~long_gap, :].interpolate(limit=SHORT_GAP_LIMIT - 1,
                                            limit_direction="both"))

    # 4. 飞点检测（逐列 3×MAD）----------------------------------------------
    for col in pivot.columns.drop("class"):
        med = pivot[col].median(skipna=True)
        mad = (np.abs(pivot[col] - med)).median(skipna=True)
        thr = MAD_K * 1.4826 * mad
        outlier_mask = np.abs(pivot[col] - med) > thr
        pivot["is_noise_" + col] = outlier_mask.astype(int)

    # 5. 长表输出（方便后续滑窗）--------------------------------------------
    raw_cols = [c for c in pivot.columns
                if c not in ["class"] and not c.startswith("is_noise_")]

    long_df = (pivot
               .reset_index(names="time")
               .melt(id_vars=["time", "class"],
                     value_vars=raw_cols,
                     var_name="signal_type",
                     value_name="value")
               .dropna(subset=["value"]))

    long_df["value_z"] = (long_df.groupby("signal_type")["value"]
                          .transform(lambda s: (s - s.mean()) / s.std()))

    long_df["is_noise"] = long_df.apply(
        lambda r: pivot.at[r["time"], "is_noise_" + r["signal_type"]],
        axis=1
    )
    return long_df


def main():
    OUT_DIR.mkdir(exist_ok=True)
    for f in INPUT_FILES:
        print(f"▶ Processing {f} …")
        raw = read_one_file(f)
        clean_long = qc_clean(raw)

        # 保存
        stem = Path(f).stem
        clean_long.to_parquet(OUT_DIR / f"{stem}_clean_long.parquet")

        # 简要 QC 报告
        miss_pct = clean_long["value"].isna().mean() * 100
        noise_pct = clean_long["is_noise"].mean() * 100
        print(f"   ↳ missing {miss_pct:.2f} %,  noise {noise_pct:.2f} %")
    print("✅ All done!  清洗后文件已保存到", OUT_DIR.resolve())


if __name__ == "__main__":
    main()
# ───────────────────────────────────────────────────────────────────────────
