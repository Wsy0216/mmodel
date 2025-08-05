# 00_feature.py -------------------------------------------------------------
import pandas as pd
import numpy as np

# ===== feature_utils.py  顶部  =====
WIN  = "5min"     # ← 原来是 10min
STEP = "150s"     # ← 2.5 min（150 秒）

def slide_features(df_sig: pd.DataFrame) -> pd.DataFrame:
    """
    输入：含 ['time','value'] 的单一路径 DataFrame
    输出：以窗口末端时间戳为索引的特征 DataFrame
    """
    import scipy.signal as sg                    # ☆ 只在函数里 import
    s = df_sig.set_index("time")["value"]

    # —— ① 时域统计 ——
    stat = (s.rolling(WIN)
              .agg(["mean", "std", "min", "max", "skew", "kurt"])
              .resample(STEP).last())
    stat["p95"] = (s.rolling(WIN).quantile(0.95)
                     .resample(STEP).last())

    # —— ② 零交叉率（ZCR） ——
    zcr = (s.diff().fillna(0).gt(0) != s.diff().fillna(0).shift().gt(0)).astype(int)
    stat["zcr"] = (zcr.rolling(WIN).sum()
                     .resample(STEP).last())

    # —— ③ 频带能量占比 ——
    import scipy.signal as sg
    bands = [(0, 0.02), (0.02, 0.05), (0.05, 0.1), (0.1, 0.2), (0.2, 0.5)]
    fs = 1 / 30  # 30-s 采样

    for i, (a, b) in enumerate(bands, start=1):
        def band_energy_scalar(x, lo=a, hi=b):
            if x.isna().any():
                return np.nan
            f, P = sg.welch(x, fs=fs, nperseg=len(x))
            return P[(f >= lo) & (f < hi)].sum()

        stat[f"band{i}"] = (s.rolling(WIN)
                            .apply(band_energy_scalar, raw=False)
                            .resample(STEP).last())

    # 归一化为占比
    band_cols = [f"band{i}" for i in range(1, 6)]
    stat[band_cols] = stat[band_cols].div(stat[band_cols].sum(axis=1), axis=0)

    # ---------- 这里就直接返回 ----------
    return stat
