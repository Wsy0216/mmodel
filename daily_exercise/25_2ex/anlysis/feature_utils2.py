# feature_utils.py
import pandas as pd, numpy as np
import scipy.signal as sg

# ─── 全局窗长与步长 ────────────────────────────────────────────────
WIN  = "2min"     # 滑窗长度
STEP = "60s"     # 步长（=2.5 min，50 % overlap）

# ─── 主函数 ───────────────────────────────────────────────────────
def slide_features(df_sig: pd.DataFrame) -> pd.DataFrame:
    """
    df_sig 必含 ['time','value']；返回窗口末端时间戳索引的特征 DataFrame
    """
    s = df_sig.set_index("time")["value_z"]

    # ① 时域统计
    stat = (s.rolling(WIN)
              .agg(["mean", "std", "min", "max", "skew", "kurt"])
              .resample(STEP).last())
    stat["p95"] = (s.rolling(WIN).quantile(0.95)
                     .resample(STEP).last())

    # ── 衍生动态特征 ───────────────────────────
    stat["range"] = stat["max"] - stat["min"]

    stat["rms"] = (
            s.pow(2).rolling(WIN).mean()  # 先得到 Series
            ** 0.5  # 完整括号
    ).resample(STEP).last()

    stat["absdiff_sum"] = (
        s.diff().abs().rolling(WIN).sum()
    ).resample(STEP).last()

    # 超阈尖峰计数
    thr_value = s.mean() + 3 * s.std()  # 这里结束，绝不连 resample
    spike = (s > thr_value).astype(int)
    stat["spike_cnt"] = (
        spike.rolling(WIN).sum()
    ).resample(STEP).last()

    # 能量
    stat["energy"] = (
        s.pow(2).rolling(WIN).sum()
    ).resample(STEP).last()

    # ② 零交叉率
    sign_change = (s.diff().fillna(0).gt(0) !=
                   s.diff().fillna(0).shift().gt(0)).astype(int)
    stat["zcr"] = (sign_change.rolling(WIN).sum()
                                 .resample(STEP).last())

    # ③ 频带能量占比（5 段）
    bands = [(0,0.02), (0.02,0.05), (0.05,0.1),
             (0.1,0.2), (0.2,0.5)]
    fs = 1/30        # 30 s 采样

    for i, (lo, hi) in enumerate(bands, start=1):
        def _band_energy(x, lo=lo, hi=hi):
            if x.isna().any():           # 整窗若有缺失 → NaN
                return np.nan
            f, Pxx = sg.welch(x, fs=fs, nperseg=len(x))
            return Pxx[(f>=lo)&(f<hi)].sum()

        stat[f"band{i}"] = (s.rolling(WIN)
                              .apply(_band_energy, raw=False)
                              .resample(STEP).last())

    # 占比归一化
    band_cols = [f"band{i}" for i in range(1,6)]
    stat[band_cols] = stat[band_cols].div(
        stat[band_cols].sum(axis=1), axis=0
    )

    # 最大单步跳变幅度
    stat["max_change"] = (
        s.diff().abs().rolling(WIN).max()
    ).resample(STEP).last()

    # 连续单调上升长度
    up_seq = (s.diff() > 0).astype(int)
    stat["up_len"] = (
            up_seq.groupby((up_seq == 0).cumsum()).cumcount() + 1
    ).rolling(WIN).max().resample(STEP).last()

    # 高频能量 (f > 0.15 Hz)
    def hf_energy(x):
        if x.isna().any(): return np.nan
        f, P = sg.welch(x, 1 / 30, nperseg=len(x))
        return P[f > 0.15].sum()

    stat["hf"] = (
        s.rolling(WIN).apply(hf_energy, raw=False)
    ).resample(STEP).last()

    # RMS 的增量
    stat["rms_diff"] = stat["rms"].diff()

    # 尖峰均值
    spk = s.where(s > thr_value)
    stat["spike_mean"] = (
        spk.rolling(WIN).mean()
    ).resample(STEP).last()

    return stat
