# 02_predict_interf.py -----------------------------------------------------
import pandas as pd
from pathlib import Path
import lightgbm as lgb
from feature_utils import slide_features

DATA_TEST = [Path("clean_data/tidy2.xlsx"), Path("clean_data/tidy3.xlsx")]
SHEETS = ["EMR", "AE"]
model = lgb.Booster(model_file="old/interf_model.txt")

def detect_one_file(path: Path) -> pd.DataFrame:
    dfs = []
    for sh in SHEETS:
        t = pd.read_excel(path, sheet_name=sh, parse_dates=["time"])
        t["signal_type"] = sh
        t["signal_type"] = t["signal_type"].str.upper()
        dfs.append(t)
    df = pd.concat(dfs, ignore_index=True)

    fe_emr = slide_features(df.query("signal_type=='EMR'")).add_suffix("_EMR")
    fe_ae  = slide_features(df.query("signal_type=='AE'")).add_suffix("_AE")
    X = fe_emr.join(fe_ae, how="outer")

    p = model.predict(X)
    flag = (p > 0.5).astype(int)          # ☆阈值可根据训练 PR 曲线细调

    # 合并区段
    segments, run = [], None
    for t, f in zip(X.index, flag):
        if f==1 and run is None:
            run = [t, t]
        elif f==1:
            run[1] = t
        elif f==0 and run is not None:
            segments.append(run); run=None
    if run is not None:
        segments.append(run)

    seg_df = (pd.DataFrame(segments, columns=["start","end"])
                .assign(duration=lambda d: (d["end"]-d["start"]
                        ).dt.total_seconds()/60)
                .query("duration>=2")
                .sort_values("start"))
    seg_df["file"] = path.name
    return seg_df

out = pd.concat([detect_one_file(p) for p in DATA_TEST], ignore_index=True)
out.to_csv("interf_segments.csv", index=False)
print("干扰区段已输出 → interf_segments.csv")
print(out.head())
