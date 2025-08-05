# predict.py
import pandas as pd
from pathlib import Path
import lightgbm as lgb, joblib
from feature_utils import slide_features

DATA_TEST = [Path("clean_data/tidy2.xlsx"),
             Path("clean_data/tidy3.xlsx")]
THRESHOLD = 0.5

emr_model = lgb.Booster(model_file="models/emr_model.txt")
stack_lr  = joblib.load("models/stack_lr.pkl")

def detect_one_file(path: Path):
    # —— EMR
    emr = pd.read_excel(path, sheet_name="EMR", parse_dates=["time"])
    fe_emr = slide_features(emr[["time","value"]]).add_suffix("_EMR")
    fe_emr = fe_emr.fillna(fe_emr.median())

    p1 = emr_model.predict(fe_emr)

    # —— AE 三维
    ae = pd.read_excel(path, sheet_name="AE", parse_dates=["time"])
    fe_ae = slide_features(ae[["time","value"]]) \
              .add_suffix("_AE")[["max_AE","p95_AE","zcr_AE"]]
    fe_ae = fe_ae.fillna(0.0)

    X2 = pd.concat([pd.Series(p1, index=fe_emr.index, name="p_emr"),
                    fe_ae], axis=1).fillna(0.0)
    p_final = stack_lr.predict_proba(X2)[:,1]

    # —— 区段化
    flag = (p_final > THRESHOLD).astype(int)
    seg, run = [], None
    for t, f in zip(X2.index, flag):
        if f and run is None:   run = [t, t]
        elif f:                 run[1] = t
        elif not f and run is not None:
            seg.append(run); run=None
    if run is not None: seg.append(run)

    df_seg = (pd.DataFrame(seg, columns=["start","end"])
                .assign(duration=lambda d:
                        (d["end"]-d["start"]).dt.total_seconds()/60)
                .query("duration>=2")
                .assign(file=path.name))
    return df_seg

out = pd.concat([detect_one_file(p) for p in DATA_TEST],
                ignore_index=True)
out.to_csv("interf_segments.csv", index=False)
print("done -> interf_segments.csv")
print(out.head())
