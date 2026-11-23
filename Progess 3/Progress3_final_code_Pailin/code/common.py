"""common.py - shared settings and data preparation (used by every script).

ไฟล์นี้: อ่านข้อมูล, เติมค่าที่หาย (ใช้ค่าจากอดีตเท่านั้น), สร้าง feature, แบ่ง train/validation/test ตามเวลา


Changes from Dew's v2 (bif633_pipeline_v2.py):
  1. No interpolation over the whole series. Missing values are filled with the
     last known value (forward fill = past only). Anything still missing is
     filled inside each model Pipeline with the TRAIN median (SimpleImputer).
  2. Feature sets chosen from validation results (D_slide3, E_slide3_chl) are removed.
  3. Models are chosen on validation. The test set is opened once, for the
     chosen model and the baselines only (see 02_ and 03_).
"""
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

SEED = 42
HERE = Path(__file__).resolve().parent
DATA = HERE.parent / "HABs_ScrippsPier_v3.xlsx"
OUT = HERE.parent / "results"              # tables (CSV)
OUT.mkdir(exist_ok=True)
FIG = OUT / "slide_figures"                # figures used in the slides
FIG.mkdir(exist_ok=True)

TARGET = "Avg_Chloro"
NUTR = ["Phosphate", "Silicate", "Nitrite", "Ammonium", "Nitrate"]
ENV_F = ["Temp"] + NUTR
SEASON_F = ["month_sin", "month_cos"]
CHL_F = ["Chloro_current"] + [f"Chloro_lag{k}" for k in range(1, 6)] + ["Chloro_roll4"]
SET_C = ENV_F + SEASON_F + CHL_F
GROUPS = {"Chl history": CHL_F, "Season": SEASON_F, "Environment": ENV_F}


def load_raw():
    df = pd.read_excel(DATA, sheet_name=0).iloc[1:]          # row 0 = units
    df["time"] = pd.to_datetime(df["time"], utc=True)
    df = df.sort_values("time").drop_duplicates("time").set_index("time")
    df = df[[TARGET] + ENV_F].apply(pd.to_numeric, errors="coerce")
    n_missing = int(df.isna().sum().sum())
    df = df.ffill()                                            # past only, no future values
    return df, n_missing


def build():
    """X at week t, y = log1p(Chl-a) at week t+1, raw Chl-a at t+1 for labels."""
    raw, n_missing = load_raw()
    df = raw.copy()
    for c in [TARGET] + NUTR:
        df[c] = np.log1p(df[c])

    X = pd.DataFrame(index=df.index)
    X[ENV_F] = df[ENV_F]
    nxt = df.index.to_series().shift(-1)                      # date of the week we predict
    X["month_sin"] = np.sin(2 * np.pi * nxt.dt.month / 12)
    X["month_cos"] = np.cos(2 * np.pi * nxt.dt.month / 12)
    X["Chloro_current"] = df[TARGET]
    for k in range(1, 6):
        X[f"Chloro_lag{k}"] = df[TARGET].shift(k)
    X["Chloro_roll4"] = df[TARGET].rolling(4).mean()
    y = df[TARGET].shift(-1).rename("y")
    y_raw = raw[TARGET].shift(-1).rename("y_raw")
    d = pd.concat([X, y, y_raw], axis=1).dropna()
    return d[X.columns], d["y"], d["y_raw"], n_missing


def split_masks(n):
    i_tr, i_va = int(n * 0.70), int(n * 0.85)
    idx = np.arange(n)
    return idx < i_tr, (idx >= i_tr) & (idx < i_va), idx >= i_va


def vif_keep(X, limit=10.0):
    cols, log = list(X.columns), []
    while True:
        v = {}
        for c in cols:
            o = [k for k in cols if k != c]
            r2 = LinearRegression().fit(X[o], X[c]).score(X[o], X[c])
            v[c] = 1 / max(1 - r2, 1e-9)
        worst = max(v, key=v.get)
        if v[worst] <= limit:
            return cols, pd.Series(v).round(2), log
        log.append((worst, round(v[worst], 1)))
        cols.remove(worst)


def onset_mask(label):
    """First week of each bloom episode: bloom now, no bloom the week before."""
    lab = np.asarray(label).astype(int)
    prev = np.r_[0, lab[:-1]]
    return (lab == 1) & (prev == 0)
