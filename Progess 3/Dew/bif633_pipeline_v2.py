"""BIF633 v2 - same features as Progress 2 slides + fair top-3 test.

Run:  py bif633_pipeline_v2.py
Out:  results_v2/

What is new vs v1
 - Slide features: Chloro_current, Chloro_lag1..5, Chloro_roll4, month_sin/cos
 - Slide feature sets: A env, B env+season, C env+season+Chl history
 - VIF check (train only) -> LR uses VIF-kept columns of set C
 - Top-3 test:
     D_slide3      month_sin, Silicate, Nitrate          (picked from slide 7)
     E_slide3_chl  D + Chloro_current
     F_cv3         top-3 picked by CV on TRAIN only (fair version)
 - Importance on TRAIN CV folds (stable, no peeking at val/test),
   per feature and per group (Chl history / Season / Environment)
 - Partial dependence plots (direction of effect)
 - Every score in log scale AND mg/m3
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import PartialDependenceDisplay
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

SEED = 42
HERE = Path(__file__).parent
DATA = HERE / "HABs_ScrippsPier_v3__3_.xlsx"
OUT = HERE / "results_v2"
OUT.mkdir(exist_ok=True)

TARGET = "Avg_Chloro"
NUTR = ["Phosphate", "Silicate", "Nitrite", "Ammonium", "Nitrate"]
ENV_F = ["Temp"] + NUTR
SEASON_F = ["month_sin", "month_cos"]
CHL_F = ["Chloro_current"] + [f"Chloro_lag{k}" for k in range(1, 6)] + ["Chloro_roll4"]
GROUPS = {"Chl history": CHL_F, "Season": SEASON_F, "Environment": ENV_F}
SLIDE_TOP3 = ["month_sin", "Silicate", "Nitrate"]


# ------------------------------------------------------------ data
def build():
    df = pd.read_excel(DATA, sheet_name=0).iloc[1:]
    df["time"] = pd.to_datetime(df["time"], utc=True)
    df = df.sort_values("time").drop_duplicates("time").set_index("time")
    df = df[[TARGET] + ENV_F].apply(pd.to_numeric, errors="coerce")
    df = df.interpolate(method="time", limit_direction="both")

    # outliers: flag with IQR for the report, keep them, use log1p
    q1, q3 = df[TARGET].quantile([0.25, 0.75])
    n_out = int((df[TARGET] > q3 + 1.5 * (q3 - q1)).sum())

    for c in [TARGET] + NUTR:
        df[c] = np.log1p(df[c])

    X = pd.DataFrame(index=df.index)
    X[ENV_F] = df[ENV_F]                                   # week t
    nxt = df.index.to_series().shift(-1)                   # week we predict
    X["month_sin"] = np.sin(2 * np.pi * nxt.dt.month / 12)
    X["month_cos"] = np.cos(2 * np.pi * nxt.dt.month / 12)
    X["Chloro_current"] = df[TARGET]                       # week t (known)
    for k in range(1, 6):
        X[f"Chloro_lag{k}"] = df[TARGET].shift(k)          # t-1 .. t-5
    X["Chloro_roll4"] = df[TARGET].rolling(4).mean()       # t-3 .. t
    y = df[TARGET].shift(-1).rename("y")                   # week t+1
    d = pd.concat([X, y], axis=1).dropna()
    return d.drop(columns="y"), d["y"], n_out


def vif_keep(X: pd.DataFrame, limit=10.0):
    cols = list(X.columns)
    log = []
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


def models():
    return {
        "LinearRegression": (Pipeline([("sc", StandardScaler()), ("m", LinearRegression())]), {}),
        "RandomForest": (
            RandomForestRegressor(n_estimators=300, random_state=SEED, n_jobs=-1),
            {"max_depth": [3, 5, 8], "min_samples_leaf": [1, 5, 10], "max_features": ["sqrt", 0.5, 1.0]},
        ),
        "SVR": (
            Pipeline([("sc", StandardScaler()), ("m", SVR())]),
            {"m__C": [0.1, 1, 10, 100], "m__epsilon": [0.05, 0.1, 0.2], "m__gamma": ["scale", 0.01, 0.1]},
        ),
    }


def score(yt, yp):
    return {"RMSE_log": mean_squared_error(yt, yp) ** 0.5,
            "R2_log": r2_score(yt, yp),
            "RMSE_mgm3": mean_squared_error(np.expm1(yt), np.expm1(yp)) ** 0.5,
            "MAE_mgm3": mean_absolute_error(np.expm1(yt), np.expm1(yp))}


def perm_rmse_rise(model, X, y, cols_or_groups, rng, n_rep=20):
    base = mean_squared_error(y, model.predict(X)) ** 0.5
    out = {}
    for name, cols in cols_or_groups.items():
        rises = []
        for _ in range(n_rep):
            Xp = X.copy()
            idx = rng.permutation(len(X))
            Xp[cols] = X[cols].values[idx]            # shuffle group jointly
            rises.append(mean_squared_error(y, model.predict(Xp)) ** 0.5 - base)
        out[name] = np.mean(rises)
    return out


def cv_importance(est, X, y, cv, units):
    """Fit on each TRAIN fold, permute on its next period. Returns folds x units."""
    rng = np.random.default_rng(SEED)
    rows = []
    for f, (a, b) in enumerate(cv.split(X)):
        m = clone(est).fit(X.iloc[a], y.iloc[a])
        r = perm_rmse_rise(m, X.iloc[b], y.iloc[b], units, rng)
        r["fold"] = f
        rows.append(r)
    return pd.DataFrame(rows).set_index("fold")


# ------------------------------------------------------------ main
def main():
    X_all, y, n_out = build()
    n = len(y)
    i_tr, i_va = int(n * 0.70), int(n * 0.85)
    tr = np.arange(n) < i_tr
    va = (np.arange(n) >= i_tr) & (np.arange(n) < i_va)
    te = np.arange(n) >= i_va
    print(f"Rows {n} | train {tr.sum()} | val {va.sum()} | test {te.sum()} | IQR outliers kept: {n_out}")

    cv = TimeSeriesSplit(n_splits=5)
    Xtr, ytr = X_all[tr], y[tr]

    # VIF on set C, train only
    setC = ENV_F + SEASON_F + CHL_F
    vif_cols, vif_vals, vif_log = vif_keep(Xtr[setC])
    print(f"VIF: kept {len(vif_cols)}/{len(setC)}  dropped {vif_log}")
    vif_vals.to_csv(OUT / "01_vif_kept.csv")

    # tune once on set C to get params used for importance
    tuned = {}
    for m, (est, grid) in models().items():
        cols = vif_cols if m == "LinearRegression" else setC
        gs = GridSearchCV(est, grid, cv=cv, scoring="neg_root_mean_squared_error", n_jobs=-1).fit(Xtr[cols], ytr)
        tuned[m] = (gs.best_estimator_, cols)

    # importance on train CV folds (set C)
    feat_imp, grp_imp = [], []
    for m, (est, cols) in tuned.items():
        fi = cv_importance(est, Xtr[cols], ytr, cv, {c: [c] for c in cols})
        gi = cv_importance(est, Xtr[cols], ytr, cv, {g: [c for c in v if c in cols] for g, v in GROUPS.items()})
        for c in fi:
            feat_imp.append({"model": m, "feature": c, "mean": fi[c].mean(), "std": fi[c].std(),
                             "folds_positive": int((fi[c] > 0).sum())})
        for g in gi:
            grp_imp.append({"model": m, "group": g, "mean": gi[g].mean(), "std": gi[g].std(),
                            "folds_positive": int((gi[g] > 0).sum())})
    feat_imp, grp_imp = pd.DataFrame(feat_imp), pd.DataFrame(grp_imp)
    feat_imp.to_csv(OUT / "02_feature_importance_cv.csv", index=False)
    grp_imp.to_csv(OUT / "03_group_importance_cv.csv", index=False)

    avg = feat_imp.groupby("feature")["mean"].mean().sort_values(ascending=False)
    cv_top3 = [c for c in avg.index if c not in CHL_F][:3]  # top-3 non-Chl, comparable to slide
    print("CV top-3 (non-Chl):", cv_top3)

    sets = {
        "A_env": ENV_F,
        "B_env_season": ENV_F + SEASON_F,
        "C_env_season_chl": setC,
        "D_slide3": SLIDE_TOP3,
        "E_slide3_chl": SLIDE_TOP3 + ["Chloro_current"],
        "F_cv3": cv_top3,
        "G_cv3_chl": cv_top3 + ["Chloro_current"],
    }

    rows = []
    # baselines
    for name, p in {"Persistence": X_all["Chloro_current"],
                    "Train_mean": pd.Series(ytr.mean(), index=y.index)}.items():
        for s, mk in [("train", tr), ("validation", va), ("test", te)]:
            rows.append({"set": "-", "model": name, "split": s, **score(y[mk], p[mk])})

    best_C = {}
    for s_name, base_cols in sets.items():
        for m, (est, grid) in models().items():
            cols = [c for c in base_cols if c in vif_cols] if (m == "LinearRegression" and s_name.startswith("C")) else base_cols
            gs = GridSearchCV(est, grid, cv=cv, scoring="neg_root_mean_squared_error", n_jobs=-1)
            gs.fit(X_all[tr][cols], ytr)
            b = gs.best_estimator_
            for s, mk in [("train", tr), ("validation", va)]:
                rows.append({"set": s_name, "model": m, "split": s, **score(y[mk], b.predict(X_all[mk][cols])),
                             "n_feat": len(cols), "cv_RMSE_log": -gs.best_score_, "params": str(gs.best_params_)})
            fin = clone(b).fit(X_all[tr | va][cols], y[tr | va])
            rows.append({"set": s_name, "model": m, "split": "test", **score(y[te], fin.predict(X_all[te][cols])),
                         "n_feat": len(cols), "params": str(gs.best_params_)})
            if s_name == "C_env_season_chl":
                best_C[m] = (fin, cols)
        print("  done", s_name)

    res = pd.DataFrame(rows)
    res.to_csv(OUT / "04_all_metrics.csv", index=False)

    def piv(metric):
        p = res.pivot_table(index=["set", "model"], columns="split", values=metric)
        return p.reindex(columns=["train", "validation", "test"])
    t = piv("RMSE_log").round(3)
    t["gap"] = (t["validation"] - t["train"]).round(3)
    t["val_mgm3"] = piv("RMSE_mgm3")["validation"].round(3)
    t["test_mgm3"] = piv("RMSE_mgm3")["test"].round(3)
    t["cv_log"] = res[res.split == "train"].set_index(["set", "model"])["cv_RMSE_log"].round(3)
    t.to_csv(OUT / "05_summary.csv")
    pd.set_option("display.width", 200)
    print("\nRMSE (log) train/val/test, gap, and mg/m3 val/test\n", t.to_string())

    print("\nGroup importance (train CV, rise in RMSE_log):")
    print(grp_imp.pivot(index="group", columns="model", values="mean").round(4).to_string())
    print("\nTop 8 features (mean of 3 models, train CV):")
    top8 = feat_imp.groupby("feature").agg(mean=("mean", "mean"), folds_pos=("folds_positive", "min"))
    print(top8.sort_values("mean", ascending=False).head(8).round(4).to_string())

    # ------------------------------------------------ figures
    g = grp_imp.pivot(index="group", columns="model", values="mean")
    ge = grp_imp.pivot(index="group", columns="model", values="std")
    ax = g.plot.barh(xerr=ge, figsize=(7, 3.5), capsize=3)
    ax.axvline(0, color="k", lw=0.8)
    ax.set_xlabel("Rise in RMSE (log) when group shuffled  (train CV, mean ± sd over 5 folds)")
    ax.set_title("Group importance, feature set C")
    plt.tight_layout(); plt.savefig(OUT / "fig1_group_importance.png", dpi=150); plt.close()

    fa = feat_imp.groupby("feature")[["mean", "std"]].mean().sort_values("mean")
    color = ["#8e3b5f" if c in CHL_F else "#b5a33a" if c in SEASON_F else "#2a9d8f" for c in fa.index]
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.barh(fa.index, fa["mean"], xerr=fa["std"], color=color, capsize=2)
    ax.axvline(0, color="k", lw=0.8)
    ax.set_xlabel("Rise in RMSE (log) when shuffled  (train CV, mean of 3 models)")
    ax.set_title("Feature importance, set C  (purple Chl / gold season / teal env)")
    fig.tight_layout(); fig.savefig(OUT / "fig2_feature_importance.png", dpi=150); plt.close()

    rf, cols = best_C["RandomForest"]
    show = [c for c in ["Chloro_current", "Temp", "Silicate", "Nitrate", "month_sin", "Ammonium"] if c in cols]
    fig, ax = plt.subplots(figsize=(12, 6))
    PartialDependenceDisplay.from_estimator(rf, X_all[tr | va][cols], show, ax=ax, n_cols=3)
    fig.suptitle("Partial dependence (Random Forest, set C): effect on log1p(next-week Chl-a)")
    fig.tight_layout(); fig.savefig(OUT / "fig3_partial_dependence.png", dpi=150); plt.close()

    print(f"\nSaved to {OUT}")


if __name__ == "__main__":
    main()
