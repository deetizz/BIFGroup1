"""02 Regression, round 2: predict log1p(Chl-a) next week (slide 5, slide 8 top).

ทายค่า Chl-a สัปดาห์หน้า: 5 ชุด input x 3 โมเดล เลือกจาก validation แล้วเปิด test ครั้งเดียว
The 4 inputs (G_cv3_chl) = the 3 most useful non-Chl inputs (feature importance on TRAIN folds:
shuffle one input, see how much the error rises, as in Progress 2) + this week's Chl-a.

Feature sets: A env, B env+season, C env+season+Chl history,
F_cv3 top-3 non-Chl features from TRAIN CV importance, G_cv3_chl = F + Chloro_current.
Models: Linear Regression (L12-13), Random Forest, SVR (L15).
Model choice: lowest validation RMSE (mg/m3). Test is used once, for that model.
"""
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

from common import (CHL_F, ENV_F, OUT, SEASON_F, SEED, SET_C, build, split_masks, vif_keep)


def models():
    imp = ("imp", SimpleImputer(strategy="median"))
    return {
        "LinearRegression": (Pipeline([imp, ("sc", StandardScaler()), ("m", LinearRegression())]), {}),
        "RandomForest": (Pipeline([imp, ("m", RandomForestRegressor(n_estimators=300, random_state=SEED, n_jobs=-1))]),
                         {"m__max_depth": [3, 5, 8], "m__min_samples_leaf": [1, 5, 10],
                          "m__max_features": ["sqrt", 0.5, 1.0]}),
        "SVR": (Pipeline([imp, ("sc", StandardScaler()), ("m", SVR())]),
                {"m__C": [0.1, 1, 10, 100], "m__epsilon": [0.05, 0.1, 0.2], "m__gamma": ["scale", 0.01, 0.1]}),
    }


def score(yt, yp):
    a, b = np.expm1(yt), np.expm1(yp)
    return {"RMSE_log": mean_squared_error(yt, yp) ** 0.5,
            "RMSE_mgm3": mean_squared_error(a, b) ** 0.5,
            "MAE_mgm3": mean_absolute_error(a, b)}


def cv_perm_importance(est, X, y, cv, n_rep=10):
    """Fit on each train fold, shuffle one feature at a time on the next period."""
    rng = np.random.default_rng(SEED)
    rows = []
    for a, b in cv.split(X):
        m = clone(est).fit(X.iloc[a], y.iloc[a])
        Xb, yb = X.iloc[b], y.iloc[b]
        base = mean_squared_error(yb, m.predict(Xb)) ** 0.5
        r = {}
        for c in X.columns:
            rise = []
            for _ in range(n_rep):
                Xp = Xb.copy()
                Xp[c] = Xb[c].values[rng.permutation(len(Xb))]
                rise.append(mean_squared_error(yb, m.predict(Xp)) ** 0.5 - base)
            r[c] = np.mean(rise)
        rows.append(r)
    return pd.DataFrame(rows).mean()


def main():
    X, y, y_raw, n_missing = build()
    tr, va, te = split_masks(len(y))
    print(f"Rows {len(y)} | train {tr.sum()} | val {va.sum()} | test {te.sum()} | missing values filled (past only): {n_missing}")
    cv = TimeSeriesSplit(n_splits=5)
    Xtr, ytr = X[tr], y[tr]

    vif_cols, vif_vals, vif_log = vif_keep(Xtr[SET_C])
    print("VIF dropped:", vif_log)
    vif_vals.to_csv(OUT / "reg_01_vif_kept.csv")

    # fair top-3: importance on TRAIN CV folds only (mean of 3 tuned models on set C)
    imps = []
    for m, (est, grid) in models().items():
        cols = vif_cols if m == "LinearRegression" else SET_C
        best = GridSearchCV(est, grid, cv=cv, scoring="neg_root_mean_squared_error", n_jobs=-1).fit(Xtr[cols], ytr).best_estimator_
        imps.append(cv_perm_importance(best, Xtr[cols], ytr, cv).rename(m))
    imp = pd.concat(imps, axis=1)
    imp["mean"] = imp.mean(axis=1)
    imp = imp.sort_values("mean", ascending=False)
    imp.to_csv(OUT / "reg_02_feature_importance_trainCV.csv")
    cv_top3 = [c for c in imp.index if c not in CHL_F][:3]
    print("Train-CV top-3 non-Chl:", cv_top3)

    sets = {"A_env": ENV_F, "B_env_season": ENV_F + SEASON_F, "C_env_season_chl": SET_C,
            "F_cv3": cv_top3, "G_cv3_chl": cv_top3 + ["Chloro_current"]}

    rows, fitted = [], {}
    for s_name, base_cols in sets.items():
        for m, (est, grid) in models().items():
            cols = [c for c in base_cols if c in vif_cols] if (m == "LinearRegression" and s_name.startswith("C")) else base_cols
            gs = GridSearchCV(est, grid, cv=cv, scoring="neg_root_mean_squared_error", n_jobs=-1).fit(Xtr[cols], ytr)
            b = gs.best_estimator_
            r = {"set": s_name, "model": m, "n_feat": len(cols), "params": str(gs.best_params_)}
            for s, mk in [("train", tr), ("val", va)]:
                for k, v in score(y[mk], b.predict(X[mk][cols])).items():
                    r[f"{k}_{s}"] = v
            rows.append(r)
            fitted[(s_name, m)] = (b, cols)
        print(" done", s_name)
    res = pd.DataFrame(rows)
    res["gap_log"] = res["RMSE_log_val"] - res["RMSE_log_train"]

    # baselines on validation
    month_next = (X.index + pd.Timedelta(days=7)).month
    seas = pd.Series(y_raw[tr].groupby(month_next[tr]).median())
    base = {"Persistence": np.log1p(np.expm1(X["Chloro_current"])),
            "Train_median": pd.Series(np.log1p(y_raw[tr].median()), index=y.index),
            "Seasonal_median": pd.Series(np.log1p(seas.reindex(month_next).values), index=y.index)}
    for name, p in base.items():
        r = {"set": "-", "model": name}
        for s, mk in [("train", tr), ("val", va)]:
            for k, v in score(y[mk], p[mk]).items():
                r[f"{k}_{s}"] = v
        rows.append(r)
    res = pd.DataFrame(rows)
    res["gap_log"] = res["RMSE_log_val"] - res["RMSE_log_train"]
    res.round(4).to_csv(OUT / "reg_03_train_val_all.csv", index=False)

    models_only = res[res.set != "-"]
    pick = models_only.sort_values("RMSE_mgm3_val").iloc[0]
    print("\nValidation table (sorted by RMSE mg/m3):")
    print(res.sort_values("RMSE_mgm3_val")[["set", "model", "RMSE_mgm3_val", "MAE_mgm3_val", "RMSE_log_train", "RMSE_log_val", "gap_log"]].round(3).to_string(index=False))
    print(f"\nChosen on validation: {pick.model} / {pick.set}")

    # ---- test, once: chosen model refit on train+val, plus baselines
    b, cols = fitted[(pick.set, pick.model)]
    fin = clone(b).fit(X[tr | va][cols], y[tr | va])
    p_te = pd.Series(fin.predict(X[te][cols]), index=y[te].index)
    seas2 = pd.Series(y_raw[tr | va].groupby(month_next[tr | va]).median())
    test_preds = {f"{pick.model} ({pick.set})": p_te,
                  "Persistence": X["Chloro_current"][te],
                  "Train_median": pd.Series(np.log1p(y_raw[tr | va].median()), index=p_te.index),
                  "Seasonal_median": pd.Series(np.log1p(seas2.reindex(month_next[te]).values), index=p_te.index)}
    trows = []
    yt = y[te]
    grp = pd.cut(y_raw[te], [-1, 5, 10, np.inf], labels=["<=5", "5-10", ">10"])
    for name, p in test_preds.items():
        r = {"model": name, **score(yt, p)}
        for g in grp.cat.categories:
            mk = (grp == g).values
            r[f"MAE_{g}"] = mean_absolute_error(np.expm1(yt[mk]), np.expm1(p[mk]))
        trows.append(r)
    tt = pd.DataFrame(trows)
    print("\nTEST (used once):\n", tt.round(3).to_string(index=False))
    print("Test weeks per group:", grp.value_counts().to_dict())
    tt.round(4).to_csv(OUT / "reg_04_test_once.csv", index=False)
    pd.DataFrame({"y_true_mgm3": np.expm1(yt), "pred_mgm3": np.expm1(p_te),
                  "persistence_mgm3": np.expm1(X["Chloro_current"][te])}).to_csv(OUT / "reg_05_test_predictions.csv")


if __name__ == "__main__":
    main()
