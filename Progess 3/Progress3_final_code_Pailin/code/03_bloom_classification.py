"""03 Bloom classification, round 2: will next week be a bloom week? (slide 6, slide 8 bottom)

ทาย bloom หรือไม่: 4 โมเดล x 2 ชุด input, เลือกด้วย cost matrix (Lecture 9) แล้วเปิด test ครั้งเดียว

Bloom label: next-week Chl-a above the 90th percentile of the TRAIN weeks (P90),
and, as a check, above a fixed 10 mg/m3.
Models: Logistic Regression, Random Forest, SVM (L15), MLP neural network (L14).
Baselines: Persistence (bloom next week if bloom this week), Never-bloom.

Steps (no peeking at validation or test):
  1. Tune each model on TRAIN with TimeSeriesSplit(5), scoring = PR-AUC.
  2. Choose the probability cutoff from out-of-fold TRAIN predictions (best F1).
  3. Compare models on VALIDATION. Choose the final model by the lowest validation cost
     (cost matrix, Lecture 9: missed bloom = 5, false alarm = 1).
  4. Refit the chosen model on train+validation and open the TEST set once.
"""
import warnings




import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, confusion_matrix, roc_auc_score
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from common import CHL_F, OUT, SEED, SET_C, build, onset_mask, split_masks, vif_keep

warnings.filterwarnings("ignore")
COST_FN, COST_FP = 5, 1
VIF_COLS = None          # a missed bloom costs 5 times a false alarm


def models():
    imp = ("imp", SimpleImputer(strategy="median"))
    sc = ("sc", StandardScaler())
    return {
        "LogisticRegression": (Pipeline([imp, sc, ("m", LogisticRegression(class_weight="balanced", max_iter=5000))]),
                               {"m__C": [0.01, 0.1, 1, 10]}),
        "RandomForest": (Pipeline([imp, ("m", RandomForestClassifier(n_estimators=300, class_weight="balanced",
                                                                      random_state=SEED, n_jobs=-1))]),
                         {"m__max_depth": [3, 5, 8], "m__min_samples_leaf": [1, 5, 10]}),
        "SVM": (Pipeline([imp, sc, ("m", SVC(class_weight="balanced", probability=True, random_state=SEED))]),
                {"m__C": [0.1, 1, 10], "m__gamma": ["scale", 0.01, 0.1]}),
        "MLP": (Pipeline([imp, sc, ("m", MLPClassifier(max_iter=2000, random_state=SEED))]),
                {"m__hidden_layer_sizes": [(8,), (16,), (16, 8)], "m__alpha": [0.001, 0.01, 0.1]}),
    }


def oof_proba(est, X, y, cv):
    """Out-of-fold probabilities on the train part (each fold predicts its next period)."""
    p = pd.Series(np.nan, index=y.index)
    for a, b in cv.split(X):
        if y.iloc[a].sum() == 0:
            continue
        m = clone(est).fit(X.iloc[a], y.iloc[a])
        p.iloc[b] = m.predict_proba(X.iloc[b])[:, 1]
    return p


def best_cutoff(y, p):
    ok = p.notna()
    y, p = y[ok], p[ok]
    best, best_f1 = 0.5, -1
    for c in np.arange(0.05, 0.96, 0.01):
        pred = (p >= c).astype(int)
        tp = ((pred == 1) & (y == 1)).sum(); fp = ((pred == 1) & (y == 0)).sum(); fn = ((pred == 0) & (y == 1)).sum()
        f1 = 2 * tp / max(2 * tp + fp + fn, 1)
        if f1 > best_f1:
            best, best_f1 = c, f1
    return round(float(best), 2)


def evaluate(y, pred, prob=None):
    y, pred = np.asarray(y).astype(int), np.asarray(pred).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    on = onset_mask(y)
    r = {"bloom_weeks": int(y.sum()), "TP": tp, "FN": fn, "FP": fp, "TN": tn,
         "precision": tp / (tp + fp) if tp + fp else 0.0,
         "recall": tp / (tp + fn) if tp + fn else 0.0,
         "F1": 2 * tp / (2 * tp + fp + fn) if tp else 0.0,
         "cost": COST_FN * fn + COST_FP * fp,
         "onsets": int(on.sum()), "onsets_caught": int((pred[on] == 1).sum())}
    if prob is not None and 0 < y.sum() < len(y):
        r["PR_AUC"] = average_precision_score(y, prob)
        r["ROC_AUC"] = roc_auc_score(y, prob)
    return r


def run(threshold_name, thr, X, y_raw, tr, va, te, sets, cv):
    lab = (y_raw > thr).astype(int)
    cur_bloom = (np.expm1(X["Chloro_current"]) > thr).astype(int)       # persistence baseline
    print(f"\n=== {threshold_name}: bloom if next-week Chl-a > {thr:.2f} mg/m3 | "
          f"bloom weeks train {lab[tr].sum()}, val {lab[va].sum()}, test {lab[te].sum()}")
    rows, keep = [], {}
    for name, pred in {"Persistence": cur_bloom, "Never_bloom": pd.Series(0, index=lab.index)}.items():
        rows.append({"threshold": threshold_name, "set": "-", "model": name, "cutoff": None,
                     **evaluate(lab[va], pred[va], None if name == "Never_bloom" else pred[va])})
    for s_name, cols_all in sets.items():
        for m, (est, grid) in models().items():
            # VIF only matters for the linear model; tree, SVM and MLP keep all 15 features
            cols = [c for c in cols_all if c in VIF_COLS] if (m == "LogisticRegression" and s_name == "C_all") else cols_all
            Xtr, ytr = X[tr][cols], lab[tr]
            gs = GridSearchCV(est, grid, cv=cv, scoring="average_precision", n_jobs=-1).fit(Xtr, ytr)
            cut = best_cutoff(ytr, oof_proba(gs.best_estimator_, Xtr, ytr, cv))
            p_va = gs.best_estimator_.predict_proba(X[va][cols])[:, 1]
            rows.append({"threshold": threshold_name, "set": s_name, "model": m, "cutoff": cut,
                         "params": str(gs.best_params_), **evaluate(lab[va], p_va >= cut, p_va)})
            keep[(s_name, m)] = (gs.best_estimator_, cols, p_va)
        print("  done", s_name)
    res = pd.DataFrame(rows)
    show = ["set", "model", "cutoff", "bloom_weeks", "TP", "FN", "FP", "precision", "recall", "F1", "cost",
            "onsets_caught", "onsets", "PR_AUC", "ROC_AUC"]
    print("VALIDATION:\n", res[show].round(3).to_string(index=False))
    return res, keep, lab, cur_bloom


def main():
    X, y, y_raw, _ = build()
    tr, va, te = split_masks(len(y))
    cv = TimeSeriesSplit(n_splits=5)

    global VIF_COLS
    VIF_COLS, _, _ = vif_keep(X[tr][SET_C])
    try:
        imp = pd.read_csv(OUT / "reg_02_feature_importance_trainCV.csv", index_col=0)
        top3 = [c for c in imp.index if c not in CHL_F][:3]
    except FileNotFoundError:
        raise SystemExit("Run 02_regression_v2.py first (it makes the train-CV top-3 features).")
    sets = {"C_all": SET_C, "G_4feat": top3 + ["Chloro_current"]}
    print("Sets:", sets)

    thr_p90 = float(np.percentile(y_raw[tr], 90))
    all_res, final_rows = [], []
    for t_name, thr in [("P90_train", thr_p90), ("Fixed_10", 10.0)]:
        res, keep, lab, cur_bloom = run(t_name, thr, X, y_raw, tr, va, te, sets, cv)
        all_res.append(res)

        # ---- choose on validation, then open test once
        cand = res[res.set != "-"].sort_values(["cost", "F1"], ascending=[True, False]).iloc[0]   # cost matrix, Lecture 9
        est, cols, p_va_best = keep[(cand.set, cand.model)]
        trva = tr | va
        fin = clone(est).fit(X[trva][cols], lab[trva])
        cut = best_cutoff(lab[trva], oof_proba(est, X[trva][cols], lab[trva], cv))
        p_te = fin.predict_proba(X[te][cols])[:, 1]
        print(f"Chosen on validation ({t_name}): {cand.model} / {cand.set}; test cutoff {cut}")
        final_rows.append({"threshold": t_name, "model": f"{cand.model} ({cand.set})", "cutoff": cut,
                           **evaluate(lab[te], p_te >= cut, p_te)})
        final_rows.append({"threshold": t_name, "model": "Persistence", **evaluate(lab[te], cur_bloom[te], cur_bloom[te])})
        final_rows.append({"threshold": t_name, "model": "Never_bloom", **evaluate(lab[te], np.zeros(te.sum()))})

        if t_name == "P90_train":
            # saved for the figures in 06_slide_figures.py
            pd.DataFrame({"y_raw": y_raw[te], "bloom": lab[te], "prob": p_te, "cutoff": cut,
                          "model": f"{cand.model} ({cand.set})"}).to_csv(OUT / "bloom_03_test_probability.csv")
            pd.Series({"val_bloom_share": lab[va].mean(), "threshold_mgm3": thr}).to_csv(OUT / "bloom_04_info.csv")

    pd.concat(all_res).round(4).to_csv(OUT / "bloom_01_validation_all.csv", index=False)
    ft = pd.DataFrame(final_rows)
    ft.round(4).to_csv(OUT / "bloom_02_test_once.csv", index=False)
    print("\nTEST (used once):\n", ft[["threshold", "model", "cutoff", "bloom_weeks", "TP", "FN", "FP", "precision",
                                      "recall", "F1", "cost", "onsets_caught", "onsets", "PR_AUC", "ROC_AUC"]].round(3).to_string(index=False))


if __name__ == "__main__":
    main()
