"""06 Figures for slides 5, 6 and 8, made from the result tables of scripts 02 and 03.

  s05_regression_validation.png - best model of each type vs simple baselines (validation RMSE)
  s06_bloom_cost.png            - cost matrix result (Lecture 9) of 4 classifiers, 15 vs 4 inputs
  s08_final_test.png            - test weeks (used once): Chl-a prediction and bloom probability

วาดรูปสำหรับสไลด์ 5, 6, 8 จากไฟล์ CSV ของ 02 และ 03 (ตัวเลขในรูป = ตัวเลขในตาราง)
"""
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pandas as pd

import style
from common import FIG, OUT

# =============== slide 5: regression, validation RMSE
r = pd.read_csv(OUT / "reg_03_train_val_all.csv")
best = r[r.set != "-"].sort_values("RMSE_mgm3_val").groupby("model").head(1)   # best input set of each model
base = r[r.set == "-"]                                                          # simple baselines
t = pd.concat([best, base]).sort_values("RMSE_mgm3_val", ascending=False).reset_index(drop=True)
names = {"LinearRegression": "Linear Regression", "RandomForest": "Random Forest", "SVR": "SVR",
         "Persistence": "Persistence", "Train_median": "Train median", "Seasonal_median": "Seasonal median"}
sets = {"G_cv3_chl": "4 inputs", "F_cv3": "3 inputs", "C_env_season_chl": "set C", "A_env": "set A", "B_env_season": "set B"}
labels = [names[m] + (f"  ({sets[s]})" if s != "-" else "") for m, s in zip(t.model, t.set)]
fig, ax = plt.subplots(figsize=(10.4, 6.4))
ax.barh(range(len(t)), t["RMSE_mgm3_val"], color=[style.GREY if s == "-" else style.TEAL for s in t.set], height=0.62)
for i, v in enumerate(t["RMSE_mgm3_val"]):
    ax.text(v + 0.05, i, f"{v:.2f}", va="center", fontsize=17)
ax.set_yticks(range(len(t)), labels)
ax.set_xlim(0, 5.4)
ax.set_xlabel("Validation RMSE (mg/m³), lower is better")
ax.xaxis.grid(True, color=style.GRID)
ax.set_axisbelow(True)
fig.tight_layout()
style.save(fig, FIG / "s05_regression_validation.png")
lr, pers = t.iloc[-1]["RMSE_mgm3_val"], base.set_index("model").loc["Persistence", "RMSE_mgm3_val"]
print(f"slide 5: best {t.iloc[-1].model} {lr:.2f} vs persistence {pers:.2f} -> {100 * (1 - lr / pers):.0f}% lower")

# =============== slide 6: bloom classification, cost matrix (missed bloom = 5, false alarm = 1)
b = pd.read_csv(OUT / "bloom_01_validation_all.csv")
b = b[b.threshold == "P90_train"]
order = ["MLP", "SVM", "RandomForest", "LogisticRegression"]
nm = {"MLP": "MLP (neural net)", "SVM": "SVM", "RandomForest": "Random Forest", "LogisticRegression": "Logistic Regression"}
cost15 = b[b.set == "C_all"].set_index("model")["cost"]
cost4 = b[b.set == "G_4feat"].set_index("model")["cost"]
cost_pers = b.loc[b.model == "Persistence", "cost"].item()
cost_never = b.loc[b.model == "Never_bloom", "cost"].item()
fig, ax = plt.subplots(figsize=(10.4, 6.4))
for i, m in enumerate(order):
    ax.plot([cost15[m], cost4[m]], [i, i], color=style.GRID, lw=3, zorder=1)
    ax.scatter(cost15[m], i, s=170, color=style.BG, edgecolor=style.OLIVE, lw=2.5, zorder=3, label="15 inputs" if i == 0 else None)
    ax.scatter(cost4[m], i, s=170, color=style.PLUM, zorder=3, label="4 inputs" if i == 0 else None)
    ax.text(cost4[m], i + 0.22, f"{cost4[m]:.0f}", ha="center", fontsize=16, color=style.PLUM)
ax.axvline(cost_pers, color=style.MUTED, ls="--", lw=1.5)
ax.text(cost_pers + 0.4, 3.6, f"persistence {cost_pers:.0f}", color=style.MUTED, fontsize=16)
ax.axvline(cost_never, color=style.GREY, ls=":", lw=1.5)
ax.text(cost_never - 0.4, 3.6, f"never bloom {cost_never:.0f}", color=style.GREY, fontsize=16, ha="right")
ax.set_yticks(range(4), [nm[m] for m in order])
ax.set_ylim(-0.6, 4.0)
ax.set_xlim(28, 58)
ax.set_xlabel("Validation cost (missed bloom = 5, false alarm = 1), lower is better")
ax.xaxis.grid(True, color=style.GRID)
ax.set_axisbelow(True)
ax.legend(frameon=False, loc="lower right", fontsize=16)
fig.tight_layout()
style.save(fig, FIG / "s06_bloom_cost.png")
print("slide 6 cost, 4 inputs:", cost4.to_dict(), "| persistence", cost_pers, "| never bloom", cost_never)

# =============== slide 8: final test (used once)
d = pd.read_csv(OUT / "reg_05_test_predictions.csv", index_col=0, parse_dates=True)
p = pd.read_csv(OUT / "bloom_03_test_probability.csv", index_col=0, parse_dates=True)
d.index = d.index.tz_localize(None)
p.index = p.index.tz_localize(None)
fig, (a1, a2) = plt.subplots(2, 1, figsize=(12.0, 7.0), sharex=True, gridspec_kw={"height_ratios": [1.35, 1]})
a1.plot(d.index, d["y_true_mgm3"], color=style.PLUM, lw=2.4, label="Observed")
a1.plot(d.index, d["pred_mgm3"], color=style.TEAL, lw=2.4, label="Linear Regression (4 inputs)")
a1.set_ylabel("Chl-a (mg/m³)")
a1.set_ylim(0, 24)
a1.yaxis.grid(True, color=style.GRID)
a1.set_axisbelow(True)
a1.legend(frameon=False, loc="upper right", fontsize=16)
cut = p["cutoff"].iloc[0]
half_week = pd.Timedelta(days=3.5)
for k, week in enumerate(p.index[p["bloom"] == 1]):
    a2.axvspan(week - half_week, week + half_week, color=style.PLUM, alpha=0.22, lw=0, label="real bloom week" if k == 0 else None)
a2.plot(p.index, p["prob"], color=style.TEAL, lw=2.4, label="bloom probability")
a2.axhline(cut, color=style.MUTED, ls="--", lw=1.5, label=f"cut-off {cut}")
a2.set_ylim(0, 1)
a2.set_ylabel("probability")
a2.yaxis.grid(True, color=style.GRID)
a2.set_axisbelow(True)
a2.legend(frameon=False, loc="upper right", fontsize=15, ncol=3)
a2.xaxis.set_major_locator(mdates.MonthLocator(bymonth=[1, 4, 7, 10]))
a2.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
fig.tight_layout()
style.save(fig, FIG / "s08_final_test.png")
tt = pd.read_csv(OUT / "reg_04_test_once.csv")
bt = pd.read_csv(OUT / "bloom_02_test_once.csv")
print("slide 8 regression test:\n", tt[["model", "RMSE_mgm3"]].round(2).to_string(index=False))
print("slide 8 bloom test:\n", bt[bt.threshold == "P90_train"][["model", "F1", "ROC_AUC", "onsets_caught", "onsets"]].round(2).to_string(index=False))
