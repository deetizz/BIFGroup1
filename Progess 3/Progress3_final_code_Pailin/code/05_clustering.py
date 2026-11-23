"""05 Clustering of weekly water conditions (Lectures 12-13), slide 7 right.

Question: do the weeks fall into a few "water types", and do blooms happen more in one type?
  - Inputs: temperature + 5 nutrients of the same week (log for nutrients), standardised.
  - Chl-a is NOT used to make the groups; we only use it afterwards to count bloom weeks.
  - k-means for k = 2..8; choose k with the highest silhouette (Lecture 12-13).
  - Check with hierarchical clustering (Ward): do we get similar groups?
This is descriptive (no prediction), so all weeks are used.

จัดกลุ่มสัปดาห์ตามสภาพน้ำ (ไม่ใช้ Chl-a) แล้วดูว่ากลุ่มไหนเกิด bloom บ่อย
Output: results/clu_01_silhouette.csv, results/clu_02_profiles.csv, results/clu_03_centres.csv,
        results/slide_figures/s07b_clusters_bloom_rate.png
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import fcluster, linkage
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, silhouette_score
from sklearn.preprocessing import StandardScaler

import style
from common import ENV_F, FIG, NUTR, OUT, SEED, TARGET, build, load_raw, split_masks

raw, _ = load_raw()
_, y, y_raw, _ = build()
tr, _, _ = split_masks(len(y))
bloom_line = float(np.percentile(y_raw[tr], 90))          # same bloom line as script 03 (6.2 mg/m3)

# ---- 1. inputs: water conditions only
Z = raw[ENV_F].copy()
Z[NUTR] = np.log1p(Z[NUTR])
Zs = StandardScaler().fit_transform(Z)

# ---- 2. choose k by silhouette (closer to 1 = clear groups, close to 0 = groups overlap)
sil = {}
for k in range(2, 9):
    labels = KMeans(n_clusters=k, n_init=20, random_state=SEED).fit_predict(Zs)
    sil[k] = silhouette_score(Zs, labels)
sil = pd.Series(sil, name="silhouette")
sil.round(3).to_csv(OUT / "clu_01_silhouette.csv")
k_best = int(sil.idxmax())
print("Silhouette by k:", sil.round(3).to_dict(), "-> k =", k_best)

km = KMeans(n_clusters=k_best, n_init=20, random_state=SEED).fit(Zs)
order = raw.groupby(km.labels_)["Temp"].median().sort_values().index      # Type 1 = coldest
name = {c: f"Type {i + 1}" for i, c in enumerate(order)}
raw = raw.assign(cluster=pd.Series(km.labels_, index=raw.index).map(name))
raw["bloom"] = (raw[TARGET] > bloom_line).astype(int)

# ---- 3. check with hierarchical clustering (Ward)
ward = fcluster(linkage(Zs, method="ward"), t=k_best, criterion="maxclust")
print(f"k-means vs hierarchical agreement (adjusted Rand index): {adjusted_rand_score(km.labels_, ward):.2f}")

# ---- 4. profile of each water type + bloom rate
prof = raw.groupby("cluster").agg(weeks=("Temp", "size"), Temp=("Temp", "median"),
                                  **{n: (n, "median") for n in NUTR},
                                  bloom_weeks=("bloom", "sum"), bloom_rate=("bloom", "mean"))
prof.round(3).to_csv(OUT / "clu_02_profiles.csv")
print(f"Bloom line {bloom_line:.2f} mg/m3\n", prof.round(2).to_string())

# ---- 5. slide figure: cluster profile (Lectures 12-13) + bloom rate
#      bar height = cluster centre in standard units (what k-means uses; 0 = average week)
#      number     = the same centre turned back into real units (°C, µM), so it is easy to read
from matplotlib.patches import Rectangle
scaler = StandardScaler().fit(Z)
centre_std = pd.DataFrame(km.cluster_centers_, columns=ENV_F)
centre_real = pd.DataFrame(scaler.inverse_transform(km.cluster_centers_), columns=ENV_F)
centre_real[NUTR] = np.expm1(centre_real[NUTR])                   # undo log
type_of = {c: name[c] for c in range(k_best)}
centre_std.index = centre_real.index = [type_of[c] for c in centre_std.index]
centre_std, centre_real = centre_std.loc[prof.index], centre_real.loc[prof.index]
pd.concat({"standard_units": centre_std, "real_units": centre_real}, axis=1).round(3).to_csv(OUT / "clu_03_centres.csv")
print("Cluster centres (standard units):\n", centre_std.round(2).to_string())
print("Cluster centres (real units):\n", centre_real.round(2).to_string())

show = ["Temp", "Nitrate", "Silicate", "Phosphate", "Ammonium", "Nitrite"]
head = ["Temp", "NO₃", "SiO₄", "PO₄", "NH₄", "NO₂"]
names = ["Cold,\nnutrient-rich", "Average", "Warm,\nlow-nutrient"]
TEMP = "#C8553D"


def real(v, var):
    if var == "Temp":
        return f"{v:.0f}"
    return "0" if v < 0.005 else (f"{v:.1f}" if v >= 1 else f"{v:.2f}")


fig = plt.figure(figsize=(8.9, 5.0))
ax = fig.add_axes([0, 0, 1, 1])
ax.set_xlim(0, 8.9); ax.set_ylim(5.0, 0); ax.axis("off")
ax.text(4.45, 0.32, f"{k_best} water types (k-means, no Chl-a)", ha="center", va="center", fontsize=22)
x0, cw, top, rh, sc = 2.25, 0.8, 1.05, 1.12, 0.115
for j, h in enumerate(head):
    ax.text(x0 + cw * (j + .5), top - 0.05, h, ha="center", va="bottom", fontsize=20, color=TEMP if j == 0 else style.TEAL)
ax.text(x0 + cw * 6 + 0.2, top - 0.05, "Bloom weeks", ha="left", va="bottom", fontsize=20, color=style.MUTED)
for i, t in enumerate(prof.index):
    mid = top + rh * (i + .5) + 0.3
    ax.text(x0 - 0.12, mid, names[i], ha="right", va="center", fontsize=21, fontweight="bold", color=style.INK)
    ax.plot([x0, x0 + cw * 6], [mid, mid], color=style.MUTED, lw=0.8)
    for j, var in enumerate(show):
        z = centre_std.loc[t, var]; h = z * sc
        ax.add_patch(Rectangle((x0 + cw * j + 0.16, mid - h if h > 0 else mid), cw - 0.32, abs(h),
                               color=TEMP if j == 0 else style.TEAL, alpha=1 if z > 0 else 0.45, lw=0))
        ax.text(x0 + cw * (j + .5), mid - h - 0.17 if h > 0 else mid - h + 0.17, real(centre_real.loc[t, var], var),
                ha="center", va="center", fontsize=18, color=style.INK)
    r, nb_, nw_ = prof.loc[t, "bloom_rate"] * 100, prof.loc[t, "bloom_weeks"], prof.loc[t, "weeks"]
    bw = r / 25 * 0.95
    ax.add_patch(Rectangle((x0 + cw * 6 + 0.2, mid - 0.24), bw, 0.48, color=style.PLUM if i == 0 else style.GREY, lw=0))
    ax.text(x0 + cw * 6 + 0.27 + bw, mid, f"{r:.0f}%", va="center", fontsize=23, fontweight="bold")
    ax.text(x0 + cw * 6 + 0.2, mid + 0.42, f"{nb_} of {nw_} weeks", va="center", fontsize=16, color=style.MUTED)
ax.text(x0 - 0.3, 4.82, "Line = average week · numbers = group centre (°C, µM)",
        fontsize=16, color=style.MUTED)
style.save(fig, FIG / "s07b_clusters_bloom_rate.png")
