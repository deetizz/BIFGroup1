"""01 Progress 1 in one figure (slide 3): what the data showed.

Three small panels, top to bottom:
  1. Very skewed   - histogram of weekly Chl-a (log axis): median about 2, max 468
  2. Spring peak   - median Chl-a for each month: highest in March to May
  3. 5-week memory - autocorrelation: this week's value is linked to the next 5 weeks

สรุปผล Progress 1 ในรูปเดียว: ข้อมูลเบ้มาก, สูงช่วงฤดูใบไม้ผลิ, และมีความจำ 5 สัปดาห์
Output: results/slide_figures/s03_progress1_summary.png
"""
import numpy as np
import matplotlib.pyplot as plt

import style
from common import FIG, TARGET, load_raw

raw, _ = load_raw()
chl = raw[TARGET]                                   # weekly Chl-a (mg/m3)

fig, ax = plt.subplots(3, 1, figsize=(3.7, 5.2), gridspec_kw={"hspace": 0.95})
fig.patch.set_facecolor("white")
for a in ax:
    a.set_facecolor("white")
small = 12

# ---- 1. Very skewed: histogram on a log axis
bins = np.logspace(np.log10(chl.min()), np.log10(chl.max()), 30)
ax[0].hist(chl, bins=bins, color=style.TEAL, alpha=0.75)
ax[0].set_xscale("log")
ax[0].set_xticks([1, 10, 100], ["1", "10", "100"], fontsize=13)
ax[0].minorticks_off()
ax[0].set_yticks([])
median = chl.median()
ax[0].axvline(median, color=style.PLUM, lw=1.6, ls="--")
ax[0].annotate(f"median {median:.0f}", (median, ax[0].get_ylim()[1] * 0.8), xytext=(6, 0),
               textcoords="offset points", color=style.PLUM, fontsize=small)
ax[0].annotate(f"max {chl.max():.0f}", (chl.max(), 2), xytext=(-8, 18), textcoords="offset points",
               ha="right", color=style.PLUM, fontsize=small, arrowprops=dict(arrowstyle="->", color=style.PLUM, lw=1))
ax[0].set_xlabel("Chl-a (mg/m³)", fontsize=small, labelpad=1)

# ---- 2. Spring peak: median Chl-a by month
monthly_median = chl.groupby(chl.index.month).median()
ax[1].axvspan(2.5, 5.5, color=style.PLUM, alpha=0.12, lw=0)          # March to May
ax[1].plot(monthly_median.index, monthly_median.values, color=style.TEAL, lw=2.2, marker="o", ms=4)
ax[1].set_xticks(range(1, 13), list("JFMAMJJASOND"), fontsize=11)
ax[1].set_yticks([0, 2.5], ["0.0", "2.5"], fontsize=13)
ax[1].set_ylim(0, 4.3)
ax[1].text(4, 0.35, "Mar–May", ha="center", color=style.PLUM, fontsize=small)
ax[1].set_ylabel("median", fontsize=small, labelpad=1)
ax[1].yaxis.grid(True, color=style.GRID)

# ---- 3. 5-week memory: autocorrelation of log Chl-a, lags 0-10
y = np.log1p(chl.values)
y = y - y.mean()
acf = [1.0] + [np.corrcoef(y[:-k], y[k:])[0, 1] for k in range(1, 11)]
band = 1.96 / np.sqrt(len(y))                      # 95% band: bars inside it are not significant
colours = [style.PLUM if 1 <= k <= 5 else "#B3B0AA" for k in range(11)]
ax[2].axhspan(-band, band, color=style.GRID, lw=0, zorder=0)
ax[2].bar(range(11), acf, color=colours, width=0.65, zorder=2)
ax[2].set_xticks(range(11), [str(k) for k in range(11)], fontsize=11)
ax[2].set_yticks([0, 1], ["0", "1"], fontsize=13)
ax[2].set_ylim(-0.1, 1.05)
ax[2].text(3, 0.75, "lag 1–5", color=style.PLUM, fontsize=small, ha="center")
ax[2].set_xlabel("weeks back", fontsize=small, labelpad=1)
ax[2].set_ylabel("autocorr.", fontsize=small, labelpad=1)

for a, title in zip(ax, ["Very skewed", "Spring peak", "5-week memory"]):
    a.set_title(title, fontsize=14, loc="left", color=style.INK, fontweight="bold", pad=4)
fig.subplots_adjust(left=0.14, right=0.97, top=0.95, bottom=0.08)
fig.savefig(FIG / "s03_progress1_summary.png", dpi=250, facecolor="white")

print(f"median {median:.2f}, max {chl.max():.1f}")
print("autocorrelation lag 1-5:", [round(v, 2) for v in acf[1:6]], "| 95% band", round(band, 3))
print("monthly median:", monthly_median.round(2).to_dict())
