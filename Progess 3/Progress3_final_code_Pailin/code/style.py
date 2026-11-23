"""style.py - one look for every figure (same colours as the slides).

ทุกกราฟใช้สีและฟอนต์เดียวกัน และกำหนดขนาดรูป (นิ้ว) เท่ากับพื้นที่บนสไลด์ ตัวอักษรจึงอ่านได้จริง
"""
import logging
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

logging.getLogger("matplotlib.font_manager").setLevel(logging.ERROR)

BG = "#FAF9F7"      # slide background
INK = "#2B2B2B"     # text
MUTED = "#555555"   # axis lines
GRID = "#E4E1DC"    # light grid
PLUM = "#8A3560"    # observed / bloom / highlight
TEAL = "#00958F"    # model
OLIVE = "#A4983A"   # second group
GREY = "#9A968F"    # baselines

plt.rcParams.update({
    "font.family": ["Carlito", "Calibri", "DejaVu Sans"], "font.size": 18,
    "axes.facecolor": BG, "figure.facecolor": BG, "savefig.facecolor": BG,
    "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": INK, "ytick.color": INK,
    "axes.spines.top": False, "axes.spines.right": False,
})


def save(fig, path):
    """Save at 200 dpi (sharp on a projector)."""
    fig.savefig(path, dpi=200)
    plt.close(fig)
