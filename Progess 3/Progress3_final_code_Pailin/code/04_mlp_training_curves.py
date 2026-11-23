"""04 Neural network training curves (Lecture 14), slide 7 left.

Lecture 14: "Training curves - important to plot!"
We train a small and a large MLP one epoch at a time on the TRAIN weeks and record the loss
(log loss) on train and on validation after each epoch.
  - lines stay close            -> the network learns real patterns
  - validation loss goes up      -> overfitting (the network memorises the training weeks)

กราฟ loss ของ train และ validation ทุก epoch: network ใหญ่ overfit หลัง epoch ~16
Task: bloom next week (Chl-a > P90 of train weeks). Inputs: set C after the VIF check.
Output: results/slide_figures/s07a_mlp_training_curves.png
"""
import warnings

import matplotlib.pyplot as plt
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.metrics import log_loss
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

import style
from common import FIG, SEED, SET_C, build, split_masks, vif_keep

warnings.filterwarnings("ignore")
EPOCHS = 400


def curve(hidden, alpha, Xtr, ytr, Xva, yva):
    """Train one epoch at a time; return train and validation loss per epoch."""
    net = MLPClassifier(hidden_layer_sizes=hidden, alpha=alpha, learning_rate_init=0.001, random_state=SEED)
    train_loss, val_loss = [], []
    for _ in range(EPOCHS):
        net.partial_fit(Xtr, ytr, classes=[0, 1])
        train_loss.append(log_loss(ytr, net.predict_proba(Xtr)[:, 1], labels=[0, 1]))
        val_loss.append(log_loss(yva, net.predict_proba(Xva)[:, 1], labels=[0, 1]))
    return np.array(train_loss), np.array(val_loss)


# ---- data
X, y, y_raw, _ = build()
tr, va, _ = split_masks(len(y))
cols, _, _ = vif_keep(X[tr][SET_C])
bloom_line = np.percentile(y_raw[tr], 90)                 # 6.2 mg/m3
bloom = (y_raw > bloom_line).astype(int).values
prep = make_pipeline(SimpleImputer(strategy="median"), StandardScaler()).fit(X[tr][cols])   # fit on TRAIN only
Xtr, Xva = prep.transform(X[tr][cols]), prep.transform(X[va][cols])

# ---- two networks
nets = {"Small network": ((8,), 0.1),             # 8 neurons, strong penalty
        "Large network": ((64, 32), 0.0001)}      # 64 + 32 neurons, weak penalty
fig, axes = plt.subplots(1, 2, figsize=(8.9, 5.0), sharey=True)
for ax, (title, (hidden, alpha)) in zip(axes, nets.items()):
    t_loss, v_loss = curve(hidden, alpha, Xtr, bloom[tr], Xva, bloom[va])
    ax.plot(t_loss, color=style.TEAL, lw=2.6, label="training")
    ax.plot(v_loss, color=style.PLUM, lw=2.6, label="validation")
    ax.set_title(title, fontsize=19)
    ax.set_xlabel("epoch")
    ax.set_ylim(0, 1)
    ax.set_xticks([0, 200, 400])
    ax.yaxis.grid(True, color=style.GRID)
    ax.set_axisbelow(True)
    best = int(v_loss.argmin())
    print(f"{title}: lowest validation loss {v_loss.min():.3f} at epoch {best}; "
          f"at epoch {EPOCHS}: train {t_loss[-1]:.3f}, validation {v_loss[-1]:.3f}")
axes[0].set_ylabel("loss")
axes[0].legend(frameon=False, loc="upper right", fontsize=16)
fig.tight_layout()
style.save(fig, FIG / "s07a_mlp_training_curves.png")
