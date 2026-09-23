"""Redraw the log loss curves from outputs/logloss_curves.npz (no refitting).

Bold line is the mean across the 5 cluster-grouped folds, shaded band is +/- 1 sd.
"""
from pathlib import Path
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
EMBEDS = ["ProtT5", "ESM-2", "ESM-3", "ProstT5"]
SERIES = [("train", "#1f77b4", "train"),
          ("val", "#d62728", "validation (inner)"),
          ("held", "#2ca02c", "held-out fold (outer)")]
NULL_LL = 0.542
PATIENCE = 40


def main():
    z = np.load(ROOT / "outputs/logloss_curves.npz")
    n_rounds = z[f"{EMBEDS[0]}|train"].shape[1]
    x = np.arange(n_rounds)

    fig, axes = plt.subplots(1, 4, figsize=(19, 4.6), sharey=True)
    for ax, name in zip(axes, EMBEDS):
        for key, col, lab in SERIES:
            C = z[f"{name}|{key}"]
            m, s = C.mean(0), C.std(0)
            ax.plot(x, m, color=col, lw=1.8, label=lab)
            ax.fill_between(x, m - s, m + s, color=col, alpha=.15, lw=0)
        ax.axhline(NULL_LL, ls=":", c="k", lw=1.2)
        ax.text(n_rounds - 8, NULL_LL + .012, f"null {NULL_LL:.2f}",
                fontsize=8, ha="right", color="k")
        stop = int(np.median([min(int(r.argmin()) + PATIENCE, n_rounds - 1)
                              for r in z[f"{name}|val"]]))
        ax.axvline(stop, ls="--", c="grey", lw=1.1)
        ax.text(stop + n_rounds * .012, .93, f"early stop ~{stop}", fontsize=8,
                color="grey", rotation=90, va="top")
        gap = z[f"{name}|held"].mean(0)[-1] - z[f"{name}|train"].mean(0)[-1]
        ax.set_title(f"{name}   final gap {gap:+.2f}", fontweight="bold")
        ax.set_xlabel("boosting round"); ax.grid(alpha=.25)
    axes[0].set_ylabel("log loss"); axes[0].set_ylim(0, 1.0); axes[0].legend(fontsize=9)
    fig.suptitle("XGBoost log loss vs boosting round -- mean +/- sd over 5 cluster-grouped folds",
                 fontweight="bold", y=1.02)
    fig.text(.5, -.06, "Dev set only, whole-protein MEAN pooling. Early stopping disabled so the "
             "full trajectory is visible; dashed line marks where it would have fired. "
             "Dotted line is the log loss of always predicting the base rate.",
             ha="center", fontsize=9)
    fig.savefig(ROOT / "images/logloss_curves.png", dpi=150, bbox_inches="tight",
                facecolor="white")
    print("-> images/logloss_curves.png")


if __name__ == "__main__":
    main()
