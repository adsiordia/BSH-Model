"""Every fold's log loss curve on its own axes -- nothing averaged, nothing shaded.

plot_logloss.py collapses the five folds into a mean and a band, which hides that some
folds turn upward while others never do. This draws all 20 curves separately: one row per
embedding, one column per fold.
"""
from pathlib import Path
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
EMBEDS = ["ProtT5", "ESM-2", "ESM-3", "ProstT5"]
SERIES = [("train", "#1f77b4", "train"),
          ("val", "#d62728", "inner check"),
          ("held", "#2ca02c", "held-out fold")]
NULL_LL = 0.542


def main():
    z = np.load(ROOT / "outputs/logloss_curves.npz")
    n_folds = z[f"{EMBEDS[0]}|train"].shape[0]
    n_rounds = z[f"{EMBEDS[0]}|train"].shape[1]
    x = np.arange(n_rounds)

    fig, axes = plt.subplots(len(EMBEDS), n_folds, figsize=(4.0 * n_folds, 3.1 * len(EMBEDS)),
                             sharex=True, sharey=True)
    for r, name in enumerate(EMBEDS):
        for f in range(n_folds):
            ax = axes[r, f]
            for key, col, lab in SERIES:
                y = z[f"{name}|{key}"][f]
                ax.plot(x, y, color=col, lw=1.5, label=lab if (r == 0 and f == 0) else None)
            v = z[f"{name}|val"][f]
            h = z[f"{name}|held"][f]
            best = int(v.argmin())
            ax.axvline(best, ls="--", c="grey", lw=1)
            ax.axhline(NULL_LL, ls=":", c="k", lw=1)
            rise = h[-1] - h.min()
            ax.set_title(f"{name} · fold {f}\nbest val {v.min():.3f} @ {best}"
                         f"   held-out drift {rise:+.3f}",
                         fontsize=9, fontweight="bold" if rise > 0.03 else "normal",
                         color="#b5341f" if rise > 0.03 else "black")
            ax.grid(alpha=.25)
            if f == 0:
                ax.set_ylabel("log loss")
            if r == len(EMBEDS) - 1:
                ax.set_xlabel("boosting round")
    axes[0, 0].set_ylim(0, 0.75)
    axes[0, 0].legend(fontsize=9, loc="upper right")
    fig.suptitle("Every fold separately — no averaging. Dashed line marks that fold's best "
                 "inner-check round; dotted line is the score for ignoring the data.",
                 fontweight="bold", y=0.998, fontsize=11)
    fig.text(0.5, -0.012, "Titles in red mark folds whose held-out curve drifts more than "
             "0.03 above its own minimum — those are the folds that genuinely overfit.",
             ha="center", fontsize=9.5)
    out = ROOT / "images/logloss_curves_raw.png"
    out.parent.mkdir(exist_ok=True)
    fig.savefig(out, dpi=140, bbox_inches="tight", facecolor="white")
    print(f"-> {out}")
    print(f"\n{len(EMBEDS)} embeddings x {n_folds} folds = {len(EMBEDS)*n_folds} panels, "
          f"{n_rounds} rounds each\n")
    print(f"{'':10s} " + "  ".join(f"fold {f}      " for f in range(n_folds)))
    for name in EMBEDS:
        cells = []
        for f in range(n_folds):
            h = z[f"{name}|held"][f]
            cells.append(f"{h.min():.3f}{'  +' if h[-1]-h.min() > .03 else '   '}"
                         f"{h[-1]-h.min():.3f}")
        print(f"{name:10s} " + "  ".join(cells))
    print("\n(held-out minimum, then how far it drifts by round "
          f"{n_rounds}; + marks a real turn)")


if __name__ == "__main__":
    main()
