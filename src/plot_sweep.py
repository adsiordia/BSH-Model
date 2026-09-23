"""PR, ROC and calibration curves for every embedding x algorithm combination."""
from pathlib import Path
import sys
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve, roc_curve, average_precision_score, roc_auc_score

ROOT = Path(__file__).resolve().parent.parent
SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
COLORS = {"XGBoost": "#2a78d6", "MLP": "#eb6834",
          "RandomForest": "#1baf7a", "LogisticRegression": "#eda100"}
ORDER = ["XGBoost", "MLP", "RandomForest", "LogisticRegression"]
EMBEDS = ["ProtT5", "ESM-2", "ESM-3", "ProstT5"]


def style(ax):
    ax.set_facecolor(SURFACE); ax.grid(True, color=GRID, lw=.8, zorder=0)
    ax.set_axisbelow(True)
    for s in ("top", "right"): ax.spines[s].set_visible(False)
    for s in ("left", "bottom"): ax.spines[s].set_color(GRID)
    ax.tick_params(colors=INK2, labelsize=8.5, length=0)


def legend(ax, title, loc):
    lg = ax.legend(frameon=False, fontsize=7.8, loc=loc, title=title, labelcolor=INK2)
    lg.get_title().set_fontsize(7.8); lg.get_title().set_color(INK2)


if __name__ == "__main__":
    d = np.load(ROOT / "outputs/sweep_oof.npz"); y = d["y"]; base = y.mean()
    met = pd.read_csv(ROOT / "outputs/sweep_metrics.csv")

    fig, axes = plt.subplots(3, 4, figsize=(17, 11.5), facecolor=SURFACE)
    for col, emb in enumerate(EMBEDS):
        # PR
        ax = axes[0, col]; style(ax); ax.set_xlim(0, 1); ax.set_ylim(0, 1)
        ax.axhline(base, color=INK2, lw=1, ls=(0, (4, 3)), zorder=1)
        for m in ORDER:
            p = d[f"{emb}|{m}"]
            pr, rc, _ = precision_recall_curve(y, p)
            ax.plot(rc, pr, color=COLORS[m], lw=1.8, zorder=3,
                    label=f"{m[:12]}  {average_precision_score(y,p):.3f}")
        ax.set_title(emb, color=INK, fontsize=12, fontweight="bold", loc="left", pad=8)
        if col == 0: ax.set_ylabel("precision", color=INK2, fontsize=9.5)
        ax.set_xlabel("recall", color=INK2, fontsize=9.5)
        legend(ax, "PR-AUC", "lower left")

        # ROC
        ax = axes[1, col]; style(ax); ax.set_xlim(0, 1); ax.set_ylim(0, 1)
        ax.plot([0, 1], [0, 1], color=INK2, lw=1, ls=(0, (4, 3)), zorder=1)
        for m in ORDER:
            p = d[f"{emb}|{m}"]
            fpr, tpr, _ = roc_curve(y, p)
            ax.plot(fpr, tpr, color=COLORS[m], lw=1.8, zorder=3,
                    label=f"{m[:12]}  {roc_auc_score(y,p):.3f}")
        if col == 0: ax.set_ylabel("true positive rate", color=INK2, fontsize=9.5)
        ax.set_xlabel("false positive rate", color=INK2, fontsize=9.5)
        legend(ax, "ROC-AUC", "lower right")

        # calibration
        ax = axes[2, col]; style(ax); ax.set_xlim(0, 1); ax.set_ylim(0, 1)
        ax.plot([0, 1], [0, 1], color=INK2, lw=1, ls=(0, (4, 3)), zorder=1)
        edges = np.linspace(0, 1, 11)
        for m in ORDER:
            p = d[f"{emb}|{m}"]; xs, ys = [], []
            for lo, hi in zip(edges[:-1], edges[1:]):
                k = (p >= lo) & (p < hi)
                if k.sum() > 25: xs.append(p[k].mean()); ys.append(y[k].mean())
            ll = met[(met.embedding == emb) & (met.model == m)].log_loss.iloc[0]
            ax.plot(xs, ys, color=COLORS[m], lw=1.8, marker="o", ms=3.5, zorder=3,
                    label=f"{m[:12]}  {ll:.3f}")
        if col == 0: ax.set_ylabel("observed rate", color=INK2, fontsize=9.5)
        ax.set_xlabel("predicted probability", color=INK2, fontsize=9.5)
        legend(ax, "log loss", "upper left")

    for r, lab in enumerate(["Precision–recall", "ROC", "Calibration"]):
        axes[r, 0].text(-0.28, 0.5, lab, transform=axes[r, 0].transAxes, rotation=90,
                        va="center", ha="center", fontsize=11, fontweight="bold", color=INK)
    fig.text(.006, .008,
             f"Dev set, whole-protein MEAN pooling, 5-fold CV grouped on 70%-identity clusters. "
             f"Base rate {base:.3f}. Calibration: a point on the diagonal means the predicted "
             f"probability matches the observed rate.", fontsize=8.5, color=INK2)
    fig.tight_layout(rect=(0.012, .022, 1, 1))
    out = ROOT / "images/sweep_curves.png"
    fig.savefig(out, dpi=130, facecolor=SURFACE); print(f"wrote {out}")
