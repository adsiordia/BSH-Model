"""PR, ROC and XGBoost log-loss curves for both label rules."""
from pathlib import Path
import sys
import numpy as np, pandas as pd, warnings
warnings.filterwarnings("ignore")
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.model_selection import GroupKFold, train_test_split
from sklearn.metrics import precision_recall_curve, roc_curve, average_precision_score, roc_auc_score
import xgboost as xgb
sys.path.insert(0, str(Path(__file__).resolve().parent))
import dataset as ds, holdout
from splits import identity_matrix, cdhit_merged_clusters

ROOT = Path(__file__).resolve().parent.parent
SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
COLORS = {"XGBoost": "#2a78d6", "MLP": "#eb6834",
          "RandomForest": "#1baf7a", "LogisticRegression": "#eda100"}
ORDER = ["XGBoost", "MLP", "RandomForest", "LogisticRegression"]
TRAIN_C, VAL_C = "#2a78d6", "#eb6834"
RULES = [("ruleA_1of3", "Rule A · >10,000 in ≥1 of 3 replicates", 1),
         ("ruleB_2of3", "Rule B · >10,000 in ≥2 of 3 replicates", 2)]


def style(ax):
    ax.set_facecolor(SURFACE); ax.grid(True, color=GRID, lw=.8, zorder=0)
    ax.set_axisbelow(True)
    for s in ("top", "right"): ax.spines[s].set_visible(False)
    for s in ("left", "bottom"): ax.spines[s].set_color(GRID)
    ax.tick_params(colors=INK2, labelsize=9, length=0)


def legend(ax, title, loc):
    lg = ax.legend(frameon=False, fontsize=8.5, loc=loc, title=title, labelcolor=INK2)
    lg.get_title().set_fontsize(8.5); lg.get_title().set_color(INK2)


def logloss_curves(min_reps, n_rounds=400):
    """Train/val log loss per boosting round, per dev fold."""
    X, y, meta, *_ = ds.build(min_reps=min_reps)
    keep = meta.Enzyme.map(holdout.assignment()).eq("dev").to_numpy()
    X, y, meta = X[keep], y[keep], meta[keep].reset_index(drop=True)
    enz = meta.Enzyme.to_numpy()
    cl = cdhit_merged_clusters(identity=0.70, M=identity_matrix(set(enz)))
    groups = np.array([cl[e] for e in enz])
    out = []
    for fold, (tr, te) in enumerate(GroupKFold(n_splits=5).split(X, y, groups)):
        g = groups[tr]; uq = np.unique(g)
        a, b = train_test_split(uq, test_size=0.2, random_state=fold)
        itr, iva = tr[np.isin(g, a)], tr[np.isin(g, b)]
        spw = max((y[itr] == 0).sum() / max((y[itr] == 1).sum(), 1), 1.0)
        m = xgb.XGBClassifier(n_estimators=n_rounds, max_depth=4, learning_rate=0.05,
                              subsample=0.8, colsample_bytree=0.8, reg_alpha=1.0,
                              reg_lambda=5.0, scale_pos_weight=spw, eval_metric="logloss",
                              n_jobs=-1, verbosity=0, random_state=0)
        m.fit(X[itr], y[itr], eval_set=[(X[itr], y[itr]), (X[iva], y[iva])], verbose=False)
        r = m.evals_result()
        out.append((np.array(r["validation_0"]["logloss"]), np.array(r["validation_1"]["logloss"])))
    return out


if __name__ == "__main__":
    curves = {tag: logloss_curves(mr) for tag, _, mr in RULES}
    fig, axes = plt.subplots(2, 3, figsize=(17, 10), facecolor=SURFACE)
    for row, (tag, title, mr) in enumerate(RULES):
        d = np.load(ROOT/f"outputs/dev_oof_{tag}.npz"); y = d["y"]; base = y.mean()

        ax = axes[row, 0]; style(ax); ax.set_xlim(0, 1); ax.set_ylim(0, 1)
        ax.axhline(base, color=INK2, lw=1.2, ls=(0, (4, 3)), zorder=1)
        ax.text(.985, base + .012, f"random {base:.3f}", fontsize=8, color=INK2, ha="right")
        for m in ORDER:
            pr, rc, _ = precision_recall_curve(y, d[m])
            ax.plot(rc, pr, color=COLORS[m], lw=2, zorder=3,
                    label=f"{m}  {average_precision_score(y, d[m]):.3f}")
        ax.set_xlabel("recall", color=INK2, fontsize=10)
        ax.set_ylabel("precision", color=INK2, fontsize=10)
        ax.set_title(f"{title}\nPrecision–recall  (base {base:.3f})", color=INK,
                     fontsize=11, fontweight="bold", loc="left", pad=10)
        legend(ax, "PR-AUC", "lower left")

        ax = axes[row, 1]; style(ax); ax.set_xlim(0, 1); ax.set_ylim(0, 1)
        ax.plot([0, 1], [0, 1], color=INK2, lw=1.2, ls=(0, (4, 3)), zorder=1)
        for m in ORDER:
            fpr, tpr, _ = roc_curve(y, d[m])
            ax.plot(fpr, tpr, color=COLORS[m], lw=2, zorder=3,
                    label=f"{m}  {roc_auc_score(y, d[m]):.3f}")
        ax.set_xlabel("false positive rate", color=INK2, fontsize=10)
        ax.set_ylabel("true positive rate", color=INK2, fontsize=10)
        ax.set_title("ROC", color=INK, fontsize=11, fontweight="bold", loc="left", pad=10)
        legend(ax, "ROC-AUC", "lower right")

        ax = axes[row, 2]; style(ax)
        cur = curves[tag]
        for tr_c, va_c in cur:
            ax.plot(tr_c, color=TRAIN_C, lw=.7, alpha=.3, zorder=2)
            ax.plot(va_c, color=VAL_C, lw=.7, alpha=.3, zorder=2)
        n = min(len(t) for t, _ in cur)
        mtr = np.mean([t[:n] for t, _ in cur], axis=0)
        mva = np.mean([v[:n] for _, v in cur], axis=0)
        ax.plot(mtr, color=TRAIN_C, lw=2, zorder=3)
        ax.plot(mva, color=VAL_C, lw=2, zorder=3)
        best = int(np.argmin(mva))
        ax.axvline(best, color=INK2, lw=1, ls=(0, (4, 3)), zorder=1)
        ax.scatter([best], [mva[best]], s=42, color=VAL_C, ec=SURFACE, lw=2, zorder=4)
        ax.annotate(f"best round {best}\nval {mva[best]:.3f}\ngap {mva[best]-mtr[best]:+.3f}",
                    (best, mva[best]), xytext=(12, 14), textcoords="offset points",
                    fontsize=8.5, color=INK2)
        ax.text(n*.97, mtr[int(n*.97)]-.012, "train", color=TRAIN_C, fontsize=9.5,
                fontweight="bold", ha="right", va="top")
        ax.text(n*.97, mva[int(n*.97)]+.010, "validation", color=VAL_C, fontsize=9.5,
                fontweight="bold", ha="right", va="bottom")
        ax.set_xlabel("boosting round", color=INK2, fontsize=10)
        ax.set_ylabel("log loss", color=INK2, fontsize=10)
        ax.set_title("XGBoost log loss", color=INK, fontsize=11, fontweight="bold",
                     loc="left", pad=10)

    fig.text(.006, .008,
             "Dev set only (92 enzymes), 5-fold CV grouped on cd-hit clusters merged at 70% identity "
             "(0 leaking pairs). Log-loss faint lines are individual folds. The locked test set is untouched.",
             fontsize=8.5, color=INK2)
    fig.tight_layout(rect=(0, .022, 1, 1))
    out = ROOT/"images/curves_both_rules.png"
    fig.savefig(out, dpi=140, facecolor=SURFACE); print(f"wrote {out}")
