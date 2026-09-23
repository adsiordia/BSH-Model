"""Train vs validation log loss per boosting round, one panel per embedding.

Model is fixed to XGBoost (the sweep winner) so the only thing varying is the
enzyme representation. Early stopping is DISABLED here so the full 400-round
trajectory is visible -- the point of the figure is to see where the training
curve keeps falling while validation flattens or turns up. The round where
early stopping would have fired is marked.
"""
from pathlib import Path
import numpy as np, pandas as pd, warnings
warnings.filterwarnings("ignore")
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.model_selection import GroupKFold, train_test_split
from sklearn.preprocessing import StandardScaler
import xgboost as xgb

# respect the SLURM allocation rather than taking the whole node
N_JOBS = int(__import__('os').environ.get('SLURM_CPUS_PER_TASK', 8))

import labels as lb, representations as rep, holdout, embeddings as EM
from splits import identity_matrix, cdhit_merged_clusters, balanced_group_split, load_clusters

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "outputs"; IMG = ROOT / "images"
EMBEDS = ["ProtT5", "ESM-2", "ESM-3", "ProstT5"]
N_ROUNDS = 1500
PATIENCE = 40


def main():
    act = lb.build(min_reps=1)
    E = {n: EM.load(n, pooling="mean") for n in EMBEDS}
    common = set.intersection(*(set(v) for v in E.values())) & set(act.Enzyme)
    act = act[act.Enzyme.isin(common)].reset_index(drop=True)
    act = act[act.Enzyme.map(holdout.assignment()).eq("dev").to_numpy()].reset_index(drop=True)
    y = act.active.to_numpy(int)

    sm = rep.load_smiles("Amine")
    A, _ = rep.morgan({a: sm[a] for a in sorted(act.Amine.unique()) if a in sm}, n_bits=512)
    C, _ = rep.build_core("onehot", labels=sorted(act.Hydroxyl.unique()))
    cl = load_clusters(set(act.Enzyme))
    g = np.array([cl[e] for e in act.Enzyme])

    base = y.mean()
    null_ll = -(base*np.log(base) + (1-base)*np.log(1-base))
    print(f"{len(act):,} dev cells, {act.Enzyme.nunique()} enzymes, "
          f"base rate {base:.3f}, null log loss {null_ll:.3f}\n")

    ids = sorted(common)
    curves = {}
    for name in EMBEDS:
        Z = StandardScaler().fit_transform(np.stack([E[name][i] for i in ids]))
        vec = dict(zip(ids, Z.astype(np.float32)))
        X = np.hstack([np.stack([vec[e] for e in act.Enzyme]),
                       A.loc[act.Amine].to_numpy(np.float32),
                       C.loc[act.Hydroxyl].to_numpy(np.float32)]).astype(np.float32)
        tr_c, va_c, te_c, stops = [], [], [], []
        for fold, (tr, te) in enumerate(GroupKFold(n_splits=5).split(X, y, g)):
            a, b = balanced_group_split(g[tr], test_size=0.2, seed=fold)
            itr, iva = tr[a], tr[b]
            spw = max((y[itr] == 0).sum() / max((y[itr] == 1).sum(), 1), 1.0)
            mdl = xgb.XGBClassifier(n_estimators=N_ROUNDS, max_depth=4, learning_rate=0.05,
                subsample=0.8, colsample_bytree=0.8, reg_alpha=1.0, reg_lambda=5.0,
                scale_pos_weight=spw, eval_metric="logloss", n_jobs=N_JOBS, verbosity=0,
                random_state=0)
            mdl.fit(X[itr], y[itr], verbose=False,
                    eval_set=[(X[itr], y[itr]), (X[iva], y[iva]), (X[te], y[te])])
            r = mdl.evals_result()
            tr_c.append(r["validation_0"]["logloss"])
            va_c.append(r["validation_1"]["logloss"])
            te_c.append(r["validation_2"]["logloss"])
            v = np.array(r["validation_1"]["logloss"])
            best = int(v.argmin())
            stops.append(min(best + PATIENCE, N_ROUNDS - 1))
        curves[name] = dict(train=np.array(tr_c), val=np.array(va_c),
                            held=np.array(te_c), stop=int(np.median(stops)))
        t, v, h = curves[name]["train"], curves[name]["val"], curves[name]["held"]
        print(f"  {name:9s} final train {t[:,-1].mean():.3f}  val {v[:,-1].mean():.3f}  "
              f"held-out fold {h[:,-1].mean():.3f}   best val {v.mean(0).min():.3f} "
              f"@ round {int(v.mean(0).argmin())}   median stop {curves[name]['stop']}")

    x = np.arange(N_ROUNDS)
    fig, axes = plt.subplots(1, 4, figsize=(19, 4.6), sharey=True)
    for ax, name in zip(axes, EMBEDS):
        c = curves[name]
        for key, col, lab in [("train", "#1f77b4", "train"),
                              ("val", "#d62728", "validation (inner)"),
                              ("held", "#2ca02c", "held-out fold (outer)")]:
            m, s = c[key].mean(0), c[key].std(0)
            ax.plot(x, m, color=col, lw=1.8, label=lab)
            ax.fill_between(x, m - s, m + s, color=col, alpha=.15, lw=0)
        ax.axhline(null_ll, ls=":", c="k", lw=1.2)
        ax.axvline(c["stop"], ls="--", c="grey", lw=1.1)
        ax.text(c["stop"] + 6, .93, f"early stop ~{c['stop']}", fontsize=8,
                color="grey", rotation=90, va="top")
        ax.text(N_ROUNDS - 8, null_ll + .012, f"null {null_ll:.2f}", fontsize=8,
                ha="right", color="k")
        gap = c["held"].mean(0)[-1] - c["train"].mean(0)[-1]
        ax.set_title(f"{name}   final gap {gap:+.2f}", fontweight="bold")
        ax.set_xlabel("boosting round"); ax.grid(alpha=.25)
    axes[0].set_ylabel("log loss"); axes[0].set_ylim(0, 1.0); axes[0].legend(fontsize=9)
    fig.suptitle("XGBoost log loss vs boosting round -- mean +/- sd over 5 cluster-grouped folds",
                 fontweight="bold", y=1.02)
    fig.text(.5, -.06, "Dev set only, whole-protein MEAN pooling. Early stopping disabled so the "
             "full trajectory is visible; dashed line marks where it would have fired. "
             "Dotted line is the log loss of always predicting the base rate.",
             ha="center", fontsize=9)
    IMG.mkdir(exist_ok=True); OUT.mkdir(exist_ok=True)
    fig.savefig(IMG / "logloss_curves.png", dpi=150, bbox_inches="tight", facecolor="white")
    np.savez(OUT / "logloss_curves.npz", **{f"{n}|{k}": curves[n][k]
             for n in EMBEDS for k in ("train", "val", "held")})
    print(f"\n-> images/logloss_curves.png")


if __name__ == "__main__":
    main()
