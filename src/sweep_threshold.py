"""Re-assess the model under different definitions of "active".

The label is currently intensity >= 10,000 in at least 1 of 3 replicates. 10,000 came
from the negative controls (highest ever seen in a no-enzyme well: 10,320), so it is a
noise floor rather than a statement about signal quality. By mass-spectrometry
convention 10^4 counts is weak, and detection reproducibility -- measured directly --
climbs steeply to 50,000 and is flat above it:

    peak intensity      fraction replicating in >=2 of 3 runs
    10,000 - 25,000                 34%
    25,000 - 50,000                 66%
    50,000 - 100,000                82%     <- plateau
    over 1,000,000                  82%

So 50,000 is where a detection becomes as reliable as it will ever be. This refits the
production recipe under each candidate label and reports the full metric set.

CAUTION on reading the output: changing the label changes the task. Positives get rarer,
and PR-AUC's floor is the positive rate, so raw PR-AUC is NOT comparable between rows.
Within-substrate AUC and ROC-AUC have a floor of 0.5 whatever the rate, and normalised
PR-AUC corrects for it. Those three are the fair comparisons.

Development enzymes only -- the 23 locked-away are not touched.
"""
from pathlib import Path
import sys, json, numpy as np, pandas as pd, warnings
warnings.filterwarnings("ignore")
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import (average_precision_score, roc_auc_score, log_loss,
                             brier_score_loss, f1_score, precision_score, recall_score,
                             matthews_corrcoef, balanced_accuracy_score, confusion_matrix)
import xgboost as xgb

sys.path.insert(0, str(Path(__file__).resolve().parent))
import labels as lb, representations as rep, embeddings as EM, holdout
from splits import balanced_group_split

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "outputs"
N_JOBS = int(__import__("os").environ.get("SLURM_CPUS_PER_TASK", 8))
PARAMS = dict(max_depth=4, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8,
              reg_alpha=1.0, reg_lambda=5.0, eval_metric="logloss", n_jobs=N_JOBS,
              verbosity=0, random_state=0)
RULES = [(10_000, 1), (25_000, 1), (50_000, 1), (100_000, 1), (50_000, 2)]


def features(act, emb):
    ids = sorted(act.Enzyme.unique())
    Z = StandardScaler().fit_transform(np.stack([emb[i] for i in ids])).astype(np.float32)
    v = dict(zip(ids, Z))
    sm = rep.load_smiles("Amine")
    A, _ = rep.morgan({a: sm[a] for a in sorted(act.Amine.unique()) if a in sm}, n_bits=512)
    C, _ = rep.build_core("onehot", labels=sorted(act.Hydroxyl.unique()))
    return np.hstack([np.stack([v[e] for e in act.Enzyme]),
                      A.loc[act.Amine].to_numpy(np.float32),
                      C.loc[act.Hydroxyl].to_numpy(np.float32)]).astype(np.float32)


def run(X, y, g, act):
    oof, rounds, curves = np.zeros(len(y)), [], []
    for f, (tr, te) in enumerate(GroupKFold(n_splits=5).split(X, y, g)):
        a, b = balanced_group_split(g[tr], test_size=0.2, seed=f)
        itr, iva = tr[a], tr[b]
        spw = max((y[itr] == 0).sum() / max((y[itr] == 1).sum(), 1), 1.0)
        m = xgb.XGBClassifier(n_estimators=3000, scale_pos_weight=spw,
                              early_stopping_rounds=50, **PARAMS)
        m.fit(X[itr], y[itr], verbose=False,
              eval_set=[(X[itr], y[itr]), (X[iva], y[iva]), (X[te], y[te])])
        oof[te] = m.predict_proba(X[te])[:, 1]
        rounds.append(m.best_iteration + 1)
        r = m.evals_result()
        curves.append({k: np.array(r[f"validation_{i}"]["logloss"], dtype=np.float32)
                       for i, k in enumerate(("train", "check", "held"))})
    return oof, rounds, curves


def metrics(y, p, act):
    pred = (p >= 0.5).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    w = [roc_auc_score(s.y, s.p) for _, s in act.assign(y=y, p=p).groupby(["Amine", "Hydroxyl"])
         if s.y.sum() >= 5 and (1 - s.y).sum() >= 5]
    base = float(y.mean())
    pr = average_precision_score(y, p)
    return dict(base_rate=round(base, 4), n_positive=int(y.sum()),
                pr_auc=round(pr, 4), pr_auc_normalised=round((pr - base) / (1 - base), 4),
                roc_auc=round(roc_auc_score(y, p), 4),
                log_loss=round(log_loss(y, np.clip(p, 1e-6, 1 - 1e-6)), 4),
                brier=round(brier_score_loss(y, p), 4),
                mcc=round(matthews_corrcoef(y, pred), 4),
                bal_acc=round(balanced_accuracy_score(y, pred), 4),
                f1=round(f1_score(y, pred), 4),
                precision=round(precision_score(y, pred, zero_division=0), 4),
                recall=round(recall_score(y, pred), 4),
                specificity=round(tn / max(tn + fp, 1), 4),
                within_mean=round(float(np.mean(w)), 4),
                within_median=round(float(np.median(w)), 4), n_groups=len(w),
                tp=int(tp), fp=int(fp), fn=int(fn), tn=int(tn))


def clusters():
    """Precomputed 70%-identity clusters.

    cd-hit is on the head node but not the compute nodes, and the clustering is the
    same for every rule, so it is built once by src/precompute_clusters.py.
    """
    c = pd.read_csv(ROOT / "data/derived/clusters_70.csv")
    return dict(zip(c.Enzyme, c.cluster))


def main():
    CLUSTERS = clusters()
    emb = EM.load("ProstT5", pooling="mean")
    assign = holdout.assignment()
    rows, store = [], {}
    print(f"n_jobs={N_JOBS}\n")
    for thr, reps in RULES:
        act = lb.build(threshold=float(thr), min_reps=reps)
        act = act[act.Enzyme.isin(emb)]
        act = act[act.Enzyme.map(assign).eq("dev")].reset_index(drop=True)
        y = act.active.to_numpy(int)
        X = features(act, emb)
        g = np.array([CLUSTERS[e] for e in act.Enzyme])
        oof, rounds, curves = run(X, y, g, act)
        iso = IsotonicRegression(out_of_bounds="clip").fit(oof, y)
        m = metrics(y, oof, act)
        m.update(threshold=thr, min_reps=reps, rounds=int(np.median(rounds)),
                 rounds_per_fold=str(rounds),
                 log_loss_calibrated=round(log_loss(y, np.clip(iso.predict(oof), 1e-6, 1-1e-6)), 4))
        rows.append(m)
        store[f"{thr}|{reps}|oof"] = oof
        store[f"{thr}|{reps}|y"] = y
        for f, c in enumerate(curves):
            for k, v in c.items():
                store[f"curve|{thr}|{reps}|{f}|{k}"] = v
        print(f"  {thr:>7,} / {reps} of 3   {m['n_positive']:>5,} positive ({m['base_rate']:.1%})"
              f"   PR {m['pr_auc']:.3f} (norm {m['pr_auc_normalised']:.3f})"
              f"   ROC {m['roc_auc']:.3f}   within-sub {m['within_mean']:.3f}")

    df = pd.DataFrame(rows)
    OUT.mkdir(exist_ok=True)
    df.to_csv(OUT / "threshold_sweep.csv", index=False)
    np.savez(OUT / "threshold_sweep_oof.npz", **store)
    print(f"\n-> outputs/threshold_sweep.csv, threshold_sweep_oof.npz")


if __name__ == "__main__":
    main()
