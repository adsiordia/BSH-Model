"""Evaluation with a locked test set.

  run_dev(...)    5-fold cluster-grouped CV inside the DEV set.
                  Use this for every design decision, comparison and tuning.
  run_final(...)  trains on ALL of dev, evaluates ONCE on the locked test set.
                  Every call is recorded in outputs/test_set_evaluations.log so
                  repeated peeking is visible.

The test rows are never loaded by run_dev.
"""
from pathlib import Path
from datetime import datetime
import numpy as np, pandas as pd, warnings
warnings.filterwarnings("ignore")
from sklearn.model_selection import GroupKFold, train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import (average_precision_score, roc_auc_score, f1_score,
                             precision_score, recall_score, matthews_corrcoef,
                             balanced_accuracy_score, log_loss)
import xgboost as xgb

import dataset as ds
import holdout
from splits import identity_matrix, cdhit_merged_clusters, check_leakage

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "outputs"
N_FOLDS = 5
METRICS = ["pr_auc", "roc_auc", "mcc", "bal_acc", "f1", "precision", "recall", "log_loss"]


def models(spw):
    return {
        "LogisticRegression": make_pipeline(StandardScaler(),
            LogisticRegression(max_iter=3000, C=0.1, class_weight="balanced")),
        "RandomForest": RandomForestClassifier(n_estimators=400, min_samples_leaf=3,
            class_weight="balanced_subsample", n_jobs=-1, random_state=0),
        "XGBoost": xgb.XGBClassifier(n_estimators=600, max_depth=4, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8, reg_alpha=1.0, reg_lambda=5.0,
            scale_pos_weight=spw, eval_metric="logloss", early_stopping_rounds=40,
            n_jobs=-1, verbosity=0, random_state=0),
        # NOTE: MLP's early_stopping carves a random 10% of its TRAINING data, not a
        # cluster-aware split. No test data is involved, so the reported metric stays
        # unbiased; the only cost is a possibly suboptimal stopping point.
        "MLP": make_pipeline(StandardScaler(),
            MLPClassifier(hidden_layer_sizes=(256, 64), alpha=1e-3, batch_size=256,
                learning_rate_init=1e-3, max_iter=300, early_stopping=True,
                n_iter_no_change=15, random_state=0)),
    }


def metrics(y, p, thr=0.5):
    pred = (p >= thr).astype(int)
    if not y.sum() or not (1 - y).sum():
        return dict(n=len(y), n_pos=int(y.sum()))
    return dict(n=len(y), n_pos=int(y.sum()),
        pr_auc=average_precision_score(y, p), roc_auc=roc_auc_score(y, p),
        mcc=matthews_corrcoef(y, pred), bal_acc=balanced_accuracy_score(y, pred),
        f1=f1_score(y, pred, zero_division=0),
        precision=precision_score(y, pred, zero_division=0),
        recall=recall_score(y, pred, zero_division=0),
        log_loss=log_loss(y, np.clip(p, 1e-6, 1 - 1e-6)))


def _fit(name, mdl, X, y, itr, iva):
    if name == "XGBoost":
        mdl.fit(X[itr], y[itr], eval_set=[(X[iva], y[iva])], verbose=False)
    else:
        mdl.fit(X[np.concatenate([itr, iva])], y[np.concatenate([itr, iva])])
    return mdl


def _table(df, header):
    print(f"\n  {header}")
    for half in (METRICS[:4], METRICS[4:]):
        print(f"  {'model':20s} " + "".join(f"{m:>16s}" for m in half))
        for mdl in df.model.unique():
            r = df[df.model == mdl]
            print(f"  {mdl:20s} " + "".join(
                f"{r[m].mean():>9.3f}+/-{r[m].std():.3f}" if len(r) > 1 else f"{r[m].mean():>16.3f}"
                for m in half))
        print()


def run_dev(min_reps, tag):
    """Cross-validated development estimate. Never touches the test rows."""
    X, y, meta, names, blocks = ds.build(min_reps=min_reps)
    split = holdout.assignment()
    keep = meta.Enzyme.map(split).eq("dev").to_numpy()
    X, y, meta = X[keep], y[keep], meta[keep].reset_index(drop=True)
    enz = meta.Enzyme.to_numpy()
    # CV grouping inside dev uses cd-hit clusters merged wherever a >=70% pair joins
    # two of them. Same guarantee as MSA components (0 leaking pairs, worst 69.5%)
    # but it distributes the large family more evenly across folds, roughly halving
    # fold-to-fold variance. The dev/test boundary itself is NOT recomputed here --
    # it stays as written in holdout_split.csv.
    M_dev = identity_matrix(set(enz))
    cl = cdhit_merged_clusters(identity=0.70, M=M_dev)
    groups = np.array([cl[e] for e in enz])
    print(f"\n{'='*78}\nDEV  |  {tag}\n{'='*78}")
    print(f"  X={X.shape}  active={y.sum():,} ({100*y.mean():.1f}%)  "
          f"{len(set(enz))} enzymes, {len(set(groups))} clusters")

    rows, preds = [], {}
    for fold, (tr_i, te_i) in enumerate(GroupKFold(n_splits=N_FOLDS).split(X, y, groups)):
        g = groups[tr_i]; uq = np.unique(g)
        a, b = train_test_split(uq, test_size=0.2, random_state=fold)
        itr, iva = tr_i[np.isin(g, a)], tr_i[np.isin(g, b)]
        spw = max((y[itr] == 0).sum() / max((y[itr] == 1).sum(), 1), 1.0)
        for name, mdl in models(spw).items():
            p = _fit(name, mdl, X, y, itr, iva).predict_proba(X[te_i])[:, 1]
            preds.setdefault(name, np.zeros(len(y)))[te_i] = p
            rows.append({**metrics(y[te_i], p), "model": name, "fold": fold})
    df = pd.DataFrame(rows)
    _table(df, f"5-fold cluster-grouped CV within dev  (mean +/- sd)")
    OUT.mkdir(exist_ok=True)
    df.to_csv(OUT/f"dev_folds_{tag}.csv", index=False)
    np.savez(OUT/f"dev_oof_{tag}.npz", y=y, **preds)
    meta.to_csv(OUT/f"dev_meta_{tag}.csv", index=False)
    return df


def run_final(min_reps, tag, note=""):
    """Train on ALL of dev, evaluate ONCE on the locked test set."""
    X, y, meta, names, blocks = ds.build(min_reps=min_reps)
    split = meta.Enzyme.map(holdout.assignment()).to_numpy()
    cl = holdout.clusters()
    dev, test = split == "dev", split == "test"
    enz = meta.Enzyme.to_numpy()
    worst, _ = check_leakage({e: holdout.assignment()[e] for e in set(enz)},
                             identity_matrix(set(enz)), set(enz))
    print(f"\n{'='*78}\nFINAL TEST  |  {tag}\n{'='*78}")
    print(f"  dev  {dev.sum():,} rows / {len(set(enz[dev]))} enzymes")
    print(f"  test {test.sum():,} rows / {len(set(enz[test]))} enzymes / "
          f"{len(set(cl[e] for e in enz[test]))} clusters")
    print(f"  max sequence identity across the boundary: {worst:.1%}")
    print(f"  test active rate: {100*y[test].mean():.1f}%")

    g = np.array([cl[e] for e in enz[dev]]); uq = np.unique(g)
    a, b = train_test_split(uq, test_size=0.2, random_state=0)
    idx = np.where(dev)[0]
    itr, iva = idx[np.isin(g, a)], idx[np.isin(g, b)]
    spw = max((y[itr] == 0).sum() / max((y[itr] == 1).sum(), 1), 1.0)
    rows, preds = [], {}
    for name, mdl in models(spw).items():
        p = _fit(name, mdl, X, y, itr, iva).predict_proba(X[test])[:, 1]
        preds[name] = p
        rows.append({**metrics(y[test], p), "model": name})
    df = pd.DataFrame(rows)
    _table(df, "LOCKED TEST SET -- single evaluation")
    OUT.mkdir(exist_ok=True)
    df.to_csv(OUT/f"test_{tag}.csv", index=False)
    np.savez(OUT/f"test_oof_{tag}.npz", y=y[test], **preds)
    with open(OUT/"test_set_evaluations.log", "a") as f:
        f.write(f"{datetime.now().isoformat(timespec='seconds')}  tag={tag}  "
                f"min_reps={min_reps}  best_pr_auc={df.pr_auc.max():.4f}  {note}\n")
    return df


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "final":
        run_final(1, "ruleA_1of3", note="first look")
    else:
        run_dev(1, "ruleA_1of3")
        run_dev(2, "ruleB_2of3")
