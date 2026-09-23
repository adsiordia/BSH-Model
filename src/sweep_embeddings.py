"""4 embeddings x 4 algorithms, whole-protein mean pooling, cluster-grouped CV.

Writes outputs/sweep_metrics.csv and outputs/sweep_oof.npz so the curves can be
drawn without refitting.
"""
from pathlib import Path
import numpy as np, pandas as pd, warnings
warnings.filterwarnings("ignore")
from sklearn.model_selection import GroupKFold, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (average_precision_score, roc_auc_score, f1_score, precision_score,
                             recall_score, matthews_corrcoef, balanced_accuracy_score,
                             log_loss, brier_score_loss, confusion_matrix)
import xgboost as xgb

# respect the SLURM allocation rather than taking the whole node
N_JOBS = int(__import__('os').environ.get('SLURM_CPUS_PER_TASK', 8))

import labels as lb, representations as rep, holdout, embeddings as EM
from splits import identity_matrix, cdhit_merged_clusters, balanced_group_split, load_clusters

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "outputs"
EMBEDS = ["ProtT5", "ESM-2", "ESM-3", "ProstT5"]


def models(spw):
    return {
        "LogisticRegression": make_pipeline(StandardScaler(),
            LogisticRegression(max_iter=3000, C=0.1, class_weight="balanced")),
        "RandomForest": RandomForestClassifier(n_estimators=400, min_samples_leaf=3,
            class_weight="balanced_subsample", n_jobs=N_JOBS, random_state=0),
        "XGBoost": xgb.XGBClassifier(n_estimators=400, max_depth=4, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8, reg_alpha=1.0, reg_lambda=5.0,
            scale_pos_weight=spw, eval_metric="logloss", early_stopping_rounds=40,
            n_jobs=N_JOBS, verbosity=0, random_state=0),
        "MLP": make_pipeline(StandardScaler(),
            MLPClassifier(hidden_layer_sizes=(256, 64), alpha=1e-3, batch_size=256,
                learning_rate_init=1e-3, max_iter=300, early_stopping=True,
                n_iter_no_change=15, random_state=0)),
    }


def main():
    act = lb.build(min_reps=1)
    E = {n: EM.load(n, pooling="mean") for n in EMBEDS}
    common = set.intersection(*(set(v) for v in E.values())) & set(act.Enzyme)
    act = act[act.Enzyme.isin(common)].reset_index(drop=True)
    dev = act.Enzyme.map(holdout.assignment()).eq("dev").to_numpy()
    act = act[dev].reset_index(drop=True)
    y = act.active.to_numpy(int)

    sm = rep.load_smiles("Amine")
    A, _ = rep.morgan({a: sm[a] for a in sorted(act.Amine.unique()) if a in sm}, n_bits=512)
    C, _ = rep.build_core("onehot", labels=sorted(act.Hydroxyl.unique()))
    cl = load_clusters(set(act.Enzyme))
    g = np.array([cl[e] for e in act.Enzyme])
    print(f"{len(act):,} dev cells, {act.Enzyme.nunique()} enzymes, "
          f"{y.sum():,} active ({100*y.mean():.1f}%), {len(set(g))} clusters\n")

    rows, preds = [], {}
    for name in EMBEDS:
        ids = sorted(common)
        Z = StandardScaler().fit_transform(np.stack([E[name][i] for i in ids]))
        vec = dict(zip(ids, Z.astype(np.float32)))
        X = np.hstack([np.stack([vec[e] for e in act.Enzyme]),
                       A.loc[act.Amine].to_numpy(np.float32),
                       C.loc[act.Hydroxyl].to_numpy(np.float32)]).astype(np.float32)
        for mname in models(1.0):
            oof = np.zeros(len(y)); per_fold = []
            for fold, (tr, te) in enumerate(GroupKFold(n_splits=5).split(X, y, g)):
                a, b = balanced_group_split(g[tr], test_size=0.2, seed=fold)
                itr, iva = tr[a], tr[b]
                spw = max((y[itr] == 0).sum() / max((y[itr] == 1).sum(), 1), 1.0)
                mdl = models(spw)[mname]
                if mname == "XGBoost":
                    mdl.fit(X[itr], y[itr], eval_set=[(X[iva], y[iva])], verbose=False)
                else:
                    mdl.fit(X[tr], y[tr])
                p = mdl.predict_proba(X[te])[:, 1]; oof[te] = p
                per_fold.append(average_precision_score(y[te], p))
            pred = (oof >= .5).astype(int)
            tn, fp, fn, tp = confusion_matrix(y, pred).ravel()
            w = []
            for _, s in act.assign(y=y, p=oof).groupby(["Amine", "Hydroxyl"]):
                if s.y.sum() >= 5 and (1 - s.y).sum() >= 5:
                    w.append(roc_auc_score(s.y, s.p))
            rows.append(dict(embedding=name, model=mname, dim=Z.shape[1],
                pr_auc=average_precision_score(y, oof), pr_sd=float(np.std(per_fold)),
                roc_auc=roc_auc_score(y, oof), mcc=matthews_corrcoef(y, pred),
                bal_acc=balanced_accuracy_score(y, pred), f1=f1_score(y, pred),
                precision=precision_score(y, pred, zero_division=0), recall=recall_score(y, pred),
                specificity=tn / (tn + fp),
                log_loss=log_loss(y, np.clip(oof, 1e-6, 1 - 1e-6)),
                brier=brier_score_loss(y, oof), within_substrate=float(np.median(w)),
                tp=int(tp), fp=int(fp), fn=int(fn), tn=int(tn)))
            preds[f"{name}|{mname}"] = oof
            print(f"  {name:9s} {mname:20s} PR {rows[-1]['pr_auc']:.3f}  "
                  f"ROC {rows[-1]['roc_auc']:.3f}  LL {rows[-1]['log_loss']:.3f}  "
                  f"within-sub {rows[-1]['within_substrate']:.3f}")

    df = pd.DataFrame(rows)
    OUT.mkdir(exist_ok=True)
    df.to_csv(OUT / "sweep_metrics.csv", index=False)
    np.savez(OUT / "sweep_oof.npz", y=y, **preds)
    act.to_csv(OUT / "sweep_meta.csv", index=False)
    print(f"\n-> outputs/sweep_metrics.csv, sweep_oof.npz")


if __name__ == "__main__":
    main()
