"""The deployment model: same recipe as train_final.py, fitted on ALL 115 enzymes.

Why two models exist
--------------------
train_final.py fits on the 92 development enzymes only, holding back 23. That is what
makes the locked-test number (ROC-AUC 0.924) an honest estimate of performance on an
enzyme the model has never seen -- it can only be honest if those 23 were excluded.

That estimate is now banked and cannot be improved on. So for actually predicting new
BSH sequences, the right model is the one fitted on every enzyme available: 25% more
training data, same recipe. Its expected performance is the dev model's test score or
a little better -- and it can never be re-measured, because there is nothing left to
hold out. That is the accepted trade: you measure with one model and ship another.

Rounds are chosen by the same cluster-grouped cross-validation, now run over all 115.
"""
from pathlib import Path
import json, pickle, numpy as np, pandas as pd, warnings
warnings.filterwarnings("ignore")
from sklearn.model_selection import GroupKFold, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import (average_precision_score, roc_auc_score, log_loss,
                             brier_score_loss)
import xgboost as xgb

# respect the SLURM allocation rather than taking the whole node
N_JOBS = int(__import__('os').environ.get('SLURM_CPUS_PER_TASK', 8))

import labels as lb, representations as rep, holdout, embeddings as EM
from splits import identity_matrix, cdhit_merged_clusters, balanced_group_split, load_clusters

ROOT = Path(__file__).resolve().parent.parent
OUT, MODELS = ROOT / "outputs", ROOT / "models"
EMBEDDING, POOLING = "ProstT5", "mean"
MAX_ROUNDS, PATIENCE = 3000, 50
PARAMS = dict(max_depth=4, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8,
              reg_alpha=1.0, reg_lambda=5.0, eval_metric="logloss", n_jobs=N_JOBS,
              verbosity=0, random_state=0)


def main():
    act = lb.build(min_reps=1)
    emb = EM.load(EMBEDDING, pooling=POOLING)
    act = act[act.Enzyme.isin(emb)].reset_index(drop=True)
    y = act.active.to_numpy(int)

    ids = sorted(act.Enzyme.unique())
    sc = StandardScaler().fit(np.stack([emb[i] for i in ids]))
    vec = {i: v for i, v in zip(ids, sc.transform(np.stack([emb[i] for i in ids])).astype(np.float32))}

    sm = rep.load_smiles("Amine")
    A, _ = rep.morgan({a: sm[a] for a in sorted(act.Amine.unique()) if a in sm}, n_bits=512)
    C, _ = rep.build_core("onehot", labels=sorted(act.Hydroxyl.unique()))
    X = np.hstack([np.stack([vec[e] for e in act.Enzyme]),
                   A.loc[act.Amine].to_numpy(np.float32),
                   C.loc[act.Hydroxyl].to_numpy(np.float32)]).astype(np.float32)
    cl = load_clusters(set(act.Enzyme))
    g = np.array([cl[e] for e in act.Enzyme])
    assign = holdout.assignment()
    print(f"{EMBEDDING} {POOLING}-pooled | X={X.shape} | {act.Enzyme.nunique()} enzymes "
          f"({sum(assign.get(e) == 'dev' for e in ids)} dev + "
          f"{sum(assign.get(e) == 'test' for e in ids)} test, both now used for training)")
    print(f"{y.sum():,} active of {len(y):,} ({y.mean():.1%}), {len(set(g))} clusters\n")

    oof, rounds = np.zeros(len(y)), []
    for fold, (tr, te) in enumerate(GroupKFold(n_splits=5).split(X, y, g)):
        a, b = balanced_group_split(g[tr], test_size=0.2, seed=fold)
        itr, iva = tr[a], tr[b]
        spw = max((y[itr] == 0).sum() / max((y[itr] == 1).sum(), 1), 1.0)
        m = xgb.XGBClassifier(n_estimators=MAX_ROUNDS, scale_pos_weight=spw,
                              early_stopping_rounds=PATIENCE, **PARAMS)
        m.fit(X[itr], y[itr], eval_set=[(X[iva], y[iva])], verbose=False)
        oof[te] = m.predict_proba(X[te])[:, 1]
        rounds.append(m.best_iteration + 1)
        print(f"  fold {fold}: {rounds[-1]:4d} rounds   PR-AUC {average_precision_score(y[te], oof[te]):.3f}")
    n_est = int(np.median(rounds))

    iso = IsotonicRegression(out_of_bounds="clip").fit(oof, y)
    cal = iso.predict(oof)
    w = [roc_auc_score(s.y, s.p) for _, s in act.assign(y=y, p=oof).groupby(["Amine", "Hydroxyl"])
         if s.y.sum() >= 5 and (1 - s.y).sum() >= 5]
    print(f"\nrounds per fold {rounds} -> using {n_est}")
    print(f"  cross-validated over all 115: PR-AUC {average_precision_score(y, oof):.4f}  "
          f"ROC {roc_auc_score(y, oof):.4f}  log loss {log_loss(y, np.clip(oof,1e-6,1-1e-6)):.4f}")
    print(f"  calibrated log loss {log_loss(y, np.clip(cal,1e-6,1-1e-6)):.4f}  "
          f"Brier {brier_score_loss(y, cal):.4f}  mean pred {cal.mean():.4f} (truth {y.mean():.4f})")
    print(f"  within-substrate AUC mean {np.mean(w):.4f} median {np.median(w):.4f}")
    print(f"\n  NOTE: these are cross-validated numbers, not a held-out test. The honest")
    print(f"        estimate for this model remains the dev model's locked-test result.")

    spw = max((y == 0).sum() / (y == 1).sum(), 1.0)
    final = xgb.XGBClassifier(n_estimators=n_est, scale_pos_weight=spw, **PARAMS)
    final.fit(X, y, verbose=False)

    MODELS.mkdir(exist_ok=True)
    with open(MODELS / "bsh_prostt5_xgb_production.pkl", "wb") as f:
        pickle.dump(dict(model=final, isotonic=iso, scaler=sc, amine_bits=A, core_cols=C,
                         embedding=EMBEDDING, pooling=POOLING, n_estimators=n_est,
                         scale_pos_weight=spw, params=PARAMS,
                         feature_layout=["enzyme:1024", "amine:512", "core:3"],
                         trained_on=dict(enzymes=int(act.Enzyme.nunique()), rows=int(len(y)),
                                         active=int(y.sum()), split="dev+test (all)"),
                         label_rule=dict(threshold=lb.THRESHOLD, min_reps=1, n_reps=lb.N_REPS),
                         held_out_estimate=dict(
                             source="models/bsh_prostt5_xgb.pkl evaluated on 23 locked enzymes",
                             roc_auc=0.924, pr_auc=0.690, precision=0.699, recall=0.701,
                             note="This model has no held-out set of its own. Quote these.")), f)
    pd.DataFrame(dict(Enzyme=act.Enzyme, Amine=act.Amine, Hydroxyl=act.Hydroxyl, active=y,
                      raw=oof.round(4), calibrated=cal.round(4),
                      was=[assign.get(e, "?") for e in act.Enzyme])
                 ).to_csv(OUT / "production_model_oof.csv", index=False)
    (OUT / "production_model.json").write_text(json.dumps(dict(
        embedding=EMBEDDING, pooling=POOLING, n_estimators=n_est, rounds_per_fold=rounds,
        enzymes=int(act.Enzyme.nunique()), rows=int(len(y)),
        cv_pr_auc=round(float(average_precision_score(y, oof)), 4),
        cv_roc_auc=round(float(roc_auc_score(y, oof)), 4),
        cv_log_loss=round(float(log_loss(y, np.clip(oof, 1e-6, 1-1e-6))), 4),
        cv_within_substrate=round(float(np.mean(w)), 4),
        base_rate=round(float(y.mean()), 4)), indent=2))
    print(f"\n-> models/bsh_prostt5_xgb_production.pkl")
    print(f"-> outputs/production_model_oof.csv, production_model.json")


if __name__ == "__main__":
    main()
