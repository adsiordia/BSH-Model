"""Fit the production model: ProstT5 whole-protein mean + amine Morgan + core one-hot, XGBoost.

Two things happen here that the sweep did not do.

  Rounds.   The sweep capped n_estimators at 400 and the validation log loss was still
            falling at the last round, so every sweep model was under-trained. The number
            of rounds is instead chosen per fold by early stopping against a held-out
            slice of that fold's training clusters, then the median is used for the model
            fitted on all of dev.

  Calibration.  scale_pos_weight stops the model ignoring the minority class but inflates
            its probabilities -- mean predicted 0.311 against a true base rate of 0.232.
            An isotonic regression fitted on the out-of-fold predictions maps the raw
            score back onto an honest probability. Both are saved; the raw score is the
            better ranking, the calibrated one is the number to quote.

The locked test set is NOT touched. Run src/evaluate.py run_final() for that, once.
"""
from pathlib import Path
import json, numpy as np, pandas as pd, warnings, pickle
warnings.filterwarnings("ignore")
from sklearn.model_selection import GroupKFold, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import (average_precision_score, roc_auc_score, log_loss,
                             brier_score_loss, f1_score, matthews_corrcoef)
import xgboost as xgb

import labels as lb, representations as rep, holdout, embeddings as EM
from splits import identity_matrix, cdhit_merged_clusters, balanced_group_split, load_clusters

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "outputs"; MODELS = ROOT / "models"
EMBEDDING, POOLING = "ProstT5", "mean"
MAX_ROUNDS, PATIENCE = 3000, 50
PARAMS = dict(max_depth=4, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8,
              reg_alpha=1.0, reg_lambda=5.0, eval_metric="logloss", n_jobs=-1,
              verbosity=0, random_state=0)


def features(act, vec, A, C):
    return np.hstack([np.stack([vec[e] for e in act.Enzyme]),
                      A.loc[act.Amine].to_numpy(np.float32),
                      C.loc[act.Hydroxyl].to_numpy(np.float32)]).astype(np.float32)


def main():
    act = lb.build(min_reps=1)
    emb = EM.load(EMBEDDING, pooling=POOLING)
    act = act[act.Enzyme.isin(emb)].reset_index(drop=True)
    dev = act[act.Enzyme.map(holdout.assignment()).eq("dev")].reset_index(drop=True)
    y = dev.active.to_numpy(int)

    # the scaler is fitted on DEV enzymes only -- a scaler fitted on all 115 would carry
    # information about the test sequences into the model
    ids = sorted(dev.Enzyme.unique())
    sc = StandardScaler().fit(np.stack([emb[i] for i in ids]))
    vec = {i: v for i, v in zip(sorted(emb), sc.transform(np.stack(
        [emb[i] for i in sorted(emb)])).astype(np.float32))}

    sm = rep.load_smiles("Amine")
    A, _ = rep.morgan({a: sm[a] for a in sorted(act.Amine.unique()) if a in sm}, n_bits=512)
    C, _ = rep.build_core("onehot", labels=sorted(act.Hydroxyl.unique()))
    X = features(dev, vec, A, C)
    cl = load_clusters(set(dev.Enzyme))
    g = np.array([cl[e] for e in dev.Enzyme])
    print(f"{EMBEDDING} {POOLING}-pooled | X={X.shape} | {y.sum():,} active of {len(y):,} "
          f"({y.mean():.1%}) | {dev.Enzyme.nunique()} enzymes, {len(set(g))} clusters\n")

    # --- per-fold early stopping: how many rounds does this data actually want? ---
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
        print(f"  fold {fold}: stopped at {rounds[-1]:4d} rounds   "
              f"PR-AUC {average_precision_score(y[te], oof[te]):.3f}")
    n_est = int(np.median(rounds))

    iso = IsotonicRegression(out_of_bounds="clip").fit(oof, y)
    cal = iso.predict(oof)
    print(f"\nrounds per fold {rounds} -> using {n_est}")
    print(f"  raw        PR-AUC {average_precision_score(y, oof):.3f}  "
          f"ROC {roc_auc_score(y, oof):.3f}  log loss {log_loss(y, np.clip(oof,1e-6,1-1e-6)):.3f}  "
          f"Brier {brier_score_loss(y, oof):.3f}  mean pred {oof.mean():.3f}")
    print(f"  calibrated PR-AUC {average_precision_score(y, cal):.3f}  "
          f"ROC {roc_auc_score(y, cal):.3f}  log loss {log_loss(y, np.clip(cal,1e-6,1-1e-6)):.3f}  "
          f"Brier {brier_score_loss(y, cal):.3f}  mean pred {cal.mean():.3f}   (truth {y.mean():.3f})")

    w = []
    for _, s in dev.assign(y=y, p=oof).groupby(["Amine", "Hydroxyl"]):
        if s.y.sum() >= 5 and (1 - s.y).sum() >= 5:
            w.append(roc_auc_score(s.y, s.p))
    print(f"  within-substrate AUC  mean {np.mean(w):.3f}  median {np.median(w):.3f}  "
          f"({len(w)} substrate groups)")

    # --- refit on all of dev at the agreed number of rounds ---
    spw = max((y == 0).sum() / (y == 1).sum(), 1.0)
    final = xgb.XGBClassifier(n_estimators=n_est, scale_pos_weight=spw, **PARAMS)
    final.fit(X, y, verbose=False)

    MODELS.mkdir(exist_ok=True)
    with open(MODELS / "bsh_prostt5_xgb.pkl", "wb") as f:
        pickle.dump(dict(model=final, isotonic=iso, scaler=sc, amine_bits=A, core_cols=C,
                         embedding=EMBEDDING, pooling=POOLING, n_estimators=n_est,
                         scale_pos_weight=spw, params=PARAMS,
                         feature_layout=["enzyme:1024", "amine:512", "core:3"],
                         trained_on=dict(enzymes=int(dev.Enzyme.nunique()), rows=int(len(y)),
                                         active=int(y.sum()), split="dev"),
                         label_rule=dict(threshold=lb.THRESHOLD, min_reps=1, n_reps=lb.N_REPS)), f)
    pd.DataFrame(dict(Enzyme=dev.Enzyme, Amine=dev.Amine, Hydroxyl=dev.Hydroxyl,
                      active=y, raw=oof.round(4), calibrated=cal.round(4))
                 ).to_csv(OUT / "final_model_oof.csv", index=False)
    (OUT / "final_model.json").write_text(json.dumps(dict(
        embedding=EMBEDDING, pooling=POOLING, n_estimators=n_est, rounds_per_fold=rounds,
        pr_auc=round(float(average_precision_score(y, oof)), 4),
        roc_auc=round(float(roc_auc_score(y, oof)), 4),
        log_loss_raw=round(float(log_loss(y, np.clip(oof, 1e-6, 1-1e-6))), 4),
        log_loss_calibrated=round(float(log_loss(y, np.clip(cal, 1e-6, 1-1e-6))), 4),
        brier_raw=round(float(brier_score_loss(y, oof)), 4),
        brier_calibrated=round(float(brier_score_loss(y, cal)), 4),
        within_substrate_mean=round(float(np.mean(w)), 4),
        within_substrate_median=round(float(np.median(w)), 4),
        mean_prediction_raw=round(float(oof.mean()), 4),
        mean_prediction_calibrated=round(float(cal.mean()), 4),
        base_rate=round(float(y.mean()), 4)), indent=2))
    print(f"\n-> models/bsh_prostt5_xgb.pkl")
    print(f"-> outputs/final_model_oof.csv, outputs/final_model.json")
    n_test = int((pd.read_csv(ROOT / "data/derived/holdout_split.csv").split == "test").sum())
    print(f"\nlocked test set untouched ({n_test} enzymes).")


if __name__ == "__main__":
    main()
