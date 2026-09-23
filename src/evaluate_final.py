"""ONE-SHOT evaluation of models/bsh_prostt5_xgb.pkl on the locked test set.

The model is loaded, not refitted -- it was trained on dev only, and nothing in this
script touches a training decision. Every run appends to outputs/test_set_evaluations.log
so that repeated peeking is visible in the record.

After this runs, the 23 test enzymes are no longer a clean estimate of performance on
unseen BSH. Any further tuning has to be judged on dev.
"""
from pathlib import Path
from datetime import datetime
import pickle, json, numpy as np, pandas as pd, warnings
warnings.filterwarnings("ignore")
from sklearn.metrics import (average_precision_score, roc_auc_score, f1_score, precision_score,
                             recall_score, matthews_corrcoef, balanced_accuracy_score,
                             log_loss, brier_score_loss, confusion_matrix)

import labels as lb, holdout, embeddings as EM
from splits import identity_matrix, check_leakage

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "outputs"


def score(y, p, label):
    pred = (p >= .5).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return dict(which=label, pr_auc=average_precision_score(y, p), roc_auc=roc_auc_score(y, p),
                log_loss=log_loss(y, np.clip(p, 1e-6, 1 - 1e-6)), brier=brier_score_loss(y, p),
                mcc=matthews_corrcoef(y, pred), bal_acc=balanced_accuracy_score(y, pred),
                f1=f1_score(y, pred), precision=precision_score(y, pred, zero_division=0),
                recall=recall_score(y, pred), specificity=tn / max(tn + fp, 1),
                mean_pred=p.mean(), tp=int(tp), fp=int(fp), fn=int(fn), tn=int(tn))


def main():
    d = pickle.load(open(ROOT / "models/bsh_prostt5_xgb.pkl", "rb"))
    act = lb.build(min_reps=d["label_rule"]["min_reps"])
    emb = EM.load(d["embedding"], pooling=d["pooling"])
    act = act[act.Enzyme.isin(emb)].reset_index(drop=True)
    assign = holdout.assignment()
    test = act[act.Enzyme.map(assign).eq("test")].reset_index(drop=True)
    y = test.active.to_numpy(int)

    enz = sorted(set(act.Enzyme))
    worst, off = check_leakage({e: assign[e] for e in enz}, identity_matrix(set(enz)), set(enz))

    print(f"\n{'='*78}\nLOCKED TEST SET -- single evaluation\n{'='*78}")
    print(f"  model        {d['embedding']} {d['pooling']}-pooled + amine + core, XGBoost "
          f"x{d['n_estimators']}")
    print(f"  trained on   {d['trained_on']['enzymes']} dev enzymes, "
          f"{d['trained_on']['rows']:,} rows")
    print(f"  test         {len(test):,} rows / {test.Enzyme.nunique()} enzymes / "
          f"{len(set(holdout.clusters()[e] for e in test.Enzyme))} clusters")
    print(f"  test active  {y.sum():,} ({y.mean():.1%})   dev was "
          f"{d['trained_on']['active']/d['trained_on']['rows']:.1%}")
    print(f"  max sequence identity across the dev/test boundary: {worst:.1%}"
          f"  ({'PASS' if worst < holdout.IDENTITY else 'FAIL'}, must be < "
          f"{holdout.IDENTITY:.0%})\n")

    X = np.hstack([d["scaler"].transform(np.stack([emb[e] for e in test.Enzyme])).astype(np.float32),
                   d["amine_bits"].loc[test.Amine].to_numpy(np.float32),
                   d["core_cols"].loc[test.Hydroxyl].to_numpy(np.float32)]).astype(np.float32)
    raw = d["model"].predict_proba(X)[:, 1]
    cal = d["isotonic"].predict(raw)

    rows = [score(y, raw, "raw"), score(y, cal, "calibrated")]
    df = pd.DataFrame(rows)
    cols = ["which", "pr_auc", "roc_auc", "log_loss", "brier", "mcc", "bal_acc", "f1",
            "precision", "recall", "specificity", "mean_pred"]
    print(df[cols].round(3).to_string(index=False))
    print(f"\n  at a 0.5 cut-off (raw):  tp={rows[0]['tp']}  fp={rows[0]['fp']}  "
          f"fn={rows[0]['fn']}  tn={rows[0]['tn']}")

    base = y.mean()
    print(f"\n  a coin flip would score pr_auc={base:.3f}, roc_auc=0.500, "
          f"log_loss={-(base*np.log(base)+(1-base)*np.log(1-base)):.3f}")

    dev = json.loads((OUT / "final_model.json").read_text())
    print(f"\n  dev cross-validation said: pr_auc={dev['pr_auc']:.3f}  "
          f"roc_auc={dev['roc_auc']:.3f}  log_loss={dev['log_loss_raw']:.3f}")
    print(f"  test says:                 pr_auc={rows[0]['pr_auc']:.3f}  "
          f"roc_auc={rows[0]['roc_auc']:.3f}  log_loss={rows[0]['log_loss']:.3f}")

    w = []
    for (a, h), s in test.assign(y=y, p=raw).groupby(["Amine", "Hydroxyl"]):
        if s.y.sum() >= 3 and (1 - s.y).sum() >= 3:
            w.append(roc_auc_score(s.y, s.p))
    if w:
        print(f"\n  within-substrate AUC on test: mean {np.mean(w):.3f}  "
              f"median {np.median(w):.3f}  ({len(w)} groups with >=3 of each; "
              f"only {test.Enzyme.nunique()} enzymes, so these are noisy)")
        print(f"  dev said: mean {dev['within_substrate_mean']:.3f}")

    OUT.mkdir(exist_ok=True)
    df.to_csv(OUT / "test_final_metrics.csv", index=False)
    pd.DataFrame(dict(Enzyme=test.Enzyme, Amine=test.Amine, Hydroxyl=test.Hydroxyl,
                      active=y, raw=raw.round(4), calibrated=cal.round(4))
                 ).to_csv(OUT / "test_final_predictions.csv", index=False)
    with open(OUT / "test_set_evaluations.log", "a") as f:
        f.write(f"{datetime.now().isoformat(timespec='seconds')}  "
                f"model=bsh_prostt5_xgb.pkl  embedding={d['embedding']}  "
                f"n_estimators={d['n_estimators']}  min_reps={d['label_rule']['min_reps']}  "
                f"test_rows={len(y)}  test_enzymes={test.Enzyme.nunique()}  "
                f"pr_auc={rows[0]['pr_auc']:.4f}  roc_auc={rows[0]['roc_auc']:.4f}  "
                f"log_loss_raw={rows[0]['log_loss']:.4f}  "
                f"log_loss_cal={rows[1]['log_loss']:.4f}\n")
    print(f"\n-> outputs/test_final_metrics.csv, test_final_predictions.csv")
    print(f"-> appended to outputs/test_set_evaluations.log")


if __name__ == "__main__":
    main()
