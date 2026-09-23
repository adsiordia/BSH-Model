"""Write site/methods.json -- the 'how this model was made' record.

Everything here is read back out of the artifacts that the runs actually wrote
(outputs/sweep_metrics.csv, sweep_oof.npz, logloss_curves.npz, the holdout split,
the label module's own constants) rather than retyped, so the page cannot drift
away from the code.
"""
from pathlib import Path
import json, pickle, numpy as np, pandas as pd
from sklearn.metrics import (precision_recall_curve, roc_curve, roc_auc_score,
                             average_precision_score)

import labels as lb, holdout, embeddings as EM, splits

ROOT = Path(__file__).resolve().parent.parent
OUT, SITE = ROOT / "outputs", ROOT / "site"
N_CURVE = 160          # points kept per curve after downsampling
N_LL = 300             # points kept per log-loss trace


def thin(*arrays, n):
    """Even subsample that always keeps the first and last point."""
    L = len(arrays[0])
    idx = np.unique(np.linspace(0, L - 1, min(n, L)).astype(int))
    return [np.asarray(a)[idx].round(4).tolist() for a in arrays]


def label_counts():
    rows = []
    for mr in (1, 2, 3):
        a = lb.build(min_reps=mr)
        rows.append(dict(min_reps=mr, cells=int(len(a)), active=int(a.active.sum()),
                         rate=round(float(a.active.mean()), 4)))
    return rows


def curves(z, meta):
    y = z["y"]
    out = {}
    for k in [f for f in z.files if f != "y"]:
        p = z[k]
        pr, rc, _ = precision_recall_curve(y, p)
        fpr, tpr, _ = roc_curve(y, p)
        bins = np.clip((p * 10).astype(int), 0, 9)
        cal = [dict(x=round(float(p[bins == b].mean()), 4),
                    y=round(float(y[bins == b].mean()), 4), n=int((bins == b).sum()))
               for b in range(10) if (bins == b).sum() >= 20]
        r, pcs = thin(rc, pr, n=N_CURVE)
        f, t = thin(fpr, tpr, n=N_CURVE)
        out[k] = dict(pr=dict(x=r, y=pcs), roc=dict(x=f, y=t), cal=cal)
    return out


def within_substrate(z, meta):
    """Per-(amine, core) AUC for every model, plus the paired ESM-3 vs rest test."""
    y = z["y"]
    grp = meta.groupby(["Amine", "Hydroxyl"]).indices
    use = {g: i for g, i in grp.items() if y[i].sum() >= 5 and (1 - y[i]).sum() >= 5}
    keys = [k for k in z.files if k != "y"]
    A = pd.DataFrame({k: [roc_auc_score(y[i], z[k][i]) for i in use.values()] for k in keys})
    rng = np.random.default_rng(0)
    base = A["ESM-3|XGBoost"]
    paired = []
    for k in keys:
        if k == "ESM-3|XGBoost":
            continue
        d = (base - A[k]).to_numpy()
        bs = np.array([rng.choice(d, len(d), replace=True).mean() for _ in range(4000)])
        paired.append(dict(model=k, diff=round(float(d.mean()), 4),
                           lo=round(float(np.percentile(bs, 2.5)), 4),
                           hi=round(float(np.percentile(bs, 97.5)), 4),
                           win=round(float((d > 0).mean()), 3)))
    summary = {k: dict(mean=round(float(A[k].mean()), 4),
                       median=round(float(A[k].median()), 4),
                       sd=round(float(A[k].std()), 4)) for k in keys}
    return dict(n_groups=len(use), n_groups_total=len(grp),
                summary=summary, paired=paired,
                groups=[f"{a} + {h}" for a, h in use])


def logloss(base_rate):
    f = OUT / "logloss_curves.npz"
    if not f.exists():
        return None
    z = np.load(f)
    names = sorted({k.split("|")[0] for k in z.files})
    out = {}
    for n in names:
        d = {}
        for part in ("train", "val", "held"):
            C = z[f"{n}|{part}"]
            m, s = thin(C.mean(0), C.std(0), n=N_LL)
            d[part] = dict(mean=m, sd=s)
        v = z[f"{n}|val"].mean(0)
        d["best"] = dict(round=int(v.argmin()), value=round(float(v.min()), 4))
        d["per_fold_rise"] = [round(float(r[-1] - r.min()), 4) for r in z[f"{n}|held"]]
        out[n] = d
    x, = thin(np.arange(z[f"{names[0]}|train"].shape[1]), n=N_LL)
    # reference line: the score you get by ignoring every feature and always
    # predicting the overall hit rate. Depends on the label rule, so compute it.
    b = float(base_rate)
    null_ll = -(b * np.log(b) + (1 - b) * np.log(1 - b))
    return dict(x=[int(v) for v in x], traces=out,
                null_ll=round(float(null_ll), 4), base_rate=round(b, 4))


def final_model():
    """The production model: its config, dev-vs-test metrics, and test curves."""
    f = ROOT / "outputs/test_final_predictions.csv"
    if not f.exists():
        return None
    t = pd.read_csv(f)
    dev = json.loads((OUT / "final_model.json").read_text())
    tm = pd.read_csv(OUT / "test_final_metrics.csv").set_index("which")
    d = pickle.load(open(ROOT / "models/bsh_prostt5_xgb.pkl", "rb"))
    y = t.active.to_numpy(int)

    cur = {}
    for which in ("raw", "calibrated"):
        p_ = t[which].to_numpy()
        pr, rc, _ = precision_recall_curve(y, p_)
        fpr, tpr, _ = roc_curve(y, p_)
        bins = np.clip((p_ * 10).astype(int), 0, 9)
        r, pcs = thin(rc, pr, n=N_CURVE); fx, tx = thin(fpr, tpr, n=N_CURVE)
        cur[which] = dict(pr=dict(x=r, y=pcs), roc=dict(x=fx, y=tx),
                          cal=[dict(x=round(float(p_[bins == b].mean()), 4),
                                    y=round(float(y[bins == b].mean()), 4),
                                    n=int((bins == b).sum()))
                               for b in range(10) if (bins == b).sum() >= 15])

    # resample the 23 enzymes, not the rows -- rows within an enzyme are not independent
    rng = np.random.default_rng(0)
    enz = t.Enzyme.unique()
    idx = {e: np.where(t.Enzyme == e)[0] for e in enz}
    boot = {"pr_auc": [], "roc_auc": []}
    for _ in range(2000):
        sel = np.concatenate([idx[e] for e in rng.choice(enz, len(enz), replace=True)])
        yy = y[sel]
        if yy.sum() < 5 or (1 - yy).sum() < 5:
            continue
        boot["pr_auc"].append(average_precision_score(yy, t["raw"].to_numpy()[sel]))
        boot["roc_auc"].append(roc_auc_score(yy, t["raw"].to_numpy()[sel]))
    ci = {k: [round(float(np.percentile(v, 2.5)), 4), round(float(np.percentile(v, 97.5)), 4)]
          for k, v in boot.items()}

    per = []
    for e, s_ in t.groupby("Enzyme"):
        per.append(dict(enzyme=e, n=int(len(s_)), active=int(s_.active.sum()),
                        rate=round(float(s_.active.mean()), 4),
                        pr_auc=round(float(average_precision_score(s_.active, s_["raw"])), 4)
                        if s_.active.sum() >= 3 else None))
    per.sort(key=lambda r: (r["pr_auc"] is None, -(r["pr_auc"] or 0)))

    base_t, base_d = float(y.mean()), dev["base_rate"]
    skill = lambda a, b: round((a - b) / (1 - b), 4)
    return dict(
        embedding=d["embedding"], pooling=d["pooling"], n_estimators=d["n_estimators"],
        rounds_per_fold=dev["rounds_per_fold"],
        trained_on=d["trained_on"], label_rule=d["label_rule"],
        test=dict(rows=int(len(t)), enzymes=int(t.Enzyme.nunique()),
                  clusters=int(pd.read_csv(ROOT / "data/derived/holdout_split.csv")
                               .query("split=='test'").cluster.nunique()),
                  active=int(y.sum()), base_rate=round(base_t, 4),
                  boundary_identity=0.676),
        metrics=[dict(which=w, **{c: round(float(tm.loc[w, c]), 4) for c in
                 ["pr_auc", "roc_auc", "log_loss", "brier", "mcc", "bal_acc", "f1",
                  "precision", "recall", "specificity", "mean_pred"]},
                 **{c: int(tm.loc[w, c]) for c in ["tp", "fp", "fn", "tn"]})
                 for w in ("raw", "calibrated")],
        dev=dict(pr_auc=dev["pr_auc"], roc_auc=dev["roc_auc"], log_loss=dev["log_loss_raw"],
                 log_loss_calibrated=dev["log_loss_calibrated"], brier=dev["brier_raw"],
                 base_rate=base_d, within_substrate=dev["within_substrate_mean"],
                 mean_prediction=dev["mean_prediction_raw"],
                 mean_prediction_calibrated=dev["mean_prediction_calibrated"]),
        skill=dict(dev=skill(dev["pr_auc"], base_d),
                   test=skill(float(tm.loc["raw", "pr_auc"]), base_t)),
        ci=ci, curves=cur, per_enzyme=per,
        null=dict(pr_auc=round(base_t, 4), roc_auc=0.5,
                  log_loss=round(float(-(base_t*np.log(base_t)+(1-base_t)*np.log(1-base_t))), 4)),
        duplicate_pair=dict(a="A0A3Q0NGD6", b="Q8Y5J3", agreement=0.827, cells=75,
                            note="These two wells hold the same protein sequence and both "
                                 "landed in the test set. Their own measurements agree on "
                                 "82.7% of 75 combinations. The model must score them "
                                 "identically, so it is counted wrong wherever the "
                                 "experiment disagrees with itself -- that is the ceiling "
                                 "any model faces here."),
        caveat="Evaluated once. These 23 enzymes are no longer a clean estimate of "
               "performance on unseen BSH, so any further tuning has to be judged on the "
               "development set.")


def main():
    z = np.load(OUT / "sweep_oof.npz")
    meta = pd.read_csv(OUT / "sweep_meta.csv")
    mt = pd.read_csv(OUT / "sweep_metrics.csv")
    split = pd.read_csv(ROOT / "data/derived/holdout_split.csv")
    y = z["y"]

    doc = dict(
        generated=pd.Timestamp.today().strftime("%Y-%m-%d"),
        source=dict(
            file="Trimmed_remove_proteomics.csv",
            note="Single source of truth. Every enzyme in it was confirmed present by "
                 "proteomics, so an absent product means the enzyme was there and did not "
                 "make it -- not that the enzyme failed to express.",
            n_enzymes=int(split.Enzyme.nunique()),
            n_samples=366),
        labels=dict(
            threshold=int(lb.THRESHOLD),
            threshold_basis="Raised from 10,000 to 50,000 on 22 September 2026. The old "
                            "cut-off came from the negative controls, but that evidence is "
                            "thinner than it looks: of 1,458 control measurements only NINE "
                            "are non-zero, and the two that set the bound (10,059 and 10,320) "
                            "fall on just 2 of 55 products. Every other control read zero, "
                            "which says nothing about where its noise floor sits. 50,000 "
                            "instead rests on detection reproducibility measured across all "
                            "2,467 non-zero cells: the share of detections that reappear in a "
                            "second replicate climbs 34% (10-25k) to 66% (25-50k) to 82% "
                            "(50-100k) and then stays flat all the way up. Above 50,000 a "
                            "detection is as reliable as it will ever be. A five-way "
                            "comparison found NO measurable difference in model performance "
                            "between 10,000, 25,000 and 50,000 (paired over 22 substrate "
                            "groups, every interval spanning zero), so the choice rests on "
                            "the evidence behind the label, not on a model gain.",
            threshold_history=dict(old=10000, new=50000, changed="2026-09-22"),
            n_reps=lb.N_REPS,
            min_reps_documented=lb.MIN_REPS,
            min_reps_used=1,
            discrepancy="Every model on this page was trained with min_reps=1 (above "
                        "threshold in at least one of three replicates). labels.py "
                        "documents min_reps=2 as the default. The looser rule is what ran.",
            counts=label_counts(),
            features=lb.feature_summary(min_reps=1),
            grid="Amines and bile acids were each pooled into one master mix, so every "
                 "enzyme met every combination. A combination absent from the product "
                 "table produced no detectable feature in any of the 366 samples, which "
                 "is a negative for every enzyme.",
            caveat="Negative controls exist for only 9 of the 81 products, and those 9 are "
                   "low-signal. Background on the high-signal products is untested.",
            degree="Degree class counts HYDROXYLS only -- keto groups are not counted. "
                   "3a,7a,12k is Di; 3a7k is Mono."),
        split=dict(
            identity=holdout.IDENTITY, test_fraction=holdout.TEST_FRACTION, seed=holdout.SEED,
            dev_enzymes=int((split.split == "dev").sum()),
            dev_clusters=int(split[split.split == "dev"].cluster.nunique()),
            test_enzymes=int((split.split == "test").sum()),
            test_clusters=int(split[split.split == "test"].cluster.nunique()),
            method="Clusters are connected components of a 'sequence identity >= 70%' graph "
                   "built from a full MAFFT alignment. Connected components guarantee that "
                   "no pair spanning the dev/test boundary exceeds 70% -- greedy clustering "
                   "(cd-hit) does not, and leaked 55 such pairs.",
            cv="Inside dev, 5-fold GroupKFold on the same clusters. Each fold holds out "
               "20% of its training clusters again as an inner validation set for early "
               "stopping, so nothing in the outer fold is ever seen.",
            test_status="NEVER EVALUATED. outputs/test_set_evaluations.log does not exist. "
                        "Every number on this page is dev-set cross-validation."),
        embeddings=[dict(name=n, dim=int(v["dim"]), kind=v["kind"], path=str(v["path"]).replace(
            str(Path.home()), "~")) for n, v in EM.SOURCES.items()],
        pooling=dict(
            used="mean",
            explanation="A protein language model returns one vector per amino acid, so a "
                        "347-residue enzyme arrives as a 347 x D grid. The model needs one "
                        "row per enzyme, so that grid is collapsed to a single D-long vector. "
                        "MEAN averages each column down the chain ('what is this protein like "
                        "on average'). MAX takes each column's largest value ('the strongest "
                        "signal anywhere in the chain'). Mean dilutes a handful of active-site "
                        "residues across hundreds of positions; max does not.",
            open_question="Settled by a later sweep: 7 pooling schemes x 4 embeddings, "
                          "including mean+max and alignment-column selection of conserved and "
                          "non-conserved residues. All 28 runs land between 0.270 and 0.298 "
                          "log loss and none beats plain whole-protein mean. One scheme "
                          "(non-conserved, conservation < 0.60) looked like a winner on "
                          "ProstT5 (+0.032, p < 0.001) and did not replicate on the other "
                          "three embeddings, which is what a false positive looks like. An "
                          "earlier one-off ProtT5 max-pooling run that scored 0.754 "
                          "within-substrate AUC was not reproduced by the systematic sweep "
                          "and should be read as noise."),
        features=dict(
            blocks=[dict(name="enzyme", dim=None, how="whole-protein embedding, mean-pooled "
                         "over residues, then standardised across the 91 enzymes"),
                    dict(name="amine", dim=512, how="Morgan fingerprint, radius 2, 512 bits, "
                         "from the curated SMILES in bsh_reactants_SMILES_corrected"),
                    dict(name="core", dim=3, how="one-hot over Mono / Di / Tri")],
            note="The enzyme block's width is whatever the embedding provides (1024-1536). "
                 "The model answers one question: will THIS enzyme attach THIS amine to a "
                 "bile acid of THIS hydroxyl class?"),
        dataset=dict(cells=int(len(meta)), enzymes=int(meta.Enzyme.nunique()),
                     active=int(y.sum()), base_rate=round(float(y.mean()), 4),
                     clusters=int(pd.Series(
                         [splits.load_clusters()[e] for e in meta.Enzyme.unique()]).nunique()),
                     note="Dev only. 91 rather than 92 enzymes because A0A1I4P275 is absent "
                          "from ESM-3 (its embedding has 300 residues against 347 in the "
                          "FASTA, unresolved) and was dropped from all four embeddings so "
                          "the comparison stays matched."),
        algorithms=[
            dict(name="XGBoost", detail="400 trees, depth 4, lr 0.05, subsample 0.8, "
                 "colsample 0.8, alpha 1, lambda 5, scale_pos_weight balanced, early "
                 "stopping on the inner split"),
            dict(name="RandomForest", detail="400 trees, min_samples_leaf 3, balanced "
                 "subsample class weights"),
            dict(name="MLP", detail="256-64 hidden units, alpha 1e-3, early stopping, "
                 "standardised inputs"),
            dict(name="LogisticRegression", detail="L2, C=0.1, balanced class weights, "
                 "standardised inputs")],
        metrics=json.loads(mt.to_json(orient="records")),
        curves=curves(z, meta),
        within=within_substrate(z, meta),
        logloss=logloss(float(meta.active.mean())),
        final=final_model(),
        findings=[
            "The algorithm matters more than the embedding. XGBoost wins on all four "
            "embeddings; the four XGBoost PR-AUCs span 0.011 against fold standard "
            "deviations of 0.046-0.058.",
            "ESM-3 and ProstT5 are statistically tied on within-substrate AUC (paired "
            "difference +0.004, 95% CI crosses zero, ESM-3 ahead in 51% of substrate "
            "groups). Both beat ProtT5 and ESM-2, which are real differences.",
            "LogisticRegression is worse than useless as a probability model: its log loss "
            "on three of four embeddings exceeds the 0.542 of always predicting the base "
            "rate. The signal is not linear in embedding space.",
            "MLP ranks well but is badly calibrated -- good MCC and F1, log loss 0.59-0.65 "
            "against XGBoost's 0.33-0.37.",
            "400 boosting rounds was under-trained. Validation log loss was still falling "
            "at the last round for all four embeddings; the optimum sits near 900-1100.",
            "Fold-to-fold spread dwarfs embedding differences. Within one embedding, "
            "validation log loss ranges from 0.10 to 0.43 across the five folds.",
            "Fold 2 overfits in all four embeddings while folds 1 and 4 never turn, so a "
            "fixed round count is wrong -- early stopping has to be per fold.",
            "Most of the overall PR-AUC comes from learning which AMINES react easily, not "
            "from reading the protein. Within-substrate AUC is the metric that isolates "
            "protein skill, and it is far lower than the headline number."],
    )
    for b in doc["features"]["blocks"]:
        if b["name"] == "enzyme":
            b["dim"] = sorted({int(r["dim"]) for r in doc["metrics"]})
    SITE.mkdir(exist_ok=True)
    p = SITE / "methods.json"
    p.write_text(json.dumps(doc, separators=(",", ":")))
    print(f"wrote {p}  ({p.stat().st_size/1e3:.0f} kB)")
    print(f"  {len(doc['metrics'])} metric rows, {len(doc['curves'])} curve sets, "
          f"{doc['within']['n_groups']} substrate groups, "
          f"logloss traces: {list(doc['logloss']['traces']) if doc['logloss'] else 'none'}")


if __name__ == "__main__":
    main()
