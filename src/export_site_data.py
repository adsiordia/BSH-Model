"""Export everything the local explorer needs as one JSON file.

The predictions here use the SAME recipe as the deployment model -- ProstT5 whole-protein
mean pooling + amine fingerprint + core, XGBoost with per-fold early stopping -- but they
are OUT OF FOLD. The deployment model was fitted on all 115 enzymes, so asking it about
those same 115 would return what it memorised, and the site would show near-perfect
agreement that means nothing. Instead the recipe is fitted five times, each leaving out a
fifth of the sequence clusters, and every cell is scored by the run that had not seen its
enzyme. One recipe, two uses: one model to predict with, cross-validated copies to measure
with.
"""
from pathlib import Path
import json, sys
import numpy as np, pandas as pd, warnings
warnings.filterwarnings("ignore")
from sklearn.model_selection import GroupKFold, train_test_split
import xgboost as xgb

# respect the SLURM allocation rather than taking the whole node
N_JOBS = int(__import__('os').environ.get('SLURM_CPUS_PER_TASK', 8))

import dataset as ds, labels as lb, representations as rep, holdout, trimmed as tr
from splits import identity_matrix, cdhit_merged_clusters, balanced_group_split, load_clusters

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "site/data.json"
# identical to src/train_production.py
MODEL_KW = dict(max_depth=4, learning_rate=0.05, subsample=0.8,
                colsample_bytree=0.8, reg_alpha=1.0, reg_lambda=5.0,
                eval_metric="logloss", n_jobs=N_JOBS, verbosity=0, random_state=0)
MAX_ROUNDS, PATIENCE = 3000, 50
EMBEDDING, POOLING = "ProstT5", "mean"


def fit(X, y, groups, seed=0):
    itr, iva = balanced_group_split(groups, test_size=0.2, seed=seed)
    spw = max((y[itr] == 0).sum() / max((y[itr] == 1).sum(), 1), 1.0)
    m = xgb.XGBClassifier(**MODEL_KW, n_estimators=MAX_ROUNDS, scale_pos_weight=spw,
                          early_stopping_rounds=PATIENCE)
    m.fit(X[itr], y[itr], eval_set=[(X[iva], y[iva])], verbose=False)
    return m


def features():
    """The production feature matrix: ProstT5 mean + amine Morgan-512 + core one-hot."""
    import embeddings as EM
    act = lb.build(min_reps=1)
    emb = EM.load(EMBEDDING, pooling=POOLING)
    act = act[act.Enzyme.isin(emb)].reset_index(drop=True)
    ids = sorted(act.Enzyme.unique())
    from sklearn.preprocessing import StandardScaler
    Z = StandardScaler().fit_transform(np.stack([emb[i] for i in ids])).astype(np.float32)
    vec = dict(zip(ids, Z))
    sm = rep.load_smiles("Amine")
    A, _ = rep.morgan({a: sm[a] for a in sorted(act.Amine.unique()) if a in sm}, n_bits=512)
    C, _ = rep.build_core("onehot", labels=sorted(act.Hydroxyl.unique()))
    X = np.hstack([np.stack([vec[e] for e in act.Enzyme]),
                   A.loc[act.Amine].to_numpy(np.float32),
                   C.loc[act.Hydroxyl].to_numpy(np.float32)]).astype(np.float32)
    return X, act.active.to_numpy(int), act[["Enzyme", "Amine", "Hydroxyl"]].copy()


def main():
    X, y, meta = features()
    enz = meta.Enzyme.to_numpy()
    cl = load_clusters(set(enz))
    groups = np.array([cl[e] for e in enz])
    split = meta.Enzyme.map(holdout.assignment()).to_numpy()

    # honest out-of-fold predictions for every cell
    oof = np.zeros(len(y))
    for trn, tst in GroupKFold(n_splits=5).split(X, y, groups):
        oof[tst] = fit(X[trn], y[trn], groups[trn]).predict_proba(X[tst])[:, 1]
    meta = meta.assign(active=y, p=oof.round(4), split=split,
                       cluster=[cl[e] for e in enz])
    amines = sorted(meta.Amine.unique()); cores = ["Mono", "Di", "Tri"]
    enzymes = sorted(meta.Enzyme.unique())

    # per-replicate detail, so the site can show how reproducible each call was
    raw = tr.enzymes(tr.load_long())
    raw["Hydroxyl"] = raw.Hydroxyl.map(lb.TO_DEGREE)
    rep_detail = (raw.groupby(["Code", "Amine", "Hydroxyl", "Replicate"]).Intensity.max()
                     .reset_index()
                     .pivot_table(index=["Code", "Amine", "Hydroxyl"], columns="Replicate",
                                  values="Intensity").fillna(0))
    rep_detail.columns = [f"r{i}" for i in range(1, len(rep_detail.columns) + 1)]
    rep_detail = rep_detail.round(0).astype(int).reset_index().rename(columns={"Code": "Enzyme"})

    # `detected` = a feature for this product exists somewhere in the run.
    # Where it does not, the cell is still a NEGATIVE (no signal in any of the 366
    # samples) -- it is the evidence that differs, not the label.
    det = lb.build(min_reps=1)[["Enzyme", "Amine", "Hydroxyl", "detected"]]
    cells = meta[["Enzyme", "Amine", "Hydroxyl", "active", "p", "split"]].merge(
        det, on=["Enzyme", "Amine", "Hydroxyl"], how="left")
    cells["detected"] = cells.detected.fillna(False).astype(bool)
    cells = cells.merge(rep_detail, on=["Enzyme", "Amine", "Hydroxyl"], how="left")
    for c in ("r1", "r2", "r3"):
        cells[c] = cells[c].fillna(-1).astype(int)

    cur = pd.read_csv(ROOT/"data/derived/reactants_curated.csv")
    cur = cur[cur.compound_type == "Amine"].set_index("data_name")
    amine_info = {a: dict(
        cls=str(cur.amine_class.get(a, "")), smiles=str(cur.smiles.get(a, "")),
        mw=float(cur.mw_smiles.get(a, 0) or 0),
        nuc=int(cur.n_nucleophilic_N.get(a, 0) or 0),
        inferred=str(cur.smiles_source.get(a, "")) == "INFERRED") for a in amines}

    enz_info = {}
    for e in enzymes:
        sub = cells[cells.Enzyme == e]
        enz_info[e] = dict(split=holdout.assignment()[e], cluster=int(cl[e]),
                           n_active=int(sub.active.sum()), n_tested=int(len(sub)),
                           n_searched=int(sub.detected.sum()))

    payload = dict(
        meta=dict(n_enzymes=len(enzymes), n_amines=len(amines), n_cells=int(len(meta)),
                  n_active=int(y.sum()), threshold=int(lb.THRESHOLD),
                  n_searched=int(cells.detected.sum()),
                  dev=len({e for e in enz if holdout.assignment()[e] == "dev"}),
                  test=len({e for e in enz if holdout.assignment()[e] == "test"})),
        amines=amine_info, enzymes=enz_info, cores=cores,
        cells=cells.to_dict(orient="records"))
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(payload, separators=(",", ":")))
    print(f"wrote {OUT}  ({OUT.stat().st_size/1e6:.1f} MB)")
    print(f"  {len(cells):,} cells  |  {int(cells.active.sum()):,} active "
          f"({100*cells.active.mean():.1f}%)")
    print(f"  {int(cells.detected.sum()):,} have a detected feature for that product; "
          f"{int((~cells.detected).sum()):,} had no signal in any sample (also negatives)")


if __name__ == "__main__":
    main()
