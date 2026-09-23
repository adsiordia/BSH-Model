"""Generate predictions for enzyme x amine x core combinations.

Three cases:
  1. combinations that exist in the panel but were never measured (121 of 200 cells)
  2. a new enzyme  -> needs its sequence embedded and aligned
  3. a new amine   -> needs a SMILES
"""
from pathlib import Path
import numpy as np, pandas as pd, warnings
warnings.filterwarnings("ignore")
from sklearn.model_selection import train_test_split
import xgboost as xgb

import dataset as ds, labels as lb, representations as rep, holdout
from splits import identity_matrix, cdhit_merged_clusters

ROOT = Path(__file__).resolve().parent.parent


def train_model(min_reps=1, dev_only=True):
    """Fit XGBoost. dev_only=True keeps the locked test set out."""
    X, y, meta, names, blocks = ds.build(min_reps=min_reps)
    if dev_only:
        keep = meta.Enzyme.map(holdout.assignment()).eq("dev").to_numpy()
        X, y, meta = X[keep], y[keep], meta[keep].reset_index(drop=True)
    enz = meta.Enzyme.to_numpy()
    cl = cdhit_merged_clusters(identity=0.70, M=identity_matrix(set(enz)))
    g = np.array([cl[e] for e in enz])
    a, b = train_test_split(np.unique(g), test_size=0.2, random_state=0)
    itr, iva = np.where(np.isin(g, a))[0], np.where(np.isin(g, b))[0]
    spw = max((y[itr] == 0).sum() / max((y[itr] == 1).sum(), 1), 1.0)
    m = xgb.XGBClassifier(n_estimators=600, max_depth=4, learning_rate=0.05,
                          subsample=0.8, colsample_bytree=0.8, reg_alpha=1.0,
                          reg_lambda=5.0, scale_pos_weight=spw, eval_metric="logloss",
                          early_stopping_rounds=40, n_jobs=-1, verbosity=0, random_state=0)
    m.fit(X[itr], y[itr], eval_set=[(X[iva], y[iva])], verbose=False)
    return m, meta


def featurise(pairs, morgan_bits=ds.MORGAN_BITS, core_labels=None):
    """pairs: DataFrame with Enzyme, Amine, Hydroxyl. Returns X in the model's layout."""
    emb = ds.enzyme_embeddings()
    sm = rep.load_smiles("Amine")
    from sklearn.preprocessing import StandardScaler
    ids = sorted(emb)
    E = StandardScaler().fit_transform(np.stack([emb[e] for e in ids]))
    enz_vec = dict(zip(ids, E.astype(np.float32)))
    A, _ = rep.morgan({a: sm[a] for a in sorted(set(pairs.Amine)) if a in sm}, n_bits=morgan_bits)
    C, _ = rep.build_core("onehot", labels=core_labels)
    return np.hstack([
        np.stack([enz_vec[e] for e in pairs.Enzyme]),
        A.loc[pairs.Amine].to_numpy(np.float32),
        C.loc[pairs.Hydroxyl].to_numpy(np.float32)]).astype(np.float32)


if __name__ == "__main__":
    act = lb.build(min_reps=1)
    measured = set(zip(act.Enzyme, act.Amine, act.Hydroxyl))
    enzymes = sorted(act.Enzyme.unique())
    amines = sorted(act.Amine.unique())
    cores = sorted(act.Hydroxyl.unique())

    grid = pd.DataFrame([(e, a, c) for e in enzymes for a in amines for c in cores],
                        columns=["Enzyme", "Amine", "Hydroxyl"])
    grid["measured"] = [t in measured for t in zip(grid.Enzyme, grid.Amine, grid.Hydroxyl)]
    print(f"full grid           {len(grid):,} = {len(enzymes)} enzymes x {len(amines)} amines x {len(cores)} cores")
    print(f"  measured          {int(grid.measured.sum()):,}")
    print(f"  NEVER measured    {int((~grid.measured).sum()):,}  <- the model can score these\n")

    model, _ = train_model()
    X = featurise(grid, core_labels=cores)
    grid["p_active"] = model.predict_proba(X)[:, 1]

    new = grid[~grid.measured].sort_values("p_active", ascending=False)
    print("=== highest-scoring combinations that were NEVER measured ===\n")
    print(new.head(15)[["Enzyme", "Amine", "Hydroxyl", "p_active"]].to_string(index=False))

    print("\n=== which unmeasured amine x core cells look most promising overall ===\n")
    cell = (new.groupby(["Amine", "Hydroxyl"])
               .agg(n_enzymes=("p_active", "size"), mean_p=("p_active", "mean"),
                    n_above_half=("p_active", lambda s: int((s > .5).sum())))
               .sort_values("mean_p", ascending=False))
    print(cell.head(12).to_string())

    out = ROOT/"outputs/predictions_unmeasured.csv"
    out.parent.mkdir(exist_ok=True)
    new.to_csv(out, index=False)
    print(f"\n-> {out}  ({len(new):,} rows)")
