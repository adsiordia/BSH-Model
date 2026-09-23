"""Does pooling over selected residues beat pooling over the whole protein?

The production model averages all ~332 residues into one vector. A handful of
active-site residues therefore contribute under 3% of the result, which is the
suspected reason the enzyme block adds so little (+0.04 PR-AUC over amine+core).

This tries the alternatives on the same folds:
  whole mean / whole max / whole mean+max   -- no alignment needed
  core columns                              -- the 280 present in >=90% of enzymes
  non-conserved at a range of thresholds    -- drop columns dominated by one residue
  active site only                          -- the invariant catalytic columns

Scored on WITHIN-SUBSTRATE AUC: hold the amine and bile acid fixed, ask whether the
right enzymes rank on top. That is the metric a new-sequence prediction depends on;
overall PR-AUC is dominated by which amine is easy and hides the enzyme block.

PROTOCOL
    Everything here runs on the 92 DEVELOPMENT enzymes only. The 23 locked-away
    enzymes take no part: choosing a method is a design decision, and design
    decisions are made on development data. If a scheme wins here it can then be
    evaluated once on the 23, and only after that fitted to all 115.

    Inside development, 5-fold cross-validation grouped on 70%-identity clusters,
    with a further cluster-grouped split of each training fold for early stopping.
    No enzyme is ever scored by a model that saw it or a relative above 70%.
"""
from pathlib import Path
import sys, h5py, numpy as np, pandas as pd, warnings
warnings.filterwarnings("ignore")
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (average_precision_score, roc_auc_score, log_loss,
                             brier_score_loss, f1_score, precision_score, recall_score,
                             matthews_corrcoef, balanced_accuracy_score, confusion_matrix)
import xgboost as xgb

sys.path.insert(0, str(Path(__file__).resolve().parent))
import labels as lb, representations as rep, embeddings as EM, holdout
from splits import identity_matrix, cdhit_merged_clusters, balanced_group_split, load_clusters

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "outputs"
H = Path.home()
PER_RES = {"ProstT5": (H / "prost5_embeddings/Seqs_list_total_prost5_per_residue.h5", "bf16"),
           "ProtT5":  (ROOT / "data/raw/Seqs_list_total_per_residue.h5", "f32"),
           "ESM-3":   (ROOT / "data/raw/esm3_per_residue.h5", "f32"),
           "ESM-2":   (H / "esm2_emebddings/bsh_esm2_per_res", "pt")}
THRESHOLDS = [0.30, 0.40, 0.50, 0.60, 0.70, 0.75, 0.85, 1.00]


def load_per_residue(name):
    path, kind = PER_RES[name]
    if kind == "pt":                      # esm-extract writes one .pt per sequence
        import torch
        out = {}
        for f in sorted(Path(path).glob("*.pt")):
            o = torch.load(f, map_location="cpu", weights_only=False)
            out[o["label"].split("_")[-1]] = o["representations"][33].numpy().astype(np.float32)
        return out
    with h5py.File(path, "r") as f:
        return {k.split("_")[-1]: (EM._bf16(f[k][:]) if kind == "bf16"
                                   else f[k][:].astype(np.float32)) for k in f.keys()}


def selections():
    """Each scheme -> the set of alignment columns to pool over (None = all residues)."""
    c = pd.read_csv(ROOT / "data/derived/conservation.csv")
    core = c[c.occupancy >= 0.9]
    sel = {"whole mean": None, "whole max": None, "whole mean+max": None}
    sel["core columns"] = set(core.col)
    for t in THRESHOLDS:
        sel[f"non-conserved <{t:.2f}"] = set(core[core.conservation < t].col)
    sel["active site only"] = set(core[core.conservation == 1.0].col)
    return sel


def compact():
    """A smaller set for comparing four embeddings -- both directions of the cut."""
    c = pd.read_csv(ROOT / "data/derived/conservation.csv")
    core = c[c.occupancy >= 0.9]
    return {"whole mean": None,
            "whole mean+max": None,
            "non-conserved <0.30": set(core[core.conservation < 0.30].col),
            "non-conserved <0.60": set(core[core.conservation < 0.60].col),
            "conserved >=0.60": set(core[core.conservation >= 0.60].col),
            "conserved >=0.75": set(core[core.conservation >= 0.75].col),
            "active site only": set(core[core.conservation == 1.0].col)}


def build(emb, ridx, cols, how, keep=None):
    """One vector per enzyme by pooling the chosen residues.

    The per-residue files hold 127 proteins; the alignment covers only the 115 that
    appear in Trimmed_remove_proteomics.csv, so the rest are dropped here.
    """
    out = {}
    for e, M in emb.items():
        if keep is not None and e not in keep:
            continue
        if cols is None:
            rows = M
        else:
            i = ridx.loc[e, sorted(cols)].to_numpy()
            i = i[(i >= 0) & (i < len(M))]
            rows = M[i] if len(i) else M
        if how == "max":
            out[e] = rows.max(0)
        elif how == "meanmax":
            out[e] = np.concatenate([rows.mean(0), rows.max(0)])
        else:
            out[e] = rows.mean(0)
    return out


def metrics(y, p, act):
    """Everything the summary table reports, from the out-of-fold predictions."""
    pred = (p >= 0.5).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    w = [roc_auc_score(s.y, s.p) for _, s in act.assign(y=y, p=p).groupby(["Amine", "Hydroxyl"])
         if s.y.sum() >= 5 and (1 - s.y).sum() >= 5]
    return dict(pr_auc=average_precision_score(y, p), roc_auc=roc_auc_score(y, p),
                log_loss=log_loss(y, np.clip(p, 1e-6, 1 - 1e-6)),
                brier=brier_score_loss(y, p), mcc=matthews_corrcoef(y, pred),
                bal_acc=balanced_accuracy_score(y, pred), f1=f1_score(y, pred),
                precision=precision_score(y, pred, zero_division=0),
                recall=recall_score(y, pred), specificity=tn / max(tn + fp, 1),
                within_mean=float(np.mean(w)), within_median=float(np.median(w)),
                n_groups=len(w), tp=int(tp), fp=int(fp), fn=int(fn), tn=int(tn))


def score(X, y, g, act):
    oof = np.zeros(len(y))
    rounds, curves = [], []
    for f, (tr, te) in enumerate(GroupKFold(n_splits=5).split(X, y, g)):
        a, b = balanced_group_split(g[tr], test_size=0.2, seed=f)
        itr, iva = tr[a], tr[b]
        spw = max((y[itr] == 0).sum() / max((y[itr] == 1).sum(), 1), 1.0)
        m = xgb.XGBClassifier(n_estimators=3000, max_depth=4, learning_rate=0.05,
                              subsample=0.8, colsample_bytree=0.8, reg_alpha=1.0,
                              reg_lambda=5.0, scale_pos_weight=spw, eval_metric="logloss",
                              early_stopping_rounds=50, n_jobs=-1, verbosity=0, random_state=0)
        m.fit(X[itr], y[itr], verbose=False,
              eval_set=[(X[itr], y[itr]), (X[iva], y[iva]), (X[te], y[te])])
        oof[te] = m.predict_proba(X[te])[:, 1]
        rounds.append(m.best_iteration + 1)
        r = m.evals_result()
        curves.append({k: np.array(r[f"validation_{i}"]["logloss"], dtype=np.float32)
                       for i, k in enumerate(("train", "check", "held"))})
    return oof, rounds, curves


def main(which=("ProstT5",), schemes=None):
    act = lb.build(min_reps=1)
    assign = holdout.assignment()
    act = act[act.Enzyme.map(assign).eq("dev")].reset_index(drop=True)   # 92, never the 23
    ridx = pd.read_csv(ROOT / "data/derived/alignment_residue_index.csv", index_col=0)
    ridx.columns = ridx.columns.astype(int)
    sm = rep.load_smiles("Amine")
    A, _ = rep.morgan({a: sm[a] for a in sorted(act.Amine.unique()) if a in sm}, n_bits=512)
    C, _ = rep.build_core("onehot", labels=sorted(act.Hydroxyl.unique()))
    sel = compact() if schemes == "compact" else selections()
    rows, oofs = [], {}

    for name in which:
        emb = load_per_residue(name)
        sub = act[act.Enzyme.isin(emb) & act.Enzyme.isin(ridx.index)].reset_index(drop=True)
        y = sub.active.to_numpy(int)
        cl = load_clusters(set(sub.Enzyme))
        g = np.array([cl[e] for e in sub.Enzyme])
        Am = A.loc[sub.Amine].to_numpy(np.float32)
        Co = C.loc[sub.Hydroxyl].to_numpy(np.float32)
        print(f"\n{name}: {sub.Enzyme.nunique()} enzymes, {len(sub):,} cells, "
              f"{len(set(g))} clusters")
        print(f"  {'scheme':24s} {'residues':>9} {'dim':>6} {'PR-AUC':>8} {'ROC':>7} "
              f"{'log loss':>9} {'within-sub':>11}")
        for scheme, cols in sel.items():
            how = "max" if scheme == "whole max" else "meanmax" if scheme == "whole mean+max" else "mean"
            vec = build(emb, ridx, cols, how, keep=set(sub.Enzyme))
            ids = sorted(sub.Enzyme.unique())
            Z = StandardScaler().fit_transform(np.stack([vec[i] for i in ids])).astype(np.float32)
            v = dict(zip(ids, Z))
            X = np.hstack([np.stack([v[e] for e in sub.Enzyme]), Am, Co]).astype(np.float32)
            nres = int(np.median([len(ridx.loc[e, sorted(cols)][ridx.loc[e, sorted(cols)] >= 0])
                                 for e in ids])) if cols else int(np.median(
                                 [len(emb[e]) for e in ids]))
            oof, rounds, curves = score(X, y, g, sub)
            mt = metrics(y, oof, sub)
            rows.append(dict(embedding=name, scheme=scheme, residues=nres, dim=Z.shape[1],
                             rounds=int(np.median(rounds)), rounds_per_fold=str(rounds),
                             **{k: (round(v, 4) if isinstance(v, float) else v)
                                for k, v in mt.items()}))
            oofs[f"{name}|{scheme}"] = oof
            for f, c in enumerate(curves):
                for k, v in c.items():
                    oofs[f"curve|{name}|{scheme}|{f}|{k}"] = v
            print(f"  {scheme:24s} {nres:>9d} {Z.shape[1]:>6d} {mt['pr_auc']:>8.3f} "
                  f"{mt['roc_auc']:>7.3f} {mt['log_loss']:>9.3f} {mt['within_mean']:>11.3f}")

    df = pd.DataFrame(rows)
    OUT.mkdir(exist_ok=True)
    tag = "_compact" if schemes == "compact" else ""
    df.to_csv(OUT / f"pooling_sweep{tag}.csv", index=False)
    np.savez(OUT / f"pooling_sweep{tag}_oof.npz", y=y, **oofs)
    sub[["Enzyme", "Amine", "Hydroxyl"]].to_csv(OUT / f"pooling_sweep{tag}_meta.csv", index=False)
    print(f"\n-> outputs/pooling_sweep.csv")
    b = df.loc[df.within_mean.idxmax()]
    print(f"\nbest within-substrate: {b.embedding} / {b.scheme} = {b.within_mean:.4f} "
          f"({b.residues} residues)")


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if a != "--compact"]
    main(tuple(args) or ("ProstT5",), "compact" if "--compact" in sys.argv else None)
