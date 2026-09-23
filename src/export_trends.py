"""Trend + sequence-preference data for the explorer."""
from pathlib import Path
import json, sys
import numpy as np, pandas as pd, warnings
warnings.filterwarnings("ignore")
from scipy.spatial.distance import pdist, squareform
from scipy.stats import spearmanr, rankdata
from sklearn.preprocessing import StandardScaler

import labels as lb, dataset as ds
from splits import identity_matrix, load_clusters

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "site/trends.json"


def partial(x, y, z):
    """Spearman of x vs y, controlling for z."""
    X, Y, Z = (rankdata(v) for v in (x, y, z))
    r = lambda a, b: float(np.corrcoef(a, b)[0, 1])
    rxy, rxz, ryz = r(X, Y), r(X, Z), r(Y, Z)
    return (rxy - rxz * ryz) / np.sqrt((1 - rxz ** 2) * (1 - ryz ** 2))


def main():
    act = lb.build(min_reps=1)
    cur = pd.read_csv(ROOT / "data/derived/reactants_curated.csv")
    cur = cur[cur.compound_type == "Amine"].set_index("data_name")

    a = act.groupby("Amine").agg(cells=("active", "size"), hits=("active", "sum")).reset_index()
    a["rate"] = a.hits / a.cells
    a = a.join(cur[["amine_class", "n_primary", "n_secondary", "n_tertiary", "n_aniline",
                    "n_nucleophilic_N", "has_COOH", "mw_smiles"]], on="Amine")
    # classify by the kind of nitrogen that forms the bond -- one exclusive answer per
    # amine. See src/export_amine_chem.py:ntype for why amine_class was dropped.
    a["amine_class"] = a.apply(
        lambda r: "aromatic" if r.n_aniline > 0
        else "tertiary" if (r.n_tertiary > 0 and r.n_primary == 0 and r.n_secondary == 0)
        else "secondary" if (r.n_secondary > 0 and r.n_primary == 0)
        else "primary", axis=1)
    a["has_COOH"] = a.has_COOH.astype(str).str.lower().isin(["true", "1"])
    per_core = act.groupby(["Amine", "Hydroxyl"]).active.mean().unstack().round(4)

    by_class = a.groupby("amine_class").agg(n=("Amine", "size"), cells=("cells", "sum"),
                                            hits=("hits", "sum"))
    by_class["rate"] = by_class.hits / by_class.cells
    by_nuc = a.groupby("n_nucleophilic_N").agg(n=("Amine", "size"), cells=("cells", "sum"),
                                               hits=("hits", "sum"))
    by_nuc["rate"] = by_nuc.hits / by_nuc.cells
    by_core = act.groupby("Hydroxyl").agg(cells=("active", "size"), hits=("active", "sum"))
    by_core["rate"] = by_core.hits / by_core.cells

    prof = (act.assign(v=act.active.astype(float))
              .pivot_table(index="Enzyme", columns=["Amine", "Hydroxyl"], values="v", aggfunc="max")
              .fillna(0.0).astype(float))
    emb = ds.enzyme_embeddings()
    ids = sorted(set(prof.index) & set(emb))
    prof = prof.loc[ids]
    M = identity_matrix(set(ids)).loc[ids, ids]
    E = StandardScaler().fit_transform(np.stack([emb[e] for e in ids]))
    n_act = prof.values.sum(1)
    cl = load_clusters(set(M.index))

    def pack(mask):
        idx = np.where(mask)[0]
        iu = np.triu_indices(len(idx), 1)
        seq = M.values[np.ix_(idx, idx)][iu]
        embs = (1 - squareform(pdist(E[idx], "cosine")))[iu]
        jac = (1 - squareform(pdist(prof.values[idx], "jaccard")))[iu]
        na = n_act[idx]
        lvl = (1 - squareform(pdist(na.reshape(-1, 1), "cityblock") / max(na.ptp(), 1)))[iu]
        bands = []
        for lo, hi in [(0, .25), (.25, .30), (.30, .40), (.40, .50), (.50, .60), (.60, .70), (.70, 1.01)]:
            m = (seq >= lo) & (seq < hi)
            if m.sum() > 10:
                bands.append(dict(lo=round(lo, 2), hi=round(hi, 2), n=int(m.sum()),
                                  mean=round(float(jac[m].mean()), 3),
                                  q1=round(float(np.percentile(jac[m], 25)), 3),
                                  q3=round(float(np.percentile(jac[m], 75)), 3)))
        return dict(n_enzymes=int(len(idx)), n_pairs=int(len(seq)),
                    seq_raw=round(float(spearmanr(seq, jac)[0]), 3),
                    seq_partial=round(float(partial(seq, jac, lvl)), 3),
                    emb_raw=round(float(spearmanr(embs, jac)[0]), 3),
                    emb_partial=round(float(partial(embs, jac, lvl)), 3),
                    level_raw=round(float(spearmanr(lvl, jac)[0]), 3),
                    bands=bands,
                    points=[[round(float(s), 3), round(float(j), 3)]
                            for s, j in zip(seq[::4], jac[::4])])

    live = n_act > 0
    payload = dict(
        amines=[dict(name=r.Amine, rate=round(float(r["rate"]), 4), cells=int(r.cells),
                     hits=int(r.hits), cls=str(r.amine_class), nuc=int(r.n_nucleophilic_N or 0),
                     cooh=bool(r.has_COOH), mw=round(float(r.mw_smiles or 0), 1),
                     per_core={c: (None if pd.isna(per_core.loc[r.Amine, c])
                                   else round(float(per_core.loc[r.Amine, c]), 4))
                               for c in per_core.columns})
                for _, r in a.sort_values("rate", ascending=False).iterrows()],
        by_class=[dict(k=str(k), n=int(v.n), rate=round(float(v["rate"]), 4), cells=int(v.cells))
                  for k, v in by_class.sort_values("rate", ascending=False).iterrows()],
        by_nuc=[dict(k=int(k), n=int(v.n), rate=round(float(v["rate"]), 4), cells=int(v.cells))
                for k, v in by_nuc.iterrows()],
        by_core=[dict(k=str(k), rate=round(float(v["rate"]), 4), cells=int(v.cells), hits=int(v.hits))
                 for k, v in by_core.iterrows()],
        enzymes=[dict(name=e, n_active=int(n_act[i]), cluster=int(cl[e])) for i, e in enumerate(ids)],
        seqpref=dict(all=pack(np.ones(len(ids), bool)), active=pack(live)),
        cluster_sizes={str(k): int(v) for k, v in pd.Series(cl).value_counts().items()})
    OUT.write_text(json.dumps(payload, separators=(",", ":")))
    print(f"wrote {OUT} ({OUT.stat().st_size/1e6:.2f} MB)")
    s = payload["seqpref"]["active"]
    print(f"  active enzymes: {s['n_enzymes']}, {s['n_pairs']:,} pairs")
    print(f"  sequence identity  raw {s['seq_raw']:+.3f}  partial {s['seq_partial']:+.3f}")
    print(f"  embedding          raw {s['emb_raw']:+.3f}  partial {s['emb_partial']:+.3f}")
    print(f"  scatter points {len(s['points']):,}  bands {len(s['bands'])}")


if __name__ == "__main__":
    main()
