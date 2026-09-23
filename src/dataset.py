"""Assemble the feature matrix: X = [enzyme 1024 | amine 512 | core one-hot]."""
from pathlib import Path
import h5py, numpy as np, pandas as pd
from Bio import SeqIO
from sklearn.preprocessing import StandardScaler

import labels as lb
import representations as rep

ROOT = Path(__file__).resolve().parent.parent
RAW, DERIVED = ROOT/"data/raw", ROOT/"data/derived"

CONSERVATION_THRESHOLD = 0.6      # inherited from v1, not re-verified on these labels
POOLING = "max"                   # inherited from v1, not re-verified
EMBEDDING_H5 = RAW / "Seqs_list_total_per_residue.h5"   # ProtT5-XL, 1024 dims
# alternatives, same layout:  RAW / "esm3_per_residue.h5"  (ESM-3, 1536 dims)
MORGAN_BITS = 512                 # chosen on the OLD labels -- should be re-checked

# First round groups the five keto cores into their degree class by hydroxyl count
# rather than dropping them -- see labels.TO_DEGREE. Every measurement is kept, and
# the amine x core grid goes from 28% filled to 65%.


def enzyme_embeddings(threshold=CONSERVATION_THRESHOLD, pooling=POOLING,
                      residues="noncons", embedding_h5=None):
    """Pool per-residue vectors into one vector per enzyme.

    embedding_h5 : any HDF5 with one (length, dim) dataset per accession.
                   Dimensionality is read from the file, so ProtT5 (1024) and
                   ESM-3 (1536) both work without further changes.
    residues     : "noncons" keeps alignment columns below `threshold`;
                   "full" keeps every residue.
    """
    with h5py.File(embedding_h5 or EMBEDDING_H5, "r") as f:
        per = {k.split("_")[-1]: f[k][:] for k in f.keys()}
    if residues == "full":
        fn_ = np.max if pooling == "max" else np.mean
        return {e: fn_(v, axis=0).astype(np.float32) for e, v in per.items()}
    cons = pd.read_csv(DERIVED/"conservation_scores.csv"); cons = cons[~cons.is_high_gap]
    var = cons.loc[cons.conservation_score < threshold, "alignment_position"].values
    aln = {}
    for rec in SeqIO.parse(DERIVED/"bsh_aligned.fasta", "fasta"):
        m, p = {}, 0
        for col, ch in enumerate(str(rec.seq)):
            if ch != "-":
                m[col] = p; p += 1
        aln[rec.id.split("_")[-1]] = m
    fn = np.max if pooling == "max" else np.mean
    out = {}
    for e, v in per.items():
        mp = aln.get(e)
        if mp is None:
            continue
        idx = [mp[c] for c in var if c in mp and mp[c] < len(v)]
        if idx:
            out[e] = fn(v[idx], axis=0).astype(np.float32)
    return out


def build(min_reps=lb.MIN_REPS, threshold=lb.THRESHOLD, morgan_bits=MORGAN_BITS,
          core_kind="onehot", drop_cores=(), embedding_h5=None,
          conservation=CONSERVATION_THRESHOLD, pooling=POOLING, residues="noncons"):
    """Returns X, y, meta, feature names, block index ranges."""
    act = lb.build(threshold=threshold, min_reps=min_reps, drop_cores=drop_cores)
    emb = enzyme_embeddings(threshold=conservation, pooling=pooling,
                            residues=residues, embedding_h5=embedding_h5)
    act = act[act.Enzyme.isin(emb)].reset_index(drop=True)

    ids = sorted(act.Enzyme.unique())
    # standardise each dimension: removes the shared "this is a BSH" component,
    # which is ~8x larger than the per-enzyme deviation. Does NOT reduce dimensionality.
    E = StandardScaler().fit_transform(np.stack([emb[e] for e in ids]))
    enz_vec = dict(zip(ids, E.astype(np.float32)))

    sm = rep.load_smiles("Amine")
    A, a_cols = rep.morgan({a: sm[a] for a in act.Amine.unique() if a in sm}, n_bits=morgan_bits)
    missing = sorted(set(act.Amine) - set(A.index))
    if missing:
        print(f"  WARNING: no SMILES for {missing} -- rows dropped")
        act = act[act.Amine.isin(A.index)].reset_index(drop=True)

    C, c_cols = rep.build_core(core_kind, labels=sorted(act.Hydroxyl.unique()))
    X = np.hstack([
        np.stack([enz_vec[e] for e in act.Enzyme]),
        A.loc[act.Amine].to_numpy(np.float32),
        C.loc[act.Hydroxyl].to_numpy(np.float32),
    ]).astype(np.float32)
    y = act.active.to_numpy(int)
    k = E.shape[1]
    names = [f"enz_{i}" for i in range(k)] + list(a_cols) + list(c_cols)
    blocks = {"enzyme": (0, k), "amine": (k, k+len(a_cols)), "core": (k+len(a_cols), X.shape[1])}
    meta = act[["Enzyme", "Amine", "Hydroxyl", "n_reps_above", "max_intensity"]].copy()
    return X, y, meta, names, blocks


if __name__ == "__main__":
    for mr in (1, 2):
        X, y, meta, names, blocks = build(min_reps=mr)
        print(f"  min_reps={mr}: X={X.shape}  active={y.sum():,} ({100*y.mean():.1f}%)  "
              f"blocks={blocks}")
