"""Which alignment columns are conserved across the 115 assayed enzymes?

Reads the alignment built by src/align.py (MAFFT L-INS-i, command recorded in
data/derived/msa_linsi.log). For every column it reports how many enzymes
actually have a residue there (occupancy) and how dominated that column is by its
most common amino acid (conservation). Those two are different questions: a column
present in only 3 sequences can look perfectly conserved.

Writes data/derived/conservation.csv -- one row per alignment column, with the
residue index it corresponds to in every enzyme, so a pooling scheme can select
columns and get each protein's own positions back.
"""
from pathlib import Path
from collections import Counter
import numpy as np, pandas as pd
from Bio import SeqIO

ROOT = Path(__file__).resolve().parent.parent
ALN = ROOT / "data/derived/msa_linsi.fasta"     # built by src/align.py, provenance logged
OUT = ROOT / "data/derived"


def alignment(enzymes=None):
    a = {r.id.split("_")[-1]: str(r.seq).upper() for r in SeqIO.parse(ALN, "fasta")}
    return {k: v for k, v in a.items() if enzymes is None or k in enzymes}


def columns(aln):
    """Per-column occupancy and conservation over the enzymes given."""
    names = sorted(aln)
    A = np.array([list(aln[n]) for n in names])
    n_seq, n_col = A.shape
    rows = []
    for c in range(n_col):
        col = A[:, c]
        res = col[col != "-"]
        occ = len(res) / n_seq
        if len(res) == 0:
            rows.append(dict(col=c, occupancy=0.0, conservation=0.0, top="-", n=0))
            continue
        cnt = Counter(res)
        top, k = cnt.most_common(1)[0]
        rows.append(dict(col=c, occupancy=round(occ, 4),
                         # share of the enzymes THAT HAVE a residue here carrying the
                         # commonest one -- gaps are a separate question (occupancy)
                         conservation=round(k / len(res), 4),
                         top=top, n=len(res), n_distinct=len(cnt)))
    return pd.DataFrame(rows), names, A


def residue_index(A, names):
    """alignment column -> each enzyme's own 0-based residue index (-1 where gapped)."""
    idx = np.full(A.shape, -1, dtype=int)
    for i in range(A.shape[0]):
        pos = 0
        for c in range(A.shape[1]):
            if A[i, c] != "-":
                idx[i, c] = pos
                pos += 1
    return pd.DataFrame(idx, index=names)


def main():
    import labels as lb
    need = set(lb.build(min_reps=1).Enzyme)
    aln = alignment(need)
    df, names, A = columns(aln)
    print(f"{len(names)} enzymes, {A.shape[1]} alignment columns "
          f"(sequences are {min(len(v.replace('-','')) for v in aln.values())}-"
          f"{max(len(v.replace('-','')) for v in aln.values())} residues)\n")

    core = df[df.occupancy >= 0.9]
    print(f"columns present in at least 90% of enzymes: {len(core)} of {len(df)}")
    print(f"   the rest are insertions carried by a few sequences\n")
    print("how conserved are those core columns?")
    for lo, hi, lab in [(1.0, 1.01, "identical in every enzyme"),
                        (0.95, 1.0, "95-100%"), (0.9, 0.95, "90-95%"),
                        (0.7, 0.9, "70-90%"), (0.5, 0.7, "50-70%"), (0.0, 0.5, "below 50%")]:
        s = core[(core.conservation >= lo) & (core.conservation < hi)]
        print(f"   {lab:26s} {len(s):4d} columns  ({len(s)/len(core):5.1%})")

    ident = core[core.conservation == 1.0]
    print(f"\nthe {len(ident)} fully identical positions: "
          + " ".join(f"{r.top}{r.col}" for _, r in ident.head(30).iterrows())
          + (" ..." if len(ident) > 30 else ""))

    ridx = residue_index(A, names)
    df.to_csv(OUT / "conservation.csv", index=False)
    ridx.to_csv(OUT / "alignment_residue_index.csv")
    print(f"\n-> data/derived/conservation.csv          ({len(df)} columns)")
    print(f"-> data/derived/alignment_residue_index.csv ({ridx.shape[0]} x {ridx.shape[1]})")


if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    main()
