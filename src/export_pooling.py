"""Write site/pooling.json -- the conserved-residue experiment.

Covers how the alignment was built, which columns are conserved, what each threshold
selects, and whether any residue-selection scheme beats pooling the whole protein.
It does not, and that negative result is the point of the section.
"""
from pathlib import Path
import json, numpy as np, pandas as pd, warnings
warnings.filterwarnings("ignore")
from Bio import SeqIO

ROOT = Path(__file__).resolve().parent.parent
OUT, SITE = ROOT / "outputs", ROOT / "site"
EMBS = ["ProstT5", "ProtT5", "ESM-3", "ESM-2"]
ORDER = ["whole mean", "whole mean+max", "conserved >=0.75", "conserved >=0.60",
         "active site only", "non-conserved <0.30", "non-conserved <0.60"]
N_CURVE = 110


def thin(a, n=N_CURVE):
    a = np.asarray(a)
    i = np.unique(np.linspace(0, len(a) - 1, min(n, len(a))).astype(int))
    return [round(float(x), 4) for x in a[i]]


def main():
    cons = pd.read_csv(ROOT / "data/derived/conservation.csv")
    ridx = pd.read_csv(ROOT / "data/derived/alignment_residue_index.csv", index_col=0)
    ridx.columns = ridx.columns.astype(int)
    core = cons[cons.occupancy >= 0.9]
    aln = {r.id: str(r.seq) for r in SeqIO.parse(ROOT / "data/derived/msa_linsi.fasta", "fasta")}
    log = (ROOT / "data/derived/msa_linsi.log").read_text()

    invariant = []
    for _, x in core[core.conservation == 1.0].sort_values("col").iterrows():
        i = ridx[x.col]; i = i[i >= 0] + 1
        invariant.append(dict(col=int(x.col), aa=str(x.top),
                              median_residue=int(i.median()),
                              min_residue=int(i.min()), max_residue=int(i.max())))

    ladder, whole = [], int(np.median([int((ridx.loc[e] >= 0).sum()) for e in ridx.index]))
    for t in [0.30, 0.40, 0.50, 0.60, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95, 1.00]:
        keep = core[core.conservation < t]
        n = [int((ridx[keep.col.values].loc[e] >= 0).sum()) for e in ridx.index] if len(keep) else [0]
        ladder.append(dict(threshold=t, conserved=int((core.conservation >= t).sum()),
                           variable=int(len(keep)), residues=int(np.median(n)),
                           frac=round(float(np.median(n)) / whole, 4)))

    hist = [dict(lo=lo, hi=min(hi, 1.0),
                 n=int(((core.conservation >= lo) & (core.conservation < hi)).sum()))
            for lo, hi in [(0, .2), (.2, .3), (.3, .4), (.4, .5), (.5, .6),
                           (.6, .7), (.7, .8), (.8, .9), (.9, 1.0001)]]

    res = pd.read_csv(OUT / "pooling_sweep_compact.csv")
    piv = res.pivot(index="scheme", columns="embedding", values="within_mean")[EMBS]
    base = piv.loc["whole mean"]
    delta = piv - base

    results = []
    for s in ORDER:
        row = dict(scheme=s, residues=int(res[res.scheme == s].residues.iloc[0]),
                   within={e: round(float(piv.loc[s, e]), 4) for e in EMBS})
        if s != "whole mean":
            row.update(change={e: round(float(delta.loc[s, e]), 4) for e in EMBS},
                       mean_change=round(float(delta.loc[s].mean()), 4),
                       positive_on=int((delta.loc[s] > 0).sum()))
        results.append(row)

    z = np.load(OUT / "pooling_sweep_compact_oof.npz")
    curves = {f"{e}|{s}": [thin(z[f"curve|{e}|{s}|{f}|held"]) for f in range(5)
                           if f"curve|{e}|{s}|{f}|held" in z.files]
              for e in EMBS for s in ORDER}
    ends = [c[-1] for v in curves.values() for c in v]

    doc = dict(
        generated=pd.Timestamp.today().strftime("%Y-%m-%d"),
        alignment=dict(tool="MAFFT L-INS-i", sequences=len(aln),
                       columns=len(next(iter(aln.values()))),
                       core_columns=int(len(core)),
                       command=next(l for l in log.splitlines()
                                    if l.startswith("command"))[10:].strip()),
        invariant=invariant, ladder=ladder, whole_residues=whole, histogram=hist,
        results=results,
        metrics=json.loads(res.to_json(orient="records")),
        curves=curves,
        curve_range=[round(min(ends), 3), round(max(ends), 3)],
        n_runs=int(len(res)),
        protocol="92 development enzymes only, 5-fold cross-validation grouped on "
                 "70%-identity clusters. The 23 locked-away enzymes take no part: "
                 "choosing a method is a design decision and belongs on development data.",
        verdict="No residue-selection scheme reliably beats pooling the whole protein.")
    p = SITE / "pooling.json"
    p.write_text(json.dumps(doc, separators=(",", ":"), allow_nan=False))
    print(f"wrote {p} ({p.stat().st_size/1e3:.0f} kB)")
    print(f"  {len(invariant)} invariant columns · {len(ladder)} thresholds · "
          f"{len(results)} schemes · {len(curves)} curve sets")
    print(f"  all {len(res)} runs end between {doc['curve_range'][0]} and "
          f"{doc['curve_range'][1]} log loss")


if __name__ == "__main__":
    main()
