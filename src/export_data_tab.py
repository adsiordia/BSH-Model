"""Write site/thedata.json -- what was in hand before any modelling.

Deliberately narrow: the assay, the sequences, the intensity readout and how
reproducible it is. Four things a reader needs before any model number means
anything. Everything else belongs on a different tab.
"""
from pathlib import Path
import sys, json, numpy as np, pandas as pd, warnings
warnings.filterwarnings("ignore")
from Bio import SeqIO
sys.path.insert(0, str(Path(__file__).resolve().parent))
import labels as lb, trimmed as tr
from splits import identity_matrix

ROOT = Path(__file__).resolve().parent.parent
SITE = ROOT / "site"


def main():
    act = lb.build(min_reps=1)
    need = set(act.Enzyme)
    seq = {r.id.split("_")[-1]: str(r.seq).replace("-", "")
           for r in SeqIO.parse(ROOT / "data/raw/Seqs_list_total.fasta", "fasta")}
    L = np.array([len(seq[e]) for e in sorted(need) if e in seq])
    bins = [(250, 300), (300, 325), (325, 350), (350, 375), (375, 400), (400, 470)]
    lengths = [dict(lo=lo, hi=hi, n=int(((L >= lo) & (L < hi)).sum())) for lo, hi in bins]

    M = identity_matrix(need).to_numpy()
    pid = M[np.triu_indices(len(M), 1)]
    pid_bins = [(0, .2), (.2, .3), (.3, .4), (.4, .6), (.6, .8), (.8, 1.001)]
    identity = [dict(lo=lo, hi=hi, n=int(((pid >= lo) & (pid < hi)).sum()))
                for lo, hi in pid_bins]
    cl = pd.read_csv(ROOT / "data/derived/clusters_70.csv")

    raw = tr.enzymes(tr.load_long())
    raw["Hydroxyl"] = raw.Hydroxyl.map(lb.TO_DEGREE)
    per_rep = raw.groupby(["Code", "Amine", "Hydroxyl", "Replicate"]).Intensity.max().reset_index()
    cell = per_rep.groupby(["Code", "Amine", "Hydroxyl"]).Intensity.max().to_numpy()
    nz = cell[cell > 0]
    decades = [dict(lo=10 ** e, n=int(((nz >= 10 ** e) & (nz < 10 ** (e + 1))).sum()))
               for e in range(1, 8) if ((nz >= 10 ** e) & (nz < 10 ** (e + 1))).sum()]

    w = per_rep.pivot_table(index=["Code", "Amine", "Hydroxyl"], columns="Replicate",
                            values="Intensity").fillna(0)
    w.columns = ["r1", "r2", "r3"]; w = w.reset_index()
    w["peak"] = w[["r1", "r2", "r3"]].max(1)
    repro = []
    for lo, hi in [(1e4, 2.5e4), (2.5e4, 5e4), (5e4, 1e5), (1e5, 5e5), (5e5, 1e13)]:
        s = w[(w.peak >= lo) & (w.peak < hi)]
        if not len(s):
            continue
        n = (s[["r1", "r2", "r3"]] >= 1e4).sum(1)
        repro.append(dict(lo=int(lo), hi=int(hi) if hi < 1e13 else None,
                          cells=int(len(s)), two_plus=round(float((n >= 2).mean()), 3)))

    doc = dict(
        generated=pd.Timestamp.today().strftime("%Y-%m-%d"),
        assay=dict(samples=int(raw.groupby(["Code", "Replicate"]).ngroups),
                   replicates=int(raw.Replicate.nunique()),
                   enzymes=int(act.Enzyme.nunique()),
                   amines=int(act.Amine.nunique()),
                   cores=int(act.Hydroxyl.nunique()),
                   combinations=int(len(act))),
        sequences=dict(n=int(len(L)), min=int(L.min()), max=int(L.max()),
                       median=int(np.median(L)), bins=lengths,
                       identity_median=round(float(np.median(pid)), 3),
                       identity_bins=identity,
                       clusters=int(cl.cluster.nunique()),
                       singletons=int((cl.groupby("cluster").size() == 1).sum())),
        intensity=dict(cells=int(len(cell)), zero=int((cell == 0).sum()),
                       min=int(nz.min()), max=int(nz.max()),
                       median=int(np.median(nz)), decades=decades,
                       cutoff=int(lb.THRESHOLD), control_max=10_320,
                       below_cutoff=int((nz < lb.THRESHOLD).sum())),
        features=lb.feature_summary(min_reps=1),
        reproducibility=repro,
        duplicate=dict(a="A0A3Q0NGD6", b="Q8Y5J3", agreement=0.827))
    p = SITE / "thedata.json"
    p.write_text(json.dumps(doc, separators=(",", ":"), allow_nan=False))
    print(f"wrote {p} ({p.stat().st_size/1e3:.0f} kB)")
    print(f"  {doc['assay']['samples']} samples · {doc['sequences']['n']} sequences "
          f"{doc['sequences']['min']}-{doc['sequences']['max']} aa · "
          f"{doc['sequences']['clusters']} clusters")


if __name__ == "__main__":
    main()
