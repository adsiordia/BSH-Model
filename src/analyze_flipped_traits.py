"""Do the 8 enzymes that flip inactive->active on trimming share anything?

Compared against the other proteins in the same signal-peptide set.
"""
from pathlib import Path
import sys, json, numpy as np, pandas as pd
from scipy.stats import mannwhitneyu, fisher_exact
ROOT = Path(__file__).resolve().parent.parent; sys.path.insert(0, str(ROOT/"src"))
import labels as lb, splits
pd.set_option("display.width", 210)

CAND = json.load(open(ROOT/"site/candidates.json"))
RESC = json.load(open(ROOT/"site/rescue.json"))
P = pd.DataFrame(CAND["proteins"])
flip = [r["accession"] for r in RESC["gained_detail"]]
P["group"] = np.where(P.accession.isin(flip), "FLIPPED", "other")
lab = lb.build(min_reps=1); truth = lab.groupby("Enzyme").active.sum()
P["measured_active"] = P.accession.map(truth)
print(f"{len(P)} proteins in the signal-peptide set; {len(flip)} flipped\n")
print("the 8:", ", ".join(flip), "\n")

F = P[P.group=="FLIPPED"]
O = P[P.group=="other"]

print("=== 1. who are they? ===")
print(F[["accession","organism","phylum","klass","length","cut","measured_active"]]
      .to_string(index=False))

print("\n=== 2. taxonomy: is any lineage over-represented? ===")
for col in ("phylum","klass"):
    t = pd.DataFrame({"flipped": F[col].value_counts(), "other": O[col].value_counts()}).fillna(0).astype(int)
    t = t[t.flipped > 0]
    t["flipped_share"] = t.flipped / len(F)
    t["other_share"] = t.other / len(O)
    print(f"\n  by {col}:")
    print(t.to_string(float_format=lambda v: f"{v:.2f}"))
    for lv in t.index:
        a = int(t.loc[lv,"flipped"]); b = len(F)-a
        c = int(t.loc[lv,"other"]);  d = len(O)-c
        _, p = fisher_exact([[a,b],[c,d]])
        print(f"    {lv:24s} {a}/{len(F)} vs {c}/{len(O)}   Fisher p = {p:.3f}")

print("\n=== 3. length and peptide ===")
for col in ("length","cut","mature"):
    if col not in P: continue
    a, b = F[col].dropna(), O[col].dropna()
    if len(a) and len(b):
        _, p = mannwhitneyu(a, b)
        print(f"  {col:8s} flipped median {a.median():7.1f} (range {a.min():.0f}-{a.max():.0f})   "
              f"other {b.median():7.1f}   p = {p:.3f}")

print("\n=== 4. are they related to each other? ===")
cl = splits.load_clusters()
F2 = F.assign(cluster=F.accession.map(cl))
print("  cluster membership (>=70% identity components):")
vc = F2.cluster.value_counts()
print(f"    {F2.cluster.nunique()} distinct clusters for {len(F2)} proteins"
      f"  -> {'SAME cluster' if F2.cluster.nunique()==1 else 'NOT a single clade'}")
if (vc > 1).any():
    for c, n in vc[vc>1].items():
        print(f"    cluster {c}: {', '.join(F2[F2.cluster==c].accession)}")

# pairwise identity from the alignment
seqs, name, buf = {}, None, []
for line in (ROOT/"data/derived/msa_linsi.fasta").read_text().splitlines():
    if line.startswith(">"):
        if name: seqs[name] = "".join(buf)
        name, buf = line[1:].split()[0], []
    else: buf.append(line.strip())
seqs[name] = "".join(buf)
acc2key = {k.split("_")[-1]: k for k in seqs}
def ident(a, b):
    x, y = seqs[acc2key[a]], seqs[acc2key[b]]
    pos = [(i,j) for i,j in zip(x,y) if i!="-" and j!="-"]
    return sum(i==j for i,j in pos)/len(pos) if pos else np.nan

have = [a for a in flip if a in acc2key]
within = [ident(a,b) for i,a in enumerate(have) for b in have[i+1:]]
others = [x for x in P.accession if x in acc2key and x not in flip]
rng = np.random.default_rng(0)
between = [ident(a,b) for a in have for b in rng.choice(others, 6, replace=False)]
print(f"\n  pairwise identity AMONG the flipped : median {np.median(within):.3f} "
      f"(range {min(within):.3f}-{max(within):.3f}, {len(within)} pairs)")
print(f"  flipped vs everyone else            : median {np.median(between):.3f}")
_, p = mannwhitneyu(within, between)
print(f"  Mann-Whitney p = {p:.4f}  -> "
      f"{'more similar to each other than to others' if np.median(within)>np.median(between) and p<0.05 else 'no more similar to each other than to anyone else'}")

print("\n=== 5. what did the model score them BEFORE trimming? ===")
d = pd.DataFrame(RESC["proteins"])
print(f"  flipped   mean prediction on full sequence: "
      f"{d[d.accession.isin(flip)].mean_full.median():.4f}")
print(f"  all other measured proteins              : "
      f"{d[~d.accession.isin(flip)].mean_full.median():.4f}")
print(f"  overall base rate                        : {lab.active.mean():.4f}")
