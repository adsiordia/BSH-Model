"""What the predictions actually deliver, for sharing with collaborators."""
from pathlib import Path
import sys, json, numpy as np, pandas as pd
ROOT = Path(__file__).resolve().parent.parent; sys.path.insert(0, str(ROOT/"src"))
import labels as lb
OUT = ROOT/"outputs"; CUT = 0.5; pd.set_option("display.width", 215)

lab = lb.build(min_reps=1); truth = lab.groupby("Enzyme").active.sum()
P = pd.DataFrame(json.load(open(ROOT/"site/candidates.json"))["proteins"])
FILL = {"Parabacteroides":"Bacteroidota","Paraprevotella":"Bacteroidota","Odoribacter":"Bacteroidota",
 "Barnesiella":"Bacteroidota","Prevotella":"Bacteroidota","Dysgonomonas":"Bacteroidota",
 "Butyricimonas":"Bacteroidota","Muribaculaceae":"Bacteroidota","Alistipes":"Bacteroidota",
 "Homeothermus":"Bacteroidota","Okeania":"Cyanobacteriota","Symploca":"Cyanobacteriota",
 "Moorea":"Cyanobacteriota","Obscuribacterales":"Cyanobacteriota"}
def fill(r):
    if r.phylum: return r.phylum
    for t in (r.organism or "").replace("Candidatus ","").split()[:2]:
        if t.strip("[]") in FILL: return FILL[t.strip("[]")]
    return "other"
P["phy"] = P.apply(fill, axis=1); P["n_active"] = P.accession.map(truth)
g = pd.read_csv(OUT/"candidate_predictions.csv")

print("="*78)
print("A.  GENUINELY NEW PREDICTIONS -- 8 proteins never run in the assay")
print("="*78)
nov = g[g.measured.isna()].copy()
meta = dict(zip(P.accession, P.organism.fillna("").str.split().str[:2].str.join(" ")))
phy  = dict(zip(P.accession, P.phy))
for acc, s in nov.groupby("accession"):
    hits = s[(s.raw_full>=CUT) | (s.raw_trimmed>=CUT)].sort_values("raw_full", ascending=False)
    both = int(((s.raw_full>=CUT)&(s.raw_trimmed>=CUT)).sum())
    print(f"\n  {acc}   {meta.get(acc,''):32s} {phy.get(acc,'')}")
    print(f"    {len(hits)} products called; {both} agreed by BOTH sequence versions")
    for r in hits.head(6).itertuples():
        mark = "  <- both agree" if (r.raw_full>=CUT and r.raw_trimmed>=CUT) else ""
        print(f"      {r.Amine:26s} {r.Hydroxyl:5s} full {r.raw_full:.3f} / trimmed {r.raw_trimmed:.3f}{mark}")

print("\n" + "="*78)
print("B.  ASSAY-LEVEL FINDINGS (independent of the model)")
print("="*78)
m = P[P.n_active.notna()].copy(); m["dead"] = m.n_active==0
for ph in ("Bacillota","Bacteroidota"):
    s = m[m.phy==ph]
    print(f"  {ph:14s} {len(s):2d} measured, {int(s.dead.sum())} inactive, "
          f"median {s.n_active.median():.0f} products")
gut = m[m.organism.fillna("").str.startswith(("Bacteroides","Parabacteroides","Alistipes"))]
print(f"\n  gut Bacteroidetes are bimodal: {int(gut.dead.sum())} of {len(gut)} make NOTHING, "
      f"the other {int((~gut.dead).sum())} make a median of {gut[~gut.dead].n_active.median():.0f} "
      f"products (panel median {m.n_active.median():.0f})")

print("\n  every Bacillota enzyme makes this shared core:")
mb = lab[lab.Enzyme.isin(m[m.phy=='Bacillota'].accession)]
core = mb[mb.active].groupby(["Amine","Hydroxyl"]).size()
n = mb.Enzyme.nunique()
for (a,h), c in core[core==n].items():
    print(f"      {a:16s} {h:5s}  {c}/{n}")

print("\n" + "="*78)
print("C.  WHAT THE MODEL IS RELIABLE FOR")
print("="*78)
mm = json.load(open(ROOT/"site/methods.json"))
print("  product-type prediction (which amine+core gets made) : PR-AUC 0.7642, ROC 0.9341")
print("  enzyme discrimination (which enzyme, fixed substrate): within-substrate AUC 0.6362")
print("  assay reproducibility ceiling                        : 82%")
print(f"  feature-collapse rule                                : "
      f"{mm['labels']['features']['n_multi']} of {mm['labels']['features']['n_products']} products "
      f"have 2-4 LC-MS features, max-pooled per replicate")
