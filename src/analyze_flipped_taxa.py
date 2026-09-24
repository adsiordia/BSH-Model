"""Taxonomy of the flipped enzymes, with missing labels filled from the organism name."""
from pathlib import Path
import sys, json, numpy as np, pandas as pd
from scipy.stats import fisher_exact
ROOT = Path(__file__).resolve().parent.parent; sys.path.insert(0, str(ROOT/"src"))
import labels as lb
pd.set_option("display.width", 200)

CAND = json.load(open(ROOT/"site/candidates.json"))
RESC = json.load(open(ROOT/"site/rescue.json"))
P = pd.DataFrame(CAND["proteins"])
flip = [r["accession"] for r in RESC["gained_detail"]]

# fill phylum from the genus where the record left it blank
GENUS_PHYLUM = {
    "Bacteroides":"Bacteroidota", "Parabacteroides":"Bacteroidota", "Alistipes":"Bacteroidota",
    "Prevotella":"Bacteroidota", "Porphyromonas":"Bacteroidota", "Bacteroidales":"Bacteroidota",
    "Myroides":"Bacteroidota", "Maribacter":"Bacteroidota", "Aureitalea":"Bacteroidota",
    "Mariniradius":"Bacteroidota", "Flavobacterium":"Bacteroidota",
    "Bifidobacterium":"Actinomycetota",
    "Moorea":"Cyanobacteriota", "Okeania":"Cyanobacteriota", "Symploca":"Cyanobacteriota",
    "Melainabacteria":"Cyanobacteriota",
    "Lactobacillus":"Bacillota", "Clostridium":"Bacillota", "Eubacterium":"Bacillota",
    "Lysinibacillus":"Bacillota", "Enterococcus":"Bacillota", "Blautia":"Bacillota",
    "Vibrio":"Pseudomonadota", "Francisella":"Pseudomonadota", "Marinobacter":"Pseudomonadota",
    "Halodesulfovibrio":"Thermodesulfobacteriota", "Desulfomarina":"Thermodesulfobacteriota",
    "Desulfovibrio":"Thermodesulfobacteriota", "Blastospirellula":"Planctomycetota",
}
def fill(r):
    if r.phylum: return r.phylum
    org = (r.organism or "")
    for g, ph in GENUS_PHYLUM.items():
        if org.startswith(g): return ph
    return ""
P["phy"] = P.apply(fill, axis=1)
P["grp"] = np.where(P.accession.isin(flip), "FLIPPED", "other")
F, O = P[P.grp=="FLIPPED"], P[P.grp=="other"]

print("=== the 8, with phylum filled in ===")
print(F[["accession","organism","phylum","phy"]].rename(
      columns={"phylum":"as recorded","phy":"filled"}).to_string(index=False))
print(f"\n  still unlabelled: {int((F.phy=='').sum())} of {len(F)}")

print("\n=== phylum, flipped vs the rest of the signal-peptide set ===")
t = pd.DataFrame({"flipped": F.phy.value_counts(), "other": O.phy.value_counts()}).fillna(0).astype(int)
t.index = [i if i else "(unlabelled)" for i in t.index]
print(t.to_string())
tests = []
for lv in t.index:
    a = int(t.loc[lv,"flipped"]); c = int(t.loc[lv,"other"])
    _, p = fisher_exact([[a, len(F)-a],[c, len(O)-c]])
    tests.append((lv, a, c, p))
print()
n = len(tests)
for lv, a, c, p in sorted(tests, key=lambda x: x[3]):
    print(f"  {lv:24s} {a}/{len(F)} vs {c:2d}/{len(O)}   p = {p:.4f}   "
          f"Bonferroni ({n} tests) = {min(p*n,1):.3f}")

print("\n=== the gut-Bacteroidetes genera specifically ===")
GUT = ("Bacteroides","Parabacteroides","Alistipes")
P["gut"] = P.organism.fillna("").str.startswith(GUT)
a = int(P[P.grp=="FLIPPED"].gut.sum()); c = int(P[P.grp=="other"].gut.sum())
_, p = fisher_exact([[a,len(F)-a],[c,len(O)-c]])
print(f"  Bacteroides / Parabacteroides / Alistipes: {a}/{len(F)} flipped vs "
      f"{c}/{len(O)} other   Fisher p = {p:.4f}")
print(f"  names: {', '.join(P[(P.grp=='FLIPPED')&P.gut].organism.str.split().str[:2].str.join(' '))}")

print("\n=== for context: how do these genera do overall? ===")
lab = lb.build(min_reps=1); truth = lab.groupby("Enzyme").active.sum()
P["n_active"] = P.accession.map(truth)
m = P[P.n_active.notna()]
for lbl, sub in [("gut Bacteroidetes", m[m.gut]), ("everything else", m[~m.gut])]:
    dead = int((sub.n_active==0).sum())
    print(f"  {lbl:20s} {len(sub):2d} measured, {dead} made nothing "
          f"({dead/len(sub):.0%}), median products {sub.n_active.median():.0f}")
sub, oth = m[m.gut], m[~m.gut]
_, p = fisher_exact([[int((sub.n_active==0).sum()), int((sub.n_active>0).sum())],
                     [int((oth.n_active==0).sum()), int((oth.n_active>0).sum())]])
print(f"  inactive rate, gut Bacteroidetes vs rest: Fisher p = {p:.4f}")
