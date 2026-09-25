"""site/candgrid.json -- the 66 signal-peptide proteins as a heatmap:
what the assay measured, and what the model says from each sequence version."""
from pathlib import Path
import sys, json, numpy as np, pandas as pd
ROOT = Path(__file__).resolve().parent.parent; sys.path.insert(0, str(ROOT/"src"))
import labels as lb
OUT, SITE = ROOT/"outputs", ROOT/"site"; CUT = 0.5

g = pd.read_csv(OUT/"candidate_predictions.csv")
lab = lb.build(min_reps=1)
truth = lab.groupby("Enzyme").active.sum()
P = pd.DataFrame(json.load(open(SITE/"candidates.json"))["proteins"])
FILL = {"Parabacteroides":"Bacteroidota","Paraprevotella":"Bacteroidota","Odoribacter":"Bacteroidota",
        "Barnesiella":"Bacteroidota","Prevotella":"Bacteroidota","Dysgonomonas":"Bacteroidota",
        "Butyricimonas":"Bacteroidota","Muribaculaceae":"Bacteroidota","Alistipes":"Bacteroidota",
        "Homeothermus":"Bacteroidota","Okeania":"Cyanobacteriota","Symploca":"Cyanobacteriota",
        "Moorea":"Cyanobacteriota","Obscuribacterales":"Cyanobacteriota"}
def phy(r):
    if r.phylum: return r.phylum
    for t in (r.organism or "").replace("Candidatus ","").split()[:2]:
        if t.strip("[]") in FILL: return FILL[t.strip("[]")]
    return ""
P["phy"] = P.apply(phy, axis=1)
meta = P.set_index("accession")

# fixed product order: commonest first, so the heatmap reads left to right
prev = lab.groupby(["Amine","Hydroxyl"]).active.mean().sort_values(ascending=False)
products = [[a, h] for a, h in prev.index]
pidx = {tuple(p): i for i, p in enumerate(products)}

prot, cells = [], {}
for acc, s in g.groupby("accession"):
    s = s.copy()
    s["j"] = [pidx[(a, h)] for a, h in zip(s.Amine, s.Hydroxyl)]
    s = s.sort_values("j")
    m = [None if pd.isna(v) else int(v) for v in s.measured]
    cells[acc] = dict(
        m=m,
        f=[round(float(v), 3) for v in s.raw_full],
        t=[round(float(v), 3) for v in s.raw_trimmed])
    r = meta.loc[acc] if acc in meta.index else None
    n_meas = int(s.measured.notna().sum())
    prot.append(dict(
        acc=acc,
        org=(r.organism if r is not None else ""),
        phy=(r.phy if r is not None else ""),
        cut=(None if r is None or pd.isna(r.cut) else int(r.cut)),
        tested=n_meas > 0,
        measured=(None if n_meas == 0 else int(s.measured.fillna(0).sum())),
        n_full=int((s.raw_full >= CUT).sum()),
        n_trim=int((s.raw_trimmed >= CUT).sum()),
        comparable=bool(s.comparable.iloc[0])))
prot.sort(key=lambda r: (-(r["measured"] if r["measured"] is not None else -1), -r["n_full"]))

payload = dict(
    generated=str(pd.Timestamp.today().date()),
    cut=CUT, n_proteins=len(prot), n_products=len(products),
    products=products,
    prevalence=[round(float(v), 4) for v in prev.values],
    proteins=prot, cells=cells,
    n_tested=int(sum(1 for r in prot if r["tested"])),
    n_untested=int(sum(1 for r in prot if not r["tested"])),
    trained_on=int(sum(1 for r in prot if r["tested"])),
    note=("58 of these 66 proteins were in the training set of the model that scores "
          "them here, so their columns are retrodictions, not predictions: the model "
          "has already seen the right answer. They are shown to compare the two sequence "
          "versions against a known result, not as new discoveries. Only the 8 removed "
          "by proteomics carry genuinely new information."),
    removed_note=("These 8 are absent from the proteomics table -- they were removed at "
                  "that step rather than never attempted, so there is no assay result to "
                  "compare their predictions against."))
(SITE/"candgrid.json").write_text(json.dumps(payload, allow_nan=False))
kb = (SITE/"candgrid.json").stat().st_size/1024
print(f"wrote {SITE/'candgrid.json'}  ({kb:.0f} kB)")
print(f"  {len(prot)} proteins x {len(products)} products")
print(f"  {payload['n_tested']} assayed (in training), {payload['n_untested']} never assayed")
