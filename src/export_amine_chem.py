"""Structure drawings + physicochemical analysis for every amine, for the explorer."""
from pathlib import Path
import json, sys, itertools
import numpy as np, pandas as pd
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import Draw, Descriptors, Crippen, Lipinski, rdMolDescriptors
from rdkit.Chem.Draw import rdMolDraw2D
from rdkit.Chem import rdFingerprintGenerator as rdfp
RDLogger.DisableLog("rdApp.*")

import labels as lb, representations as rep

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "site/chem.json"
W, H = 230, 170


def svg(mol, dark=False):
    d = rdMolDraw2D.MolDraw2DSVG(W, H)
    o = d.drawOptions()
    o.clearBackground = False
    o.bondLineWidth = 1.5
    o.minFontSize = 11
    o.maxFontSize = 14
    o.padding = 0.10
    if dark:
        o.setAtomPalette({-1: (0.92, 0.92, 0.90), 7: (0.45, 0.70, 0.95),
                          8: (0.95, 0.55, 0.45), 16: (0.92, 0.80, 0.35)})
    else:
        o.setAtomPalette({-1: (0.10, 0.11, 0.13), 7: (0.10, 0.36, 0.62),
                          8: (0.72, 0.25, 0.12), 16: (0.62, 0.52, 0.06)})
    m = Chem.Mol(mol)
    rdMolDraw2D.PrepareAndDrawMolecule(d, m)
    d.FinishDrawing()
    return d.GetDrawingText().replace("<?xml version='1.0' encoding='iso-8859-1'?>\n", "").strip()


DESCS = [
    ("mw", "molecular weight", lambda m: Descriptors.MolWt(m), "g/mol"),
    ("logp", "lipophilicity (logP)", Crippen.MolLogP, ""),
    ("tpsa", "polar surface area", rdMolDescriptors.CalcTPSA, "Å²"),
    ("hbd", "H-bond donors", Lipinski.NumHDonors, ""),
    ("hba", "H-bond acceptors", Lipinski.NumHAcceptors, ""),
    ("rotb", "rotatable bonds", Lipinski.NumRotatableBonds, ""),
    ("arom", "aromatic rings", Lipinski.NumAromaticRings, ""),
    ("heavy", "heavy atoms", lambda m: m.GetNumHeavyAtoms(), ""),
]


def ntype(r):
    """Which kind of nitrogen forms the bond -- one exclusive answer per amine.

    The older amine_class column mixed two questions: "amino acid" and "primary amine"
    are not alternatives (glycine is both), and "polyamine" is a count rather than a
    kind. This is the kind only; the count is carried separately as n_nucleophilic_N.
    """
    if r.n_aniline > 0:
        return "aromatic"
    if r.n_tertiary > 0 and r.n_primary == 0 and r.n_secondary == 0:
        return "tertiary"
    if r.n_secondary > 0 and r.n_primary == 0:
        return "secondary"
    return "primary"


def main():
    act = lb.build(min_reps=1)
    sm = rep.load_smiles("Amine")
    cur = pd.read_csv(ROOT/"data/derived/reactants_curated.csv")
    cur = cur[cur.compound_type == "Amine"].set_index("data_name")
    amines = sorted(set(act.Amine) & set(sm))
    mols = {a: Chem.MolFromSmiles(sm[a]) for a in amines}

    rate = act.groupby("Amine").active.mean().to_dict()
    n_cells = act.groupby("Amine").size().to_dict()
    n_hit = act.groupby("Amine").active.sum().to_dict()

    out = {}
    for a in amines:
        m = mols[a]
        out[a] = dict(
            svg=svg(m), svg_dark=svg(m, dark=True), smiles=sm[a],
            formula=rdMolDescriptors.CalcMolFormula(m),
            cls=ntype(cur.loc[a]),
            inferred=str(cur.smiles_source.get(a, "")) == "INFERRED",
            nuc=int(cur.n_nucleophilic_N.get(a, 0) or 0),
            n_primary=int(cur.n_primary.get(a, 0) or 0),
            n_secondary=int(cur.n_secondary.get(a, 0) or 0),
            has_cooh=str(cur.has_COOH.get(a, "")).lower() in ("true", "1"),
            rate=round(float(rate[a]), 4), cells=int(n_cells[a]), hits=int(n_hit[a]),
            props={k: round(float(f(m)), 2) for k, _, f, _ in DESCS})

    # pairwise Tanimoto, for the "most similar" panel
    gen = rdfp.GetMorganGenerator(radius=2, fpSize=512)
    fps = {a: gen.GetFingerprint(mols[a]) for a in amines}
    sim = {}
    for a in amines:
        s = sorted(((b, round(DataStructs.TanimotoSimilarity(fps[a], fps[b]), 3))
                    for b in amines if b != a), key=lambda x: -x[1])
        sim[a] = s[:4]

    payload = dict(amines=out, similar=sim,
                   desc_labels=[dict(k=k, label=l, unit=u) for k, l, _, u in DESCS])
    OUT.write_text(json.dumps(payload, separators=(",", ":")))
    print(f"wrote {OUT}  ({OUT.stat().st_size/1e6:.2f} MB)  {len(out)} amines")
    print(f"  example: {amines[0]} -> {out[amines[0]]['formula']}, "
          f"svg {len(out[amines[0]]['svg'])} chars")


if __name__ == "__main__":
    main()
