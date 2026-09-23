"""Draw the predicted conjugate product for every amine x bile acid core.

A BSH cuts the glycine or taurine off C24 of a bile acid and can re-attach a different
amine in its place. So the product is: bile acid, amide bond at C24, amine. This builds
that molecule for each pair and renders it.

Mono / Di / Tri count hydroxyls only, and bile acids with the same count weigh the same,
so the data cannot say which positions they occupy. Rather than draw every isomer, one
representative is drawn per degree with its hydroxyls HIGHLIGHTED, and the positions that
are actually possible are listed alongside. That keeps the ambiguity explicit without
four near-identical pictures.

That also brings the set down to 25 amines x 3 degrees, small enough to inline in the
page, so the structures need no separate file and work from a bare index.html.
"""
from pathlib import Path
import json, re, warnings
warnings.filterwarnings("ignore")
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem
from rdkit.Chem.Draw import rdMolDraw2D
import pandas as pd
RDLogger.DisableLog("rdApp.*")

ROOT = Path(__file__).resolve().parent.parent
SITE = ROOT / "site"
W, H = 340, 215

# One representative per degree. The others differ only in where the hydroxyls sit, which
# is stated in words instead -- POSITIONS below.
REPRESENTATIVE = {"Mono": "3a", "Di": "3a,12a", "Tri": "3a,7a,12a"}

# strip the glycine/taurine already on the bile acid, back to the free C24 acid
UNCONJUGATE = AllChem.ReactionFromSmarts("[C:1](=[O:2])[N;H1:3]>>[C:1](=[O:2])[OH]")
# then form the new amide. The nitrogen must be a free amine: not already an amide,
# not a sulfonamide, not aromatic, and carrying at least one hydrogen.
COUPLE = AllChem.ReactionFromSmarts(
    "[C:1](=[O:2])[OH].[N;!$(N-C=O);!$(N-S);!$(N=*);!n;H1,H2:3]>>[C:1](=[O:2])[N:3]")

CORE_NAME = {
    "3a": "lithocholic acid (LCA)",
    "3a,12a": "deoxycholic acid (DCA)",
    "3a,7a": "chenodeoxycholic acid (CDCA)",
    "3a,7b": "ursodeoxycholic acid (UDCA)",
    "3a,6a": "hyodeoxycholic acid (HDCA)",
    "3a,7a,12a": "cholic acid (CA)",
    "3a,12k": "3α-hydroxy-12-keto",
    "3a,7k": "3α-hydroxy-7-keto",
    "3k,7a": "3-keto-7α-hydroxy",
    "3a,7a,12k": "3α,7α-dihydroxy-12-keto",
}


def shrink(svg):
    svg = re.sub(r"<\?xml[^>]*\?>", "", svg)
    svg = re.sub(r"<!DOCTYPE[^>]*>", "", svg)
    svg = re.sub(r"<rdkit:[^>]*>.*?</rdkit:[^>]*>", "", svg, flags=re.S)
    svg = re.sub(r"\s*xmlns:rdkit=\S+", "", svg)
    svg = re.sub(r"(\d+\.\d)\d+", r"\1", svg)
    svg = re.sub(r">\s+<", "><", svg)
    return re.sub(r"\s{2,}", " ", svg).strip()


# RDKit writes fixed hex colours. Swapping them for CSS variables lets one drawing serve
# both themes instead of storing a light and a dark copy of every structure.
THEME = {"#000000": "var(--ink)", "#0000FF": "var(--atom-n)",
         "#FF0000": "var(--atom-o)", "#CCCC00": "var(--atom-s)"}


def hydroxyl_atoms(mol):
    """Ring-bound OH oxygens -- the ones whose position is ambiguous."""
    out = []
    for a in mol.GetAtoms():
        if a.GetSymbol() != "O" or a.GetTotalNumHs() != 1:
            continue
        nb = a.GetNeighbors()
        if len(nb) == 1 and nb[0].IsInRing():
            out.append(a.GetIdx())
    return out


def render(mol, highlight=()):
    d = rdMolDraw2D.MolDraw2DSVG(W, H)
    o = d.drawOptions()
    o.clearBackground = False
    o.bondLineWidth = 1
    o.highlightRadius = 0.32
    rdMolDraw2D.PrepareAndDrawMolecule(
        d, mol, highlightAtoms=list(highlight),
        highlightAtomColors={i: (0.86, 0.74, 0.55) for i in highlight})
    d.FinishDrawing()
    svg = shrink(d.GetDrawingText())
    for hexv, var in THEME.items():
        svg = svg.replace(hexv, var).replace(hexv.lower(), var)
    return svg


def free_acid(smiles):
    m = Chem.MolFromSmiles(smiles)
    if m is None:
        return None
    prods = UNCONJUGATE.RunReactants((m,))
    if not prods:
        return m
    p = prods[0][0]
    Chem.SanitizeMol(p)
    return p


def conjugate(acid, amine_smiles):
    a = Chem.MolFromSmiles(amine_smiles)
    if a is None:
        return None, 0
    prods = COUPLE.RunReactants((acid, a))
    if not prods:
        return None, 0
    seen = {}
    for (p,) in prods:
        try:
            Chem.SanitizeMol(p)
            seen[Chem.MolToSmiles(p)] = p
        except Exception:
            continue
    if not seen:
        return None, 0
    # several nitrogens can react; the first canonical form is shown and the count kept
    k = sorted(seen)[0]
    return seen[k], len(seen)


def main():
    r = pd.read_csv(ROOT / "data/derived/reactants_curated.csv")
    ba = r[r.compound_type.str.contains("Bile", case=False, na=False)]
    am = r[r.compound_type.str.contains("Amine", case=False, na=False)]
    am = am[am.data_name.notna()]
    # only the amines the model actually scores
    used = set(pd.read_csv(ROOT / "outputs/candidate_predictions.csv").Amine.unique())
    am = am[am.data_name.isin(used)]

    cores = {}
    for c, s in ba.groupby("core"):
        if pd.isna(s.degree.iloc[0]):
            continue                              # 3k,7k,12k has no hydroxyls at all
        acid = free_acid(s.smiles.iloc[0])
        if acid is None:
            continue
        cores[c] = dict(degree=s.degree.iloc[0], acid=acid,
                        name=CORE_NAME.get(c, c), n_oh=int(s.n_OH.iloc[0]),
                        n_keto=int(s.n_keto.iloc[0]))

    # what each degree class could actually be, in words
    positions = {}
    for deg in ("Mono", "Di", "Tri"):
        opts = [(c, v["name"]) for c, v in cores.items() if v["degree"] == deg]
        # "3a,7a,12k" -> hydroxyls at 3 and 7, a ketone at 12. The suffix is the
        # orientation (a/b) for a hydroxyl, or k for a keto group.
        oh, keto = set(), set()
        for c, _ in opts:
            for tok in c.split(","):
                num, suffix = tok[:-1], tok[-1]
                (keto if suffix == "k" else oh).add(int(num))
        positions[deg] = dict(
            options=[dict(core=c, name=n) for c, n in sorted(opts)],
            oh=sorted(oh), keto=sorted(keto),
            representative=REPRESENTATIVE[deg],
            representative_name=cores[REPRESENTATIVE[deg]]["name"])

    out, skipped = {}, []
    for _, a in am.iterrows():
        for deg, rep in REPRESENTATIVE.items():
            cd = cores[rep]
            mol, n = conjugate(cd["acid"], a.smiles)
            if mol is None:
                skipped.append((a.data_name, deg))
                continue
            out[f"{a.data_name}|{deg}"] = dict(
                svg=render(mol, hydroxyl_atoms(mol)),
                smiles=Chem.MolToSmiles(mol),
                formula=Chem.rdMolDescriptors.CalcMolFormula(mol),
                mw=round(Chem.Descriptors.MolWt(mol), 2),
                sites=n)
    doc = dict(positions=positions, products=out)
    p = SITE / "conjugates.json"
    p.write_text(json.dumps(doc, separators=(",", ":")))
    print(f"wrote {p}  ({p.stat().st_size/1e6:.2f} MB)")
    print(f"  {am.data_name.nunique()} amines x 3 degree classes -> {len(out)} structures")
    for deg, v in positions.items():
        print(f"     {deg:5s} drawn as {v['representative_name']}; hydroxyls at "
              f"{'/'.join(str(x) for x in v['oh'])}"
              + (f", keto at {'/'.join(str(x) for x in v['keto'])}" if v["keto"] else "")
              + f" ({len(v['options'])} bile acids)")
    if skipped:
        print(f"  could not build {len(skipped)}: {skipped[:6]}")


if __name__ == "__main__":
    from rdkit.Chem import Descriptors, rdMolDescriptors  # noqa: F401
    main()
