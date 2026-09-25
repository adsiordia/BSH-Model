"""Balanced BSH transamidation reactions, one per (amine, hydroxyl class).

    glyco/tauro-bile acid  +  amine  >>  amine-conjugate  +  glycine/taurine

The bile acid class is a hydroxyl COUNT, so a representative is chosen per
class -- the same ones the site draws. Both leaving groups are written, so the
two can be scored and averaged.
"""
from pathlib import Path
import json, re, pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem import rdMolDescriptors
RDLogger.DisableLog("rdApp.*")
ROOT = Path(__file__).resolve().parent.parent

LEAVING = {"glycine": "NCC(=O)O", "taurine": "NCCS(=O)(=O)O"}
# representative parent, matching site/conjugates.json
REP = {"Mono": "TLCA", "Di": "TDCA", "Tri": "TCA"}
REP_GLY = {"Mono": "GLCA", "Di": "GDCA", "Tri": "GCA"}


def formula(sm):
    m = Chem.MolFromSmiles(sm)
    return rdMolDescriptors.CalcMolFormula(m) if m else None


def atoms(sm):
    m = Chem.MolFromSmiles(sm)
    if m is None: return None
    m = Chem.AddHs(m)
    c = {}
    for a in m.GetAtoms():
        c[a.GetSymbol()] = c.get(a.GetSymbol(), 0) + 1
    return c


def main():
    enum = pd.read_excel(ROOT / "data/raw/swap_enumeration_with_core_smiles.xlsx")
    conj = json.loads((ROOT / "site/conjugates.json").read_text())["products"]
    chem = json.loads((ROOT / "site/chem.json").read_text())["amines"]

    # parent bile acids, by their short name
    parents = (enum.dropna(subset=["Parent_BA_SMILES_Original"])
                   .drop_duplicates("Parent_BA_Name")
                   .set_index("Parent_BA_Name").Parent_BA_SMILES_Original.to_dict())
    print(f"parent bile acids available: {len(parents)}")
    missing = [n for n in list(REP.values()) + list(REP_GLY.values()) if n not in parents]
    if missing:
        print(f"  not found by name: {missing}")
        print(f"  names present: {sorted(parents)[:14]}")

    rows, bad = [], []
    for key, p in conj.items():
        amine, core = key.rsplit("|", 1)
        asm = chem.get(amine, {}).get("smiles")
        psm = p.get("smiles")
        if not asm or not psm:
            bad.append((key, "no amine or product smiles")); continue
        for lg, lgs in LEAVING.items():
            pname = (REP_GLY if lg == "glycine" else REP)[core]
            sub = parents.get(pname)
            if not sub:
                bad.append((key, f"no parent {pname}")); continue
            rxn = f"{sub}.{asm}>>{psm}.{lgs}"
            L, R = atoms(f"{sub}.{asm}"), atoms(f"{psm}.{lgs}")
            rows.append(dict(amine=amine, core=core, leaving_group=lg,
                             parent=pname, reaction=rxn,
                             balanced=(L == R),
                             left=L, right=R))
    d = pd.DataFrame(rows)
    print(f"\nreactions built: {len(d)}  ({d.amine.nunique()} amines x "
          f"{d.core.nunique()} classes x {d.leaving_group.nunique()} leaving groups)")
    print(f"  atom-balanced: {int(d.balanced.sum())} of {len(d)}")
    if not d.balanced.all():
        u = d[~d.balanced].head(3)
        for r in u.itertuples():
            print(f"    {r.amine} {r.core} {r.leaving_group}: left {r.left} right {r.right}")
    if bad:
        print(f"  skipped: {len(bad)} -> {bad[:4]}")

    out = ROOT / "outputs/horizyn_reactions.csv"
    d.drop(columns=["left", "right"]).to_csv(out, index=False)
    print(f"\nwrote {out}")
    print("\nexample:")
    r = d.iloc[0]
    print(f"  {r.amine} + {r.core} ({r.leaving_group}, parent {r.parent})")
    print(f"  {r.reaction[:150]}...")


if __name__ == "__main__":
    main()
