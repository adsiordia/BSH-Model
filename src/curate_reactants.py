"""Build a curated reactants table from the raw spreadsheet.

Corrections applied here (raw file is left untouched):
  * 2,3-Diaminopropionic acid  -- stored SMILES was a cyclic C5H11N3O2 (MW 145.16);
    the intended compound is linear C3H8N2O2 (MW 104.11), which matches the
    spreadsheet's own MW column. Stereochemistry is NOT asserted: unlike the other
    amino acids the name carries no L-/D- prefix, so the corrected SMILES is achiral.

Noted but NOT changed:
  * TLCA -- spreadsheet MW (505.7) is the sodium salt; the SMILES is the free acid
    (483.7). The structure is correct, so the SMILES stands.
"""

from pathlib import Path

import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem import Descriptors, rdMolDescriptors

RDLogger.DisableLog("rdApp.*")

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data/raw/bsh_reactants_SMILES_corrected.xlsx"
OUT = ROOT / "data/derived/reactants_curated.csv"

SMILES_FIXES = {
    "2,3-Diaminopropinoic Acid": ("NC(CN)C(=O)O", "cyclic C5H11N3O2 in raw file; MW column says 104.11"),
}

# Amines that appear in the activity data but are absent from the reactants sheet.
# SMILES are INFERRED from the common meaning of the name, NOT read from the source
# file, and are flagged `smiles_source == "INFERRED"` so they are easy to revisit.
# Pending confirmation with whoever produced the LC-MS results.
INFERRED_AMINES = {
    "serotonin": ("NCCc1c[nH]c2ccc(O)cc12", "5-hydroxytryptamine; not in reactants sheet"),
    "tyramine":  ("NCCc1ccc(O)cc1",         "4-hydroxyphenethylamine; not in reactants sheet"),
    "cystine":   ("OC(=O)C(N)CSSCC(N)C(=O)O", "disulfide dimer of L-cysteine, which IS a reactant"),
}

# core identity after deconjugation, and which conjugate forms were supplied
BA_CORE = {
    "GLCA": ("3a", "Gly"), "TLCA": ("3a", "Tau"),
    "GDCA": ("3a,12a", "Gly"), "TDCA": ("3a,12a", "Tau"),
    "GCDCA": ("3a,7a", "Gly"), "TCDCA": ("3a,7a", "Tau"),
    "GUDCA": ("3a,7b", "Gly"), "TUDCA": ("3a,7b", "Tau"),
    "GHDCA": ("3a,6a", "Gly"), "THDCA": ("3a,6a", "Tau"),
    "GCA": ("3a,7a,12a", "Gly"), "TCA": ("3a,7a,12a", "Tau"),
    "3a12k-Gly": ("3a,12k", "Gly"), "3a12k-Tau": ("3a,12k", "Tau"),
    "3a7k-Gly": ("3a,7k", "Gly"), "3k7a-Gly": ("3k,7a", "Gly"),
    "3a7a12k-Gly": ("3a,7a,12k", "Gly"), "3a7a12k-Tau": ("3a,7a,12k", "Tau"),
    "3k7k12k-Tau": ("3k,7k,12k", "Tau"),
}

# spreadsheet name -> name used in the activity table
DATA_NAME = {
    "2,3-Diaminopropinoic Acid": "2,3_diaminopropionic acid", "2-aminophenol": "2_aminophenol",
    "3-methoxytyramine HCl": "3_methoxytyramine", "4-aminophenol": "4_aminophenol",
    "L-Alanine": "alanine", "L-Arginine": "arginine", "Asparagine": "asparagine",
    "Cadaverine": "cadaverine", "L-Citrulline": "citrulline", "L-Cysteine": "cysteine",
    "Dopamine HCl": "dopamine", "gamma-Aminobutyric acid >99%": "gaba",
    "L-Glutamine": "glutamine", "Glycyl-L-Valine": "glyglycine", "L-Histidine": "histidine",
    "L-Lysine": "lysine", "L-Methionine": "methionine",
    "L-Ornithine monohydrochloride": "ornithine", "L-Phenylalanine": "phenylalanine",
    "DL-Proline": "proline", "Putrescine": "putrescine", "L-Serine": "serine",
    "L-Threonine": "threonine", "Tryptamine": "tryptamine",
}

OH = Chem.MolFromSmarts("[OX2H][CX4;R]")
KETO = Chem.MolFromSmarts("[CX3;R]=[OX1]")
COOH = Chem.MolFromSmarts("C(=O)[OX2H1]")
GLY = Chem.MolFromSmarts("C(=O)NCC(=O)[OX2H1]")
TAU = Chem.MolFromSmarts("C(=O)NCCS(=O)(=O)[OX2H1]")


def classify_nitrogens(mol):
    """(degree, environment) for every N."""
    out = []
    for at in mol.GetAtoms():
        if at.GetSymbol() != "N":
            continue
        heavy = [n for n in at.GetNeighbors() if n.GetSymbol() != "H"]
        carbonyl = any(
            n.GetSymbol() == "C" and any(
                b.GetBondType() == Chem.BondType.DOUBLE and b.GetOtherAtom(n).GetSymbol() == "O"
                for b in n.GetBonds())
            for n in heavy)
        amidine = any(
            n.GetSymbol() == "C" and sum(
                1 for b in n.GetBonds() if b.GetOtherAtom(n).GetSymbol() == "N") >= 2
            for n in heavy)
        if at.GetIsAromatic():                      env = "aromatic_ring_N"
        elif carbonyl:                              env = "amide_N"
        elif amidine:                               env = "guanidine_urea_N"
        elif any(n.GetIsAromatic() for n in heavy): env = "aniline_N"
        else:                                       env = "aliphatic"
        deg = {0: "tertiary", 1: "secondary", 2: "primary"}.get(at.GetTotalNumHs(), "quaternary")
        out.append((deg, env))
    return out


def amine_class(prim, sec, anil, has_acid):
    if anil:            return "aryl_amine"
    if sec and not prim: return "secondary_amine"
    if prim >= 2:       return "polyamine"
    if has_acid:        return "amino_acid"
    return "primary_amine"


def main():
    df = pd.read_excel(RAW, sheet_name="Reactants")
    rows = []
    for _, r in df.iterrows():
        name = r["Compound_Name"]
        kind = r["Compount Type"]
        raw_smiles = str(r["SMILES"])
        fixed, reason = SMILES_FIXES.get(name, (None, ""))
        smiles = fixed if fixed else max(raw_smiles.split("."), key=len)
        mol = Chem.MolFromSmiles(smiles)
        rec = dict(
            compound_name=name, compound_type=kind,
            data_name=DATA_NAME.get(name),
            smiles=smiles,
            smiles_source="CORRECTED" if fixed else ("desalted" if "." in raw_smiles else "raw"),
            correction_note=reason,
            mw_file=r["Molecular Weight (g/mol)"],
        )
        if mol is None:
            rows.append({**rec, "parse_ok": False})
            continue
        rec.update(parse_ok=True,
                   formula=rdMolDescriptors.CalcMolFormula(mol),
                   mw_smiles=round(Descriptors.MolWt(mol), 2))
        if kind == "Amine":
            Ns = classify_nitrogens(mol)
            prim = sum(1 for d, e in Ns if d == "primary" and e == "aliphatic")
            sec = sum(1 for d, e in Ns if d == "secondary" and e == "aliphatic")
            tert = sum(1 for d, e in Ns if d == "tertiary" and e == "aliphatic")
            anil = sum(1 for d, e in Ns if e == "aniline_N")
            acid = mol.HasSubstructMatch(COOH)
            nuc = sum(1 for d, e in Ns if e in ("aliphatic", "aniline") and d in ("primary", "secondary"))
            rec.update(n_N=len(Ns), n_primary=prim, n_secondary=sec, n_tertiary=tert,
                       n_aniline=anil, has_COOH=acid, n_nucleophilic_N=nuc,
                       amine_class=amine_class(prim, sec, anil, acid))
        else:
            core, conj = BA_CORE.get(name, (None, None))
            n_oh = len({a for a, _ in mol.GetSubstructMatches(OH)})
            n_k = len({m[0] for m in mol.GetSubstructMatches(KETO)})
            detected = "Gly" if mol.HasSubstructMatch(GLY) else ("Tau" if mol.HasSubstructMatch(TAU) else None)
            rec.update(core=core, conjugate=conj, conjugate_detected=detected,
                       n_OH=n_oh, n_keto=n_k,
                       degree={0: None, 1: "Mono", 2: "Di", 3: "Tri"}.get(n_oh),
                       conjugate_matches_name=(conj == detected))
        rows.append(rec)

    for data_name, (smiles, note) in INFERRED_AMINES.items():
        mol = Chem.MolFromSmiles(smiles)
        Ns = classify_nitrogens(mol)
        prim = sum(1 for d, e in Ns if d == "primary" and e == "aliphatic")
        sec = sum(1 for d, e in Ns if d == "secondary" and e == "aliphatic")
        tert = sum(1 for d, e in Ns if d == "tertiary" and e == "aliphatic")
        anil = sum(1 for d, e in Ns if e == "aniline_N")
        acid = mol.HasSubstructMatch(COOH)
        rows.append(dict(
            compound_name=data_name, compound_type="Amine", data_name=data_name,
            smiles=smiles, smiles_source="INFERRED", correction_note=note,
            mw_file=None, parse_ok=True,
            formula=rdMolDescriptors.CalcMolFormula(mol),
            mw_smiles=round(Descriptors.MolWt(mol), 2),
            n_N=len(Ns), n_primary=prim, n_secondary=sec, n_tertiary=tert,
            n_aniline=anil, has_COOH=acid,
            n_nucleophilic_N=sum(1 for d, e in Ns
                                 if e in ("aliphatic", "aniline") and d in ("primary", "secondary")),
            amine_class=amine_class(prim, sec, anil, acid)))

    out = pd.DataFrame(rows)
    # how many conjugate forms each core was supplied as (1 mM each -> concentration differs)
    forms = out[out.compound_type == "Bile Acid"].groupby("core")["conjugate"].nunique()
    out["n_conjugate_forms"] = out["core"].map(forms)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT, index=False)
    return out


if __name__ == "__main__":
    t = main()
    print(f"wrote {OUT}  ({len(t)} reactants)\n")
    print("--- corrections applied ---")
    c = t[t.smiles_source == "CORRECTED"]
    print(c[["compound_name", "smiles", "formula", "mw_smiles", "mw_file", "correction_note"]].to_string(index=False))
    print("\n--- INFERRED (not in the reactants sheet; pending confirmation) ---")
    i = t[t.smiles_source == "INFERRED"]
    print(i[["compound_name", "smiles", "formula", "mw_smiles", "amine_class", "correction_note"]].to_string(index=False))
    print("\n--- amine classes ---")
    print(t[t.compound_type == "Amine"]["amine_class"].value_counts().to_string())
    print("\n--- bile acids: all conjugates match the name? ---")
    print(t[t.compound_type == "Bile Acid"]["conjugate_matches_name"].value_counts().to_string())
    print("\n--- parse failures ---")
    print(int((~t.parse_ok).sum()))
