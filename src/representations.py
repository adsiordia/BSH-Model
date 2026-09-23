"""Molecular representations for amines and bile acid cores.

All builders share one signature: they take a {name: smiles} mapping and return
(DataFrame indexed by name, list of column names). Rows are in sorted-name order
so the feature matrix is reproducible across runs.
"""

from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem import Crippen, Descriptors, Lipinski, rdMolDescriptors
from rdkit.Chem import rdFingerprintGenerator as rdfp

RDLogger.DisableLog("rdApp.*")

ROOT = Path(__file__).resolve().parent.parent
CURATED = ROOT / "data/derived/reactants_curated.csv"

MORGAN_RADIUS = 2      # radius 2 == ECFP4
MORGAN_BITS = 1024


def load_smiles(compound_type="Amine", key="data_name"):
    """{name: smiles} from the curated reactants table, for rows that have a data_name."""
    df = pd.read_csv(CURATED)
    df = df[(df.compound_type == compound_type) & df[key].notna() & df.parse_ok]
    return dict(zip(df[key], df.smiles))


def _mols(smiles_map):
    names = sorted(smiles_map)
    mols = {n: Chem.MolFromSmiles(smiles_map[n]) for n in names}
    bad = [n for n, m in mols.items() if m is None]
    if bad:
        raise ValueError(f"SMILES failed to parse: {bad}")
    return names, mols


# ── Morgan fingerprint ─────────────────────────────────────────────

def morgan(smiles_map, radius=MORGAN_RADIUS, n_bits=MORGAN_BITS, counts=False):
    """Circular (ECFP-style) fingerprint. counts=True gives occurrence counts, not bits."""
    names, mols = _mols(smiles_map)
    gen = rdfp.GetMorganGenerator(radius=radius, fpSize=n_bits)
    fn = gen.GetCountFingerprintAsNumPy if counts else gen.GetFingerprintAsNumPy
    X = np.stack([fn(mols[n]) for n in names]).astype(np.float32)
    cols = [f"morgan_{i}" for i in range(n_bits)]
    return pd.DataFrame(X, index=names, columns=cols), cols


def morgan_bit_info(smiles_map, radius=MORGAN_RADIUS, n_bits=MORGAN_BITS):
    """Which bits are actually set by this set of molecules, and how often.

    Worth checking before modelling: a 1024-bit vector over ~25 small molecules is
    mostly dead columns, and bits set by every molecule carry no signal either.
    """
    df, _ = morgan(smiles_map, radius, n_bits)
    freq = df.sum(axis=0)
    return pd.DataFrame({
        "n_molecules_setting_bit": freq.astype(int),
        "fraction": (freq / len(df)).round(3),
    }).query("n_molecules_setting_bit > 0").sort_values("n_molecules_setting_bit", ascending=False)


# ── physicochemical descriptors ────────────────────────────────────

DESCRIPTORS = [
    ("mol_weight", Descriptors.MolWt), ("logp", Crippen.MolLogP),
    ("tpsa", rdMolDescriptors.CalcTPSA), ("h_donors", Lipinski.NumHDonors),
    ("h_acceptors", Lipinski.NumHAcceptors), ("rotatable_bonds", Lipinski.NumRotatableBonds),
    ("aromatic_rings", Lipinski.NumAromaticRings), ("rings", rdMolDescriptors.CalcNumRings),
    ("heavy_atoms", lambda m: m.GetNumHeavyAtoms()),
    ("fraction_csp3", rdMolDescriptors.CalcFractionCSP3),
    ("molar_refractivity", Crippen.MolMR),
    ("heteroatoms", rdMolDescriptors.CalcNumHeteroatoms),
    ("n_count", lambda m: sum(a.GetSymbol() == "N" for a in m.GetAtoms())),
    ("o_count", lambda m: sum(a.GetSymbol() == "O" for a in m.GetAtoms())),
    ("formal_charge", Chem.GetFormalCharge),
]


def descriptors(smiles_map):
    """15 interpretable physicochemical properties."""
    names, mols = _mols(smiles_map)
    cols = [f"desc_{n}" for n, _ in DESCRIPTORS]
    X = np.array([[fn(mols[n]) for _, fn in DESCRIPTORS] for n in names], dtype=np.float32)
    return pd.DataFrame(X, index=names, columns=cols), cols


# ── identity ───────────────────────────────────────────────────────

def onehot(smiles_map):
    """One-hot identity. Only meaningful under enzyme hold-out, where every
    amine is seen in training; useless for generalising to a new amine."""
    names = sorted(smiles_map)
    cols = [f"is_{n}" for n in names]
    return pd.DataFrame(np.eye(len(names), dtype=np.float32), index=names, columns=cols), cols


# ── chemistry annotations from the curated table ───────────────────

def amine_class_features():
    """Class / nucleophile counts curated in reactants_curated.csv."""
    df = pd.read_csv(CURATED)
    df = df[(df.compound_type == "Amine") & df.data_name.notna()].set_index("data_name")
    num = df[["n_primary", "n_secondary", "n_tertiary", "n_aniline", "n_nucleophilic_N"]].astype(np.float32)
    num["has_COOH"] = df["has_COOH"].astype(np.float32)
    dummies = pd.get_dummies(df["amine_class"], prefix="class").astype(np.float32)
    out = pd.concat([num, dummies], axis=1).sort_index()
    return out, list(out.columns)


BUILDERS = {
    "morgan": morgan,
    "descriptors": descriptors,
    "onehot": onehot,
}


def build(kind, smiles_map, **kw):
    if kind == "descriptors_onehot":
        d, dc = descriptors(smiles_map)
        o, oc = onehot(smiles_map)
        out = pd.concat([d, o], axis=1)
        return out, dc + oc
    return BUILDERS[kind](smiles_map, **kw)


# ── bile acid core ─────────────────────────────────────────────────
import re as _re

# the eight substituent tokens that actually occur in this panel
CORE_TOKENS = ["3a", "3k", "6a", "7a", "7b", "7k", "12a", "12k"]
_TOK = _re.compile(r"(\d+)([abk])")

# product label -> resolved core, or None where the label is ambiguous.
# Mono and Tri are unique in this panel (only 3a is monohydroxy, only 3a,7a,12a is
# trihydroxy), so they resolve; Di covers four isobaric isomers and does not.
LABEL_TO_CORE = {
    "Mono": "3a",
    "Tri": "3a,7a,12a",
    "3a12k": "3a,12k",
    "3a7k": "3a,7k",
    "3k7a": "3k,7a",
    "3a,7a,12k": "3a,7a,12k",
    "3k12a": "3k,12a",      # NOT a supplied reactant -- provisional, see PROVENANCE
    "Di": None,             # one of 3a,12a / 3a,7a / 3a,7b / 3a,6a
}
DI_CANDIDATES = ["3a,12a", "3a,7a", "3a,7b", "3a,6a"]
DI_SHARED = ["3a"]          # every Di candidate carries 3-alpha-OH
DI_N_OH, DI_N_KETO = 2, 0


def core_positional(labels=None):
    """Positional encoding that degrades gracefully where the label is ambiguous.

    8 substituent bits + n_OH + n_keto + unresolved flag = 11 dims.

    For `Di` the shared 3-alpha-OH bit is set (all four candidates have it) and the
    second hydroxyl is left unset with `core_unresolved = 1`. So the model sees
    "3a present, 2 hydroxyls, one position unknown" rather than a vector that is
    indistinguishable from an unsubstituted core.
    """
    labels = sorted(LABEL_TO_CORE) if labels is None else sorted(labels)
    cols = [f"core_{t}" for t in CORE_TOKENS] + ["core_n_OH", "core_n_keto", "core_unresolved"]
    rows = []
    for lab in labels:
        core = LABEL_TO_CORE.get(lab, None)
        vec = dict.fromkeys(cols, 0.0)
        if core is None:                                   # ambiguous (Di)
            for t in DI_SHARED:
                vec[f"core_{t}"] = 1.0
            vec["core_n_OH"], vec["core_n_keto"] = float(DI_N_OH), float(DI_N_KETO)
            vec["core_unresolved"] = 1.0
        else:
            toks = [f"{p}{s}" for p, s in _TOK.findall(core.replace(",", ""))]
            for t in toks:
                vec[f"core_{t}"] = 1.0
            vec["core_n_OH"] = float(sum(1 for t in toks if t.endswith(("a", "b"))))
            vec["core_n_keto"] = float(sum(1 for t in toks if t.endswith("k")))
        rows.append(vec)
    return pd.DataFrame(rows, index=labels)[cols], cols


def core_onehot(labels=None):
    """Identity of the measured label. The ceiling under enzyme hold-out, but it
    cannot score a core that was never measured."""
    labels = sorted(LABEL_TO_CORE) if labels is None else sorted(labels)
    cols = [f"core_is_{l}" for l in labels]
    return pd.DataFrame(np.eye(len(labels), dtype=np.float32), index=labels, columns=cols), cols


def core_counts(labels=None):
    """Degree only: n_OH, n_keto. Collapses every isomer -- a floor, not a contender."""
    df, _ = core_positional(labels)
    cols = ["core_n_OH", "core_n_keto"]
    return df[cols].copy(), cols


CORE_BUILDERS = {"positional": core_positional, "onehot": core_onehot, "counts": core_counts}


def build_core(kind, labels=None):
    return CORE_BUILDERS[kind](labels)
