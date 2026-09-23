"""Load the Trimmed proteomics table — the single source of intensity data.

Wide (366 x 85) -> long (Code, Replicate, ProductName, Amine, Hydroxyl, Intensity).
Enzyme rows are complete; control rows are largely NaN (see control_coverage()).
"""
from pathlib import Path
import re
import numpy as np, pandas as pd

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data/raw"
SRC = RAW / "Trimmed_remove_proteomics.csv"

META = ["filename", "Code", "Replicate"]
CANONICAL = ["taurine", "glycine", "unconjugated"]
# Penicillin amidase is a functional enzyme, not a blank -- excluded from controls.
NEGATIVE_CONTROLS = ["Negative_ctrl_1", "Negative_ctrl_2", "Negative_ctrl_3",
                     "Only_substrate_1", "Only_substrate_2", "Only_substrate_3"]
_DEG = re.compile(r"^(Mono|Di|Tri)_")


def _split_product(p):
    """'Di_2_aminophenol_36589' -> core 'Di', amine '2_aminophenol', feature id '36589'."""
    body = re.sub(r"_(\d+)$", "", p)
    feat = p[len(body) + 1:] if len(p) > len(body) else None
    core, rest = body.split("_", 1)
    rest = re.sub(r"_M\+\w+$", "", rest)          # strip adduct annotation
    return core, rest.lower(), feat


def load_long(drop_canonical=True):
    t = pd.read_csv(SRC).dropna(subset=["filename"]).copy()
    stem = t["filename"].str.replace(r"_rep\d+$", "", regex=True)
    t["Code"] = t["Code"].fillna(stem.str.split("_").str[-1])
    prods = [c for c in t.columns if c not in META]

    lg = t.melt(id_vars=["Code", "Replicate"], value_vars=prods,
                var_name="ProductName", value_name="Intensity")
    lg["Intensity"] = pd.to_numeric(lg["Intensity"], errors="coerce")
    parts = lg["ProductName"].map(_split_product)
    lg["Hydroxyl"] = [p[0] for p in parts]
    lg["Amine"] = [p[1] for p in parts]
    lg["FeatureId"] = [p[2] for p in parts]
    lg["is_control"] = lg["Code"].str.contains("ctrl|substrate|amidase", case=False, na=False)
    lg["is_negative_control"] = lg["Code"].isin(NEGATIVE_CONTROLS)
    if drop_canonical:
        lg = lg[~lg["Amine"].isin(CANONICAL)]
    return lg.reset_index(drop=True)


def enzymes(lg=None):
    lg = load_long() if lg is None else lg
    return lg[~lg.is_control]


def negative_controls(lg=None):
    """Only rows that were actually measured (NaN means not reported)."""
    lg = load_long() if lg is None else lg
    return lg[lg.is_negative_control & lg.Intensity.notna()]


def control_coverage(lg=None):
    """Which products have any true-negative-control measurement."""
    lg = load_long() if lg is None else lg
    nc = lg[lg.is_negative_control]
    g = nc.groupby("ProductName")["Intensity"].agg(
        n_measured=lambda s: int(s.notna().sum()),
        n_nonzero=lambda s: int((s.fillna(0) > 0).sum()),
        max_intensity=lambda s: float(np.nanmax(s)) if s.notna().any() else np.nan)
    return g.sort_values("n_measured", ascending=False)
