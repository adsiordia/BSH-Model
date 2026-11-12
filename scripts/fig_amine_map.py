#!/usr/bin/env python3
"""Panel C: the amine half of the site, position by conjugate.

One cell is one (position, conjugate) pair: how many of the 26 modelled enzymes
place that contact in at least 3 of their 5 AlphaFold samples. Same >=3 of 5
threshold as every number on the site, and the same 27 amine-side positions that
clear >=3 enzymes in analysis/pocket_columns.json.

Conjugates run anionic-aliphatic -> anionic-aromatic -> neutral-aromatic ->
neutral-aliphatic, so a position that cares about charge bands vertically and one
that cares about the ring bands horizontally. Rows are sorted by how much more
they engage aromatic than aliphatic amines, which is the structure the figure is
there to show; the position number stays on the label so it is still citable.

Sequential single blue hue, 100->700, zero receding to the surface. The dominant
interaction class rides in muted text beside each row rather than as a second
colour, so the only colour in the plot means magnitude.
"""
import json
import os
import sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import numpy as np
import pandas as pd

import amine_strip

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import sys as _sys, pathlib as _pl
_sys.path.insert(0, str(_pl.Path(__file__).resolve().parent))
from _paths import AF3_ROOT, DERIVED, ANALYSIS as _ANALYSIS, REPO as _REPO
AF3 = str(AF3_ROOT)
BSH = str(_REPO)
OUT = f"{BSH}/figures"
SOLID = 3

SURF = "#fcfcfb"
INK, MUTED, RULE = "#1f2328", "#6b7177", "#e3e3e0"
BLUE = ["#cde2fb", "#b7d3f6", "#9ec5f4", "#86b6ef", "#6da7ec", "#5598e7",
        "#3987e5", "#2a78d6", "#256abf", "#1c5cab", "#184f95", "#104281",
        "#0d366b"]
CMAP = LinearSegmentedColormap.from_list("seqblue", [SURF] + BLUE)

AROM = {"Phe", "His", "Tyramine", "Dopamine", "3MeOTyramine", "Tryptamine",
        "2Aminophenol"}
PI = {"alkyl_pi", "carbon_pi", "pi_pi_stacking", "donor_pi", "amide_pi",
      "pi_cation", "cation_pi"}
FAM = {"hydrogen_bonds": "H-bond", "hydrophobic": "hydrophobic",
       "attractive_charge": "ionic", "repulsion": "repulsion",
       "covalent": "covalent"}

d = pd.read_pickle(f"{DERIVED}/contacts_full.pkl")
d = d[d.protein != "A0A414Q275"]              # no catalytic cysteine
charge = json.load(open(f"{_ANALYSIS}/amine_charge.json"))
POS = json.load(open(f"{_ANALYSIS}/pocket_columns.json"))["moiety_ref"]
NENZ = d.protein.nunique()

m = d[d.part == "moiety"]

# ---- the matrix: enzymes with a >=3 of 5 contact ---------------------------
f = m.groupby(["protein", "token", "pos"])["sample"].nunique().reset_index(name="f")
f = f[f.f >= SOLID]
M = (f.groupby(["pos", "token"]).protein.nunique().unstack(fill_value=0)
     .reindex(index=[float(p) for p in POS], fill_value=0))

# ---- conjugate order: charge, then ring -----------------------------------
def ckey(t):
    anion = charge.get(t, "neutral") != "neutral"
    return (0 if t not in AROM else 1, 0 if anion else 1, t)

TOK = sorted(M.columns, key=ckey)
M = M[TOK]
RING = [(t, "aromatic amines" if t in AROM else "aliphatic amines") for t in TOK]
CHG = [(t, "anionic" if charge.get(t, "neutral") != "neutral" else "neutral")
       for t in TOK]

# ---- rows: modal residue, conservation, dominant interaction class --------
pm = pd.read_csv(f"{BSH}/data/position_map.csv")
lab = {}
for p in POS:
    aa = pm[pm.position == p].aa
    aa = aa[aa != "-"]
    top = aa.value_counts()
    lab[p] = (top.index[0], top.iloc[0] / len(pm[pm.position == p]))

AA3 = {"C": "Cys", "W": "Trp", "Y": "Tyr", "R": "Arg", "K": "Lys", "D": "Asp",
       "E": "Glu", "N": "Asn", "Q": "Gln", "P": "Pro", "V": "Val", "L": "Leu",
       "I": "Ile", "M": "Met", "F": "Phe", "G": "Gly", "A": "Ala", "S": "Ser",
       "T": "Thr", "H": "His"}

# ionic and attractive_charge are duplicate rows -- count the class once
comp = m[m.itype != "ionic"]
itop = {}
for p in POS:
    x = comp[comp.pos == p]
    if not len(x):
        itop[p] = ""
        continue
    shares = {"pi-family": x.itype.isin(PI).mean()}
    for k, v in FAM.items():
        shares[v] = (x.itype == k).mean()
    k = max(shares, key=shares.get)
    itop[p] = f"{k} {shares[k]*100:.0f}%"

arom = [t for t in TOK if t in AROM]
alip = [t for t in TOK if t not in AROM]
sel = M[arom].mean(axis=1) / NENZ - M[alip].mean(axis=1) / NENZ
M = M.loc[sel.sort_values(ascending=False).index]
ROWS = [int(p) for p in M.index]

# ---- draw -----------------------------------------------------------------
Z = M.values / NENZ
fig, (axm, ax) = plt.subplots(
    2, 1, figsize=(16.8, 10.4), facecolor=SURF, sharex=True,
    gridspec_kw=dict(height_ratios=[.102, 1], hspace=.10))
im = ax.imshow(Z, cmap=CMAP, vmin=0, vmax=1, aspect="auto",
               interpolation="nearest")

# the conjugates drawn as structures, cut from the SMILES AlphaFold was given.
# sharex plus an extent in column units puts tile i under column i exactly.
png = amine_strip.strip(TOK, f"{OUT}/amine_strip.png")
axm.imshow(plt.imread(png), extent=(-.5, len(TOK) - .5, 0, 1),
           aspect="auto", interpolation="antialiased")
axm.set_facecolor(SURF)
axm.set_yticks([])
axm.set_xticks(range(len(TOK)))
axm.set_xticklabels(TOK, rotation=90, fontsize=8, color=INK)
# sharex hands axm the grid's minor ticks too, which print as stray dashes
# between the structures
axm.set_xticks([], minor=True)
# the name reads first, then the structure, then the grid it belongs to
axm.xaxis.set_label_position("top")
axm.xaxis.tick_top()
axm.tick_params(which="both", length=0, labelbottom=False, labeltop=True)
for sp in axm.spines.values():
    sp.set_visible(False)
axm.axvline(len(TOK) - len(arom) - .5, color=MUTED, lw=1.2, zorder=5)

# three label columns rather than one string, so each can carry a heading in
# the same register as the brackets over the conjugates
LCOL = ((-4.3, "right", "position", lambda p: str(p)),
        (-4.0, "left", "commonest\nresidue",
         lambda p: AA3.get(lab[p][0], lab[p][0])),
        (-0.9, "right", "% of BSH\nsequences",
         lambda p: f"{lab[p][1]*100:.0f}%"))
ax.set_yticks([])
for i, p in enumerate(ROWS):
    for x, ha, _h, val in LCOL:
        ax.text(x, i, val(p), ha=ha, va="center", fontsize=8.5, color=INK,
                fontfamily="DejaVu Sans Mono")
ax.tick_params(length=0)
for s in ax.spines.values():
    s.set_visible(False)

# 2 px surface gap between cells, drawn as a grid on the minor ticks
ax.set_xticks(np.arange(-.5, len(TOK), 1), minor=True)
ax.set_yticks(np.arange(-.5, len(ROWS), 1), minor=True)
ax.tick_params(axis="x", labelbottom=False)
ax.grid(which="minor", color=SURF, linewidth=1.6)
ax.tick_params(which="minor", length=0)

# chemical-class brackets over the conjugate axis
def brackets(spans, y, size, lw, below=False):
    """One tier of labelled brackets along the conjugate axis."""
    seen, start = spans[0][1], 0
    for i in range(1, len(spans) + 1):
        here = spans[i][1] if i < len(spans) else None
        if here == seen:
            continue
        ax.plot([start - .4, i - .6], [y, y], color=RULE, lw=lw,
                clip_on=False, solid_capstyle="butt")
        ax.text((start + i - 1) / 2, y + (.55 if below else -.55), seen,
                ha="center", va="top" if below else "bottom",
                fontsize=size, color=MUTED)
        seen, start = here, i


# under the grid: the structures and their names are already above it, and a
# third tier up there pushed the heatmap too far from its own column headings
brackets(RING, len(ROWS) - .1, 9.0, 2.8, below=True)

for x, ha, head, _v in LCOL:
    ax.text(x, -.85, head, ha=ha, va="bottom", fontsize=8, color=MUTED,
            linespacing=1.3)

nalip = sum(1 for t in TOK if t not in AROM)
ax.axvline(nalip - .5, color=MUTED, lw=1.2, ymin=-.02, ymax=1.02,
           clip_on=False, zorder=5)

cb = fig.colorbar(im, ax=(ax, axm), fraction=.022, pad=.03, ticks=[0, .5, 1])
cb.ax.set_yticklabels(["0", "50", "100%"], fontsize=8, color=MUTED)
cb.ax.tick_params(length=0)
cb.outline.set_visible(False)
cb.set_label(f"of {NENZ} enzymes with the contact in ≥3 of 5 models",
             fontsize=8, color=MUTED)


fig.suptitle("The amine half of the site is chemically structured",
             fontsize=12, color=INK, x=.06, ha="left", y=.985)
fig.text(.06, .957, "one column per conjugated product, drawn as the amine it "
         "carries \u2014 only its amine-side contacts are counted",
         fontsize=9, color=MUTED, ha="left", va="top")




# The structures must be square: a tile is one column wide, so it has to be one
# column tall too. Guessing a height_ratio gets this wrong every time the gap or
# the colorbar changes, so measure the grid once it is laid out and size the
# strip from it. ax.get_position() is read after the colorbar, so it is the real
# plotting box rather than the pre-colorbar one.
fig.canvas.draw()
gw, fw, fh = ax.get_position().width, *fig.get_size_inches()
tile = gw * fw / len(TOK)                      # one column, in inches
box = ax.get_position()
axm.set_position([box.x0, box.y1 + .018, box.width, tile / fh])

for ext in ("png", "pdf"):
    fig.savefig(f"{OUT}/panelC_amine_map.{ext}", dpi=400, facecolor=SURF,
                bbox_inches="tight")
print(f"wrote {OUT}/panelC_amine_map.png and .pdf")
print(f"  {len(ROWS)} amine positions x {len(TOK)} conjugates, n={NENZ} enzymes")
print("\n  most aromatic-selective positions:")
for p in ROWS[:6]:
    print(f"    {p:>4} {AA3.get(lab[p][0],lab[p][0])} cons {lab[p][1]*100:4.0f}%  "
          f"arom-aliph {sel.loc[float(p)]*100:+5.1f} pts   {itop[p]}")
print("\n  most aliphatic-selective positions:")
for p in ROWS[-4:]:
    print(f"    {p:>4} {AA3.get(lab[p][0],lab[p][0])} cons {lab[p][1]*100:4.0f}%  "
          f"arom-aliph {sel.loc[float(p)]*100:+5.1f} pts   {itop[p]}")
