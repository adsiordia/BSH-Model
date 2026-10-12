#!/usr/bin/env python3
"""Panel B: the arginines switch interaction mode with the amine's chemistry.

Stacked composition, so each bar is 100% of that arginine's amine-side contacts
for that class of amine. Three categories, coloured with the same hues the site
uses for these interactions (validated all-pairs, light and dark): ionic teal,
cation-pi olive, hydrogen bond blue. A 2 px surface gap separates segments, the
legend is always present, and only segments above 8% are labelled so the bars
stay readable.
"""
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import pandas as pd

AF3 = "/home/adsiordia/AF3/BSH_AF3"
SITE = "/home/adsiordia/BSH-Model/site"
OUT = "/home/adsiordia/BSH-Model/figures"

SURF = "#fcfcfb"
INK, MUTED, RULE = "#1f2328", "#6b7177", "#e3e3e0"
COL = {"ionic": "#0f8f7d", "cation-π": "#8a6b12", "hydrogen bond": "#2a5fa8"}

AROM = {"Phe", "Tyramine", "Dopamine", "3MeOTyramine", "Tryptamine",
        "2Aminophenol", "His"}
CATPI = {"pi_cation", "cation_pi"}
IONIC = {"ionic", "attractive_charge"}

d = pd.read_pickle(f"{AF3}/analysis/contacts_full.pkl")
d = d[d.protein != "A0A414Q275"]
charge = json.load(open(f"{AF3}/analysis/amine_charge.json"))
m = d[d.part == "moiety"].copy()


def cls(t):
    a, c = t in AROM, charge.get(t, "neutral") != "neutral"
    return ("aromatic\n+ carboxylate" if a and c else
            "aromatic,\nneutral" if a else
            "carboxylate,\nnon-aromatic" if c else
            "neutral,\nnon-aromatic")


m["cls"] = m.token.map(cls)
ORDER = ["carboxylate,\nnon-aromatic", "aromatic\n+ carboxylate",
         "aromatic,\nneutral", "neutral,\nnon-aromatic"]

fig, axes = plt.subplots(1, 2, figsize=(7.4, 3.5), facecolor=SURF, sharey=True)
for ax, pos in zip(axes, (222, 206)):
    x = m[(m.pos == pos) & (m.resname == "ARG")]
    labels, bars = [], []
    for c in ORDER:
        y = x[x.cls == c]
        if len(y) < 20:
            continue
        n = len(y)
        bars.append((c, len(y),
                     100 * y.itype.isin(IONIC).mean(),
                     100 * y.itype.isin(CATPI).mean(),
                     100 * (y.itype == "hydrogen_bonds").mean()))
    ys = range(len(bars))
    for i, (c, n, io, cp, hb) in enumerate(bars):
        left = 0.0
        for val, key in ((io, "ionic"), (cp, "cation-π"), (hb, "hydrogen bond")):
            if val <= 0:
                left += val
                continue
            ax.barh(i, val, left=left, height=.62, color=COL[key],
                    edgecolor=SURF, linewidth=2)       # 2 px surface gap
            if val >= 8:
                ax.text(left + val / 2, i, f"{val:.0f}", ha="center", va="center",
                        color="white", fontsize=8.5, fontweight="bold")
            left += val
        ax.text(101, i, f"n={n}", va="center", fontsize=7.5, color=MUTED)
    ax.set_yticks(list(ys))
    ax.set_yticklabels([b[0] for b in bars], fontsize=8.5, color=INK)
    ax.set_ylim(len(bars) - .5, -.5)   # top-down, so the gradient reads downward
    ax.set_xlim(0, 118)
    ax.set_xticks([0, 50, 100])
    ax.set_xticklabels(["0", "50", "100%"], fontsize=8, color=MUTED)
    ax.set_title(f"Arg{pos}", fontsize=10.5, color=INK, fontweight="bold", pad=8)
    ax.set_facecolor(SURF)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color(RULE)
    ax.tick_params(length=0)
    ax.grid(False)

axes[0].set_xlabel("share of that arginine's amine-side contacts",
                   fontsize=8.5, color=MUTED, loc="left")
fig.legend(handles=[Patch(facecolor=COL[k], label=k) for k in COL],
           loc="lower center", ncol=3, frameon=False, fontsize=8.5,
           bbox_to_anchor=(.5, -.02))
fig.suptitle("A conserved arginine pair re-engages according to the amine's chemistry",
             fontsize=11, color=INK, y=.99)
fig.tight_layout(rect=[0, .07, 1, .93])
for ext in ("png", "pdf"):
    fig.savefig(f"{OUT}/panelB_mode_switch.{ext}", dpi=300,
                facecolor=SURF, bbox_inches="tight")
print(f"wrote {OUT}/panelB_mode_switch.png and .pdf")
for ax, pos in zip(axes, (222, 206)):
    pass
x = m[(m.pos == 222) & (m.resname == "ARG")]
print(f"  Arg222 amine-side records: {len(x)}")
x = m[(m.pos == 206) & (m.resname == "ARG")]
print(f"  Arg206 amine-side records: {len(x)}")
