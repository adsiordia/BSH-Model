"""Intensity distribution in the Trimmed table: enzymes vs negative controls."""
from pathlib import Path
import sys
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
sys.path.insert(0, str(Path(__file__).resolve().parent))
import trimmed as tr

ROOT = Path(__file__).resolve().parent.parent
SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
ENZ, CTRL = "#2a78d6", "#eb6834"


def style(ax):
    ax.set_facecolor(SURFACE); ax.grid(True, color=GRID, lw=.8, zorder=0)
    ax.set_axisbelow(True)
    for s in ("top", "right"): ax.spines[s].set_visible(False)
    for s in ("left", "bottom"): ax.spines[s].set_color(GRID)
    ax.tick_params(colors=INK2, labelsize=9, length=0)


lg = tr.load_long()
e = tr.enzymes(lg)
nc = tr.negative_controls(lg)
cov = tr.control_coverage(lg)
ctl_products = cov[cov.n_measured > 0].index.tolist()

fig, ax = plt.subplots(3, 1, figsize=(11, 12.5), facecolor=SURFACE)
for a in ax: style(a)
bins = np.linspace(2.4, 7.6, 53)

# A — all enzyme measurements
enz_nz = e.Intensity[e.Intensity > 0]
ax[0].hist(np.log10(enz_nz), bins=bins, color=ENZ, alpha=.8, zorder=3)
ax[0].set_ylabel("measurements", color=INK2, fontsize=10)
ax[0].set_title(f"A · All enzyme measurements — {len(enz_nz):,} detected of {len(e):,} "
                f"({100*(e.Intensity==0).mean():.0f}% are exactly zero)",
                color=INK, fontsize=11.5, fontweight="bold", loc="left", pad=10)
for q, lab in [(50, "median"), (90, "p90"), (99, "p99")]:
    v = np.percentile(enz_nz, q)
    ax[0].axvline(np.log10(v), color=INK2, lw=1, ls=(0, (4, 3)), zorder=4)
    ax[0].annotate(f"{lab}\n{v:,.0f}", (np.log10(v), ax[0].get_ylim()[1]*.95),
                   fontsize=8.5, color=INK2, ha="center", va="top")

# B — the only fair comparison: the 9 products controls cover
sub_e = e[e.ProductName.isin(ctl_products)]
be = sub_e.Intensity[sub_e.Intensity > 0]
bc = nc.Intensity[nc.Intensity > 0]
ax[1].hist(np.log10(be), bins=bins, density=True, color=ENZ, alpha=.75, zorder=3,
           label=f"enzymes on these 9 products (n={len(be)} detected of {len(sub_e)})")
ax[1].hist(np.log10(bc), bins=bins, density=True, color=CTRL, alpha=.85, zorder=4,
           label=f"negative controls (n={len(bc)} detected of {len(nc)})")
ax[1].set_ylabel("density", color=INK2, fontsize=10)
ax[1].set_title("B · Enzymes vs negative controls — restricted to the 9 products "
                "the controls actually cover",
                color=INK, fontsize=11.5, fontweight="bold", loc="left", pad=10)
ax[1].legend(frameon=False, fontsize=9, labelcolor=INK2, loc="upper right")

# C — per-product control coverage
ax[2].axis("off")
rows = [("product", "ctrl measured", "ctrl nonzero", "ctrl max", "enzyme detected")]
for p in ctl_products:
    c = cov.loc[p]; ee = e[e.ProductName == p]
    rows.append((p, f"{int(c.n_measured)}/18", str(int(c.n_nonzero)),
                 f"{c.max_intensity:,.0f}", f"{int((ee.Intensity>0).sum())}/{len(ee)}"))
tbl = ax[2].table(cellText=rows[1:], colLabels=rows[0], loc="upper center", cellLoc="left")
tbl.auto_set_font_size(False); tbl.set_fontsize(8.5); tbl.scale(1, 1.45)
for (r, c), cell in tbl.get_celld().items():
    cell.set_edgecolor(GRID)
    cell.set_text_props(color=INK if r == 0 else INK2,
                        fontweight="bold" if r == 0 else "normal")
    cell.set_facecolor(SURFACE)
ax[2].set_title("C · The 9 products with any negative-control measurement "
                "(the other 72 have none)",
                color=INK, fontsize=11.5, fontweight="bold", loc="left", pad=14)

for a in ax[:2]:
    a.set_xlim(2.4, 7.6); a.set_xticks(range(3, 8))
    a.set_xticklabels([f"1e{i}" for i in range(3, 8)])
ax[1].set_xlabel("LC-MS intensity (log scale, zeros excluded)", color=INK2, fontsize=10)
fig.text(.008, .008, "Source: data/raw/Trimmed_remove_proteomics.csv. "
         "Penicillin amidase excluded from controls (it is a functional enzyme).",
         fontsize=8.5, color=INK2)
fig.tight_layout(rect=(0, .02, 1, 1))
out = ROOT/"images/trimmed_distribution.png"; out.parent.mkdir(exist_ok=True)
fig.savefig(out, dpi=150, facecolor=SURFACE); print(f"wrote {out}")
