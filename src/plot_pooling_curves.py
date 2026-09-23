"""Log loss curves for every pooling scheme, from the dev-only sweep.

Each panel is one embedding x one scheme, with all five folds drawn. Early stopping
means folds end at different rounds, so the lines have different lengths -- that is
real information, not a plotting artefact.
"""
from pathlib import Path
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
EMBEDS = ["ProstT5", "ProtT5", "ESM-3", "ESM-2"]
SCHEMES = ["whole mean", "whole mean+max", "conserved >=0.75", "conserved >=0.60",
           "active site only", "non-conserved <0.30", "non-conserved <0.60"]
SERIES = [("train", "#1f77b4"), ("check", "#d62728"), ("held", "#2ca02c")]
NULL_LL = 0.542


def main():
    z = np.load(ROOT / "outputs/pooling_sweep_compact_oof.npz")
    fig, axes = plt.subplots(len(SCHEMES), len(EMBEDS),
                             figsize=(3.6 * len(EMBEDS), 2.5 * len(SCHEMES)),
                             sharex=True, sharey=True)
    for r, sch in enumerate(SCHEMES):
        for c, emb in enumerate(EMBEDS):
            ax = axes[r, c]
            ends = []
            for f in range(5):
                for key, col in SERIES:
                    k = f"curve|{emb}|{sch}|{f}|{key}"
                    if k not in z.files:
                        continue
                    y = z[k]
                    ax.plot(np.arange(len(y)), y, color=col, lw=1, alpha=.75,
                            label=key if (r == 0 and c == 0 and f == 0) else None)
                    if key == "held":
                        ends.append(len(y))
            ax.axhline(NULL_LL, ls=":", c="k", lw=.9)
            ax.grid(alpha=.22)
            if r == 0:
                ax.set_title(emb, fontweight="bold", fontsize=11)
            if c == 0:
                ax.set_ylabel(sch, fontsize=9)
            if ends:
                ax.text(.97, .93, f"stopped {min(ends)}-{max(ends)}", transform=ax.transAxes,
                        ha="right", va="top", fontsize=7.5, color="grey")
            if r == len(SCHEMES) - 1:
                ax.set_xlabel("boosting round", fontsize=9)
    axes[0, 0].set_ylim(0, .75)
    axes[0, 0].legend(fontsize=8, loc="lower left")
    fig.suptitle("Log loss per fold, every pooling scheme — 92 development enzymes\n"
                 "blue = fitted on · red = stopping check · green = held-out fold",
                 fontweight="bold", y=.998, fontsize=12)
    out = ROOT / "images/pooling_curves.png"
    fig.savefig(out, dpi=125, bbox_inches="tight", facecolor="white")
    print(f"-> {out}")

    print(f"\n{'scheme':22s}" + "".join(f"{e:>22s}" for e in EMBEDS))
    print(f"{'':22s}" + "".join(f"{'held-out end / drift':>22s}" for _ in EMBEDS))
    for sch in SCHEMES:
        cells = []
        for emb in EMBEDS:
            hs = [z[f"curve|{emb}|{sch}|{f}|held"] for f in range(5)
                  if f"curve|{emb}|{sch}|{f}|held" in z.files]
            if not hs:
                cells.append("n/a"); continue
            end = np.mean([h[-1] for h in hs])
            drift = np.mean([h[-1] - h.min() for h in hs])
            cells.append(f"{end:.3f} / {drift:+.3f}")
        print(f"{sch:22s}" + "".join(f"{c:>22s}" for c in cells))


if __name__ == "__main__":
    main()
