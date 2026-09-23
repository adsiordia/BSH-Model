"""Does collapsing feature ids by MAX rather than SUM change the labels?"""
from pathlib import Path
import sys, numpy as np, pandas as pd
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT/"src"))
import labels as lb, trimmed as tr
pd.set_option("display.width", 200)
TH, MINREP = lb.THRESHOLD, 1

e = tr.enzymes(tr.load_long())


e = e.assign(Hydroxyl=e.Hydroxyl.map(lb.TO_DEGREE))
e = e[e.Hydroxyl.notna()]
keys = ["Code","Amine","Hydroxyl"]

nf = e.groupby(["Amine","Hydroxyl"]).ProductName.nunique()
print(f"\n(amine, core) cells: {len(nf)}")
print(f"  with 1 feature id : {(nf==1).sum()}")
print(f"  with >1           : {(nf>1).sum()}   (max {nf.max()})")
print("\nthe multi-feature products:")
print(nf[nf>1].sort_values(ascending=False).to_string())

def build(how):
    pr = e.groupby(keys+["Replicate"]).agg(Intensity=("Intensity", how)).reset_index()
    pr["above"] = pr.Intensity > TH
    o = pr.groupby(keys).agg(n_reps_above=("above","sum"),
                             max_intensity=("Intensity","max")).reset_index()
    o["active"] = o.n_reps_above >= MINREP
    return o.set_index(keys)

mx, sm = build("max"), build("sum")
j = mx.join(sm, lsuffix="_max", rsuffix="_sum")
print(f"\n=== labels under each rule (threshold {TH:,.0f}, min_reps {MINREP}) ===")
print(f"  active by MAX : {int(j.active_max.sum()):,} of {len(j):,}")
print(f"  active by SUM : {int(j.active_sum.sum()):,} of {len(j):,}")
flip = j[j.active_max != j.active_sum]
print(f"  cells that disagree: {len(flip)}  ({len(flip)/len(j):.2%})")
if len(flip):
    print("\n  where they differ:")
    f = flip.reset_index()
    print(f.groupby(["Amine","Hydroxyl"]).size().sort_values(ascending=False).to_string())
    print(f"\n  all are SUM-active / MAX-inactive: "
          f"{bool((flip.active_sum & ~flip.active_max).all())}")
    print(f"  their intensities: max-rule median {flip.max_intensity_max.median():,.0f}, "
          f"sum-rule median {flip.max_intensity_sum.median():,.0f}")

print("\n=== how much does summing inflate multi-feature cells? ===")
multi = [k for k in j.index if nf.get((k[1],k[2]), 1) > 1]
if multi:
    sub = j.loc[multi]
    r = (sub.max_intensity_sum / sub.max_intensity_max.replace(0, np.nan)).dropna()
    print(f"  {len(sub):,} cells belong to a multi-feature product")
    print(f"  sum/max ratio: median {r.median():.2f}, 90th pct {r.quantile(.9):.2f}, max {r.max():.2f}")
    print(f"  ratio == 1 (one feature dominates): {(r<1.01).mean():.1%} of them")
