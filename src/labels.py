"""Activity labels from the Trimmed proteomics table.

RULE
    A product is called present if its intensity exceeds THRESHOLD in at least
    MIN_REPS of the 3 replicates.

    THRESHOLD = 50,000, raised from 10,000 on 2026-09-22.

    The old 10,000 came from the negative controls, but that evidence is thinner
    than it looks: of 1,458 control measurements only NINE are non-zero, and the two
    that set the bound (10,059 and 10,320) fall on just 2 of 55 products. Every other
    product's control read zero, which says nothing about where its noise floor sits.

    50,000 rests on detection reproducibility, measured across all 2,467 non-zero
    cells rather than two products. The fraction of detections that reappear in a
    second replicate climbs steeply and then stops:

        peak intensity        seen again in >=2 of 3 runs
        10,000 -  25,000                 34%
        25,000 -  50,000                 66%
        50,000 - 100,000                 82%   <- plateau
        over 1,000,000                   82%

    Above 50,000 a detection is as reliable as it will ever be. 10^4 counts is also
    weak signal by mass-spectrometry convention.

    This costs positives: 1,915 -> 1,381 at min_reps=1. A five-way comparison of
    thresholds found NO measurable difference in model performance between 10,000,
    25,000 and 50,000 (paired over 22 substrate groups, every interval spanning
    zero), so the choice rests on the evidence behind the label, not on a model gain.

    MIN_REPS: one-off detections are 8x weaker than reproducible ones
    (median max-intensity 12,234 for 1-of-3 vs 97,670 for 3-of-3), and 32% of all
    detections appear in a single replicate. Requiring 2 removes them.

COMPLETE GRID
    Amines and bile acids were both pooled -- one Amine Master Mix and one Bile Acid
    Master Mix went into every well with a single enzyme. So every enzyme had the
    opportunity to make every amine x core combination, and the panel is complete
    by construction.

    The product table lists 55 (amine, core) cells, not the full 75. Those 55 are
    what LC-MS feature detection FOUND: product names carry feature ids, every one
    of the 82 product columns has at least one non-zero detection, and the
    enumeration file annotates detected features rather than listing planned
    targets. A combination absent from the table therefore produced no detectable
    feature in ANY of the 366 samples -- which is a negative for every enzyme, at
    the same operational definition as a below-threshold measurement.

    full_grid=True adds those combinations as explicit negatives.

CAVEAT carried with these labels: negative controls exist for only 9 of the 81
products in this table, and those 9 are low-signal products. Background on the
high-signal products is untested.

Feature ids are merged first: 17 of 55 (amine, core) cells were detected under
2-4 feature ids. The feature vector is [enzyme | amine | core] and has no slot
for a feature id, so those rows would be identical in X yet could disagree in y.
Intensity is maxed across feature ids within each replicate before thresholding.
"""
from pathlib import Path
import numpy as np, pandas as pd

import trimmed as tr

THRESHOLD = 50_000.0
MIN_REPS = 2
N_REPS = 3

# Degree class = number of HYDROXYLS only; keto groups are not counted. So
# 3a,7a,12k is Di (3a and 7a are hydroxyls, 12k is a ketone) and 3a7k is Mono.
# Grouping to degree keeps every measurement while removing core categories that
# were each backed by only 1-4 amines and could not be learned or extrapolated to.
TO_DEGREE = {"Mono": "Mono", "3a7k": "Mono", "3k7a": "Mono",
             "3a12k": "Mono", "3k12a": "Mono",
             "Di": "Di", "3a,7a,12k": "Di",
             "Tri": "Tri"}


def build(threshold=THRESHOLD, min_reps=MIN_REPS, collapse=True, drop_cores=(),
          group_to_degree=True, full_grid=True):
    """One row per (enzyme, amine, core) -- or per product if collapse=False.

    Returns Enzyme, Amine, Hydroxyl, active, n_reps_above, max_intensity, n_features.
    """
    e = tr.enzymes(tr.load_long())
    if drop_cores:
        e = e[~e.Hydroxyl.isin(drop_cores)]
    if group_to_degree:
        e = e.assign(Hydroxyl=e.Hydroxyl.map(TO_DEGREE))
    keys = ["Code", "Amine", "Hydroxyl"] if collapse else ["Code", "ProductName", "Amine", "Hydroxyl"]

    # max across feature ids within each replicate, then count replicates above threshold
    per_rep = e.groupby(keys + ["Replicate"]).agg(
        Intensity=("Intensity", "max"),
        n_features=("ProductName", "nunique")).reset_index()
    per_rep["above"] = per_rep["Intensity"] > threshold

    out = per_rep.groupby(keys).agg(
        n_reps_above=("above", "sum"),
        max_intensity=("Intensity", "max"),
        n_features=("n_features", "max")).reset_index()
    out["active"] = out["n_reps_above"] >= min_reps
    out = out.rename(columns={"Code": "Enzyme"})
    out["detected"] = True
    if not full_grid or not collapse:
        return out

    # combinations no feature was ever detected for -> negative for every enzyme
    import itertools
    full = pd.DataFrame(itertools.product(sorted(out.Enzyme.unique()),
                                          sorted(out.Amine.unique()),
                                          sorted(out.Hydroxyl.unique())),
                        columns=["Enzyme", "Amine", "Hydroxyl"])
    out = full.merge(out, on=["Enzyme", "Amine", "Hydroxyl"], how="left")
    out["detected"] = out["detected"].fillna(False)
    out["active"] = out["active"].fillna(False).astype(bool)
    out["n_reps_above"] = out["n_reps_above"].fillna(0).astype(int)
    out["max_intensity"] = out["max_intensity"].fillna(0.0)
    out["n_features"] = out["n_features"].fillna(0).astype(int)
    return out


if __name__ == "__main__":
    for fg in (False, True):
        a = build(min_reps=1, full_grid=fg)
        print(f"  full_grid={str(fg):5s}: {len(a):>5,} cells, {int(a.active.sum()):>5,} active "
              f"({100*a.active.mean():>5.1f}%)   searched-for cells: {int(a.detected.sum()):,}")
    print()
    for mr in (1, 2, 3):
        a = build(min_reps=mr)
        dead = (a.groupby("Enzyme").active.sum() == 0).sum()
        print(f"  >{THRESHOLD:,.0f} in >={mr} of 3 reps: {int(a.active.sum()):>5,} active "
              f"of {len(a):,} ({100*a.active.mean():>5.1f}%)   "
              f"enzymes with no activity: {dead}/{a.Enzyme.nunique()}")
    a = build()
    print(f"\n  cells {len(a):,} = {a.Enzyme.nunique()} enzymes x "
          f"{a.groupby(['Amine','Hydroxyl']).ngroups} products")
    print(f"  cores {sorted(a.Hydroxyl.unique())}")


def feature_summary(threshold=None, min_reps=1):
    """How feature ids are collapsed, and whether the choice of rule matters.

    Several LC-MS features can map to the same (amine, core) product -- different
    ids for what the table calls e.g. di_alanine. build() takes the MAX across
    feature ids within each replicate before thresholding. This reports how many
    products that affects and how many labels would change under SUM instead.
    """
    th = THRESHOLD if threshold is None else threshold
    e = tr.enzymes(tr.load_long())
    e = e.assign(Hydroxyl=e.Hydroxyl.map(TO_DEGREE))
    e = e[e.Hydroxyl.notna()]
    keys = ["Code", "Amine", "Hydroxyl"]

    nf = e.groupby(["Amine", "Hydroxyl"]).ProductName.nunique()

    def rule(how):
        pr = e.groupby(keys + ["Replicate"]).agg(Intensity=("Intensity", how)).reset_index()
        pr["above"] = pr.Intensity > th
        o = pr.groupby(keys).agg(n_reps_above=("above", "sum"),
                                 peak=("Intensity", "max")).reset_index()
        o["active"] = o.n_reps_above >= min_reps
        return o.set_index(keys)

    mx, sm = rule("max"), rule("sum")
    j = mx.join(sm, lsuffix="_max", rsuffix="_sum")
    flip = j[j.active_max != j.active_sum]
    multi = [k for k in j.index if nf.get((k[1], k[2]), 1) > 1]
    sub = j.loc[multi]
    ratio = (sub.peak_sum / sub.peak_max.replace(0, np.nan)).dropna()

    return dict(
        rule="max within each replicate, then threshold",
        threshold=float(th), min_reps=int(min_reps),
        n_products=int(len(nf)),
        n_multi=int((nf > 1).sum()),
        max_features=int(nf.max()),
        multi_products=[dict(amine=a, core=c, features=int(v))
                        for (a, c), v in nf[nf > 1].sort_values(ascending=False).items()],
        active_max=int(j.active_max.sum()), active_sum=int(j.active_sum.sum()),
        n_disagree=int(len(flip)), pct_disagree=float(len(flip) / len(j)),
        disagree_peak_max=(None if not len(flip) else float(flip.peak_max.median())),
        disagree_peak_sum=(None if not len(flip) else float(flip.peak_sum.median())),
        one_dominates=float((ratio < 1.01).mean()) if len(ratio) else None,
        ratio_median=float(ratio.median()) if len(ratio) else None,
        ratio_p90=float(ratio.quantile(.9)) if len(ratio) else None)
