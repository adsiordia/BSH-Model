"""Write site/limits.json -- what the model can and cannot do, and why.

Three things that belong together:
  1. how much each part of the feature vector actually contributes
  2. the raw intensity distribution the yes/no label is built from
  3. how reproducible the assay is, which sets a ceiling no model can pass

Read back out of the saved artifacts; nothing is refitted here.
"""
from pathlib import Path
import sys, json, numpy as np, pandas as pd, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parent))
import trimmed as tr, labels as lb

ROOT = Path(__file__).resolve().parent.parent
OUT, SITE = ROOT / "outputs", ROOT / "site"

# from the block ablation: same recipe, feature blocks removed one at a time,
# 5-fold cluster-grouped CV over all 115
ABLATION = [
    dict(block="nothing (random guessing)", pr_auc=0.222, roc_auc=0.500, within=0.500),
    dict(block="bile acid type only", pr_auc=0.241, roc_auc=0.544, within=0.503),
    dict(block="the protein only", pr_auc=0.258, roc_auc=0.560, within=0.588),
    dict(block="the amine only", pr_auc=0.628, roc_auc=0.834, within=0.357),
    dict(block="amine + bile acid", pr_auc=0.738, roc_auc=0.911, within=0.347),
    dict(block="everything", pr_auc=0.780, roc_auc=0.924, within=0.645),
]


def main():
    raw = tr.enzymes(tr.load_long())
    raw["Hydroxyl"] = raw.Hydroxyl.map(lb.TO_DEGREE)
    per_rep = raw.groupby(["Code", "Amine", "Hydroxyl", "Replicate"]).Intensity.max().reset_index()
    cell = per_rep.groupby(["Code", "Amine", "Hydroxyl"]).Intensity.max()
    v = cell.to_numpy()
    nz = v[v > 0]

    decades = []
    for e in range(1, 8):
        lo, hi = 10 ** e, 10 ** (e + 1)
        n = int(((nz >= lo) & (nz < hi)).sum())
        if n:
            decades.append(dict(lo=lo, hi=hi, n=n))

    # does a detection show up again in the other runs?
    w = per_rep.pivot_table(index=["Code", "Amine", "Hydroxyl"], columns="Replicate",
                            values="Intensity").fillna(0)
    w.columns = ["r1", "r2", "r3"]
    w = w.reset_index()
    w["peak"] = w[["r1", "r2", "r3"]].max(1)
    repro = []
    for lo, hi in [(1e4, 2.5e4), (2.5e4, 5e4), (5e4, 1e5), (1e5, 2.5e5),
                   (2.5e5, 5e5), (5e5, 1e6), (1e6, 1e13)]:
        s = w[(w.peak >= lo) & (w.peak < hi)]
        if not len(s):
            continue
        n = (s[["r1", "r2", "r3"]] >= 1e4).sum(1)
        repro.append(dict(lo=int(lo), hi=int(hi) if hi < 1e13 else None, cells=int(len(s)),
                          one=round(float((n == 1).mean()), 3),
                          two_plus=round(float((n >= 2).mean()), 3),
                          all_three=round(float((n == 3).mean()), 3)))

    # the same protein sequenced into two separate wells
    cells = pd.read_csv(OUT / "candidate_predictions.csv")
    dup = dict(a="A0A3Q0NGD6", b="Q8Y5J3", agreement=0.827, cells=75)

    # what a confidence cut-off buys you
    oof = pd.read_csv(OUT / "production_model_oof.csv")
    y, p = oof.active.to_numpy(int), oof.calibrated.to_numpy()
    cutoffs = []
    for t in [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
        pred = p >= t
        tp = int((pred & (y == 1)).sum()); fp = int((pred & (y == 0)).sum())
        fn = int((~pred & (y == 1)).sum())
        prec = tp / max(tp + fp, 1); rec = tp / max(tp + fn, 1)
        cutoffs.append(dict(cutoff=t, called=tp + fp, share=round((tp + fp) / len(y), 4),
                            precision=round(prec, 4), recall=round(rec, 4),
                            f1=round(2 * prec * rec / max(prec + rec, 1e-9), 4)))

    plateau = [r for r in repro if r["lo"] >= 50_000]
    doc = dict(
        generated=pd.Timestamp.today().strftime("%Y-%m-%d"),
        ablation=ABLATION,
        ablation_note="Same recipe, feature blocks removed one at a time, cross-validated "
                      "over all 115 enzymes. Describes where the performance comes from; "
                      "it is not a design decision, so it was not restricted to development.",
        intensity=dict(cells=int(len(v)), zero=int((v == 0).sum()),
                       nonzero=int(len(nz)), min=int(nz.min()), max=int(nz.max()),
                       decades=decades,
                       pct={str(q): int(np.percentile(nz, q))
                            for q in (1, 5, 10, 25, 50, 75, 90, 95, 99)},
                       cutoff=int(lb.THRESHOLD),
                       cutoff_percentile=round(float((nz < lb.THRESHOLD).mean()), 3),
                       control_max=10_320),
        reproducibility=repro,
        plateau=dict(floor=50_000,
                     low=round(float(np.mean([r["two_plus"] for r in repro if r["lo"] < 50_000])), 3),
                     high=round(float(np.mean([r["two_plus"] for r in plateau])), 3)),
        duplicate=dup,
        model_precision=0.78,
        cutoffs=cutoffs)
    p_ = SITE / "limits.json"
    p_.write_text(json.dumps(doc, separators=(",", ":"), allow_nan=False))
    print(f"wrote {p_} ({p_.stat().st_size/1e3:.0f} kB)")
    print(f"  intensity {doc['intensity']['min']:,}-{doc['intensity']['max']:,}, "
          f"{doc['intensity']['zero']:,} zeros of {doc['intensity']['cells']:,}")
    print(f"  reproducibility below 50k {doc['plateau']['low']:.0%}, "
          f"above {doc['plateau']['high']:.0%}")


if __name__ == "__main__":
    main()
