"""Does feeding the model trimmed sequences make it more or less right?
Scored against what was actually measured."""
from pathlib import Path
import json, numpy as np, pandas as pd, pickle, sys
from sklearn.metrics import average_precision_score, roc_auc_score, log_loss

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
OUT = ROOT / "outputs"
CUT = 0.5

g = pd.read_csv(OUT / "candidate_predictions.csv")
m = g[g.measured.notna()].copy()
m["measured"] = m.measured.astype(int)
print(f"{len(m):,} measured cells across {m.entry.nunique()} proteins "
      f"({m.measured.mean():.3f} active)\n")

def score(d, tag):
    y = d.measured.values
    if y.sum() == 0 or y.sum() == len(y):
        return None
    r = {"n": len(d), "base": y.mean()}
    for k, col in (("full", "raw_full"), ("trim", "raw_trimmed")):
        p = d[col].values
        r[f"pr_{k}"]  = average_precision_score(y, p)
        r[f"roc_{k}"] = roc_auc_score(y, p)
        r[f"ll_{k}"]  = log_loss(y, np.clip(p, 1e-6, 1 - 1e-6))
        r[f"acc_{k}"] = ((p >= CUT).astype(int) == y).mean()
    return tag, r

print("=== scored against measured truth ===")
hdr = f"{'subset':14s} {'n':>6s} {'base':>6s} | {'PR full':>8s} {'PR trim':>8s} | " \
      f"{'ROC full':>8s} {'ROC trim':>8s} | {'LL full':>8s} {'LL trim':>8s}"
print(hdr); print("-" * len(hdr))
for tag, d in [("all measured", m), ("dev", m[m.split == "dev"]),
               ("test (honest)", m[m.split == "test"])]:
    s = score(d, tag)
    if not s: continue
    _, r = s
    print(f"{tag:14s} {r['n']:6,d} {r['base']:6.3f} | {r['pr_full']:8.4f} {r['pr_trim']:8.4f} | "
          f"{r['roc_full']:8.4f} {r['roc_trim']:8.4f} | {r['ll_full']:8.4f} {r['ll_trim']:8.4f}")

print("\n=== calls at 0.5, against truth ===")
for tag, d in [("all measured", m), ("test (honest)", m[m.split == "test"])]:
    for k, col in (("full", "raw_full"), ("trimmed", "raw_trimmed")):
        p = (d[col] >= CUT).astype(int); y = d.measured
        tp = int(((p == 1) & (y == 1)).sum()); fp = int(((p == 1) & (y == 0)).sum())
        fn = int(((p == 0) & (y == 1)).sum())
        prec = tp / (tp + fp) if tp + fp else float("nan")
        rec = tp / (tp + fn) if tp + fn else float("nan")
        print(f"  {tag:14s} {k:8s}  calls {tp+fp:4d}  hits {tp:4d}  false {fp:4d}  "
              f"missed {fn:4d}  precision {prec:.3f}  recall {rec:.3f}")

print("\n=== is having a signal peptide associated with being inactive? ===")
per = pd.DataFrame(json.load(open(ROOT / "site/trimming.json"))["per_protein"])
truth = m.groupby(["entry", "accession"]).measured.agg(["sum", "size"]).reset_index()
truth = truth.merge(per[["entry", "cut", "full_len"]], on="entry", how="left")
truth["rate"] = truth["sum"] / truth["size"]
print(f"  proteins with a signal peptide annotated: {truth.cut.notna().sum()} "
      f"of {len(truth)} measured")
if truth.cut.notna().any():
    t = truth[truth.cut.notna()]
    dead = t[t["sum"] == 0]; live = t[t["sum"] > 0]
    print(f"  made nothing at all : {len(dead):3d} proteins, median cut "
          f"{dead.cut.median():.0f} aa, median length {dead.full_len.median():.0f}")
    print(f"  made something      : {len(live):3d} proteins, median cut "
          f"{live.cut.median():.0f} aa, median length {live.full_len.median():.0f}")
    if len(dead) and len(live):
        from scipy.stats import mannwhitneyu
        u, p = mannwhitneyu(dead.cut, live.cut)
        print(f"  cut length, dead vs live: Mann-Whitney p = {p:.3f}")

print("\n=== how many proteins does each version call completely silent? ===")
for k, col in (("full", "raw_full"), ("trimmed", "raw_trimmed")):
    n = g.groupby("entry")[col].max().lt(CUT).sum()
    print(f"  {k:8s}: {n} of {g.entry.nunique()} proteins predicted to make nothing")
