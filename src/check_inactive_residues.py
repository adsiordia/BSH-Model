"""Orthogonal check, read straight from the MSA: do the assay-inactive proteins
carry the invariant residues, and how complete is their alignment coverage?"""
from pathlib import Path
import sys, numpy as np, pandas as pd
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT/"src"))
import labels as lb
D = ROOT/"data/derived"; pd.set_option("display.width", 200)

# read the alignment
seqs, name, buf = {}, None, []
for line in (D/"msa_linsi.fasta").read_text().splitlines():
    if line.startswith(">"):
        if name: seqs[name] = "".join(buf)
        name, buf = line[1:].strip().split()[0], []
    else: buf.append(line.strip())
seqs[name] = "".join(buf)
ids = list(seqs); M = np.array([list(seqs[i]) for i in ids])
print(f"alignment: {M.shape[0]} proteins x {M.shape[1]} columns")

acc = {i: i.split("_")[-1] for i in ids}
lab = lb.build(min_reps=1); truth = lab.groupby("Enzyme").active.sum()
state = np.array(["active" if truth.get(acc[i], np.nan) > 0 else
                  ("INACTIVE" if truth.get(acc[i], -1) == 0 else "?") for i in ids])
print(f"  inactive {int((state=='INACTIVE').sum())}, active {int((state=='active').sum())}, "
      f"unmeasured {int((state=='?').sum())}\n")

gap = M == "-"
occ = 1 - gap.mean(0)
# a column every single protein occupies with the same residue
inv = []
for c in np.where(occ >= 0.999)[0]:
    col = M[:, c]
    u = set(col) - {"-"}
    if len(u) == 1: inv.append((c, u.pop()))
print(f"columns present in every protein with a single identical residue: {len(inv)}")
print("  " + ", ".join(f"{r}@col{c}" for c, r in inv))

print("\n=== those residues in the inactive proteins ===")
cols = [c for c, _ in inv]; want = [r for _, r in inv]
hdr = f"{'protein':14s} {'state':9s} " + " ".join(f"c{c}" for c in cols) + "   matches"
print(hdr); print("-"*len(hdr))
for k, i in enumerate(ids):
    if state[k] != "INACTIVE": continue
    got = [M[k, c] for c in cols]
    print(f"{acc[i]:14s} {'INACTIVE':9s} " + "  ".join(f"{g:>3s}" for g in got) +
          f"   {sum(1 for g,w in zip(got,want) if g==w)}/{len(cols)}")

print("\n=== alignment coverage: how much of the family fold each protein spans ===")
cov = 1 - gap.mean(1)
t = pd.DataFrame({"coverage": cov, "state": state, "acc": [acc[i] for i in ids]})
print(t[t.state!="?"].groupby("state").coverage.describe()[
      ["count","mean","50%","min"]].to_string(float_format=lambda v: f"{v:.3f}"))
from scipy.stats import mannwhitneyu
d_, l_ = t[t.state=="INACTIVE"].coverage, t[t.state=="active"].coverage
u, p = mannwhitneyu(d_, l_)
print(f"\n  inactive vs active coverage: Mann-Whitney p = {p:.3f}")
print("\n  the inactive proteins, least complete first:")
print(t[t.state=="INACTIVE"].sort_values("coverage")[["acc","coverage"]].to_string(
      index=False, float_format=lambda v: f"{v:.3f}"))
print(f"\n  for reference, active proteins span {l_.min():.3f}-{l_.max():.3f} "
      f"(median {l_.median():.3f})")
