"""Proteins the model calls silent on the full sequence but active once the
signal peptide is cut. Which are real leads, and which are the model simply
losing its opinion?"""
from pathlib import Path
import json, numpy as np, pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT, SITE = ROOT / "outputs", ROOT / "site"
CUT = 0.5

g = pd.read_csv(OUT / "candidate_predictions.csv")
tj = json.load(open(SITE / "trimming.json"))
per = pd.DataFrame(tj["per_protein"])          # rho, cos, full_len, cut
act = pd.DataFrame(tj["activity"]["per_protein"])  # nf, nt, d, mean_full, mean_trim
P = act.merge(per[["entry", "rho", "cos", "full_len", "cut", "top10"]],
              on="entry", how="left")

silent = P[(P.nf == 0) & (P.nt > 0)].sort_values("nt", ascending=False)
print(f"=== proteins silent on the full sequence, active once trimmed: {len(silent)} ===")
print(f"(population: {len(P)} proteins, median products predicted when trimmed "
      f"{P.nt.median():.0f}, sd {P.nt.std():.1f})\n")

rows = []
for r in silent.itertuples():
    sub = g[g.entry == r.entry]
    m = sub[sub.measured.notna()]
    truth = int(m.measured.sum()) if len(m) else None
    # did trimming move the call toward or away from what was measured?
    if len(m):
        err_f = float((( m.raw_full  >= CUT).astype(int) - m.measured).abs().mean())
        err_t = float((( m.raw_trimmed>= CUT).astype(int) - m.measured).abs().mean())
        rec_f = int(((m.raw_full   >= CUT) & (m.measured == 1)).sum())
        rec_t = int(((m.raw_trimmed>= CUT) & (m.measured == 1)).sum())
        fp_f  = int(((m.raw_full   >= CUT) & (m.measured == 0)).sum())
        fp_t  = int(((m.raw_trimmed>= CUT) & (m.measured == 0)).sum())
    else:
        err_f = err_t = rec_f = rec_t = fp_f = fp_t = None
    rows.append(dict(entry=r.entry, accession=r.accession, split=r.split,
                     n_measured=len(m), truth_active=truth,
                     nt=int(r.nt), mean_full=r.mean_full, mean_trim=r.mean_trim,
                     max_trim=float(sub.raw_trimmed.max()),
                     cut=r.cut, full_len=r.full_len, cos=r.cos,
                     hits_full=rec_f, hits_trim=rec_t,
                     false_full=fp_f, false_trim=fp_t,
                     err_full=err_f, err_trim=err_t))
S = pd.DataFrame(rows)

pd.set_option("display.width", 200)
print(S[["accession", "split", "truth_active", "n_measured", "nt", "mean_full",
         "mean_trim", "max_trim", "cut", "cos"]].to_string(index=False))

print("\n=== did trimming move them toward the truth? (measured proteins only) ===")
M = S[S.n_measured > 0]
if len(M):
    print(M[["accession", "split", "truth_active", "hits_full", "hits_trim",
             "false_full", "false_trim", "err_full", "err_trim"]].to_string(index=False))
    print(f"\n  real products recovered : {M.hits_full.sum():>3} -> {M.hits_trim.sum():>3}")
    print(f"  false calls added       : {M.false_full.sum():>3} -> {M.false_trim.sum():>3}")
    print(f"  mean error rate         : {M.err_full.mean():.3f} -> {M.err_trim.mean():.3f}")

print("\n=== is this a real activation, or the model losing its opinion? ===")
pop_nt, pop_mean = P.nt.median(), P.mean_trim.median()
print(f"  population median when trimmed : {pop_nt:.0f} products, mean prob {pop_mean:.3f}")
print(f"  the silent->active set         : {S.nt.median():.0f} products, mean prob {S.mean_trim.median():.3f}")
print(f"  their strongest single call    : max prob {S.max_trim.max():.3f} "
      f"(median across the set {S.max_trim.median():.3f})")
print(f"  population strongest call      : median {g.groupby('entry').raw_trimmed.max().median():.3f}")

print("\n=== the actual new calls, ranked by confidence ===")
new = g[g.entry.isin(silent.entry) & (g.raw_trimmed >= CUT)].copy()
new["gain"] = new.raw_trimmed - new.raw_full
new = new.sort_values("raw_trimmed", ascending=False)
show = new[["accession", "Amine", "Hydroxyl", "measured", "raw_full",
            "raw_trimmed", "gain"]].head(25)
print(show.to_string(index=False))
print(f"\n  {len(new)} new calls total; "
      f"{int(new.measured.eq(1).sum())} were measured active, "
      f"{int(new.measured.eq(0).sum())} measured inactive, "
      f"{int(new.measured.isna().sum())} never measured")

print("\n=== which amines do the new calls land on? ===")
print(new.Amine.value_counts().head(10).to_string())

S.to_csv(OUT / "gained_proteins.csv", index=False)
new.to_csv(OUT / "gained_calls.csv", index=False)
print(f"\nwrote {OUT/'gained_proteins.csv'} and {OUT/'gained_calls.csv'}")
