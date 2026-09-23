"""Leads among proteins that were never measured, under both sequence versions."""
from pathlib import Path
import pandas as pd
ROOT = Path(__file__).resolve().parents[1]; OUT = ROOT / "outputs"; CUT = 0.5
g = pd.read_csv(OUT / "candidate_predictions.csv")
novel = g[g.measured.isna()]
print(f"never-measured proteins: {novel.entry.nunique()}  ({len(novel):,} cells)\n")
if novel.entry.nunique():
    a = novel.groupby(["entry", "accession"]).agg(
        n_full=("raw_full", lambda s: int((s >= CUT).sum())),
        n_trim=("raw_trimmed", lambda s: int((s >= CUT).sum())),
        max_full=("raw_full", "max"), max_trim=("raw_trimmed", "max")).reset_index()
    a["d"] = a.n_trim - a.n_full
    print(a.sort_values("max_full", ascending=False).to_string(index=False))
print("\n=== agreement between the two versions on never-measured proteins ===")
if len(novel):
    both = ((novel.raw_full >= CUT) & (novel.raw_trimmed >= CUT)).sum()
    only_f = ((novel.raw_full >= CUT) & (novel.raw_trimmed < CUT)).sum()
    only_t = ((novel.raw_full < CUT) & (novel.raw_trimmed >= CUT)).sum()
    print(f"  called by both: {both}   only by full: {only_f}   only by trimmed: {only_t}")
print("\n=== headline counts over every candidate cell ===")
for k, col in (("full", "raw_full"), ("trimmed", "raw_trimmed")):
    print(f"  {k:8s}: {int((g[col] >= CUT).sum())} of {len(g):,} cells called active")
