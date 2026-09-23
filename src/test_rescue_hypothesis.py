"""HYPOTHESIS: using trimmed sequences, does the model predict activity for
proteins the assay calls inactive?

The raw answer is yes. The control asks whether that rise is SPECIFIC to
inactive proteins, or one half of a symmetric collapse toward the mean.
"""
from pathlib import Path
import sys, json, numpy as np, pandas as pd
from scipy.stats import pearsonr, mannwhitneyu
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT/"src"))
import labels as lb
OUT, SITE = ROOT/"outputs", ROOT/"site"; CUT = 0.5
pd.set_option("display.width", 200)

g = pd.read_csv(OUT/"candidate_predictions.csv")
g = g[g.comparable]                      # only proteins with BOTH embeddings
lab = lb.build(min_reps=1)
truth = lab.groupby("Enzyme").active.sum()

P = g.groupby("accession").agg(
    mean_full=("raw_full","mean"), mean_trim=("raw_trimmed","mean"),
    n_full=("raw_full", lambda s:int((s>=CUT).sum())),
    n_trim=("raw_trimmed", lambda s:int((s>=CUT).sum()))).reset_index()
P["truth"] = P.accession.map(truth)
P = P[P.truth.notna()]                   # measured proteins only
P["assay"] = np.where(P.truth==0, "inactive in assay", "active in assay")
P["d_mean"] = P.mean_trim - P.mean_full
P["d_n"] = P.n_trim - P.n_full

print(f"{len(P)} measured proteins with both embeddings "
      f"({(P.assay=='inactive in assay').sum()} inactive, "
      f"{(P.assay=='active in assay').sum()} active)\n")

print("=== 1. the headline: does trimming turn assay-inactives into predicted-actives? ===")
inact = P[P.assay=="inactive in assay"]
print(f"  inactive proteins predicted active (>=1 product) when FULL    : "
      f"{int((inact.n_full>0).sum())} of {len(inact)}")
print(f"  inactive proteins predicted active (>=1 product) when TRIMMED : "
      f"{int((inact.n_trim>0).sum())} of {len(inact)}")
print("  -> taken alone, this supports the hypothesis.\n")

print("=== 2. the control: what happens to the ACTIVE proteins? ===")
print(P.groupby("assay").agg(n=("accession","size"),
      mean_full=("mean_full","mean"), mean_trim=("mean_trim","mean"),
      d_mean=("d_mean","mean"), n_full=("n_full","mean"),
      n_trim=("n_trim","mean"), d_n=("d_n","mean")).to_string(
      float_format=lambda v: f"{v:,.3f}"))
act = P[P.assay=="active in assay"]
print(f"\n  active proteins whose prediction FELL when trimmed: "
      f"{int((act.d_mean<0).sum())} of {len(act)}")
print(f"  inactive proteins whose prediction ROSE when trimmed: "
      f"{int((inact.d_mean>0).sum())} of {len(inact)}")

print("\n=== 3. is this regression to the mean? ===")
r, p = pearsonr(P.mean_full, P.d_mean)
print(f"  correlation between starting score and how much it moved: r = {r:.3f}  (p = {p:.2g})")
print(f"  spread of mean prediction, full    : sd {P.mean_full.std():.4f}")
print(f"  spread of mean prediction, trimmed : sd {P.mean_trim.std():.4f}"
      f"   ({P.mean_trim.std()/P.mean_full.std()-1:+.0%})")
print(f"  base rate the predictions collapse toward: {lab.active.mean():.3f}")
print(f"  mean |prediction - base rate|, full    : "
      f"{(P.mean_full - lab.active.mean()).abs().mean():.4f}")
print(f"  mean |prediction - base rate|, trimmed : "
      f"{(P.mean_trim - lab.active.mean()).abs().mean():.4f}")

print("\n=== 4. do the rescued proteins get PROTEIN-SPECIFIC predictions? ===")
hit = g[(g.raw_trimmed>=CUT) & g.accession.isin(inact.accession)]
prod_rate = lab.groupby(["Amine","Hydroxyl"]).active.mean()
pairs = hit.groupby(["Amine","Hydroxyl"]).size()
top12 = prod_rate.sort_values(ascending=False).head(12).index
print(f"  distinct products predicted for the rescued proteins: {len(pairs)}")
print(f"  how many are among the 12 most commonly made products: "
      f"{sum(1 for k in pairs.index if k in set(top12))} of {len(pairs)}")
print(f"  median prevalence of a predicted product: "
      f"{prod_rate.loc[list(pairs.index)].median():.1%}  "
      f"(median across all {len(prod_rate)} products: {prod_rate.median():.1%})")

print("\n=== 5. could the model have learned 'signal peptide -> inactive'? ===")
per = pd.DataFrame(json.load(open(SITE/"trimming.json"))["per_protein"])
per["truth"] = per.accession.map(truth)
pm = per[per.truth.notna()]
print(f"  measured proteins carrying an annotated signal peptide: "
      f"{int(pm.cut.notna().sum())} of {len(pm)}")
d_, l_ = pm[pm.truth==0], pm[pm.truth>0]
u, pv = mannwhitneyu(d_.cut.dropna(), l_.cut.dropna())
print(f"  peptide length, inactive median {d_.cut.median():.0f} aa vs "
      f"active median {l_.cut.median():.0f} aa  (p = {pv:.2f})")
print("  -> essentially every protein has one, so 'has a signal peptide'")
print("     cannot be the feature that marks a protein as inactive.")
