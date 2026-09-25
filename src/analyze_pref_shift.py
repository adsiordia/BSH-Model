"""Does trimming change which amines a protein prefers, or only the levels?"""
from pathlib import Path
import sys, json, numpy as np, pandas as pd
from scipy.stats import spearmanr, wilcoxon
ROOT = Path(__file__).resolve().parent.parent; sys.path.insert(0, str(ROOT/"src"))
import labels as lb
OUT = ROOT/"outputs"; CUT = 0.5; pd.set_option("display.width", 215)

g = pd.read_csv(OUT/"candidate_predictions.csv")
g = g[g.comparable].copy()
P = pd.DataFrame(json.load(open(ROOT/"site/candidates.json"))["proteins"])
org = dict(zip(P.accession, P.organism.fillna("").str.split().str[:2].str.join(" ")))
lab = lb.build(min_reps=1); truth = lab.groupby("Enzyme").active.sum()
print(f"{g.accession.nunique()} proteins with both versions\n")

# amine-level preference: best score across the three cores
am = g.groupby(["accession","Amine"]).agg(full=("raw_full","max"),
                                          trim=("raw_trimmed","max")).reset_index()

rows = []
for a, s in am.groupby("accession"):
    rho = spearmanr(s.full, s.trim).statistic
    tf = s.loc[s.full.idxmax(), "Amine"]; tt = s.loc[s.trim.idxmax(), "Amine"]
    top3f = set(s.nlargest(3, "full").Amine); top3t = set(s.nlargest(3, "trim").Amine)
    top5f = set(s.nlargest(5, "full").Amine); top5t = set(s.nlargest(5, "trim").Amine)
    rows.append(dict(accession=a, organism=org.get(a,""), rho=rho,
                     top_full=tf, top_trim=tt, top_changed=tf != tt,
                     top3_shared=len(top3f & top3t), top5_shared=len(top5f & top5t),
                     measured=truth.get(a, np.nan)))
R = pd.DataFrame(rows)

print("=== does the single favourite amine change? ===")
print(f"  top amine unchanged : {int((~R.top_changed).sum())} of {len(R)}")
print(f"  top amine CHANGED   : {int(R.top_changed.sum())} of {len(R)}")
print(f"  top-3 set identical : {int((R.top3_shared==3).sum())} of {len(R)}")
print(f"  top-5 set identical : {int((R.top5_shared==5).sum())} of {len(R)}")
print(f"\n  rank correlation across the 25 amines: median {R.rho.median():.3f}, "
      f"min {R.rho.min():.3f}, below 0.90: {int((R.rho<0.90).sum())}")

print("\n=== the proteins whose preference moved most ===")
sh = R.sort_values("rho").head(10)
print(sh[["accession","organism","rho","top_full","top_trim","top3_shared","top5_shared","measured"]]
      .to_string(index=False, float_format=lambda v: f"{v:,.3f}"))

if R.top_changed.any():
    print("\n=== every protein whose FAVOURITE amine changed ===")
    c = R[R.top_changed]
    print(c[["accession","organism","top_full","top_trim","rho","measured"]]
          .to_string(index=False, float_format=lambda v: f"{v:,.3f}"))

print("\n=== amine-level: does any amine systematically gain or lose rank? ===")
am["rank_full"] = am.groupby("accession").full.rank(ascending=False)
am["rank_trim"] = am.groupby("accession").trim.rank(ascending=False)
am["drank"] = am.rank_trim - am.rank_full          # positive = demoted
prev = lab.groupby("Amine").active.mean()
t = am.groupby("Amine").agg(mean_rank_full=("rank_full","mean"),
                            mean_rank_trim=("rank_trim","mean"),
                            drank=("drank","mean")).reset_index()
t["made_by"] = t.Amine.map(prev)
t["p"] = [wilcoxon(s.rank_trim, s.rank_full).pvalue if s.drank.abs().sum() else 1.0
          for _, s in am.groupby("Amine")]
t = t.sort_values("drank")
print(t.to_string(index=False, float_format=lambda v: f"{v:,.3f}"))
print(f"\n  correlation between how common an amine is and how much it is demoted: "
      f"r = {np.corrcoef(t.made_by, t.drank)[0,1]:+.3f}")
