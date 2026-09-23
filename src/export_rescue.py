"""site/rescue.json -- what trimming the signal peptide does to the model's
calls, measured against the assay. Both directions."""
from pathlib import Path
import sys, json, numpy as np, pandas as pd
from scipy.stats import pearsonr
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT/"src"))
import labels as lb
OUT, SITE = ROOT/"outputs", ROOT/"site"; CUT = 0.5

def clean(o):
    if isinstance(o, dict):  return {k: clean(v) for k, v in o.items()}
    if isinstance(o, list):  return [clean(v) for v in o]
    if isinstance(o, (np.floating, float)):
        return None if not np.isfinite(o) else round(float(o), 4)
    if isinstance(o, (np.integer,)): return int(o)
    if isinstance(o, (np.bool_, bool)): return bool(o)
    return o

g = pd.read_csv(OUT/"candidate_predictions.csv")
g = g[g.comparable]
lab = lb.build(min_reps=1)
truth = lab.groupby("Enzyme").active.sum()
prev = lab.groupby(["Amine","Hydroxyl"]).active.mean()

m = g[g.measured.notna()].copy(); m["measured"] = m.measured.astype(int)
ma, mi = m[m.measured == 1], m[m.measured == 0]
on_f, on_t = m.raw_full >= CUT, m.raw_trimmed >= CUT

counts = dict(
    cells=len(m), measured_active=len(ma), measured_inactive=len(mi),
    kept  =int(((ma.raw_full>=CUT)&(ma.raw_trimmed>=CUT)).sum()),
    lost  =int(((ma.raw_full>=CUT)&(ma.raw_trimmed< CUT)).sum()),
    regain=int(((ma.raw_full< CUT)&(ma.raw_trimmed>=CUT)).sum()),
    missed=int(((ma.raw_full< CUT)&(ma.raw_trimmed< CUT)).sum()),
    gained=int(((mi.raw_full< CUT)&(mi.raw_trimmed>=CUT)).sum()),
    false_both=int(((mi.raw_full>=CUT)&(mi.raw_trimmed>=CUT)).sum()),
    cleared=int(((mi.raw_full>=CUT)&(mi.raw_trimmed< CUT)).sum()),
    right_both=int(((mi.raw_full< CUT)&(mi.raw_trimmed< CUT)).sum()))

def pr(col):
    p = m[col] >= CUT
    tp=int((p&(m.measured==1)).sum()); fp=int((p&(m.measured==0)).sum())
    fn=int((~p&(m.measured==1)).sum())
    return dict(calls=tp+fp, hits=tp, false=fp, missed=fn,
                precision=tp/(tp+fp) if tp+fp else None,
                recall=tp/(tp+fn) if tp+fn else None)
scores = dict(full=pr("raw_full"), trimmed=pr("raw_trimmed"))

# per protein, for the scatter
P = g.groupby("accession").agg(
    mean_full=("raw_full","mean"), mean_trim=("raw_trimmed","mean"),
    n_full=("raw_full", lambda s:int((s>=CUT).sum())),
    n_trim=("raw_trimmed", lambda s:int((s>=CUT).sum()))).reset_index()
P["truth"] = P.accession.map(truth)
P = P[P.truth.notna()]
P["assay"] = np.where(P.truth==0, "inactive", "active")
r, pv = pearsonr(P.mean_full, P.mean_trim - P.mean_full)
base = float(lab.active.mean())
regression = dict(
    r=float(r), p=float(pv), n=len(P), base_rate=base,
    sd_full=float(P.mean_full.std()), sd_trim=float(P.mean_trim.std()),
    sd_change=float(P.mean_trim.std()/P.mean_full.std()-1),
    dist_full=float((P.mean_full-base).abs().mean()),
    dist_trim=float((P.mean_trim-base).abs().mean()),
    active_fell=int(((P.assay=="active")&(P.mean_trim<P.mean_full)).sum()),
    active_n=int((P.assay=="active").sum()),
    inactive_rose=int(((P.assay=="inactive")&(P.mean_trim>P.mean_full)).sum()),
    inactive_n=int((P.assay=="inactive").sum()))
by = P.groupby("assay").agg(n=("accession","size"), mean_full=("mean_full","mean"),
     mean_trim=("mean_trim","mean"), n_full=("n_full","mean"),
     n_trim=("n_trim","mean")).reset_index()

def table(d, lo_col, hi_col):
    t = d.groupby(["Amine","Hydroxyl"]).agg(n=("Amine","size"),
            mean_full=("raw_full","mean"), mean_trim=("raw_trimmed","mean")).reset_index()
    t["made_by"] = [float(prev.get((a,h), np.nan)) for a,h in zip(t.Amine,t.Hydroxyl)]
    return t.sort_values("n", ascending=False).to_dict("records")

lost_d = ma[(ma.raw_full>=CUT)&(ma.raw_trimmed<CUT)]
gain_d = mi[(mi.raw_full<CUT)&(mi.raw_trimmed>=CUT)]
kept_d = ma[(ma.raw_full>=CUT)&(ma.raw_trimmed>=CUT)]
kp = prev.loc[list(kept_d.groupby(["Amine","Hydroxyl"]).size().index)]
lp = prev.loc[list(lost_d.groupby(["Amine","Hydroxyl"]).size().index)]

# the enzymes called silent on the full sequence and active once trimmed:
# their full 75-combination ranking, so the site can show what is predicted and in what order
per = pd.DataFrame(json.load(open(SITE/"trimming.json"))["per_protein"])
cutmap = dict(zip(per.accession, per.cut))
silent = P[(P.n_full == 0) & (P.n_trim > 0)].accession.tolist()
detail = []
for a in sorted(silent, key=lambda x: -int(P.loc[P.accession==x, "n_trim"].iloc[0])):
    sub = g[g.accession == a].copy()
    sub["made_by"] = [float(prev.get((am, h), np.nan))
                      for am, h in zip(sub.Amine, sub.Hydroxyl)]
    sub = sub.sort_values("raw_trimmed", ascending=False)
    detail.append(dict(
        accession=a, cut=(None if pd.isna(cutmap.get(a)) else int(cutmap.get(a))),
        n_full=int((sub.raw_full >= CUT).sum()), n_trim=int((sub.raw_trimmed >= CUT).sum()),
        measured_active=int(sub.measured.fillna(0).sum()),
        n_measured=int(sub.measured.notna().sum()),
        products=[dict(amine=r.Amine, core=r.Hydroxyl,
                       full=float(r.raw_full), trim=float(r.raw_trimmed),
                       measured=(None if pd.isna(r.measured) else int(r.measured)),
                       made_by=(None if pd.isna(r.made_by) else float(r.made_by)))
                  for r in sub.itertuples()]))

payload = dict(
    generated=str(pd.Timestamp.today().date()), cut=CUT,
    n_proteins=int(g.accession.nunique()), counts=counts, scores=scores,
    regression=regression, by_assay=by.to_dict("records"),
    proteins=P.to_dict("records"),
    lost=table(lost_d,0,0), gained=table(gain_d,0,0),
    prevalence=dict(lost_median=float(lp.median()), kept_median=float(kp.median()),
                    all_median=float(prev.median()), n_products=int(len(prev))),
    inactive=P[P.assay=="inactive"].to_dict("records"),
    gained_detail=detail)
(SITE/"rescue.json").write_text(json.dumps(clean(payload), allow_nan=False))
print(f"wrote {SITE/'rescue.json'}  ({(SITE/'rescue.json').stat().st_size/1e3:.1f} kB)")
print(f"  {counts['lost']} lost, {counts['gained']} gained, {counts['kept']} kept")
print(f"  regression r={r:.3f}, sd {regression['sd_full']:.4f}->{regression['sd_trim']:.4f}")
print(f"  lost products median prevalence {lp.median():.1%}, kept {kp.median():.1%}")
