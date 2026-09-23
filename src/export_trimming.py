"""Write site/trimming.json -- the trimmed vs untrimmed comparison.

Read back out of outputs/candidate_predictions.csv and the two ProstT5 h5 files, so
the page cannot drift from what was actually computed.
"""
from pathlib import Path
import json, pickle, h5py, numpy as np, pandas as pd, warnings
warnings.filterwarnings("ignore")
from scipy.stats import spearmanr
from sklearn.metrics import average_precision_score, roc_auc_score

import embeddings as EM

ROOT = Path(__file__).resolve().parent.parent
OUT, SITE = ROOT / "outputs", ROOT / "site"
_E = Path(__file__).resolve().parent.parent / "data/embeddings"
H = _E if (_E / "Seqs_list_total_prost5.h5").exists() else Path.home() / "prost5_embeddings"
GRID = 44           # resolution of the 2-D density used for the scatter


def read_h5(p):
    with h5py.File(p, "r") as f:
        # int16 in the file, bfloat16 bit patterns in fact -- see embeddings._bf16
        return {k: EM._bf16(f[k][:]) for k in f.keys()}


def activity(c):
    """Does the model's yes/no answer change -- not just the order of its answers."""
    af = (c.raw_full >= .5).to_numpy()
    at = (c.raw_trimmed >= .5).to_numpy()
    p = c.assign(af=af, at=at).groupby("entry").agg(
        nf=("af", "sum"), nt=("at", "sum"), split=("split", "first"),
        mean_full=("raw_full", "mean"), mean_trim=("raw_trimmed", "mean"))
    p["d"] = p.nt - p.nf

    spread = {}
    for label, sub in [("all", p), ("dev", p[p.split == "dev"]), ("test", p[p.split == "test"])]:
        # dev/test is kept only to label the bars; the production model trained on both
        if len(sub) < 2:
            continue
        spread[label] = dict(n=int(len(sub)),
                             full_min=int(sub.nf.min()), full_max=int(sub.nf.max()),
                             full_sd=round(float(sub.nf.std()), 3),
                             trim_min=int(sub.nt.min()), trim_max=int(sub.nt.max()),
                             trim_sd=round(float(sub.nt.std()), 3),
                             change=round(float(sub.nt.std() / sub.nf.std() - 1), 4),
                             silent_full=int((sub.nf == 0).sum()),
                             silent_trim=int((sub.nt == 0).sum()))

    rows = [dict(entry=e, accession=r.name.split("_")[-1] if False else None,
                 nf=int(r.nf), nt=int(r.nt), d=int(r.d),
                 split=None if pd.isna(r.split) else str(r.split),
                 mean_full=round(float(r.mean_full), 4), mean_trim=round(float(r.mean_trim), 4))
            for e, r in p.iterrows()]
    for row in rows:
        row["accession"] = row["entry"].split("_")[-1]
    rows.sort(key=lambda r: r["d"])

    return dict(
        both_active=int((af & at).sum()), both_inactive=int((~af & ~at).sum()),
        lost=int((af & ~at).sum()), gained=int((~af & at).sum()),
        total=int(len(c)),
        unchanged_count=int((p.d == 0).sum()),
        changed_1_5=int(((p.d.abs() > 0) & (p.d.abs() <= 5)).sum()),
        changed_6_plus=int((p.d.abs() > 5).sum()),
        n_proteins=int(len(p)),
        spread=spread, per_protein=rows,
        threshold=0.5, of=int(c.groupby("entry").size().iloc[0]))


def worked_example(c, per):
    """The protein whose ordering moved most -- its top combinations, both ways.

    The point of showing it is that the order survives while the scores do not.
    """
    e = min(per, key=lambda r: r["rho"])["entry"]
    s = c[c.entry == e].copy()
    s["rank_full"] = s.raw_full.rank(ascending=False).astype(int)
    s["rank_trim"] = s.raw_trimmed.rank(ascending=False).astype(int)
    top = s.nsmallest(12, "rank_full")
    return dict(entry=e, rows=[dict(amine=r.Amine, core=r.Hydroxyl,
                                    full=round(float(r.raw_full), 4),
                                    trim=round(float(r.raw_trimmed), 4),
                                    rank_full=int(r.rank_full), rank_trim=int(r.rank_trim))
                               for _, r in top.iterrows()],
                kept=int((top.rank_trim <= 12).sum()))


def main():
    g = pd.read_csv(OUT / "candidate_predictions.csv")
    c = g[g.comparable].copy()
    man = pd.read_csv(ROOT / "data/predictions_data/signaling_peptide/"
                      "signal_peptide_manifest.csv").set_index("id")

    # 2-D density instead of 4,875 individual points: same picture, a fraction of the size
    xb = np.clip((c.raw_full * GRID).astype(int), 0, GRID - 1)
    yb = np.clip((c.raw_trimmed * GRID).astype(int), 0, GRID - 1)
    dens = pd.Series(1, index=pd.MultiIndex.from_arrays([xb, yb])).groupby(level=[0, 1]).size()
    cells = [dict(x=int(i), y=int(j), n=int(v)) for (i, j), v in dens.items()]

    per = []
    for e, s in c.groupby("entry"):
        rho = float(spearmanr(s.raw_full, s.raw_trimmed).statistic)
        tf = set(s.nlargest(10, "raw_full").index); tt = set(s.nlargest(10, "raw_trimmed").index)
        sp = s.split.iloc[0]
        row = dict(entry=e, accession=s.accession.iloc[0],
                   # the 8 proteins that are not training enzymes have no split; NaN is not
                   # valid JSON, so carry them as null
                   split=None if pd.isna(sp) else str(sp),
                   rho=round(rho, 4), top10=len(tf & tt),
                   n_full=int((s.raw_full >= .5).sum()), n_trim=int((s.raw_trimmed >= .5).sum()),
                   flips=int(((s.raw_full >= .5) != (s.raw_trimmed >= .5)).sum()),
                   cos=round(float(s.embedding_cosine.iloc[0]), 4),
                   mean_abs_delta=round(float(s.delta.abs().mean()), 4))
        if e in man.index:
            row.update(full_len=int(man.loc[e, "full_length"]),
                       cut=int(man.loc[e, "signal_peptide_length"]))
        per.append(row)
    per.sort(key=lambda r: r["rho"])

    # Accuracy against measured data. The production model trained on all 115 enzymes,
    # so every candidate with a measurement is one it has memorised -- the full-sequence
    # score is recall, not prediction, and there is no unmemorised subset left to
    # compare against. The numbers are reported with that stated rather than split.
    m = c[c.measured.notna()].copy(); m["measured"] = m.measured.astype(int)
    y = m.measured.to_numpy()
    acc = [dict(split="all", enzymes=int(m.accession.nunique()), rows=int(len(m)),
                base_rate=round(float(y.mean()), 4),
                **{f"{v}_{k}": round(float(f(y, m[f"raw_{v}"].to_numpy())), 4)
                   for v in ("full", "trimmed")
                   for k, f in [("pr", average_precision_score), ("roc", roc_auc_score)]})]
    ci = None

    # did the vector change by more than the dropped residues can account for?
    F, T = read_h5(H / "Seqs_list_total_prost5.h5"), read_h5(H / "proteins_without_signal_peptides_trimmed.h5")
    ratios, cuts = [], []
    for e in c.entry.unique():
        if e not in F or e not in man.index:
            continue
        L, k = int(man.loc[e, "full_length"]), int(man.loc[e, "signal_peptide_length"])
        imp = (F[e] * L - T[e] * (L - k)) / k
        ratios.append(dict(entry=e, cut=k, frac=round(k / L, 4),
                           ratio=round(float(np.linalg.norm(imp) / np.linalg.norm(F[e])), 3),
                           cos=round(float(np.dot(F[e], T[e]) /
                                     (np.linalg.norm(F[e]) * np.linalg.norm(T[e]))), 4)))
        cuts.append((k / L, ratios[-1]["cos"]))
    cuts = np.array(cuts)

    md = pickle.load(open(ROOT / "models/bsh_prostt5_xgb_production.pkl", "rb"))
    doc = dict(
        generated=pd.Timestamp.today().strftime("%Y-%m-%d"),
        model=dict(file="bsh_prostt5_xgb_production.pkl",
                   enzymes=int(md["trained_on"]["enzymes"]),
                   split=md["trained_on"]["split"],
                   rounds=int(md["n_estimators"]),
                   held_out=md.get("held_out_estimate", {})),
        n_proteins=int(c.entry.nunique()), n_rows=int(len(c)),
        n_amines=int(c.Amine.nunique()), n_cores=int(c.Hydroxyl.nunique()),
        trimmed_only=sorted(set(g.entry) - set(c.entry)),
        agreement=dict(
            pearson=round(float(np.corrcoef(c.raw_full, c.raw_trimmed)[0, 1]), 4),
            spearman=round(float(c[["raw_full", "raw_trimmed"]].corr("spearman").iloc[0, 1]), 4),
            mean_abs_delta=round(float(c.delta.abs().mean()), 4),
            median_abs_delta=round(float(c.delta.abs().median()), 4),
            max_abs_delta=round(float(c.delta.abs().max()), 4),
            flips=int(((c.raw_full >= .5) != (c.raw_trimmed >= .5)).sum()),
            flip_rate=round(float(((c.raw_full >= .5) != (c.raw_trimmed >= .5)).mean()), 4),
            rho_median=round(float(np.median([r["rho"] for r in per])), 4),
            rho_min=round(float(min(r["rho"] for r in per)), 4),
            rho_above_90=int(sum(r["rho"] > .90 for r in per)),
            top10_median=int(np.median([r["top10"] for r in per])),
            top10_min=int(min(r["top10"] for r in per)),
            calls_full=int(sum(r["n_full"] for r in per)),
            calls_trim=int(sum(r["n_trim"] for r in per)),
            unchanged=int(sum(r["flips"] == 0 for r in per))),
        embedding=dict(
            cos_median=round(float(c.groupby("entry").embedding_cosine.first().median()), 4),
            cos_min=round(float(c.embedding_cosine.min()), 4),
            cos_max=round(float(c.embedding_cosine.max()), 4),
            ratio_median=round(float(np.median([r["ratio"] for r in ratios])), 2),
            ratio_max=round(float(max(r["ratio"] for r in ratios)), 2),
            ratio_over_3=int(sum(r["ratio"] > 3 for r in ratios)),
            n=len(ratios),
            cut_vs_cos=round(float(spearmanr(cuts[:, 0], cuts[:, 1]).statistic), 3),
            points=ratios),
        density=dict(grid=GRID, cells=cells, max=int(max(d["n"] for d in cells))),
        activity=activity(c),
        example=worked_example(c, per),
        accuracy=acc, bootstrap_ci=ci, per_protein=per)
    p = SITE / "trimming.json"
    text = json.dumps(doc, separators=(",", ":"), allow_nan=False)   # fail loudly on NaN
    p.write_text(text)
    print(f"wrote {p} ({p.stat().st_size/1e3:.0f} kB)")
    print(f"  {doc['n_proteins']} proteins, {doc['n_rows']:,} comparable rows, "
          f"{len(cells)} density cells, {len(per)} per-protein rows")


if __name__ == "__main__":
    main()
