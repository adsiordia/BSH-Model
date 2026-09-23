"""Write site/candidates.json -- the new predictions, for collaborators.

Built for someone who wants to know which enzyme is predicted to attach which amine,
not how the model works. Everything is keyed by the entry name from the FASTA, carries
the organism where UniProt knows it, and is scored from the TRIMMED sequences.
"""
from pathlib import Path
import json, numpy as np, pandas as pd, warnings
warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parent.parent
OUT, SITE = ROOT / "outputs", ROOT / "site"
THRESHOLD = 0.5


def signal_peptide_story(g, man):
    """The two questions a collaborator will ask about the signal peptide.

    1. Which enzymes does the model call active only once the peptide is gone?
    2. Does the length of the peptide correspond to anything?
    """
    from scipy.stats import mannwhitneyu, spearmanr
    c = g[g.comparable].copy()
    c["af"] = c.raw_full >= THRESHOLD
    c["at"] = c.raw_trimmed >= THRESHOLD
    p = c.groupby("entry").agg(acc=("accession", "first"), nf=("af", "sum"), nt=("at", "sum"),
                               meas=("measured", lambda s: s.sum() if s.notna().any() else np.nan))
    p["cut"] = [int(man.loc[e, "signal_peptide_length"]) for e in p.index]
    p["d"] = p.nt - p.nf

    gained = [dict(entry=e, accession=r.acc, cut=int(r.cut), full=int(r.nf), trimmed=int(r.nt),
                   measured=None if pd.isna(r.meas) else int(r.meas))
              for e, r in p[(p.nf == 0) & (p.nt > 0)].sort_values("nt", ascending=False).iterrows()]

    m = p[p.meas.notna()]
    act, sil = m[m.meas > 0], m[m.meas == 0]
    bins = [(0, 20, "20 or fewer"), (21, 40, "21 to 40"), (41, 60, "41 to 60"), (61, 200, "over 60")]
    length = dict(
        active=dict(n=int(len(act)), median=float(act.cut.median()),
                    lo=int(act.cut.min()), hi=int(act.cut.max())),
        silent=dict(n=int(len(sil)), median=float(sil.cut.median()),
                    lo=int(sil.cut.min()), hi=int(sil.cut.max())),
        p_value=round(float(mannwhitneyu(act.cut, sil.cut).pvalue), 4),
        spearman_products=round(float(spearmanr(m.cut, m.meas).statistic), 3),
        spearman_change=round(float(spearmanr(p.cut, p.d.abs()).statistic), 3),
        bins=[dict(label=lab, lo=lo, hi=hi, n=int(((m.cut >= lo) & (m.cut <= hi)).sum()),
                   active=int(((m.cut >= lo) & (m.cut <= hi) & (m.meas > 0)).sum()),
                   silent=int(((m.cut >= lo) & (m.cut <= hi) & (m.meas == 0)).sum()))
              for lo, hi, lab in bins if ((m.cut >= lo) & (m.cut <= hi)).sum()],
        cut_median=float(p.cut.median()), cut_lo=int(p.cut.min()), cut_hi=int(p.cut.max()))
    typical = int(np.median(p.nt))
    return dict(gained=gained, length=length, typical_trimmed=typical,
                n_silent_full=int((p.nf == 0).sum()), n_silent_trimmed=int((p.nt == 0).sum()))


def main():
    g = pd.read_csv(OUT / "candidate_predictions.csv")
    org = pd.read_csv(ROOT / "data/predictions_data/candidate_organisms.csv").fillna("")
    man = pd.read_csv(ROOT / "data/predictions_data/signaling_peptide/"
                      "signal_peptide_manifest.csv").set_index("id")
    O = org.set_index("accession").to_dict("index")
    # Classify by NITROGEN TYPE only. The older labels (amino acid / polyamine / simple
    # amine) mixed two different questions and overlapped -- glycine is an amino acid AND a
    # primary amine. What kind of nitrogen does the attacking is a single, exclusive property.
    cur = pd.read_csv(ROOT / "data/derived/reactants_curated.csv")
    cur = cur[cur.data_name.notna()]

    def ntype(r):
        if r.n_aniline > 0:
            return "aromatic"
        if r.n_tertiary > 0 and r.n_primary == 0 and r.n_secondary == 0:
            return "tertiary"
        if r.n_secondary > 0 and r.n_primary == 0:
            return "secondary"
        return "primary"

    CLS = {r.data_name: ntype(r) for _, r in cur.iterrows()}
    # how many reactive nitrogens the amine carries -- a separate axis from which KIND
    NUC = {r.data_name: int(r.n_nucleophilic_N or 0) for _, r in cur.iterrows()}

    proteins = []
    for e, s in g.groupby("entry"):
        acc = s.accession.iloc[0]
        o = O.get(acc, {})
        m = man.loc[e] if e in man.index else None
        hits = s[s.raw_trimmed >= THRESHOLD].sort_values("raw_trimmed", ascending=False)
        # how the predicted hits break down by amine class, against what was on offer
        cls = {}
        for a in s.Amine.unique():
            c = CLS.get(a, "other")
            cls.setdefault(c, {"offered": 0, "hit": 0})
            cls[c]["offered"] += int((s.Amine == a).sum())
            cls[c]["hit"] += int(((hits.Amine == a).sum()))
        classes = [dict(cls=c, offered=v["offered"], hit=v["hit"],
                        rate=round(v["hit"] / v["offered"], 4) if v["offered"] else 0)
                   for c, v in sorted(cls.items(), key=lambda kv: -kv[1]["hit"])]
        known = s.measured.notna().any()
        proteins.append(dict(
            entry=e, accession=acc,
            organism=o.get("organism", ""), phylum=o.get("phylum", ""),
            klass=o.get("class", ""), genus=o.get("genus", ""),
            approximate=bool(o.get("approximate", False)),
            source=o.get("source", ""),
            protein=o.get("protein", ""),
            length=int(m.full_length) if m is not None else None,
            cut=int(m.signal_peptide_length) if m is not None else None,
            mature=int(m.mature_length) if m is not None else None,
            n_pred=int(len(hits)), classes=classes,
            n_pred_full=int((s.raw_full >= THRESHOLD).sum()),
            measured=int(s.measured.sum()) if known else None,
            tested=bool(known),
            comparable=bool(s.comparable.iloc[0]),
            top=[dict(amine=r.Amine, core=r.Hydroxyl, cls=CLS.get(r.Amine, "other"),
                      p=round(float(r.raw_trimmed), 3),
                      cal=round(float(r.cal_trimmed), 3),
                      p_full=round(float(r.raw_full), 3),
                      measured=None if pd.isna(r.measured) else bool(r.measured))
                 for _, r in hits.head(20).iterrows()]))
    proteins.sort(key=lambda r: -r["n_pred"])

    # which amines the whole panel is predicted to favour
    t = g[g.raw_trimmed >= THRESHOLD]
    amine = []
    for a, s in g.groupby("Amine"):
        h = s[s.raw_trimmed >= THRESHOLD]
        known = s[s.measured.notna()]
        amine.append(dict(amine=a, n=int(len(h)), of=int(len(s)),
                          rate=round(float((s.raw_trimmed >= THRESHOLD).mean()), 4),
                          rate_full=round(float((s.raw_full >= THRESHOLD).mean()), 4),
                          proteins=int(h.entry.nunique()),
                          measured_rate=round(float(known.measured.mean()), 4) if len(known) else None,
                          mean_p=round(float(s.raw_trimmed.mean()), 4)))
    amine.sort(key=lambda r: -r["rate"])

    cls_tot = {}
    for a, sa in g.groupby("Amine"):
        c = CLS.get(a, "other")
        d = cls_tot.setdefault(c, {"offered": 0, "hit": 0, "amines": set(), "measured": [0, 0]})
        d["offered"] += len(sa); d["hit"] += int((sa.raw_trimmed >= THRESHOLD).sum())
        d["amines"].add(a)
        k = sa[sa.measured.notna()]
        d["measured"][0] += int(k.measured.sum()); d["measured"][1] += len(k)
    nuc_tot = {}
    for a, sa in g.groupby("Amine"):
        n = NUC.get(a, 0)
        d = nuc_tot.setdefault(n, {"offered": 0, "hit": 0, "amines": set(), "measured": [0, 0]})
        d["offered"] += len(sa); d["hit"] += int((sa.raw_trimmed >= THRESHOLD).sum())
        d["amines"].add(a)
        k = sa[sa.measured.notna()]
        d["measured"][0] += int(k.measured.sum()); d["measured"][1] += len(k)
    nitrogens = [dict(n=n, amines=sorted(v["amines"]), n_amines=len(v["amines"]),
                      rate=round(v["hit"] / v["offered"], 4),
                      measured_rate=round(v["measured"][0] / v["measured"][1], 4)
                      if v["measured"][1] else None)
                 for n, v in sorted(nuc_tot.items())]

    classes = [dict(cls=c, amines=sorted(v["amines"]), n_amines=len(v["amines"]),
                    rate=round(v["hit"] / v["offered"], 4),
                    measured_rate=round(v["measured"][0] / v["measured"][1], 4)
                    if v["measured"][1] else None)
               for c, v in sorted(cls_tot.items(), key=lambda kv: -kv[1]["hit"] / kv[1]["offered"])]

    core = [dict(core=c, rate=round(float((s.raw_trimmed >= THRESHOLD).mean()), 4),
                 rate_full=round(float((s.raw_full >= THRESHOLD).mean()), 4),
                 measured_rate=round(float(s[s.measured.notna()].measured.mean()), 4)
                 if s.measured.notna().any() else None)
            for c, s in g.groupby("Hydroxyl")]

    # amine x core grid of predicted hit rate
    grid = []
    for (a, c), s in g.groupby(["Amine", "Hydroxyl"]):
        grid.append(dict(amine=a, core=c,
                         rate=round(float((s.raw_trimmed >= THRESHOLD).mean()), 4),
                         n=int((s.raw_trimmed >= THRESHOLD).sum())))

    # do predictions track taxonomy?
    tax = []
    df = pd.DataFrame(proteins)
    for rank in ("phylum", "klass"):
        for v, s in df[df[rank] != ""].groupby(rank):
            if len(s) >= 2:
                tax.append(dict(rank=rank, group=v, n=int(len(s)),
                                median_pred=float(s.n_pred.median()),
                                min_pred=int(s.n_pred.min()), max_pred=int(s.n_pred.max()),
                                measured=[int(x) for x in s.measured.dropna()]))
    tax.sort(key=lambda r: (r["rank"], -r["n"]))

    doc = dict(generated=pd.Timestamp.today().strftime("%Y-%m-%d"),
               threshold=THRESHOLD, scored_on="trimmed (signal peptide removed)",
               model="bsh_prostt5_xgb_production.pkl, trained on all 115 assayed enzymes",
               n_proteins=len(proteins), n_amines=int(g.Amine.nunique()),
               n_cores=int(g.Hydroxyl.nunique()), n_rows=int(len(g)),
               n_predicted=int(len(t)),
               n_named=int(sum(1 for p in proteins if p["organism"] and not p["approximate"])),
               proteins=proteins, amines=amine, cores=core, grid=grid, taxonomy=tax,
               classes=classes, nitrogens=nitrogens,
               signal=signal_peptide_story(g, man))
    p = SITE / "candidates.json"
    p.write_text(json.dumps(doc, separators=(",", ":"), allow_nan=False))
    print(f"wrote {p} ({p.stat().st_size/1e3:.0f} kB)")
    print(f"  {len(proteins)} proteins, {len(t):,} predicted pairs of {len(g):,}, "
          f"{doc['n_named']} with a named organism")
    print(f"  top amines: " + ", ".join(f"{a['amine']} {a['rate']:.0%}" for a in amine[:5]))


if __name__ == "__main__":
    main()
