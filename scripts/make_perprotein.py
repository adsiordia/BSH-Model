#!/usr/bin/env python3
"""Rebuild perprotein.json from the complete PandaMap output.

The shipped file was built from a dataset that kept 5 of PandaMap's 13
interaction types. This rebuild reads contacts_full.pkl, so the `t` field now
carries every type -- including the pi family and covalent, which were the
majority of the contacts at several positions and were absent entirely.

Schema is unchanged, so the page needs no edit for this part:
  proteins[id] = {seq, seq_len, construct, stratum, contacts, active,
                  interface, n_itf_conj, n_own_res}
  contacts[amine] = [{r, n, ch, part, f, t}]   f = samples of 5, t = {type: n}

One threshold is used throughout, SOLID = 3 of 5, matching the alignment and
every statistic. Nothing is filtered out of `contacts` itself -- entries carry
their own f so the page can apply the threshold -- but the derived counts
(n_own_res, interface, n_itf_conj) use it.
"""
import json, csv, collections
import pandas as pd

AF3 = "/home/adsiordia/AF3/BSH_AF3"
SITE = "/home/adsiordia/BSH-Model/site"
SOLID = 3

d = pd.read_pickle(f"{AF3}/analysis/contacts_full.pkl")
old = json.load(open(f"{SITE}/perprotein.json"))
man = {r["protein"]: r for r in csv.DictReader(open(f"{AF3}/panel/manifest.csv"))}

AM = old["amines"]
PR = old["order"]
out = {}
for p in PR:
    x = d[d.protein == p]
    prev = old["proteins"][p]
    con = {}
    for a in AM:
        y = x[x.token == a]
        rows = []
        for (r, n, ch, part), g in y.groupby(["resnum", "resname", "chain", "part"]):
            f = g["sample"].nunique()
            t = {k: int(v) for k, v in g.groupby("itype")["sample"].nunique().items()}
            rows.append(dict(r=int(r), n=n, ch=ch,
                             part=("m" if part == "moiety" else "c"),
                             f=int(f), t=t))
        rows.sort(key=lambda z: (z["ch"], z["r"]))
        con[a] = rows
    solid = {a: [c for c in con[a] if c["f"] >= SOLID] for a in AM}
    own = {(c["ch"], c["r"]) for a in AM for c in solid[a] if c["ch"] == "A"}
    itf = collections.Counter()
    for a in AM:
        for c in solid[a]:
            if c["ch"] != "A":
                itf[(c["r"], c["n"])] += 1
    out[p] = dict(seq=prev["seq"], seq_len=prev["seq_len"],
                  construct=prev["construct"], stratum=prev["stratum"],
                  contacts=con, active=prev["active"],
                  interface=[dict(r=r, n=n, nconj=k)
                             for (r, n), k in itf.most_common()],
                  n_itf_conj=sum(1 for a in AM
                                 if any(c["ch"] != "A" for c in solid[a])),
                  n_own_res=len(own))

new = dict(proteins=out, amines=AM, charge=old["charge"], order=PR,
           nsamples=int(d["sample"].nunique()), solid=SOLID)
json.dump(new, open(f"{SITE}/perprotein.json", "w"), separators=(",", ":"))

print("rebuilt site/perprotein.json from contacts_full.pkl")
oc = sum(len(v) for e in old["proteins"].values() for v in e["contacts"].values())
nc = sum(len(v) for e in out.values() for v in e["contacts"].values())
print(f"  contact entries: {oc} -> {nc}")
ot = collections.Counter(k for e in old["proteins"].values()
                         for v in e["contacts"].values() for c in v for k in c["t"])
nt = collections.Counter(k for e in out.values()
                         for v in e["contacts"].values() for c in v for k in c["t"])
print(f"  interaction types: {len(ot)} -> {len(nt)}")
print(f"    new: {sorted(set(nt) - set(ot))}")
print(f"  n_own_res mean: {sum(e['n_own_res'] for e in old['proteins'].values())/len(PR):.1f}"
      f" -> {sum(e['n_own_res'] for e in out.values())/len(PR):.1f}")
print(f"  enzymes with an interface contact: "
      f"{sum(1 for e in old['proteins'].values() if e['n_itf_conj'])} -> "
      f"{sum(1 for e in out.values() if e['n_itf_conj'])}")
