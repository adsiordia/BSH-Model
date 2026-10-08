#!/usr/bin/env python3
"""Per job, which of the five AlphaFold samples showed each contact.

Everything on this site counts samples, never rows, but the counts were shipped
as a bare number ("3 of 5") with no way to see the reproducibility directly. The
underlying per-sample CSVs have it, so carry it: [0,1,2,3,4] and [0,4] are very
different pieces of evidence that both reduce to a count.

Keyed chain:resnum, listing sample indices, amine-side and core-side merged
because the table row is per residue.
"""
import csv, glob, json, os, collections, re

AF3 = "/home/adsiordia/AF3/BSH_AF3"
SITE = "/home/adsiordia/BSH-Model/site"
jobs = sorted(os.path.basename(f)[:-4]
              for f in glob.glob(f"{SITE}/pockets/*.pdb"))
out, miss = {}, 0
for job in jobs:
    acc = collections.defaultdict(set)
    files = glob.glob(f"{AF3}/analysis/contacts/{job}__seed-*_sample-*.csv")
    if not files:
        miss += 1
        continue
    for f in files:
        m = re.search(r"_sample-(\d+)\.csv$", f)
        if not m:
            continue
        s = int(m.group(1))
        for r in csv.DictReader(open(f)):
            acc[f'{r["protein_chain"]}:{r["protein_resnum"]}'].add(s)
    out[job] = {k: sorted(v) for k, v in sorted(acc.items())}

json.dump(out, open(f"{SITE}/samples.json", "w"), separators=(",", ":"))
n = sum(len(v) for v in out.values())
print(f"wrote site/samples.json  ({os.path.getsize(f'{SITE}/samples.json')/1024:.0f} KB)")
print(f"  {len(out)} jobs, {n} residue entries, {miss} jobs with no CSVs")
d = collections.Counter(len(s) for v in out.values() for s in v.values())
print(f"  residues by how many samples showed them: {dict(sorted(d.items()))}")
