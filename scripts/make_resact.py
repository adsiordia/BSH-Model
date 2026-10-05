#!/usr/bin/env python3
"""Build resact.json -- the residue table and the activity table on one index.

The Alignment tab and the Pocket & activity tab were answering two halves of one
question from opposite ends. Alignment showed, per conjugate, which aligned
columns the structures found in contact, for the 27 folded enzymes, with no assay
data. Pocket & activity showed residue letters at 11 fixed positions beside the
assay, for the 115 enzymes that have it, with no way to see any other column.

This joins them: for every enzyme with assay data, the residue it carries at any
contact-carrying column, plus what it was measured to make, plus the lineage it
belongs to -- so a residue trend and the confound can be read side by side.

Inputs are the files the site already ships, so this adds no new analysis:
  site/alignment.json       aln127, map27to127, ref127, hits, verdict, near
  site/sitesactivity.json   activity, cluster, the 11 amine positions
  <AF3>/analysis/pocket_columns.json   which columns touch the amine vs the core
"""
import json, os, collections

SITE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "site")
AF3 = "/home/adsiordia/AF3/BSH_AF3/analysis"

AL = json.load(open(f"{SITE}/alignment.json"))
SA = json.load(open(f"{SITE}/sitesactivity.json"))
PC = json.load(open(f"{AF3}/pocket_columns.json"))

m27 = {int(k): v for k, v in AL["map27to127"].items()}      # col27 -> col127
ref127 = {int(k): v for k, v in AL["ref127"].items()}       # col127 -> C0ERS1 pos
verdict = {int(k): v for k, v in AL["verdict"].items()}     # col27 -> alignment quality

# how many of the 27 folded enzymes contact each col27, split by what it touches.
# hits[enz][amine] = {om,oc,im,ic} -> {col27: samples}; o/i = own/neighbouring
# subunit, m/c = the amine moiety / the bile acid core.
touch = collections.defaultdict(lambda: {"m": set(), "c": set(), "itf": set()})
for enz, byam in AL["hits"].items():
    for am, d in byam.items():
        for key, part in (("om", "m"), ("im", "m"), ("oc", "c"), ("ic", "c")):
            for col in d.get(key, {}):
                touch[int(col)][part].add(enz)
                if key[0] == "i":
                    touch[int(col)]["itf"].add(enz)

rows = {r["id"]: r for r in SA["rows"]}
cols = sorted(set(PC["moiety"]) | set(PC["core"]))

positions = []
for c27 in cols:
    c127 = m27.get(c27)
    if c127 is None:
        continue
    letters = collections.Counter()
    for eid in rows:
        s = AL["aln127"].get(eid)
        if s and c127 < len(s):
            letters[s[c127]] += 1
    non_gap = {k: v for k, v in letters.items() if k != "-"}
    top = max(non_gap.values()) if non_gap else 0
    positions.append({
        "col": c27,
        "col127": c127,
        "ref": ref127.get(c127),
        "amine": c27 in PC["moiety"],
        "core": c27 in PC["core"],
        "n_amine": len(touch[c27]["m"]),
        "n_core": len(touch[c27]["c"]),
        "n_itf": len(touch[c27]["itf"]),
        "verdict": verdict.get(c27, "ok"),
        # "invariant" = one residue in >=95% of enzymes that have one here
        "invariant": bool(non_gap) and top / sum(non_gap.values()) >= 0.95,
        "nres": len(non_gap),
    })

out = {
    "positions": positions,
    "amines": SA["amines"],
    "charge": SA["charge"],
    "clusters": SA["clusters"],
    "cluster_sizes": SA["cluster_sizes"],
    "rows": [{
        "id": eid,
        "cluster": r["cluster"],
        "modelled": r["modelled"],
        "n": r["n"],
        "near": AL["near"].get(eid),
        "act": r["act"],
        "res": {str(p["col"]): (AL["aln127"].get(eid, "") or "-")[p["col127"]]
                if AL["aln127"].get(eid) and p["col127"] < len(AL["aln127"][eid]) else "-"
                for p in positions},
    } for eid, r in rows.items()],
}

dst = f"{SITE}/resact.json"
json.dump(out, open(dst, "w"), separators=(",", ":"))
print(f"wrote {dst}  ({os.path.getsize(dst)/1024:.0f} KB)")
print(f"  {len(out['rows'])} enzymes with assay data")
print(f"  {len(positions)} contact-carrying columns "
      f"({sum(p['amine'] for p in positions)} touch the amine, "
      f"{sum(p['core'] for p in positions)} the core)")
print(f"  {sum(p['invariant'] for p in positions)} are invariant across the enzymes")
print(f"  {len(out['amines'])} conjugates measured")
