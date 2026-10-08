#!/usr/bin/env python3
"""Rebuild alignment.json's `hits` from the complete PandaMap output.

`hits[enzyme][amine] = {om, oc, im, ic}` maps a 27-sequence alignment column to
how many of the 5 samples showed a contact there. o/i is own versus a
neighbouring subunit, m/c is the amine moiety versus the bile acid core.

Only `hits` is rewritten. Every other field in alignment.json -- the two
alignments, the reference maps, the identity-to-nearest-folded, the sequence
provenance -- is left exactly as it was, since none of it derives from contacts.
"""
import json, collections
import pandas as pd

AF3 = "/home/adsiordia/AF3/BSH_AF3"
SITE = "/home/adsiordia/BSH-Model/site"
SOLID = 3

AL = json.load(open(f"{SITE}/alignment.json"))
M27 = {int(k): v for k, v in AL["map27to127"].items()}      # col27 -> col127
REF = {int(a): b for a, b in AL["ref127"].items()}          # col127 -> C0ERS1
POS2C27 = {}
for c27, c127 in M27.items():
    p = REF.get(c127)
    if p is not None:
        POS2C27[p] = c27

d = pd.read_pickle(f"{AF3}/analysis/contacts_full.pkl")
hits = {}
for p in AL["order"]:
    x = d[d.protein == p]
    byam = {}
    for a in AL["amines"]:
        y = x[x.token == a]
        o = {"om": {}, "oc": {}, "im": {}, "ic": {}}
        if len(y):
            g = y.groupby(["pos", "chain", "part"])["sample"].nunique()
            for (pos, ch, part), f in g.items():
                if pd.isna(pos):
                    continue
                c27 = POS2C27.get(int(pos))
                if c27 is None:
                    continue
                k = ("o" if ch == "A" else "i") + ("m" if part == "moiety" else "c")
                o[k][str(c27)] = max(o[k].get(str(c27), 0), int(f))
        # the page expects only contacts at or above the threshold here
        byam[a] = {k: {c: f for c, f in v.items() if f >= SOLID} for k, v in o.items()}
    hits[p] = byam

oldhits = AL["hits"]
AL["hits"] = hits
AL["solid"] = SOLID
json.dump(AL, open(f"{SITE}/alignment.json", "w"), separators=(",", ":"))

cnt = lambda h: sum(len(v) for e in h.values() for a in e.values() for v in a.values())
print("rebuilt alignment.json hits from contacts_full.pkl")
print(f"  columns flagged: {cnt(oldhits)} -> {cnt(hits)}")
for k, lab in (("om", "own / amine"), ("im", "neighbour / amine"),
               ("oc", "own / core"), ("ic", "neighbour / core")):
    o = sum(len(a[k]) for e in oldhits.values() for a in e.values())
    n = sum(len(a[k]) for e in hits.values() for a in e.values())
    print(f"    {lab:<20}{o:>7} -> {n}")
print("  every other field in alignment.json left untouched")
