#!/usr/bin/env python3
"""Build validation.json: the two crystal structures against our predicted pocket.

8BLT  L. salivarius BSH + taurocholate (intact substrate), 2.10 A, 8 copies.
2BJF  C. perfringens CBAH + taurine AND deoxycholate (the products), 1.67 A,
      1 chain deposited -- the tetramer is rebuilt from the 4 crystal symmetry
      operators, without which the inter-subunit contacts do not exist at all.

For 8BLT the ligand is one molecule, so amine vs core is split with the same
geometric rule used on the AF3 panel. For 2BJF the products are already separate
molecules, so taurine contacts are amine-subsite contacts by construction.
"""
import json, collections

PC = json.load(open("../analysis/pocket_columns.json"))
MOI = PC["moiety_ref"]
CORE = {x for x in PC["core_ref"] if x is not None}

def load(path, which):
    rows = json.load(open(path))
    if which == "8BLT":
        amine = [r for r in rows if r["part"] == "moiety"]
        core = [r for r in rows if r["part"] == "core"]
    else:
        amine = [r for r in rows if r["lig"] == "TAU"]
        core = [r for r in rows if r["lig"] == "DXC"]
    return amine, core

def summarise(rows, key="ligchain"):
    by = collections.defaultdict(list)
    for r in rows:
        by[r["ref"]].append(r)
    out = {}
    for pos, g in by.items():
        if pos is None:
            continue
        out[pos] = dict(
            res=f'{g[0]["resname"]}{g[0]["resnum"]}',
            copies=len({r[key] for r in g}),
            itf=len({r[key] for r in g if r["itf"]}),
            itypes=sorted({r["itype"] for r in g}),
            mind=round(min(r["dist"] for r in g), 2))
    return out

S = {}
for pdb, path, ncopy, meta in [
        ("8BLT", "8BLT_contacts.json", 8,
         dict(enzyme="Lactobacillus salivarius BSH", res=2.10,
              ligand="taurocholate (intact substrate)",
              note="catalytic Cys trapped as the oxidised sulfonic acid (OCS), "
                   "which is how the uncleaved substrate was captured")),
        ("2BJF", "2BJF_contacts.json", 4,
         dict(enzyme="Clostridium perfringens CBAH", res=1.67,
              ligand="taurine + deoxycholate (the reaction products)",
              note="one chain deposited; the tetramer is rebuilt from the four "
                   "crystal symmetry operators, without which no inter-subunit "
                   "contact exists"))]:
    amine, core = load(path, pdb)
    S[pdb] = dict(meta=meta, ncopy=ncopy,
                  n_amine=len(amine), n_core=len(core),
                  amine=summarise(amine), core=summarise(core),
                  amine_itf=sum(r["itf"] for r in amine),
                  core_itf=sum(r["itf"] for r in core))

rows = []
for pos in MOI:
    row = dict(pos=pos, invariant=pos in (1, 15, 169, 222))
    for pdb in ("8BLT", "2BJF"):
        hit = S[pdb]["amine"].get(pos)
        row[pdb] = hit
    rows.append(row)

extra = {}
for pdb in ("8BLT", "2BJF"):
    extra[pdb] = sorted(p for p in S[pdb]["amine"] if p not in MOI)

core_stats = {}
for pdb in ("8BLT", "2BJF"):
    pos = sorted(S[pdb]["core"])
    core_stats[pdb] = dict(n=len(pos), ours=sum(1 for p in pos if p in CORE),
                           positions=pos)

# per ligand copy, the residues to light up in the 3D viewer
copies = {}
for pdb, path, which in [("8BLT", "8BLT_contacts.json", "8BLT"),
                         ("2BJF", "2BJF_contacts.json", "2BJF")]:
    raw = json.load(open(path))
    by = collections.defaultdict(list)
    for r in raw:
        part = r["part"] if which == "8BLT" else ("moiety" if r["lig"] == "TAU" else "core")
        lg = "TCH" if which == "8BLT" else r["lig"]
        by[r["ligchain"]].append(dict(ch=r["chain"], rn=r["resnum"], nm=r["resname"],
                                      itf=bool(r["itf"]), part=part,
                                      ref=r["ref"], it=r["itype"],
                                      pa=r["pa"], la=r["la"], lg=lg,
                                      d=round(r["dist"], 2)))
    # collapse duplicates (same residue, several interaction types)
    # one entry per residue for the table and the sticks, but every atom pair
    # kept under "pairs" so each interaction can be drawn as its own line with
    # PandaMap's own distance
    out = {}
    for ch, g in by.items():
        seen = {}
        for r in g:
            k = (r["ch"], r["rn"], r["part"], r["lg"])
            if k not in seen:
                seen[k] = dict(ch=r["ch"], rn=r["rn"], nm=r["nm"], itf=r["itf"],
                               part=r["part"], ref=r["ref"], it=[r["it"]],
                               d=r["d"], lg=r["lg"], pairs=[])
            else:
                if r["it"] not in seen[k]["it"]:
                    seen[k]["it"].append(r["it"])
                seen[k]["d"] = min(seen[k]["d"], r["d"])
            seen[k]["pairs"].append([r["pa"], r["la"], r["d"], r["it"], r["lg"]])
        out[ch] = sorted(seen.values(), key=lambda x: (x["part"] != "moiety", x["rn"]))
    copies[pdb] = out

LIGS = {"8BLT": ["TCH"], "2BJF": ["TAU", "DXC"]}

json.dump(dict(structures={k: {kk: vv for kk, vv in v.items() if kk != "core"}
                           for k, v in S.items()},
               rows=rows, extra=extra, core=core_stats, copies=copies, ligs=LIGS,
               moiety_ref=MOI, n_core_ref=len(CORE)),
          open("/home/adsiordia/BSH-Model/site/validation.json", "w"),
          separators=(",", ":"))
print("wrote site/validation.json")
for pdb in S:
    a = S[pdb]
    hits = sum(1 for p in a["amine"] if p in MOI)
    print(f"  {pdb}: {len(a['amine'])} amine positions, {hits} in our 11; "
          f"core {core_stats[pdb]['ours']}/{core_stats[pdb]['n']} in our {len(CORE)}")
