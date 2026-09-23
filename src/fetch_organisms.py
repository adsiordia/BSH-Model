"""Look up organism and taxonomy for the candidate accessions from UniProt.

Writes data/predictions_data/candidate_organisms.csv. The raw TSV is cached alongside
it so the lookup does not have to be repeated, and so the exact response is on record.

44 of the 66 accessions are flagged "deleted" in current UniProt and carry no fields in
the search API. UniSave keeps their last version, so the protein name, organism and
length are recovered from there -- that is the record the sequences were taken from, so
it is the right description for them.

The taxonomy endpoint fills in lineage from the entry-name mnemonic. Mnemonics beginning
with a digit (9CLOT, 9BACE) are UniProt placeholders for "some unclassified member of
this group" rather than a species, so those are kept at group level and flagged
approximate -- never presented as a species name.
"""
from pathlib import Path
import csv, json, time, urllib.request
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "data/predictions_data"
RAW = OUT / "uniprot_raw.tsv"
FIELDS = "accession,id,protein_name,organism_name,lineage,length"
CHUNK = 25
RANKS = ["phylum", "class", "order", "family", "genus"]


def fetch(accs):
    rows = []
    for i in range(0, len(accs), CHUNK):
        q = "+OR+".join("accession:" + a for a in accs[i:i + CHUNK])
        url = (f"https://rest.uniprot.org/uniprotkb/search?query={q}"
               f"&fields={FIELDS}&format=tsv&size=200")
        txt = urllib.request.urlopen(url, timeout=90).read().decode()
        rows += [l for l in txt.split("\n") if l and not l.startswith("Entry\t")]
        time.sleep(0.4)
    return rows


def parse_lineage(s):
    """'... Bacteroidota (phylum), Flavobacteriia (class), ...' -> {rank: name}."""
    out = {}
    for part in (s or "").split(", "):
        if "(" in part and part.endswith(")"):
            name, rank = part.rsplit(" (", 1)
            out[rank[:-1]] = name.strip()
    return out


# matched by name rather than by word ending: "Bacteria" ends in -ia but is a domain
PHYLA = {"Bacillota", "Bacteroidota", "Pseudomonadota", "Actinomycetota", "Cyanobacteriota",
         "Planctomycetota", "Fusobacteriota", "Verrucomicrobiota", "Thermodesulfobacteriota",
         "Chlorobiota", "Spirochaetota", "Campylobacterota", "Deinococcota", "Mycoplasmatota",
         "Chloroflexota", "Acidobacteriota", "Firmicutes", "Proteobacteria"}
CLASSES = {"Clostridia", "Bacteroidia", "Gammaproteobacteria", "Alphaproteobacteria",
           "Betaproteobacteria", "Deltaproteobacteria", "Flavobacteriia", "Bacilli",
           "Actinomycetes", "Actinobacteria", "Cyanophyceae", "Planctomycetia",
           "Fusobacteriia", "Negativicutes", "Erysipelotrichia", "Sphingobacteriia",
           "Cytophagia", "Desulfovibrionia", "Chitinophagia", "Verrucomicrobiae"}

GROUP = {"9ACTN": "Actinomycetota", "9BACE": "Bacteroides and relatives",
         "9BACT": "Bacteria (unclassified)", "9CLOT": "Clostridium and relatives",
         "9CYAN": "Cyanobacteria", "9FIRM": "Bacillota (Firmicutes)",
         "9FLAO": "Flavobacteriaceae", "9GAMM": "Gammaproteobacteria"}


def unisave(acc):
    """Last archived version of a deleted entry: protein name, organism, length."""
    url = f"https://rest.uniprot.org/unisave/{acc}?format=txt&versions=1"
    txt = urllib.request.urlopen(url, timeout=40).read().decode()
    out = {}
    de = []
    for line in txt.split("\n"):
        if line.startswith("ID   "):
            parts = line.split()
            out["entry_name"] = parts[1]
            if "AA." in line:
                out["length"] = parts[-2]
        elif line.startswith("DE   "):
            de.append(line[5:].strip())
        elif line.startswith("OS   "):
            out["organism"] = out.get("organism", "") + " " + line[5:].strip()
    # "SubName: Full=Linear amide C-N hydrolase... {ECO:...}" -> the name only
    name = " ".join(de)
    for tag in ("RecName: Full=", "SubName: Full="):
        if tag in name:
            name = name.split(tag, 1)[1]
            break
    name = name.split("{")[0].split(";")[0].strip()
    if name:
        out["protein"] = name
    if "organism" in out:
        out["organism"] = out["organism"].strip().rstrip(".")
    return out


def taxonomy(mnemonic):
    """Resolve a UniProt organism mnemonic to a scientific name and lineage."""
    url = (f"https://rest.uniprot.org/taxonomy/search?query=mnemonic:{mnemonic}"
           f"&fields=scientific_name,lineage,rank&format=tsv")
    txt = urllib.request.urlopen(url, timeout=30).read().decode()
    lines = [l for l in txt.split("\n") if l and not l.startswith("Scientific")]
    if not lines:
        return None
    p = lines[0].split("\t")
    return dict(name=p[0], lineage=p[1] if len(p) > 1 else "", rank=p[2] if len(p) > 2 else "")


def main(refresh=False):
    accs = sorted(pd.read_csv(ROOT / "outputs/candidate_predictions.csv").accession.unique())
    if refresh or not RAW.exists():
        RAW.write_text("\n".join(fetch(accs)))
        print(f"fetched {len(accs)} accessions -> {RAW.relative_to(ROOT)}")

    seen, rows = set(), []
    for line in RAW.read_text().split("\n"):
        if not line or line.startswith("Entry\t"):
            continue
        p = line.split("\t")
        if len(p) < 6 or p[0] in seen:
            continue
        seen.add(p[0])
        lin = parse_lineage(p[4])
        rows.append(dict(accession=p[0], entry_name=p[1], protein=p[2],
                         organism=p[3], length=p[5],
                         **{r: lin.get(r, "") for r in RANKS}))
    # fill in the deleted entries: UniSave for the record itself, mnemonic for lineage
    cache = OUT / "uniprot_mnemonics.json"
    known = json.loads(cache.read_text()) if cache.exists() else {}
    arch_cache = OUT / "uniprot_archived.json"
    arch = json.loads(arch_cache.read_text()) if arch_cache.exists() else {}
    for r in rows:
        if r["organism"]:
            r["source"] = "uniprot entry"
            continue
        acc = r["accession"]
        if acc not in arch:
            try:
                arch[acc] = unisave(acc)
            except Exception:
                arch[acc] = {}
            time.sleep(0.3)
        a = arch.get(acc) or {}
        if a.get("protein"):
            r["protein"] = a["protein"]
        if a.get("length"):
            r["length"] = a["length"]
        archived_organism = a.get("organism", "")
        mn = r["entry_name"].rsplit("_", 1)[-1]
        r["mnemonic"] = mn
        if mn not in known:
            try:
                known[mn] = taxonomy(mn)
            except Exception:
                known[mn] = None
            time.sleep(0.3)
        t = known.get(mn)
        if archived_organism:
            # the archived record names the actual strain -- better than any mnemonic
            r["organism"] = archived_organism
            r["source"] = "archived entry (deleted from current UniProt)"
            r["approximate"] = False
        elif mn in GROUP:
            # a placeholder code: the group is real, the species is not knowable from it
            r["organism"] = GROUP[mn]
            r["source"] = "group only (entry deleted)"
            lin = {}
            if t:
                names = [x.strip() for x in t["lineage"].split(",")]
                for rk, nm in zip(RANKS, ["", "", "", "", ""]):
                    pass
                lin = {}
            r["approximate"] = True
        elif t:
            r["organism"] = t["name"]
            r["source"] = "mnemonic (entry deleted)"
            r["approximate"] = False
        else:
            r["source"] = "unresolved"
            r["approximate"] = True
        # lineage from the mnemonic's own record. The taxonomy endpoint returns names
        # without ranks, so phylum and class are matched against known lists rather than
        # guessed from word endings -- "Bacteria" ends in -ia but is not a class.
        for rk in RANKS:
            r.setdefault(rk, "")
        if t:
            parts = [x.strip() for x in t["lineage"].split(",") if x.strip()]
            for nm in parts:
                if nm in PHYLA and not r["phylum"]:
                    r["phylum"] = nm
                elif nm in CLASSES and not r["class"]:
                    r["class"] = nm
                elif nm.endswith("ales") and not r["order"]:
                    r["order"] = nm
                elif nm.endswith("aceae") and not r["family"]:
                    r["family"] = nm
    cache.write_text(json.dumps(known, indent=1))
    arch_cache.write_text(json.dumps(arch, indent=1))

    df = pd.DataFrame(rows).sort_values("accession")
    df["approximate"] = df.get("approximate", False)
    df["source"] = df.get("source", "uniprot entry")
    missing = sorted(set(accs) - set(df.accession))
    if missing:
        print(f"  WARNING: no record for {missing}")
    df.to_csv(OUT / "candidate_organisms.csv", index=False)
    print(f"-> {(OUT/'candidate_organisms.csv').relative_to(ROOT)}  ({len(df)} of {len(accs)})")
    df["protein"] = df.protein.replace("deleted", "")
    print(f"  organism named for {(df.organism.fillna('') != '').sum()} of {len(df)}, "
          f"protein name for {(df.protein.fillna('') != '').sum()}")
    for k, v in df.source.value_counts().items():
        print(f"     {v:2d}  {k}")
    for r in ["phylum", "class", "genus"]:
        n = df[r].replace("", pd.NA).nunique()
        top = df[r].replace("", pd.NA).value_counts().head(3)
        print(f"  {r:8s} {n:2d} distinct   most common: "
              + ", ".join(f"{k} ({v})" for k, v in top.items()))


if __name__ == "__main__":
    import sys
    main(refresh="--refresh" in sys.argv)
