"""Build the multiple sequence alignment from scratch, with provenance.

An alignment already existed at data/derived/bsh_aligned.fasta but with no record of
how it was produced, so nothing downstream could be trusted to be reproducible. This
rebuilds it from the source FASTA with a recorded command.

MAFFT L-INS-i is used rather than the default FFT-NS-2: it is the most accurate mode
MAFFT offers and is affordable at this size (115 sequences, ~350 residues). MMseqs2 is
not the right tool for this step -- it searches and clusters, and its MSAs are
profiles built around a query rather than a balanced alignment of a set, which is what
column-wise conservation needs.

Writes data/derived/msa_linsi.fasta plus a .log recording the exact invocation.
"""
from pathlib import Path
import subprocess, datetime, sys
from Bio import SeqIO

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "data/raw/Seqs_list_total.fasta"
OUT = ROOT / "data/derived"
ENV = Path.home() / "miniconda3/envs/mafft_env/bin"
MODE = "mafft-linsi"          # L-INS-i: iterative refinement, most accurate


def main(enzymes=None):
    recs = {r.id.split("_")[-1]: (r.id, str(r.seq).replace("-", "").upper())
            for r in SeqIO.parse(SRC, "fasta")}
    if enzymes is not None:
        recs = {k: v for k, v in recs.items() if k in enzymes}
    OUT.mkdir(parents=True, exist_ok=True)
    tmp = OUT / "_to_align.fasta"
    tmp.write_text("".join(f">{k}\n{s}\n" for k, (_, s) in sorted(recs.items())))

    exe = ENV / MODE
    if not exe.exists():
        sys.exit(f"{exe} not found -- mafft_env is missing")
    cmd = [str(exe), "--anysymbol", "--thread", "8", str(tmp)]
    print(f"{len(recs)} sequences, {min(len(s) for _, s in recs.values())}-"
          f"{max(len(s) for _, s in recs.values())} residues")
    print("running: " + " ".join(cmd) + "\n")
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        sys.exit("mafft failed:\n" + r.stderr[-2000:])
    dest = OUT / "msa_linsi.fasta"
    dest.write_text(r.stdout)
    (OUT / "msa_linsi.log").write_text(
        f"generated {datetime.datetime.now().isoformat(timespec='seconds')}\n"
        f"source    {SRC.relative_to(ROOT)}\n"
        f"sequences {len(recs)}\n"
        f"command   {' '.join(cmd)}\n"
        f"version   {subprocess.run([str(exe), '--version'], capture_output=True, text=True).stderr.strip()}\n"
        f"\nmafft stderr (tail):\n{r.stderr[-3000:]}\n")
    tmp.unlink()

    aln = {r.id: str(r.seq) for r in SeqIO.parse(dest, "fasta")}
    L = {len(v) for v in aln.values()}
    print(f"-> {dest.relative_to(ROOT)}  ({len(aln)} sequences, "
          f"{L.pop() if len(L) == 1 else L} columns)")
    print(f"-> {(OUT / 'msa_linsi.log').relative_to(ROOT)}")
    return dest


if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import labels as lb
    main(set(lb.build(min_reps=1).Enzyme))
