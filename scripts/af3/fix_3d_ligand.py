#!/usr/bin/env python3
"""Repair PandaMap --3d HTML files whose ligand does not render.

PandaMap writes the ligand residue name verbatim from the mmCIF.  AF3 names
ligands 'LIG_L' / 'LIG_M' (5 chars), but the PDB resName field is 3 chars
(cols 18-20).  The overflow shifts every later field right by 2, so resSeq
becomes garbage and the x/y/z columns no longer parse -- 3Dmol loads the
atoms but cannot place them, and the ligand is invisible.

This rewrites the embedded PDB with a 3-char resName and correct columns.
Files already generated with a patched PandaMap are left unchanged.

Usage:  python3 fix_3d_ligand.py <dir-or-file> [...]
"""
import base64, pathlib, re, sys

PDB_RE = re.compile(r'var pdbData = atob\("([^"]+)"\)')


def repair_line(line):
    """Re-lay a HETATM/ATOM record whose resName overflowed its field."""
    if not line.startswith(("ATOM", "HETATM")):
        return line, False
    # A correct record has an integer resSeq at cols 23-26 and parseable
    # y/z fields.  Testing x alone is not enough: a 2-column shift leaves
    # '    -0.2' in the x field, which still parses as a float.
    try:
        int(line[22:26])
        float(line[38:46])
        float(line[46:54])
        return line, False
    except ValueError:
        pass
    # Overflowed: find the resName by reading from col 18 up to the chain id.
    m = re.match(r"^(.{17})([A-Za-z0-9_]+?)\s([A-Za-z0-9])\s*(\d+)\s+(.*)$", line)
    if not m:
        return line, False
    head, resname, chain, resseq, rest = m.groups()
    nums = rest.split()
    if len(nums) < 3:
        return line, False
    x, y, z = (float(v) for v in nums[:3])
    occ = float(nums[3]) if len(nums) > 3 else 1.0
    bfac = float(nums[4]) if len(nums) > 4 else 0.0
    elem = nums[5] if len(nums) > 5 else line[-4:].strip()
    fixed = (f"{head}{resname[:3]:>3} {chain}{int(resseq):>4}    "
             f"{x:8.3f}{y:8.3f}{z:8.3f}{occ:6.2f}{bfac:6.2f}          {elem:>2}  ")
    return fixed, True


def fix(path):
    html = path.read_text()
    m = PDB_RE.search(html)
    if not m:
        return None
    pdb = base64.b64decode(m.group(1)).decode()
    out, n = [], 0
    for line in pdb.splitlines():
        new, changed = repair_line(line)
        out.append(new)
        n += changed
    if not n:
        return 0
    enc = base64.b64encode("\n".join(out).encode()).decode()
    path.write_text(html[:m.start(1)] + enc + html[m.end(1):])
    return n


if __name__ == "__main__":
    targets = [pathlib.Path(a) for a in (sys.argv[1:] or ["."])]
    files = []
    for t in targets:
        files += [t] if t.is_file() else sorted(t.rglob("*.html"))
    for f in files:
        n = fix(f)
        if n is None:
            print(f"skipped (no embedded PDB): {f}")
        elif n:
            print(f"repaired {n} atom records: {f}")
        else:
            print(f"already correct: {f}")
