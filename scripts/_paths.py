"""Where the AlphaFold inputs live, so the scripts run from a fresh clone.

Until November 2026 every script here hard-coded /home/adsiordia/AF3/BSH_AF3.
That directory is not under version control, so 8 of the 23 data payloads the
site inlines could only be rebuilt on one machine. The small inputs are now
tracked in this repo and resolved through here instead.

TRACKED, in this repo -- enough to rebuild the Crystal check tab and to read the
pocket definition every structural claim is quoted against:

    REFSTRUCT   data/af3/refstruct   crystal CIFs, the MAFFT --add alignments,
                                     PandaMap CSVs, the per-structure contact
                                     JSONs
    ANALYSIS    data/af3/analysis    pocket_columns.json and the other small
                                     derived JSON/CSV/FASTA

NOT tracked, and deliberately so -- these are large and genuinely regenerable
from the scripts in scripts/af3/:

    contacts_full.pkl      215 MB   every PandaMap row, 127,807 of them
    out_panel/             8.4 GB   the 675 AlphaFold predictions
    msa_struct.pkl                  column <-> residue maps per structure

Scripts needing those resolve AF3_ROOT, which defaults to the path on the
machine the work was done on and can be pointed elsewhere with the
BSH_AF3_ROOT environment variable:

    BSH_AF3_ROOT=/mnt/data/BSH_AF3 python scripts/make_resamine.py
"""
import os
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
REFSTRUCT = REPO / "data/af3/refstruct"
ANALYSIS = REPO / "data/af3/analysis"
SITE = REPO / "site"

AF3_ROOT = Path(os.environ.get("BSH_AF3_ROOT", "/home/adsiordia/AF3/BSH_AF3"))
AF3_ANALYSIS = AF3_ROOT / "analysis"


def need_af3(*what):
    """Fail with a useful message rather than a FileNotFoundError 40 lines later."""
    if not AF3_ROOT.is_dir():
        raise SystemExit(
            f"This script needs the large AlphaFold artefacts ({', '.join(what)}),\n"
            f"which are not in the repo. Expected them under:\n    {AF3_ROOT}\n"
            f"Set BSH_AF3_ROOT to wherever that directory lives, or regenerate it\n"
            f"with the scripts in scripts/af3/.")
    return AF3_ROOT
