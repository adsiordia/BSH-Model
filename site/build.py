"""Inline all data files into template.html -> index.html (self-contained)."""
from pathlib import Path
HERE = Path(__file__).resolve().parent
html = (HERE/"template.html").read_text()
for ph, f in [("/*__DATA__*/{}", "data.json"), ("/*__CHEM__*/{}", "chem.json"),
              ("/*__TRENDS__*/{}", "trends.json"),
              ("/*__METHODS__*/{}", "methods.json"),
              ("/*__TRIMMING__*/{}", "trimming.json"),
              ("/*__RESCUE__*/{}", "rescue.json"),
              ("/*__POOLING__*/{}", "pooling.json"),
              ("/*__THEDATA__*/{}", "thedata.json"),
              ("/*__CANDIDATES__*/{}", "candidates.json"),
              ("/*__CONJUGATES__*/{}", "conjugates.json")]:
    html = html.replace(ph, (HERE/f).read_text())
out = HERE/"index.html"; out.write_text(html)
print(f"wrote {out}  ({out.stat().st_size/1e6:.2f} MB)")
