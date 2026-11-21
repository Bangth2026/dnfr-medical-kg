"""Check that every number in the JBI manuscript (and supplement) also occurs in the v5.0 manuscript or supplement,
which manuscript3/build_manuscript.py generated from the released result files.
Usage: python3 check_numbers.py JBI.docx [JBI_supp.docx] -- V5_manuscript.docx V5_supp.docx"""
import re, sys
from docx import Document
def text(path):
    d = Document(path)
    t = [p.text for p in d.paragraphs] + [c.text for tb in d.tables for r in tb.rows for c in r.cells]
    return "\n".join(t).replace("−", "-").replace(" ", " ")
NUM = re.compile(r"(?<![\w.])[-+]?\d[\d,]*(?:\.\d+)?%?")
i = sys.argv.index("--")
new = "\n".join(text(p) for p in sys.argv[1:i]); ref = "\n".join(text(p) for p in sys.argv[i + 1:])
refnums = set(m.group().lstrip("+") for m in NUM.finditer(ref))
refnums |= set(x.lstrip("-") for x in refnums)
missing = {}
for m in NUM.finditer(new):
    tok = m.group().lstrip("+").rstrip(",")
    if tok in refnums or tok.lstrip("-") in refnums:
        continue
    missing.setdefault(tok, new[max(0, m.start() - 60): m.end() + 30].replace("\n", " "))
for k, v in missing.items():
    print(f"{k!r:>12}  …{v}…")
print(f"{len(missing)} numbers not found in the v5.0 documents")
