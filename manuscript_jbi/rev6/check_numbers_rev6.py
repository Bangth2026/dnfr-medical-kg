"""Check that every number in the revised manuscript and supplement occurs either in the submitted JBI version (v6.0-jbi,
whose numbers check_numbers.py traced to the v5.0 documents generated from the result files) or in numbers.json, which
compute_numbers.py writes from results/ and results_rev6/. Citation numbers, years and bibliographic details are
reported too and have to be read by eye.
Usage: check_numbers_rev6.py REV_ms.docx REV_supp.docx -- V6_ms.docx V6_supp.docx numbers.json"""
import re, sys, json
from docx import Document
def text(path):
    if path.endswith(".json"):
        return json.dumps(json.load(open(path)), ensure_ascii=False).replace("−", "-")
    d = Document(path)
    t = [p.text for p in d.paragraphs] + [c.text for tb in d.tables for r in tb.rows for c in r.cells]
    return "\n".join(t).replace("−", "-").replace(" ", " ")
NUM = re.compile(r"(?<![\w.])[-+]?\d[\d,]*(?:\.\d+)?%?")
i = sys.argv.index("--")
new = "\n".join(text(p) for p in sys.argv[1:i]); ref = "\n".join(text(p) for p in sys.argv[i + 1:])
refnums = set(m.group().lstrip("+") for m in NUM.finditer(ref)); refnums |= set(x.lstrip("-") for x in refnums)
missing = {}
for m in NUM.finditer(new):
    tok = m.group().lstrip("+").rstrip(",")
    if tok in refnums or tok.lstrip("-") in refnums:
        continue
    missing.setdefault(tok, new[max(0, m.start() - 60): m.end() + 30].replace("\n", " "))
for k, v in missing.items():
    print(f"{k!r:>12}  …{v}…")
print(f"{len(missing)} numbers not found in the submitted version or numbers.json")
