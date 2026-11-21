# -*- coding: utf-8 -*-
"""Build the revised supplement: substring fixes, Table S8/S13/S16 caption fixes, new Section S8 with Tables S21-S24,
references renumbered with the map of the main text. Usage: build_supp.py IN.docx OUT.docx numbers.json refmap.json"""
import sys, os, json, copy, re
from docx import Document
from docx.text.paragraph import Paragraph
from docx.table import Table
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from docxpatch import set_text, insert_after, delete, cell_text, _runs
import supp_text_rev as T

IN, OUT, NUMS, REFMAP = sys.argv[1:5]
N = json.load(open(NUMS)); RM = {int(k): v for k, v in json.load(open(REFMAP)).items()}
d = Document(IN)
P = list(d.paragraphs)

def fill(s):
    for _ in range(5):
        s = re.sub(r"<<([A-Z0-9_]+)>>", lambda m: N[m.group(1)], s)
    return s

def replace_in(p, old, new):
    """Replace a substring that lies inside one run (or rewrite the paragraph if it spans runs)."""
    for r in p.runs:
        if old in r.text:
            r.text = r.text.replace(old, new); return True
    full = p.text
    if old in full:
        # keep a bold lead-in run if present
        lead = p.runs[0].text if p.runs and p.runs[0].bold else ""
        rest = full[len(lead):].replace(old, new)
        set_text(p, (f"**{lead.strip()}** " if lead else "") + rest)
        return True
    raise KeyError(old[:60])

for i, subs in T.SUBS.items():
    for old, new in subs:
        replace_in(P[i], old, new)
rename = [("DNFR 2.0 (R1–R4)", "DNFR + R1–R4"), ("DNFR 2.0", "DNFR + R1–R4")]
def all_pars():
    for p in d.paragraphs: yield p
    for tb in d.tables:
        for row in tb.rows:
            for c in row.cells:
                for p in c.paragraphs: yield p
for p in all_pars():
    for old, new in rename:
        for r in p.runs:
            if old in r.text: r.text = r.text.replace(old, new)

# Table S8: split 0 has two disjoint ranges within the threshold
def table_after_caption(prefix):
    for el in d.element.body.iterchildren():
        pass
    cap = next(p for p in d.paragraphs if p.text.startswith(prefix))
    el = cap._p.getnext()
    while el is not None and not el.tag.endswith("}tbl"):
        el = el.getnext()
    return cap, Table(el, d._body)
cap8, t8 = table_after_caption("Table S8.")
for r in t8.rows:
    if r.cells[0].text.strip() == "0" and "[−5, 0.25]" in r.cells[2].text:
        cell_text(r.cells[2], "[−5, −2.5] and [0.1, 0.25]")
    if r.cells[2].text.strip() == "[0.5, 0.5]":
        cell_text(r.cells[2], "{0.5}")
replace_in(cap8, "a measure of the precision of the profile.",
           "a measure of the precision of the profile; it is dominated by single chains that settled in a poorer basin, "
           "and Table S21 gives the agreement of the two best chains at every point. On split 0 the points within the "
           "threshold form two disjoint ranges.")
cap16, t16 = table_after_caption("Table S16.")
replace_in(cap16, "a measure of the precision of the profile.",
           "a measure of the precision of the profile; on splits 0 and 2 it is dominated by single chains in a poorer "
           "basin (Table S21), while on split 1 the two best chains also disagree, so that profile is not resolved.")
cap13 = next(p for p in d.paragraphs if p.text.startswith("Table S13."))
replace_in(cap13, "available for the runs made after the second version of this analysis",
           "available only for the runs whose logs recorded it")

# ---------------- new tables from the template of Table S8
TPL = t8._tbl
def make_table(after_el, header, rows, widths=None):
    new = copy.deepcopy(TPL)
    trs = new.findall("{http://schemas.openxmlformats.org/wordprocessingml/2006/main}tr")
    h_tr, b_tr, l_tr = trs[0], trs[1], trs[-1]
    grid = new.find("{http://schemas.openxmlformats.org/wordprocessingml/2006/main}tblGrid")
    for tr in trs: new.remove(tr)
    W = "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}"
    n = len(header); total = 9000
    widths = widths or [total // n] * n
    for g in list(grid): grid.remove(g)
    for w in widths:
        g = copy.deepcopy(TPL.find(W + "tblGrid")[0]); g.set(W + "w", str(w)); grid.append(g)
    def mk(tpl, cells):
        tr = copy.deepcopy(tpl)
        tcs = tr.findall(W + "tc")
        for tc in tcs[1:]: tr.remove(tc)
        base = tcs[0]; tr.remove(base)
        for txt, w in zip(cells, widths):
            tc = copy.deepcopy(base)
            tc.find(W + "tcPr").find(W + "tcW").set(W + "w", str(w))
            tr.append(tc)
        new.append(tr)
        tb = Table(new, d._body)
        for c, txt in zip(tb.rows[-1].cells, cells):
            cell_text(c, txt)
    mk(h_tr, header)
    for k, r in enumerate(rows):
        mk(l_tr if k == len(rows) - 1 else b_tr, r)
    after_el.addnext(new)
    return new

# ---------------- Section S8 before the references
refh = next(p for p in d.paragraphs if p.text.strip() == "References" and p.style.name.startswith("Heading"))
h1_tpl = next(p for p in d.paragraphs if p.style.name == "Heading 1")
h2_tpl = next(p for p in d.paragraphs if p.style.name == "Heading 2")
body_tpl = P[104]
cap_tpl = cap8
prev = refh._p.getprevious()
anchor = Paragraph(prev, d._body) if prev.tag.endswith("}p") else None
def add_par(after_el, text, tpl):
    new = copy.deepcopy(tpl._p)
    for el in list(new):
        if not el.tag.endswith("}pPr"): new.remove(el)
    after_el.addnext(new)
    p = Paragraph(new, d._body)
    rpr = None
    for r in tpl.runs:
        if r._r.rPr is not None: rpr = copy.deepcopy(r._r.rPr)
    if rpr is not None:
        for tag in ("b", "i", "bCs", "iCs", "vertAlign"):
            for e in rpr.findall("{http://schemas.openxmlformats.org/wordprocessingml/2006/main}" + tag): rpr.remove(e)
    _runs(p, text, rpr)
    return new
cur = prev
cur = add_par(cur, T.NEW_SECTION_TITLE, h1_tpl)
for kind, val in T.NEW_SECTION:
    if kind == "h2":
        cur = add_par(cur, val, h2_tpl)
    elif kind == "p":
        txt = fill(val)
        for para in txt.split("\n\n"):
            cur = add_par(cur, para, body_tpl)
    elif kind == "table":
        spec = N.get("TAB_" + val)
        if not spec: continue
        cur = add_par(cur, spec["caption"], cap_tpl)
        cur = make_table(cur, spec["header"], spec["rows"], spec.get("widths"))

# ---------------- references: renumber with the main-text map, rewrite the list
CIT = re.compile(r"\[(\d+(?:\s*[–,-]\s*\d+)*)\]")
def expand(s):
    out = []
    for part in re.split(r"\s*,\s*", s):
        if re.search(r"[–-]", part):
            a, b = re.split(r"\s*[–-]\s*", part); out += list(range(int(a), int(b) + 1))
        else: out.append(int(part))
    return out
def compress(nums):
    nums = sorted(set(nums)); out = []; i = 0
    while i < len(nums):
        j = i
        while j + 1 < len(nums) and nums[j + 1] == nums[j] + 1: j += 1
        out.append(f"{nums[i]}–{nums[j]}" if j - i >= 2 else (f"{nums[i]},{nums[j]}" if j == i + 1 else f"{nums[i]}"))
        i = j + 1
    return ",".join(out)
ref_start = next(i for i, p in enumerate(d.paragraphs) if p.text.strip() == "References" and p.style.name.startswith("Heading"))
ref_pars = [p for p in d.paragraphs[ref_start + 1:] if re.match(r"\[\d+\]", p.text)]
refs = {int(re.match(r"\[(\d+)\]", p.text).group(1)): re.sub(r"^\[\d+\]\s*", "", p.text) for p in ref_pars}
refs.update({int(k): v for k, v in N["NEW_REFS"].items()})
used = set()
body_pars = [p for p in all_pars() if p not in ref_pars]
for p in list(all_pars()):
    if any(p._p is q._p for q in ref_pars): continue
    for r in p.runs:
        def rep(m):
            nums = expand(m.group(1))
            if not all(n in RM for n in nums): return m.group(0)
            used.update(nums)
            return "[" + compress([RM[n] for n in nums]) + "]"
        if CIT.search(r.text):
            r.text = CIT.sub(rep, r.text)
# list only the references the supplement cites, in main-text numbering order
tpl = ref_pars[0]
anchor = d.paragraphs[ref_start]
for p in ref_pars: delete(p)
last = anchor._p
for old in sorted(used, key=lambda o: RM[o]):
    last = add_par(last, f"[{RM[old]}] {refs[old]}", tpl)
d.save(OUT)
print("saved", OUT, "refs cited in supplement:", len(used))
