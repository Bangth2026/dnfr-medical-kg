# -*- coding: utf-8 -*-
"""Build the JBI manuscript (.docx) from jbi_text.py, reusing the styles, equations, figures and tables of the
v5.0 manuscript (which were generated from the released result files)."""
import copy, re, sys
from docx import Document
from docx.shared import Pt
from docx.enum.text import WD_ALIGN_PARAGRAPH
import jbi_text as J, jbi_patch as P

SRC, OUT = sys.argv[1], sys.argv[2]
d = Document(SRC)
body = d.element.body
src_par = list(d.paragraphs)
src_tab = list(d.tables)
texts = [p.text for p in src_par]

# elements to keep from the source
def find(prefix):
    return next(p for p in src_par if p.text.startswith(prefix))
EQ = [find("μ(e) = σ("), find("Tp(μ1"), find("P(h, t) = 1 − (1 − β0)")]
FIG = {1: src_par[46], 2: src_par[125]}
assert FIG[1]._p.xpath('.//w:drawing') and FIG[2]._p.xpath('.//w:drawing')
CAP = {k: find(f"Table {k}.") for k in range(1, 7)}
TABLE1_NOTE = find("HKG, health knowledge graph")
TITLE_BLOCK = src_par[0:8]
saved = {"eq": [copy.deepcopy(p._p) for p in EQ], "fig": {k: copy.deepcopy(v._p) for k, v in FIG.items()},
         "cap": {k: copy.deepcopy(v._p) for k, v in CAP.items()}, "note1": copy.deepcopy(TABLE1_NOTE._p),
         "title": [copy.deepcopy(p._p) for p in TITLE_BLOCK],
         "tab": {k: copy.deepcopy(src_tab[k + 1]._tbl) for k in range(1, 7)}}   # src table i+1 = Table i
body_par = copy.deepcopy(find("A clinician asked")._p)
h1 = copy.deepcopy(find("1. Introduction")._p); h2 = copy.deepcopy(find("2.1 Medical")._p)
bullet = copy.deepcopy(src_par[9]._p)

# clear the body (keep sectPr)
sect = body.sectPr
for el in list(body):
    if el is not sect:
        body.remove(el)

def clone(tpl):
    el = copy.deepcopy(tpl)
    for r in el.xpath('./w:r|./w:hyperlink'):
        el.remove(r)
    body.insert(len(body) - 1, el)
    from docx.text.paragraph import Paragraph
    return Paragraph(el, d._body)

TOK = re.compile(r"(\*\*.+?\*\*|\*.+?\*|_\{.+?\}|\^\{.+?\})")
def runs(p, text, size=None, bold_all=False):
    for part in TOK.split(text):
        if not part:
            continue
        b = i = sub = sup = False
        if part.startswith("**"): part, b = part[2:-2], True
        elif part.startswith("*"): part, i = part[1:-1], True
        elif part.startswith("_{"): part, sub = part[2:-1], True
        elif part.startswith("^{"): part, sup = part[2:-1], True
        for sp in TOK.split(part) if (b or i) else [part]:      # nested sub/sup inside bold/italic
            if not sp: continue
            s2 = sup2 = False
            if sp.startswith("_{"): sp, s2 = sp[2:-1], True
            elif sp.startswith("^{"): sp, sup2 = sp[2:-1], True
            r = p.add_run(sp)
            r.bold = b or bold_all or None; r.italic = i or None
            r.font.subscript = (sub or s2) or None; r.font.superscript = (sup or sup2) or None
            if size: r.font.size = size

def para(text, tpl=None, **k):
    p = clone(tpl if tpl is not None else body_par); runs(p, text, **k); return p

def put(el):
    body.insert(len(body) - 1, copy.deepcopy(el))

# ---- front matter
for el in saved["title"]:
    put(el)
# fix the article-type line is fine; add a line for the journal
p = para("**Highlights**", tpl=body_par); p.alignment = None
for h in J.HIGHLIGHTS:
    para(h, tpl=bullet)
para("**Abstract**").alignment = None
for k, v in J.ABSTRACT:
    para(f"**{k}:** {v}")
para("**Keywords:** " + "; ".join(J.KEYWORDS)).alignment = None

BODY = list(J.BODY)
for i, t in P.NEW.items():
    BODY[i] = (BODY[i][0], t)
BODY = [b for i, b in enumerate(BODY) if i not in P.DROP]

eq_i = 0
for b in BODY:
    kind = b[0]
    if kind == "h1":
        para(b[1], tpl=h1)
    elif kind == "h2":
        para(b[1], tpl=h2)
    elif kind == "p":
        para(b[1])
    elif kind == "li":
        para(b[1], tpl=bullet)
    elif kind == "eq":
        put(saved["eq"][eq_i]); eq_i += 1
    elif kind == "fig":
        put(saved["fig"][b[1]])
        para(J.CAPTIONS[f"fig{b[1]}"], size=Pt(10))
    elif kind == "table":
        k = b[1] - 1          # BODY uses source-table index (1-based incl. significance/algorithm offset)
        tabno = {2: 1, 3: 2, 4: 3, 5: 4, 6: 5, 7: 6}[b[1]]
        put(saved["cap"][tabno]); put(saved["tab"][tabno])
        if tabno == 1:
            put(saved["note1"])
        else:
            para("", )
    elif kind == "sig":
        para("**Statement of significance**", size=Pt(10)).alignment = None
        for k, v in J.SIGNIFICANCE:
            para(f"**{k}.** {v}")
assert eq_i == 3

para("7. Declarations", tpl=h1)
for k, v in J.DECLARATIONS:
    para(f"**{k}.** {v}")

# ---- references: renumber by order of first citation
from refs_v5 import REFS            # {old number: text}
REFS.update(J.NEW_REFS)
order = []
def expand(s):
    out = []
    for part in s.split(","):
        part = part.strip()
        if "–" in part:
            a, c = part.split("–"); out += list(range(int(a), int(c) + 1))
        else:
            out.append(int(part))
    return out
for el in body.iter("{http://schemas.openxmlformats.org/wordprocessingml/2006/main}t"):
    for m in re.finditer(r"\[(\d+(?:[–,]\s?\d+)*)\]", el.text or ""):
        for n in expand(m.group(1)):
            if n not in order:
                order.append(n)
new = {o: i + 1 for i, o in enumerate(order)}
import json; json.dump(new, open("refmap.json", "w"))
def fmt(nums):
    nums = sorted(nums); out = []; i = 0
    while i < len(nums):
        j = i
        while j + 1 < len(nums) and nums[j + 1] == nums[j] + 1: j += 1
        out.append(f"{nums[i]}–{nums[j]}" if j - i >= 2 else (f"{nums[i]},{nums[j]}" if j == i + 1 else f"{nums[i]}"))
        i = j + 1
    return ",".join(out)
for el in body.iter("{http://schemas.openxmlformats.org/wordprocessingml/2006/main}t"):
    if el.text and "[" in el.text:
        el.text = re.sub(r"\[(\d+(?:[–,]\s?\d+)*)\]", lambda m: "[" + fmt([new[n] for n in expand(m.group(1))]) + "]", el.text)
para("References", tpl=h1)
for o in order:
    para(f"[{new[o]}] {REFS[o]}", size=Pt(10))
d.save(OUT)
unused = sorted(set(REFS) - set(order))
print("saved", OUT, "refs", len(order), "uncited (dropped):", unused)
