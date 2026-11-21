# -*- coding: utf-8 -*-
"""Build the revised manuscript from Manuscript_JBI_DNFR_v2.docx: paragraph rewrites (ms_text_rev.py), table edits,
new Figure 2, references renumbered by first appearance. Usage: build_ms.py IN.docx OUT.docx numbers.json fig2.png"""
import sys, os, json, copy, re
from docx import Document
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from docxpatch import set_text, insert_after, delete, cell_text
import ms_text_rev as T

IN, OUT, NUMS, FIG2 = sys.argv[1:5]
N = json.load(open(NUMS))
d = Document(IN)
P = list(d.paragraphs)


def fill(s):
    def rep(m):
        k = m.group(1)
        if k not in N:
            raise KeyError(k)
        return N[k]
    for _ in range(5):
        s = re.sub(r"<<([A-Z0-9_]+)>>", rep, s)
    return s


# ---------------- paragraphs
for dct in (T.HIGHLIGHTS, T.ABSTRACT, T.SIGNIF, T.INTRO, T.RELATED, T.METHODS, T.RESULTS, T.RESULTS2, T.RESULTS3,
            T.EXPL, T.DISC, T.LIMIT, T.CONCL, T.DECL):
    for i, txt in dct.items():
        set_text(P[i], fill(txt))
for i, txt in N.get("_extra_after", {}).items():      # extra paragraphs inserted after a paragraph index
    last = P[int(i)]
    for t in txt:
        last = insert_after(last, fill(t))

# global renames
def rename_all(old, new):
    for p in d.paragraphs:
        if old in p.text:
            for r in p.runs:
                if old in r.text:
                    r.text = r.text.replace(old, new)
    for tb in d.tables:
        for row in tb.rows:
            for c in row.cells:
                for p in c.paragraphs:
                    for r in p.runs:
                        if old in r.text:
                            r.text = r.text.replace(old, new)
rename_all("DNFR 2.0 (R1–R4)", "DNFR + R1–R4")
rename_all("DNFR 2.0", "DNFR + R1–R4")

# ---------------- Table 2
t2 = d.tables[1]
rows = t2.rows
def row_index(tb, start):
    for i, r in enumerate(tb.rows):
        if r.cells[0].text.startswith(start):
            return i
    raise KeyError(start)
def add_row_after(tb, idx, cells, template_idx=None):
    tpl = tb.rows[template_idx if template_idx is not None else idx]._tr
    new = copy.deepcopy(tpl)
    tb.rows[idx]._tr.addnext(new)
    r = tb.rows[idx + 1]
    for c, txt in zip(r.cells, cells):
        cell_text(c, txt)
    return idx + 1
i = row_index(t2, "DWPC-LR + degree + prior")
i = add_row_after(t2, i, N["T2_DWPC_SM"], template_idx=i)
i = add_row_after(t2, i, N["T2_DWPC_NN"], template_idx=i)
i = row_index(t2, "Popularity")
add_row_after(t2, i, N["T2_RANDOM"], template_idx=i)
# drop the untuned embeddings (moved to the Supplement)
for start in ("TransE", "RotatE", "Knowledge graph embedd"):
    j = row_index(t2, start); tr = t2.rows[j]._tr; tr.getparent().remove(tr)
set_text(P[74], fill(N["CAP_T2"]))

# ---------------- Table 3: the cell with five splits
t3 = d.tables[2]
for r in t3.rows:
    for c in r.cells:
        if c.text.strip().endswith("1/5"):
            cell_text(c, c.text.strip() + " (5 splits)")
set_text(P[78], fill(N["CAP_T3"]))

# ---------------- Table 5
t5 = d.tables[4]
for row_start, vals in N["T5_A"]:
    j = row_index(t5, row_start)
    for c, v in zip(t5.rows[j].cells[1:], vals):
        cell_text(c, v)
j = row_index(t5, "Popularity")
add_row_after(t5, j, N["T5_A_RANDOM"], template_idx=j)
cell_text(t5.rows[row_index(t5, "A. Disease-centred")].cells[0], N["T5_A_TITLE"])
for c in t5.rows[row_index(t5, "A. Disease-centred")].cells[1:]:
    pass
# panels D and E appended at the end, using the panel-title row and a body row as templates
title_tpl = row_index(t5, "B. Random splits")
body_tpl = row_index(t5, "DNFR (learned p")
last = len(t5.rows) - 1
for panel in N["T5_NEW_PANELS"]:
    new = copy.deepcopy(t5.rows[title_tpl]._tr); t5.rows[last]._tr.addnext(new); last += 1
    for c in t5.rows[last].cells:
        cell_text(c, panel["title"])
    for cells in panel["rows"]:
        new = copy.deepcopy(t5.rows[body_tpl]._tr); t5.rows[last]._tr.addnext(new); last += 1
        for c, v in zip(t5.rows[last].cells, cells):
            cell_text(c, v)
set_text(P[95], fill(N["CAP_T5"]))

# ---------------- Table 6
t6 = d.tables[5]
j = row_index(t6, "DWPC-LR: facts covering 50%")
j = add_row_after(t6, j, N["T6_NN_SHARE"], template_idx=j)
j = add_row_after(t6, j, N["T6_NN_K50"], template_idx=j)
j = row_index(t6, "DWPC-LR: its leading f")
add_row_after(t6, j, N["T6_NN_CF"], template_idx=j)
set_text(P[100], fill(N["CAP_T6"]))

# ---------------- Figure 2: new image, same width
from docx.shared import Emu
rel = d.part.rels["rId9"]
rel.target_part._blob = open(FIG2, "rb").read()
from PIL import Image
w, h = Image.open(FIG2).size
cx = 5760720; cy = int(cx * h / w)
for el in d.element.body.iter():
    if el.tag.endswith("}extent") and el.get("cx") == "5760720" and el.get("cy") == "1637852":
        el.set("cy", str(cy))
    if el.tag.endswith("}ext") and el.get("cx") == "5760720" and el.get("cy") == "1637852":
        el.set("cy", str(cy))

# ---------------- references: drop [25], add new ones, renumber by first appearance
ref_start = next(i for i, p in enumerate(d.paragraphs) if p.text.strip() == "References")
refs = {}
ref_pars = []
for p in d.paragraphs[ref_start + 1:]:
    m = re.match(r"\[(\d+)\]\s*(.*)", p.text)
    if m:
        refs[int(m.group(1))] = m.group(2); ref_pars.append(p)
for k, v in N["NEW_REFS"].items():
    refs[int(k)] = v
refs.pop(25, None)
# citation tokens in body (paragraphs before the reference list) and tables, in document order
body_items = []
for el in d.element.body.iterchildren():
    tag = el.tag.split("}")[1]
    if tag == "p":
        from docx.text.paragraph import Paragraph
        p = Paragraph(el, d._body)
        if p.text.strip() == "References":
            break
        body_items.append(("p", p))
    elif tag == "tbl":
        from docx.table import Table
        tb = Table(el, d._body)
        for row in tb.rows:
            seen = set()
            for c in row.cells:
                if id(c._tc) in seen: continue
                seen.add(id(c._tc))
                for p in c.paragraphs:
                    body_items.append(("c", p))
CIT = re.compile(r"\[(\d+(?:\s*[–,-]\s*\d+)*)\]")
def expand(s):
    out = []
    for part in re.split(r"\s*,\s*", s):
        if re.search(r"[–-]", part):
            a, b = re.split(r"\s*[–-]\s*", part); out += list(range(int(a), int(b) + 1))
        else:
            out.append(int(part))
    return out
order = []
for _, p in body_items:
    for m in CIT.finditer(p.text):
        for n in expand(m.group(1)):
            if n in refs and n not in order:
                order.append(n)
missing = [k for k in refs if k not in order]
if missing:
    print("WARNING: references never cited:", missing)
    order += missing
newnum = {old: i + 1 for i, old in enumerate(order)}
def compress(nums):
    nums = sorted(set(nums)); out = []; i = 0
    while i < len(nums):
        j = i
        while j + 1 < len(nums) and nums[j + 1] == nums[j] + 1: j += 1
        out.append(f"{nums[i]}–{nums[j]}" if j - i >= 2 else (f"{nums[i]},{nums[j]}" if j == i + 1 else f"{nums[i]}"))
        i = j + 1
    return ",".join(out)
def renum_text(s):
    def rep(m):
        nums = [n for n in expand(m.group(1))]
        if not all(n in newnum for n in nums):
            return m.group(0)
        return "[" + compress([newnum[n] for n in nums]) + "]"
    return CIT.sub(rep, s)
def renum_par(p):
    if not CIT.search(p.text): return
    # citations may span runs; rebuild runs only where needed by operating run by run first
    ok = True
    for r in p.runs:
        r.text = renum_text(r.text)
    # verify nothing left split across runs
for _, p in body_items:
    renum_par(p)
# rewrite the reference list
anchor = d.paragraphs[ref_start]
for p in ref_pars:
    delete(p)
last = anchor
tpl_ref = None
for old in order:
    last = insert_after(last, f"[{newnum[old]}] {refs[old]}", template=ref_pars[0] if ref_pars else anchor)
json.dump({str(k): v for k, v in newnum.items()}, open(OUT + ".refmap.json", "w"))
d.save(OUT)
print("saved", OUT, "refs", len(order))
