# -*- coding: utf-8 -*-
"""Supplementary material for the JBI version: the v5.0 supplement plus
S5 (full methods, Algorithm S1, moved from the main text) and S6 (extended results: robustness, Table S19,
Figure S4, clinical assessment and cost), copied verbatim from the v5.0 manuscript."""
import copy, re, sys
from docx import Document
from docx.shared import Inches
from docx.text.paragraph import Paragraph
from docx.table import Table

MS, SUP, FIG3, OUT = sys.argv[1:5]
ms, sup = Document(MS), Document(SUP)
W = "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}"
els = list(ms.element.body.iterchildren())

def text(el):
    return "".join(t.text or "" for t in el.iter(W + "t"))
def idx(pred, start=0):
    return next(i for i in range(start, len(els)) if pred(els[i]))
def is_p(el): return el.tag == W + "p"
def is_tbl(el): return el.tag == W + "tbl"
def has_draw(el): return bool(el.xpath('.//w:drawing'))

a = idx(lambda e: is_p(e) and text(e).strip() == "3.1 Problem setting")
b = idx(lambda e: is_p(e) and text(e).strip() == "4. Results")
c = idx(lambda e: is_p(e) and text(e).startswith("4.6 Robustness"))
dd = idx(lambda e: is_p(e) and text(e).strip() == "5. Discussion")

sbody = sup.element.body
ref_head = next(p for p in sup.paragraphs if p.text.strip() == "References")._p
h1 = copy.deepcopy(next(p for p in sup.paragraphs if p.text.startswith("S4."))._p)
normal = copy.deepcopy(next(p for p in sup.paragraphs if len(p.text) > 200)._p)

def new_par(tpl, txt):
    el = copy.deepcopy(tpl)
    for r in el.xpath('./w:r'):
        el.remove(r)
    p = Paragraph(el, sup._body); p.add_run(txt)
    ref_head.addprevious(el); return el

def put(el):
    el = copy.deepcopy(el); ref_head.addprevious(el); return el

RENAME = [(r"Only correct predictions were rated, so this part of the assessment cannot show whether the explanations help to separate correct from incorrect predictions\.",
           "This first assessment covered correct predictions only; Section S7 reports a second, blinded rating of correct and incorrect predictions by a physician."),
          (r"Algorithm 1\b", "Algorithm S1"), (r"Table 7\b", "Table S19"), (r"Figure 3\b", "Figure S4")]
import json
NEWREF = {int(k): v for k, v in json.load(open("refmap.json")).items()}
def expand(s):
    out = []
    for part in s.split(","):
        part = part.strip()
        if "–" in part:
            x, y = part.split("–"); out += list(range(int(x), int(y) + 1))
        else:
            out.append(int(part))
    return out
def fmt(nums):
    nums = sorted(nums); out = []; i = 0
    while i < len(nums):
        j = i
        while j + 1 < len(nums) and nums[j + 1] == nums[j] + 1: j += 1
        out.append(f"{nums[i]}–{nums[j]}" if j - i >= 2 else (f"{nums[i]},{nums[j]}" if j == i + 1 else f"{nums[i]}"))
        i = j + 1
    return ",".join(out)
def rename(el):
    for t in el.iter(W + "t"):
        if t.text:
            for x, y in RENAME:
                t.text = re.sub(x, y, t.text)
            t.text = re.sub(r"\[(\d+(?:[–,]\s?\d+)*)\]",
                            lambda m: "[" + fmt([NEWREF[n] for n in expand(m.group(1))]) + "]", t.text)

def demote(el, new):   # heading text replacement
    ts = list(el.iter(W + "t")); ts[0].text = new
    for t in ts[1:]: t.text = ""

# ---- S5
new_par(h1, "S5. Full description of the methods")
new_par(normal, "The main text of the article describes the methods in shortened form. This section gives the full "
        "description, as written for the earlier, longer version of the article; section, equation, figure and table "
        "numbers refer to the main text, which keeps the same numbering of Sections 3.1–3.6, Equations 1–3 and "
        "Tables 1–6.")
skip_tab1 = False
for i in range(a, b):
    el = els[i]
    t = text(el)
    if is_p(el) and (has_draw(el) or t.startswith("Figure 1.") or t.startswith("Table 1.") or t.startswith("HKG, health knowledge graph")):
        continue
    if is_tbl(el) and not text(el).startswith("Algorithm 1"):
        continue      # Table 1 stays in the main text
    e2 = put(el); rename(e2)
    if is_p(el) and re.match(r"3\.\d ", t.strip()):
        demote(e2, "S5." + t.strip()[2:].replace(" ", " ", 1))
# ---- S6
new_par(h1, "S6. Extended results moved from the main text")
for i in range(c, dd):
    el = els[i]; t = text(el)
    if is_tbl(el) and text(el).strip().startswith("Concision"):
        continue                                  # Table 6 is in the main text
    if is_tbl(el):
        # the only other table in this range is Table 7, now Table S19
        pass
    if is_p(el) and t.startswith("Table 6."):
        continue
    if is_p(el) and has_draw(el):
        p = Paragraph(put(normal), sup._body)
        for r in list(p._p.xpath('./w:r')): p._p.remove(r)
        p.add_run().add_picture(FIG3, width=Inches(6.0)); continue
    e2 = put(el); rename(e2)
    m = re.match(r"4\.(\d) (.*)", t.strip())
    if is_p(el) and m:
        demote(e2, f"S6.{int(m.group(1)) - 5} {m.group(2)}")

sup.save(OUT)
print("saved", OUT)

# ---- S7: second clinical rater (results/expert_review2.json)
import json as _j
X = _j.load(open("../results/expert_review2.json"))
sup = Document(OUT)
ref_head = next(p for p in sup.paragraphs if p.text.strip() == "References")._p
h1 = copy.deepcopy(next(p for p in sup.paragraphs if p.text.startswith("S4."))._p)
normal = copy.deepcopy(next(p for p in sup.paragraphs if len(p.text) > 200)._p)
new_par(h1, "S7. Second clinical rater")
A = X["agreement"]
new_par(normal, "A physician (internal medicine, seven years of clinical practice, not an author) rated a second, "
        "blinded sheet of 40 disease–symptom pairs with their three leading paths: on each task, 10 correct predictions "
        "that the first rater had also assessed and 10 incorrect top-10 predictions, that is, symptoms that DNFR ranked in "
        "its top 10 but that the graph does not record for the disease. Order and identifiers revealed neither correctness "
        "nor overlap with the first rater, and scores were hidden. On the overlapping items, agreement with the first rater "
        f"was {100*A['relations_correct']['agreement']:.0f}% for the correctness of relations (Cohen's κ = "
        f"{A['relations_correct']['kappa']:.2f}, {A['relations_correct']['n']} paths), "
        f"{100*A['rationale']['agreement']:.0f}% for the kind of rationale (κ = {A['rationale']['kappa']:.2f}) and "
        f"{100*A['pair_plausible']['agreement']:.0f}% for plausibility ({A['pair_plausible']['n']} pairs; κ is not "
        "informative because both raters judged nearly every correct pair plausible). Table S20 gives the second rater's "
        "judgements by task. The sheet (dnfr/expert_sheet2.py), its key and the ratings (results/expert_review2_filled.xlsx, "
        "with the rater's name withheld) are released; dnfr/expert_summary2.py computes these figures.")
cap = new_par(normal, "Table S20. Second rater's judgements by task and by correctness of the prediction. Relations "
              "correct: paths on which every relation was judged to agree with biomedical knowledge. Rationale: "
              "mechanistic / associative / irrelevant.")
cap_r = cap.xpath('./w:r')[0]; cap_r.insert(0, copy.deepcopy(cap_r.xpath('./w:rPr')[0]) if cap_r.xpath('./w:rPr') else cap_r.makeelement(W + "rPr", {}))
rows = [("Task", "Prediction", "Pairs judged plausible", "Relations correct (paths)", "Rationale M / A / I",
         "Mean usefulness (1–5)")]
names = {"het_DpS": "Hetionet disease–symptom", "hkg_DpS": "EHR graph (graded)"}
for t in ("het_DpS", "hkg_DpS"):
    for c in ("True", "False"):
        v = X["by_task"][f"{t}_{c}"]
        rows.append((names[t], "correct" if c == "True" else "incorrect", f"{v['plausible']} of {v['pairs']}",
                     f"{v['rel_yes']} of {v['paths']}", f"{v['M']} / {v['A']} / {v['I']}", f"{v['use']:.1f}"))
rows.append(("Both tasks", "AUC, correct vs incorrect", f"{X['auc_plausible']:.2f}", "", "", f"{X['auc_usefulness']:.2f}"))
tb = sup.add_table(rows=len(rows), cols=6)
try:
    tb.style = "Table Grid"
except Exception:
    pass
for i, r in enumerate(rows):
    for j, x in enumerate(r):
        tb.cell(i, j).text = x
        if i == 0:
            for run in tb.cell(i, j).paragraphs[0].runs: run.bold = True
ref_head.addprevious(tb._tbl)
# renumber the citations of S1-S4 (v5.0 numbering) to the main article's numbering and use its reference list
body_els = list(sup.element.body.iterchildren())
s5 = next(i for i, el in enumerate(body_els) if el.tag == W + "p" and text(el).startswith("S5. Full description"))
for el in body_els[:s5]:
    for t in el.iter(W + "t"):
        if t.text and "[" in t.text:
            t.text = re.sub(r"\[(\d+(?:[–,]\s?\d+)*)\]",
                            lambda m: "[" + fmt([NEWREF[n] for n in expand(m.group(1))]) + "]", t.text)
ref_tpl = None
for el in list(sup.element.body.iterchildren())[body_els.index(ref_head) + 1:]:
    if el.tag == W + "p" and re.match(r"\[\d+\]", text(el)):
        ref_tpl = ref_tpl if ref_tpl is not None else copy.deepcopy(el)
        sup.element.body.remove(el)
from refs_v5 import REFS
import jbi_text as _J
REFS.update(_J.NEW_REFS)
prev = ref_head
for old, new_n in sorted(NEWREF.items(), key=lambda kv: kv[1]):
    el = copy.deepcopy(ref_tpl)
    for r in el.xpath('./w:r'):
        el.remove(r)
    Paragraph(el, sup._body).add_run(f"[{new_n}] {REFS[old]}")
    prev.addnext(el); prev = el
sup.save(OUT)
print("added S7; supplement references renumbered")
