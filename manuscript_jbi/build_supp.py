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

RENAME = [(r"Algorithm 1\b", "Algorithm S1"), (r"Table 7\b", "Table S19"), (r"Figure 3\b", "Figure S4")]
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
        "Tables 1–6. Citations in Sections S5 and S6 refer to the reference list of the main article.")
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
