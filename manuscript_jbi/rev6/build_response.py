# -*- coding: utf-8 -*-
"""Response to the reviewer as .docx. Usage: build_response.py numbers.json OUT.docx"""
import sys, os, json, re
from docx import Document
from docx.shared import Pt, Cm, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import response_text as R

N = json.load(open(sys.argv[1])); OUT = sys.argv[2]
def fill(s):
    for _ in range(5):
        s = re.sub(r"<<([A-Z0-9_]+)>>", lambda m: N[m.group(1)], s)
    return s
d = Document()
st = d.styles["Normal"]; st.font.name = "Times New Roman"; st.font.size = Pt(11)
st.paragraph_format.space_after = Pt(6); st.paragraph_format.line_spacing = 1.15
for s in ("Heading 1", "Heading 2"):
    h = d.styles[s]; h.font.name = "Times New Roman"; h.font.color.rgb = RGBColor(0, 0, 0)
    h.font.size = Pt(13 if s == "Heading 1" else 11.5); h.paragraph_format.space_before = Pt(12)
    h.paragraph_format.space_after = Pt(6); h.paragraph_format.keep_with_next = True
for sec in d.sections:
    sec.left_margin = sec.right_margin = Cm(2.5); sec.top_margin = sec.bottom_margin = Cm(2.2)
t = d.add_paragraph(); r = t.add_run("Response to the Reviewer"); r.bold = True; r.font.size = Pt(15)
p = d.add_paragraph(); r = p.add_run("Manuscript: Which parts of a neuro-fuzzy path reasoner should be learned? A controlled "
                                     "component analysis on medical knowledge graphs (Journal of Biomedical Informatics, "
                                     "Research Paper)"); r.italic = True
for para in R.INTRO:
    d.add_paragraph(fill(para))
for kind, txt in R.ITEMS:
    txt = fill(txt)
    if kind == "h":
        d.add_heading(txt, level=1)
    elif kind == "rc":
        p = d.add_paragraph(); p.paragraph_format.left_indent = Cm(0.6)
        r = p.add_run("Reviewer: "); r.bold = True; r.italic = True
        r = p.add_run(txt); r.italic = True
    elif kind == "r":
        p = d.add_paragraph()
        if not re.match(r"^\d+\.", txt):
            r = p.add_run("Response: "); r.bold = True
        p.add_run(txt)
for para in R.CLOSING:
    d.add_paragraph(fill(para))
d.save(OUT); print("saved", OUT)
