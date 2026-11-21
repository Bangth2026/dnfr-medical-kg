"""Helpers to rewrite paragraphs of an existing .docx with python-docx while keeping paragraph styles.
Markup in new text: **bold**, *italic*, _{sub}, ^{sup}."""
import re, copy
from docx.text.paragraph import Paragraph
TOK = re.compile(r"(\*\*.+?\*\*|\*.+?\*|_\{.+?\}|\^\{.+?\})")

def _runs(p, text, rpr=None):
    for part in TOK.split(text):
        if not part:
            continue
        b = i = sub = sup = False
        if part.startswith("**"): part, b = part[2:-2], True
        elif part.startswith("*"): part, i = part[1:-1], True
        elif part.startswith("_{"): part, sub = part[2:-1], True
        elif part.startswith("^{"): part, sup = part[2:-1], True
        inner = TOK.split(part) if (b or i) else [part]
        for sp in inner:
            if not sp: continue
            s2 = p2 = False
            if sp.startswith("_{"): sp, s2 = sp[2:-1], True
            elif sp.startswith("^{"): sp, p2 = sp[2:-1], True
            elif sp.startswith("*") and sp.endswith("*") and len(sp) > 2 and b: sp, i2 = sp[1:-1], True
            r = p.add_run(sp)
            if rpr is not None:
                r._r.insert(0, copy.deepcopy(rpr))
            if b: r.bold = True
            if i: r.italic = True
            if sub or s2: r.font.subscript = True
            if sup or p2: r.font.superscript = True

def set_text(p, text):
    """Replace all runs of paragraph p by text with markup; keeps pPr and the base run properties of its last
    non-bold run (font, size), dropping bold/italic so that markup decides them."""
    rpr = None
    for r in p.runs:
        if r._r.rPr is not None:
            rpr = copy.deepcopy(r._r.rPr)
    if rpr is not None:
        for tag in ("b", "i", "bCs", "iCs", "vertAlign"):
            for el in rpr.findall("{http://schemas.openxmlformats.org/wordprocessingml/2006/main}" + tag):
                rpr.remove(el)
    for el in list(p._p):
        if el.tag.endswith("}r") or el.tag.endswith("}hyperlink"):
            p._p.remove(el)
    _runs(p, text, rpr)
    return p

def insert_after(p, text, template=None):
    """New paragraph after p, with the pPr of template (default p)."""
    tpl = template if template is not None else p
    new = copy.deepcopy(tpl._p)
    for el in list(new):
        if not el.tag.endswith("}pPr"):
            new.remove(el)
    p._p.addnext(new)
    np_ = Paragraph(new, p._parent)
    rpr = None
    for r in tpl.runs:
        if r._r.rPr is not None:
            rpr = copy.deepcopy(r._r.rPr)
    if rpr is not None:
        for tag in ("b", "i", "bCs", "iCs", "vertAlign"):
            for el in rpr.findall("{http://schemas.openxmlformats.org/wordprocessingml/2006/main}" + tag):
                rpr.remove(el)
    _runs(np_, text, rpr)
    return np_

def delete(p):
    p._p.getparent().remove(p._p)

def cell_text(cell, text):
    """Replace the text of a table cell (first paragraph keeps its formatting; extra paragraphs removed)."""
    ps = cell.paragraphs
    set_text(ps[0], text)
    for q in ps[1:]:
        delete(q)
