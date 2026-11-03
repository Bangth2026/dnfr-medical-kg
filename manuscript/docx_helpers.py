"""Small python-docx helpers: Times New Roman 12 pt, 1.5 line spacing, three-line tables, inline markup
**bold**, *italic*, ^superscript^, ~subscript~."""
import re
from docx import Document
from docx.shared import Pt, Inches, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.oxml.ns import qn
from docx.oxml import OxmlElement

FONT = "Times New Roman"
TOKEN = re.compile(r"(\*\*.+?\*\*|\*.+?\*|\^.+?\^|~.+?~)")


def new_document(line_numbers=True):
    d = Document()
    s = d.sections[0]
    s.page_width, s.page_height = Inches(8.27), Inches(11.69)
    for side in ("left_margin", "right_margin", "top_margin", "bottom_margin"):
        setattr(s, side, Inches(1))
    st = d.styles["Normal"]; st.font.name = FONT; st.font.size = Pt(12)
    st.element.rPr.rFonts.set(qn("w:eastAsia"), FONT)
    for name, size in (("Heading 1", 14), ("Heading 2", 12), ("Heading 3", 12)):
        h = d.styles[name]; h.font.name = FONT; h.font.size = Pt(size); h.font.bold = True
        h.font.color.rgb = RGBColor(0, 0, 0); h.font.italic = (name == "Heading 3")
        rpr = h.element.get_or_add_rPr(); rf = rpr.find(qn("w:rFonts"))
        if rf is None:
            rf = OxmlElement("w:rFonts"); rpr.append(rf)
        for a in ("w:ascii", "w:hAnsi", "w:eastAsia", "w:cs"):
            rf.set(qn(a), FONT)
        h.paragraph_format.space_before = Pt(18 if name == "Heading 1" else 12)
        h.paragraph_format.space_after = Pt(6)
    if line_numbers:
        ln = OxmlElement("w:lnNumType"); ln.set(qn("w:countBy"), "1"); ln.set(qn("w:restart"), "continuous")
        ln.set(qn("w:distance"), "283")
        s._sectPr.append(ln)
    # page number footer
    p = s.footer.paragraphs[0]; p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r = p.add_run(); _field(r, "PAGE"); r.font.size = Pt(10); r.font.name = FONT
    return d


def _field(run, code):
    for kind, text in (("begin", None), (None, code), ("separate", None), ("end", None)):
        if kind:
            e = OxmlElement("w:fldChar"); e.set(qn("w:fldCharType"), kind)
        else:
            e = OxmlElement("w:instrText"); e.set(qn("xml:space"), "preserve"); e.text = text
        run._r.append(e)


def add_runs(p, text, size=None, bold=False, italic=False):
    for part in TOKEN.split(text):
        if not part:
            continue
        b, i, sup, sub = bold, italic, False, False
        if part.startswith("**") and part.endswith("**") and len(part) > 4:
            part, b = part[2:-2], True
        elif part.startswith("*") and part.endswith("*") and len(part) > 2:
            part, i = part[1:-1], True
        elif part.startswith("^") and part.endswith("^") and len(part) > 2:
            part, sup = part[1:-1], True
        elif part.startswith("~") and part.endswith("~") and len(part) > 2:
            part, sub = part[1:-1], True
        r = p.add_run(part); r.bold = b; r.italic = i
        r.font.superscript = sup; r.font.subscript = sub; r.font.name = FONT
        if size:
            r.font.size = Pt(size)
    return p


def para(d, text, align="justify", size=None, space_after=6, line=1.5, bold=False, italic=False, indent=None, keep=False):
    p = d.add_paragraph()
    pf = p.paragraph_format; pf.space_after = Pt(space_after); pf.line_spacing = line
    if keep:
        pf.keep_with_next = True
    if indent is not None:
        pf.left_indent = Inches(indent)
    p.alignment = {"justify": WD_ALIGN_PARAGRAPH.JUSTIFY, "left": WD_ALIGN_PARAGRAPH.LEFT,
                   "center": WD_ALIGN_PARAGRAPH.CENTER}[align]
    add_runs(p, text, size=size, bold=bold, italic=italic)
    return p


def heading(d, text, level=1):
    h = d.add_heading(level=level)
    add_runs(h, text)
    return h


def bullets(d, items, numbered=False):
    for it in items:
        p = d.add_paragraph(style="List Number" if numbered else "List Bullet")
        p.paragraph_format.line_spacing = 1.5; p.paragraph_format.space_after = Pt(3)
        p.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
        add_runs(p, it)


def _borders(cell, top=None, bottom=None):
    tcPr = cell._tc.get_or_add_tcPr(); b = OxmlElement("w:tcBorders")
    for side in ("top", "left", "bottom", "right", "insideH", "insideV"):
        e = OxmlElement(f"w:{side}")
        val = {"top": top, "bottom": bottom}.get(side)
        e.set(qn("w:val"), "single" if val else "nil")
        if val:
            e.set(qn("w:sz"), str(val)); e.set(qn("w:color"), "000000")
        b.append(e)
    tcPr.append(b)


def table(d, caption, header, rows, widths=None, note=None, size=8.5, bold_first_col=False, group_rows=()):
    """Three-line table. rows: list of lists of strings; a row whose first cell starts with '§' is a group label."""
    cap = para(d, caption, align="left", size=10, line=1.15, space_after=3, keep=True)
    t = d.add_table(rows=1 + len(rows), cols=len(header)); t.alignment = WD_TABLE_ALIGNMENT.CENTER
    t.autofit = False
    W = widths or [6.27 / len(header)] * len(header)
    for j, h in enumerate(header):
        c = t.rows[0].cells[j]; c.width = Inches(W[j]); c.text = ""
        add_runs(c.paragraphs[0], h, size=size, bold=True)
        _borders(c, top=8, bottom=4)
    trh = t.rows[0]._tr.get_or_add_trPr(); e = OxmlElement("w:tblHeader"); trh.append(e)
    for i, row in enumerate(rows):
        last = i == len(rows) - 1
        cells = t.rows[i + 1].cells
        if row and str(row[0]).startswith("§"):
            m = cells[0].merge(cells[-1]); m.text = ""
            add_runs(m.paragraphs[0], "*" + row[0][1:] + "*", size=size)
            for c in cells:
                _borders(c, bottom=8 if last else None)
            continue
        for j, v in enumerate(row):
            c = cells[j]; c.width = Inches(W[j]); c.text = ""
            add_runs(c.paragraphs[0], str(v), size=size, bold=(bold_first_col and j == 0))
            c.paragraphs[0].paragraph_format.space_after = Pt(0)
            _borders(c, bottom=8 if last else None)
    if note:
        para(d, note, align="justify", size=9, line=1.15, space_after=10)
    else:
        para(d, "", size=4, line=1.0, space_after=6)
    return t


def boxed(d, lines, size=9.5):
    """Single-cell bordered box (algorithm)."""
    t = d.add_table(rows=1, cols=1); t.alignment = WD_TABLE_ALIGNMENT.CENTER
    c = t.rows[0].cells[0]; c.width = Inches(6.27)
    tcPr = c._tc.get_or_add_tcPr(); b = OxmlElement("w:tcBorders")
    for side in ("top", "left", "bottom", "right"):
        e = OxmlElement(f"w:{side}"); e.set(qn("w:val"), "single"); e.set(qn("w:sz"), "6"); e.set(qn("w:color"), "000000"); b.append(e)
    tcPr.append(b)
    first = True
    for ln in lines:
        p = c.paragraphs[0] if first else c.add_paragraph(); first = False
        p.paragraph_format.space_after = Pt(1); p.paragraph_format.line_spacing = 1.1
        ind = len(ln) - len(ln.lstrip(" "))
        p.paragraph_format.left_indent = Inches(0.18 * (ind // 2))
        add_runs(p, ln.strip(), size=size)
    para(d, "", size=4, line=1.0, space_after=6)


def figure(d, path, width, caption):
    p = d.add_paragraph(); p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.keep_with_next = True
    p.add_run().add_picture(path, width=Inches(width))
    para(d, caption, align="justify", size=10, line=1.15, space_after=10)
