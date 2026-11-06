"""Files that go with the revised submission besides the manuscript, the supplement and the response letter:
- Highlights_AIM_DNFR_rev5.docx  (separate highlights file, 3-5 bullets of at most 85 characters, as Elsevier asks)
- Cover_letter_rev5.docx         (letter to the editor accompanying the revision)
- Declaration_of_interests_rev5.docx
- Manuscript_AIM_DNFR_rev5_changes_marked.docx: the revised manuscript with every paragraph and table cell that is new or
  changed relative to the previous version highlighted in yellow (paragraph-level comparison with the previous build).
Usage: python3 manuscript3/build_submission.py [previous_manuscript.docx]"""
import os, sys, re
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ms_common import *
from docx_helpers import new_document, para, bullets
import ms_text as MT
import docx
from docx.enum.text import WD_COLOR_INDEX

OUT = os.path.join(ROOT, "manuscript3", "out")


def highlights():
    d = new_document()
    para(d, "**Highlights**", align="left", size=13)
    H = MT.highlights_list()
    assert 3 <= len(H) <= 5 and all(len(h) <= 85 for h in H)
    bullets(d, H)
    d.save(f"{OUT}/Highlights_AIM_DNFR_rev5.docx")


def cover_letter():
    d = new_document()
    para(d, "Hai Bang Truong", align="left")
    para(d, "Faculty of Engineering and Computer Science, Saigon International University, 8C Tong Huu Dinh Street, An Khanh Ward, "
            "Ho Chi Minh City, Vietnam", align="left")
    para(d, "truonghaibang@siu.edu.vn", align="left", space_after=10)
    para(d, "To the Editor, *Artificial Intelligence in Medicine*", align="left", space_after=10)
    para(d, f"Re: revised manuscript \"{MT.TITLE}\"", align="left", space_after=10)
    para(d, "Dear Editor,")
    para(d, "We submit the fifth version of our manuscript, revised in response to the reviewer's fourth report and to the question "
            "on authorship. The reviewer recommended minor revision and listed three internal tensions, one sign error and the "
            "scope of the clinical assessment. We ran every analysis requested: the converged profile up to the Łukasiewicz t-norm "
            "on the graded graph and down to p = −20 on Hetionet disease–symptom, restarts of the Hamacher run on the outlying "
            "split, and an ablation of the metapaths through both the treats and the palliates relations. We corrected the sign "
            "error, which came from the script that builds the text, and limited the claim about selecting the t-norm on validation "
            "data to the graph on which the evidence supports it.")
    para(d, "For the clinical assessment we could not recruit an independent physician rater within this revision. The article now "
            "states the limits of a single author-rater trained in clinical psychology, who rated correct predictions only, and we "
            "release a blinded sheet that mixes correct and incorrect predictions for a future rating. On authorship, we confirm in "
            "the response letter that the second author's earlier CRediT statement was incomplete, and we describe which of her "
            "contributions are new.")
    para(d, "The response letter answers each point and gives every number it cites from the released result files. The code, "
            f"results and the scripts that build the manuscript are released as {MT.RELEASE} of "
            "https://github.com/Bangth2026/dnfr-medical-kg and archived at Zenodo. The manuscript has not been published and is not "
            "under consideration elsewhere, and both authors approved this version.")
    para(d, "Yours sincerely,", space_after=2)
    para(d, "Hai Bang Truong, on behalf of the authors", align="left")
    d.save(f"{OUT}/Cover_letter_rev5.docx")


def declaration():
    d = new_document()
    para(d, "**Declaration of interests**", align="left", size=13)
    para(d, "☒ The authors declare that they have no known competing financial interests or personal relationships that could "
            "have appeared to influence the work reported in this paper.")
    para(d, "☐ The authors declare the following financial interests/personal relationships which may be considered as potential "
            "competing interests:")
    para(d, f"Manuscript: {MT.TITLE}", align="left")
    para(d, "Authors: Hai Bang Truong, Van An Phan Thi", align="left")
    d.save(f"{OUT}/Declaration_of_interests_rev5.docx")


def norm(t):
    return re.sub(r"\s+", " ", t).strip()


def marked(prev):
    old = docx.Document(prev)
    seen = {norm(p.text) for p in old.paragraphs}
    for tb in old.tables:
        for row in tb.rows:
            for cl in row.cells:
                seen.add(norm(cl.text))
    new = docx.Document(f"{OUT}/Manuscript_AIM_DNFR_rev5.docx")
    n_par = n_cell = 0
    def mark(p):
        for run in p.runs:
            run.font.highlight_color = WD_COLOR_INDEX.YELLOW
    for p in new.paragraphs:
        if norm(p.text) and norm(p.text) not in seen:
            mark(p); n_par += 1
    for tb in new.tables:
        for row in tb.rows:
            for cl in row.cells:
                if norm(cl.text) and norm(cl.text) not in seen:
                    for p in cl.paragraphs:
                        mark(p)
                    n_cell += 1
    first = new.paragraphs[0]
    note = first.insert_paragraph_before("Revised manuscript with changes marked: paragraphs and table cells that are new or "
                                         "changed relative to the previous version are highlighted in yellow.")
    for run in note.runs:
        run.font.bold = True
    new.save(f"{OUT}/Manuscript_AIM_DNFR_rev5_changes_marked.docx")
    print("marked", n_par, "paragraphs and", n_cell, "table cells")


if __name__ == "__main__":
    highlights(); cover_letter(); declaration()
    prev = sys.argv[1] if len(sys.argv) > 1 else f"{OUT}/Manuscript_AIM_DNFR_rev4_submitted.docx"
    if os.path.exists(prev):
        marked(prev)
    else:
        print("no previous manuscript at", prev, "; changes-marked version not built")
    print("saved submission files in", OUT)
