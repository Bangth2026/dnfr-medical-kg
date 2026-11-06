"""Files that go with the revised submission besides the manuscript, the supplement and the response letter:
- Highlights_AIM_DNFR.docx  (separate highlights file, 3-5 bullets of at most 85 characters, as Elsevier asks)
- Cover_letter.docx              (letter to the editor accompanying a new submission)
- Declaration_of_interests.docx
- Manuscript_AIM_DNFR_changes_marked.docx: the revised manuscript with every paragraph and table cell that is new or
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
    d.save(f"{OUT}/Highlights_AIM_DNFR.docx")


def cover_letter():
    d = new_document()
    para(d, "Hai Bang Truong", align="left")
    para(d, "Faculty of Engineering and Computer Science, Saigon International University, 8C Tong Huu Dinh Street, An Khanh Ward, "
            "Ho Chi Minh City, Vietnam", align="left")
    para(d, "truonghaibang@siu.edu.vn", align="left", space_after=10)
    para(d, "To the Editor-in-Chief, *Artificial Intelligence in Medicine*", align="left", space_after=10)
    para(d, f"Re: submission of the research article \"{MT.TITLE}\"", align="left", space_after=10)
    para(d, "Dear Editor,")
    para(d, "We submit the manuscript above for consideration as a research article. Path-based reasoners over medical knowledge "
            "graphs are attractive for clinical decision support because each prediction comes with the paths that support it, "
            "and neuro-fuzzy variants go further by learning the operator that composes evidence along a path. Whether such an "
            "operator is actually determined by medical graph data had not been tested. We built a differentiable neuro-fuzzy path "
            "reasoner and took it apart component by component on four link-prediction tasks from three medical knowledge graphs "
            "(Hetionet, PrimeKG and a graph learned from emergency-department records), against NBFNet, degree-weighted path counts "
            "and other baselines, with leave-dyad-out splits and a stated decision rule for every comparison.")
    para(d, "The findings are partly negative, and we report them as such. Aggregating all paths and discounting hub nodes carried "
            "the accuracy; four plausible refinements did not help; and with every other parameter fitted to convergence the "
            "training loss hardly depends on the t-norm parameter, so a learned operator should not be read as a property of the "
            "domain. Degree-weighted path counts were at least as accurate on three of four tasks, and on drug repurposing almost "
            "all of the evidence ran through known indications. We think these results matter to readers of the journal who build "
            "or evaluate explainable decision support on knowledge graphs, because they say which parts of such a model deserve "
            "effort and what its accuracy rests on.")
    para(d, "The code, the result files and the scripts that build every number in the manuscript are released under the MIT "
            f"licence as {MT.RELEASE} of https://github.com/Bangth2026/dnfr-medical-kg and archived at Zenodo "
            "(https://doi.org/10.5281/zenodo.23998418). The work uses public data only. The use of an AI system in preparing the "
            "manuscript and the code is declared in the manuscript.")
    para(d, "The manuscript is original, has not been published and is not under consideration elsewhere. Both authors approved "
            "the submitted version, and we declare no competing interests.")
    para(d, "Yours sincerely,", space_after=2)
    para(d, "Hai Bang Truong, on behalf of the authors", align="left")
    d.save(f"{OUT}/Cover_letter.docx")


def declaration():
    d = new_document()
    para(d, "**Declaration of interests**", align="left", size=13)
    para(d, "☒ The authors declare that they have no known competing financial interests or personal relationships that could "
            "have appeared to influence the work reported in this paper.")
    para(d, "☐ The authors declare the following financial interests/personal relationships which may be considered as potential "
            "competing interests:")
    para(d, f"Manuscript: {MT.TITLE}", align="left")
    para(d, "Authors: Hai Bang Truong, Van An Phan Thi", align="left")
    d.save(f"{OUT}/Declaration_of_interests.docx")


def norm(t):
    return re.sub(r"\s+", " ", t).strip()


def marked(prev):
    old = docx.Document(prev)
    seen = {norm(p.text) for p in old.paragraphs}
    for tb in old.tables:
        for row in tb.rows:
            for cl in row.cells:
                seen.add(norm(cl.text))
    new = docx.Document(f"{OUT}/Manuscript_AIM_DNFR.docx")
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
    new.save(f"{OUT}/Manuscript_AIM_DNFR_changes_marked.docx")
    print("marked", n_par, "paragraphs and", n_cell, "table cells")


if __name__ == "__main__":
    highlights(); cover_letter(); declaration()
    if len(sys.argv) > 1:          # changes-marked version only when a previous manuscript is given (revisions)
        marked(sys.argv[1])
    print("saved submission files in", OUT)
