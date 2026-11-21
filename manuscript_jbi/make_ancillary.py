from docx import Document
from docx.shared import Pt
import jbi_text as J
def doc():
    d = Document(); st = d.styles["Normal"]; st.font.name = "Times New Roman"; st.font.size = Pt(12); return d
def P(d, t, b=False, size=None):
    p = d.add_paragraph(); r = p.add_run(t); r.bold = b
    if size: r.font.size = Pt(size)
    return p
AUTH = ("Hai Bang Truong", "Faculty of Engineering and Computer Science, Saigon International University, 8C Tong Huu Dinh "
        "Street, An Khanh Ward, Ho Chi Minh City, Vietnam", "truonghaibang@siu.edu.vn")
# cover letter
d = doc()
for x in AUTH: P(d, x)
P(d, "21 November 2026")
P(d, "To the Editor-in-Chief, Journal of Biomedical Informatics")
P(d, f"Re: submission of the research paper \"{J.TITLE}\"", b=True)
P(d, "Dear Editor,")
for t in [
 "We submit the manuscript above for consideration as a Research Paper. Path-based reasoners over medical knowledge "
 "graphs are attractive for clinical decision support because each prediction comes with the paths behind it, and "
 "neuro-fuzzy variants go further by learning the operator that composes evidence along a path. Whether such an operator "
 "is determined by medical graph data had not been tested. We built a differentiable neuro-fuzzy path reasoner and took it "
 "apart component by component on four link-prediction tasks from three medical knowledge graphs (Hetionet, PrimeKG and a "
 "graph learned from emergency-department records), against degree-weighted path counts, NBFNet, rule learning, low-rank "
 "completion and degree-only baselines, under leave-dyad-out splits and one stated decision rule for every comparison.",
 "We believe the paper fits the journal's emphasis on methodological contributions that generalise. Its lessons concern "
 "any learned operator in a path-based clinical reasoner rather than one model: aggregating all paths and discounting hubs "
 "carried the accuracy; with every other parameter fitted to convergence the training loss hardly depends on the t-norm "
 "parameter, so a learned operator should be checked for identifiability and selected on validation data rather than "
 "interpreted; on curated drug-repurposing graphs, 99% of the evidence for correct predictions ran through known "
 "indications, so claims of mechanistic reasoning need an ablation of the target relation or disease-centred splits; and "
 "an exact edge-level Shapley attribution shows what a noisy-OR path model adds over degree-weighted path counts, which "
 "were at least as accurate on three of four tasks. Several findings are negative, and we report them as such.",
 "The manuscript follows the journal's format: a structured abstract of 290 words, a body of about 5,980 words including "
 "the Statement of Significance, and two figures and six tables; further material is in the Supplementary material. The "
 "code, result files and a script that checks every number in the manuscript against the result files are released "
 "under the MIT licence as release v6.0-jbi of https://github.com/Bangth2026/dnfr-medical-kg and archived at Zenodo "
 "(https://doi.org/10.5281/zenodo.23879352); earlier releases of the same repository accompanied previous versions of "
 "this manuscript. The work uses public data only, and the use of an AI system in preparing the code and "
 "the manuscript is declared in the manuscript.",
 "Two disclosures about related submissions. First, an earlier version of this manuscript was under consideration at "
 "Artificial Intelligence in Medicine and has been withdrawn; the present version was shortened and revised for this "
 "journal. Second, in October 2026 we submitted to the Journal of Biomedical Informatics a different manuscript, "
 "\"Testing t-norm choice on a population fuzzy graph of depressive symptoms and their correlates\" (JBI-S-26-06085), "
 "which was not accepted. That study analysed a 24-concept fuzzy graph built from NHANES survey data; the present "
 "manuscript uses different data (Hetionet, PrimeKG and a graph learned from emergency-department records), a different "
 "model and different research questions, and cites the earlier analysis only as motivation (reference [25]), not as "
 "evidence. We would be glad to provide that manuscript to the editors.",
 "The manuscript is original, has not been published and is not under consideration elsewhere. Both authors approved the "
 "submitted version, and we declare no competing interests.",
 ]:
    P(d, t)
P(d, "Yours sincerely,"); P(d, "Hai Bang Truong, on behalf of the authors")
d.save("Cover_letter_JBI.docx")
# highlights
d = doc(); P(d, "Highlights", b=True)
for h in J.HIGHLIGHTS: d.add_paragraph(h, style="List Bullet")
d.save("Highlights_JBI_DNFR.docx")
# statement of significance
d = doc(); P(d, "Statement of Significance", b=True)
t = d.add_table(rows=1, cols=2); t.style = "Table Grid"
t.rows[0].cells[0].text = "Heading"; t.rows[0].cells[1].text = "Content"
for k, v in J.SIGNIFICANCE:
    r = t.add_row().cells; r[0].text = k; r[1].text = v
d.save("Statement_of_Significance_JBI_DNFR.docx")
# title page
d = doc(); P(d, J.TITLE, b=True, size=14)
P(d, "Article type: Research Paper"); P(d, "Running title: " + J.RUNNING)
P(d, "Hai Bang Truong¹*, Van An Phan Thi²")
P(d, "¹ Faculty of Engineering and Computer Science, Saigon International University, 8C Tong Huu Dinh Street, An Khanh "
     "Ward, Ho Chi Minh City, Vietnam. ORCID: https://orcid.org/0000-0001-5094-3128")
P(d, "² VNU University of Education, Vietnam National University, Hanoi, G7 Building, No. 144 Xuan Thuy Street, Cau Giay "
     "District, Hanoi, Vietnam. Email: phanthivanantlls@gmail.com. ORCID: https://orcid.org/0009-0005-4135-5689")
P(d, "* Corresponding author: Hai Bang Truong, truonghaibang@siu.edu.vn")
P(d, "Word counts: abstract 290; body (Introduction to Conclusion, including the Statement of Significance) about 5,980. "
     "Figures and tables: 2 figures, 6 tables.")
for k, v in J.DECLARATIONS:
    if k in ("Funding", "Declaration of competing interest"):
        P(d, f"{k}. {v}")
d.save("Title_page_JBI_DNFR.docx")
print("ok")
