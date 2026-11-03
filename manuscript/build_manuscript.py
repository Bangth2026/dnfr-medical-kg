"""Build the manuscript (.docx) from the statistics files. Usage: python3 manuscript/build_manuscript.py [out.docx]"""
import os, sys, re
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ms_common import ROOT, ORDER
from docx_helpers import new_document
import ms_front, ms_text

HIGHLIGHTS = [
    "Component analysis of a neuro-fuzzy path reasoner on four medical prediction tasks.",
    "Hub discounting helped on all four tasks, noisy-OR aggregation on all three tested.",
    "The t-norm parameter learned end to end reflected optimisation, not the data.",
    "Learned t-norms gave no consistent gain over fixed product or Hamacher t-norms.",
    "Degree-weighted path counts were not detectably worse on three of four tasks.",
]
CONTRIBUTIONS = [
    "A controlled component analysis on four tasks from three medical graphs that separates path aggregation, degree-aware and evidence-aware edge "
    "memberships and the t-norm, with fixed product, minimum, Hamacher and Łukasiewicz t-norms, a fixed DWPC-like degree discount and "
    "frozen memberships trained under the same protocol, and a pre-specified equivalence rule for comparing operators.",
    "Evidence that the parameter of a learned t-norm is set by optimisation rather than by the data in this setting: a profile over "
    "fixed values shows the training loss to be shallow over the non-positive range, and restarts, with and without early stopping, "
    "do not converge to a common value; the data exclude the nilpotent end of the family on graded evidence, where held-out "
    "accuracy favours a broad region of stronger t-norms that end-to-end learning does not reach.",
    "A comparison with degree-weighted path counts (with and without degree and prior features), AnyBURL, low-rank completion and two "
    "degree-only baselines under leave-dyad-out random and disease-centred splits, a sparsity analysis, and an evaluation of path "
    "explanations by concision, share-matched deletion and edge-level counterfactuals, with the implementation (DNFR, Algorithm 1), "
    "tests and all result files released.",
]


def build(out):
    d = new_document()
    ms_text.title_block(d)
    ms_text.highlights(d, HIGHLIGHTS)
    ms_text.abstract(d)
    ms_text.keywords(d)
    ms_front.introduction(d, CONTRIBUTIONS)
    ms_text.significance(d)
    ms_front.related(d)
    ms_front.methods(d)
    ms_text.results(d)
    ms_text.discussion(d)
    ms_text.conclusion(d)
    ms_text.appendix(d)
    ms_text.declarations(d)
    ms_text.references(d)
    d.save(out)
    # checks
    from docx import Document
    doc = Document(out)
    txt = [p.text for p in doc.paragraphs]
    i = txt.index("Abstract"); j = next(k for k in range(i, len(txt)) if txt[k].startswith("Keywords"))
    words = sum(len(re.findall(r"\S+", t)) for t in txt[i + 1:j]) - 4   # minus the four structure labels
    left = [t for t in txt if re.search(r"\[@|\[NA\]|CẦN|TODO|\bnan\b|\{[a-z_]+\(", t)]
    print(f"saved {out}; abstract {words} words; {len(ORDER)} references; leftover markers: {len(left)}")
    for t in left[:5]:
        print("   ", t[:120])
    return words


if __name__ == "__main__":
    out = sys.argv[1] if len(sys.argv) > 1 else os.path.join(ROOT, "manuscript", "Manuscript_AIM_DNFR_rev2.docx")
    build(out)
