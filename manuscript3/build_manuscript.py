"""Build the manuscript and its supplementary material (.docx) from the statistics files.
Usage: python3 manuscript3/build_manuscript.py [out_dir]"""
import os, sys, re
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ms_common import ROOT, ORDER
from docx_helpers import new_document
import ms_front, ms_text

CONTRIBUTIONS = [
    "A controlled component analysis on four tasks from three medical graphs that separates path aggregation, degree-aware and "
    "evidence-aware edge memberships and the t-norm, with fixed and validation-selected t-norms, frozen memberships and a fixed "
    "DWPC-like discount trained under the same protocol, and one decision rule for equivalence and difference applied to every "
    "comparison.",
    "An identifiability analysis of the t-norm parameter that separates structural redundancy between memberships and operator from "
    "practical non-identifiability by the training objective, using restarts, an unbounded parameterisation and converged "
    "profiles on two graphs, and shows that validation-based selection recovers what end-to-end learning misses.",
    "A comparison with degree-weighted path counts, AnyBURL, NBFNet, low-rank completion and degree-only baselines under "
    "leave-dyad-out random and disease-centred splits and without the metapaths through the target relation, which shows how much "
    "of the drug-repurposing accuracy rests on known indications.",
    "An evaluation of path explanations with an exact edge-level Shapley attribution and fact-level counterfactuals, compared with "
    "the same attribution for degree-weighted path counts, and a test of four refinements of the model, with the implementation, "
    "tests and all result files released.",
]


def check(out):
    from docx import Document
    doc = Document(out)
    txt = [p.text for p in doc.paragraphs] + [c.text for t in doc.tables for r in t.rows for c in r.cells]
    left = [t for t in txt if re.search(r"\[@|\[NA\]|CẦN|TODO|\bnan\b|\{[a-z_]+\(|AUTHOR TO", t)]
    return txt, left


def build(outdir):
    os.makedirs(outdir, exist_ok=True)
    d = new_document()
    ms_text.title_block(d)
    ms_text.highlights(d, ms_text.highlights_list())
    ms_text.abstract(d)
    ms_text.keywords(d)
    ms_front.introduction(d, CONTRIBUTIONS)
    ms_text.significance(d)
    ms_front.related(d)
    ms_front.methods(d)
    ms_text.results(d)
    ms_text.discussion(d)
    ms_text.conclusion(d)
    ms_text.declarations(d)
    ms_text.references(d)
    out = os.path.join(outdir, "Manuscript_AIM_DNFR_rev5.docx"); d.save(out)
    s = new_document(); ms_text.supplement(s)
    outs = os.path.join(outdir, "Supplementary_AIM_DNFR_rev5.docx"); s.save(outs)
    from docx import Document
    txt = [p.text for p in Document(out).paragraphs]
    i = txt.index("Abstract"); j = next(k for k in range(i, len(txt)) if txt[k].startswith("Keywords"))
    words = sum(len(re.findall(r"\S+", t)) for t in txt[i + 1:j]) - 4
    for f in (out, outs):
        _, left = check(f)
        print(f"saved {f}; markers left: {len(left)}")
        for t in left:
            print("   ", t[:150])
    print(f"abstract {words} words; {len(ORDER)} references")


if __name__ == "__main__":
    build(sys.argv[1] if len(sys.argv) > 1 else os.path.join(ROOT, "manuscript3", "out"))
