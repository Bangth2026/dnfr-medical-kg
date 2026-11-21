# Journal of Biomedical Informatics version

Inputs: the v5.0 manuscript and supplement (`Manuscript_AIM_DNFR.docx`, `Supplementary_AIM_DNFR.docx`), built by
`python3 manuscript3/build_manuscript.py` from the statistics files in `results/`.

```
cd manuscript_jbi
python3 build_jbi.py Manuscript_AIM_DNFR.docx Manuscript_JBI_DNFR.docx          # writes refmap.json
python3 build_supp.py Manuscript_AIM_DNFR.docx Supplementary_AIM_DNFR.docx ../figures/fig4_sparsity.png Supplementary_JBI_DNFR.docx
python3 check_numbers.py Manuscript_JBI_DNFR.docx Supplementary_JBI_DNFR.docx -- Manuscript_AIM_DNFR.docx Supplementary_AIM_DNFR.docx
python3 make_ga.py; python3 make_ancillary.py
```

`jbi_text.py` (with the shortened paragraphs in `jbi_patch.py`) holds the text of the journal version. Equations,
Figures 1–2 and Tables 1–6 are copied unchanged from the v5.0 manuscript. `check_numbers.py` lists every number of the
journal version that does not occur in the v5.0 documents; the only entries it reports are citation numbers and
bibliographic details of the references.
