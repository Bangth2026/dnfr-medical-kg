# -*- coding: utf-8 -*-
"""Revised and new text of the Supplementary material. SUBS: substring replacements inside existing paragraphs
(index -> list of (old, new)); NEW_SECTION: paragraphs appended as Section S8 before the references."""

SUBS = {
    52: [("This section gives the full description, as written for the earlier, longer version of the article; section, "
          "equation, figure and table numbers refer to the main text, which keeps the same numbering of Sections 3.1–3.6, "
          "Equations 1–3 and Tables 1–6.",
          "This section gives the full description; section, equation, figure and table numbers refer to the main text.")],
    77: [("(Section 4.8)", "(Section S6.3)")],
    82: [("(Section 5.5)", "(Section 5.3)"),
         ("AnyBURL [9] (version 23-1, 100 s of rule learning, top 200 candidates per query).",
          "AnyBURL [9] (version 23-1, 100 s of rule learning, top 200 candidates per query; 1000 s in Section S8.5)."),
         ("are untuned reference points only.",
          "are untuned reference points only and are reported in Tables S1–S3, not in the main text."),
         ],
    90: [("On compound–treats–disease the validation set has about 75 pairs, so model selection there is noisy; we report "
          "the spread of validation MRR across checkpoints in the supplementary material.",
          "On compound–treats–disease the validation set has about 75 pairs, so model selection there is noisy; Table S10 "
          "gives the spread of validation MRR across checkpoints. The expected MRR of a uniformly random ranker, H_n/n for a "
          "query with n filtered candidates averaged over queries, is 0.041, 0.016 and 0.021 on the three tasks with ten "
          "splits and about 0.0006 on PrimeKG.")],
    91: [("(4 splits; 80/10/10 of the diseases with at least one indication)",
          "(10 splits; 80/10/10 of the diseases with at least one indication; splits 0–3 were run first, splits 4–9 for "
          "the revision, Section S8.4)"),
         ("as in zero-shot repurposing [8].",
          "as in zero-shot repurposing [8]. A compound-centred split holds out every indication of 10% of the compounds "
          "instead (10 splits, Section S8.4).")],
    108: [("That is the difference in kind between the two explanations; we found no evidence that DNFR's are more "
           "faithful in the sense tested here.",
           "These differences come from the asinh transform and the unconstrained coefficients of DWPC-LR, not from path "
           "counting: a linear DWPC model with non-negative coefficients has an attribution as exact and non-negative as "
           "DNFR's (Section S8.3). We found no evidence that DNFR's explanations are more faithful in the sense tested "
           "here.")],
    117: [("rated a second, blinded sheet", "who knows the authors as a friend and colleague, rated a second, blinded "
           "sheet"),
          ("(results/expert_review2_filled.xlsx, with the rater's name withheld)",
           "(results/expert_review2_filled.xlsx, with the rater's name and workplace withheld)")],
}

# Table S8 cell fixes (table index in the supplement, row, column, new text)
CELLS = [
    # split 0 interval: two disjoint sets, not one interval
]

NEW_SECTION_TITLE = "S8. Analyses added in revision"
NEW_SECTION = [
    ("h2", "S8.1 Chain envelope of the converged profiles"),
    ("p", "<<S81>>"),
    ("table", "S21"),
    ("h2", "S8.2 Profiles of single metapaths on Hetionet disease–symptom"),
    ("p", "<<S82>>"),
    ("table", "S22"),
    ("h2", "S8.3 DWPC models that separate the model from the objective"),
    ("p", "<<S83>>"),
    ("table", "S23"),
    ("h2", "S8.4 Disease- and compound-centred splits"),
    ("p", "<<S84>>"),
    ("h2", "S8.5 AnyBURL with a longer rule-learning budget"),
    ("p", "<<S85>>"),
    ("h2", "S8.6 A worked example of the decision rule"),
    ("p", "<<S86>>"),
    ("h2", "S8.7 Confidence intervals for the clinical pilot"),
    ("p", "<<S87>>"),
    ("table", "S24"),
    ("h2", "S8.8 Path counts used to exclude metapaths"),
    ("p", "<<S88>>"),
]
