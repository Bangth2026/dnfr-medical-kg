# -*- coding: utf-8 -*-
"""Response to the reviewer (JBI). Items: ("h", text) heading, ("rc", text) reviewer comment, ("r", text) response,
("ch", text) change made. Placeholders <<KEY>> are filled from numbers.json."""

INTRO = [
    "Dear Editor, dear Reviewer,",
    "Thank you for a careful and constructive report. We agree with its three main points: the identifiability "
    "conclusion claimed more than the profile showed, the comparison with degree-weighted path counts mixed model and "
    "objective, and the drug-repurposing conclusions rested on few disease-centred splits and on a weak clinical "
    "design. We ran the requested controls on the same two-core CPU as before and rewrote the claims to match the "
    "evidence. The main changes are:",
    "• RQ3 is reformulated. The converged profiles are now reported with the disagreement between optimisation chains "
    "(new Figure 2, Table S21) and, on Hetionet disease–symptom, for the two metapaths that carry most of the evidence "
    "(Table S22). Within-split resolution, between-split variability and the mismatch between training loss and "
    "validation MRR are separated in the text.",
    "• Two DWPC controls (DWPC trained with DNFR's filtered softmax; a linear DWPC with non-negative coefficients) "
    "were run on the graded graph and on Hetionet disease–symptom with ten splits (Table 2, Table 6, Table S23). "
    "<<RESP_M2_SUMMARY>>",
    "• The disease-centred split now has ten splits, a compound-centred split was added, the disease-centred split was "
    "also run without the treats metapaths, and the chance level is reported for every task (Tables 2 and 5, "
    "Section S8.4). Training without the treats metapaths more than tripled accuracy for held-out diseases, which "
    "strengthens the point of Section 4.5.",
    "• The clinical assessment is labelled a pilot, reduced to two sentences in the main text and reported with "
    "confidence intervals in the Supplement; PrimeKG no longer enters the Highlights, the Abstract or the conclusions; "
    "the framing is that of a controlled analysis, with DNFR described as an instrument rather than a proposed method.",
    "Below we answer each comment. Page and section numbers refer to the revised manuscript; all new numbers are "
    "generated from the released result files (release <<VER>>).",
]

ITEMS = [
    ("h", "M1. RQ3: the converged profile does not establish practical non-identifiability as stated"),
    ("rc", "Chain disagreement (Table S16: 32.1, 13.1, 8.8 × 10⁻³ nats) exceeds the profile variation; the defensible "
           "conclusion is that the profile could not be resolved at the available precision."),
    ("r", "<<RESP_M1A>>"),
    ("rc", "Narrow graded-graph intervals (Table S8) mean p is identified within each split; between-split movement is "
           "sampling variability, not non-identifiability. Separate them and adjust the abstract."),
    ("r", "We agree, and the text now separates the two. <<RESP_M1B>>"),
    ("rc", "Table S11: restarts without penalty drift away from their initial values toward opposite ends; a flat profile "
           "would not do this; a ridge or several basins would."),
    ("r", "<<RESP_M1C>>"),
    ("rc", "The profile was computed only for a single shared p; profile a few influential metapaths individually or "
           "restrict the claim."),
    ("r", "<<RESP_M1D>>"),
    ("rc", "Validation MRR peaks near p ≈ −1.5 to −2 while the training loss does not: a mismatch between surrogate loss "
           "and metric. Consider profiling a ranking-aware objective or the validation log-likelihood."),
    ("r", "<<RESP_M1E>>"),
    ("rc", "Requested (a) chain envelope or agreement tolerance; (b) profiles of 2–3 high-share metapaths on Hetionet "
           "DpS; (c) rephrase Abstract, Highlight 3, Section 4.4 and Conclusion."),
    ("r", "(a) New Figure 2 and Table S21: every point with its chain envelope and a resolution flag (two best chains "
          "within 10⁻⁴ nat). (b) Table S22, two metapaths that together carry <<PERMP_SHARE2>> of DNFR's evidence on "
          "correct top-10 predictions; a third (<<PERMP_THIRD>> of the evidence) was not profiled for time. (c) The "
          "Abstract, Highlight 3, Statement of significance, Section 4.4, Section 5.1 and the Conclusion now say that the "
          "training loss located p only loosely within a split, that its optimum moved between splits and that it did not "
          "coincide with the optimum of validation MRR; the words \"not identified\" no longer appear as a finding."),

    ("h", "M2. DNFR versus DWPC-LR: separate model from objective"),
    ("rc", "DWPC-LR uses sampled logistic regression with balanced weights and an untuned damping; train it with the "
           "filtered softmax to isolate the objective."),
    ("r", "<<RESP_M2A>>"),
    ("rc", "The exact, non-negative attribution is a design choice, not intrinsic to fuzzy composition: a DWPC model "
           "without asinh and with non-negative coefficients would be as exact. Add (a) DWPC-LR with filtered softmax and "
           "(b) a linear non-negative DWPC on the graded graph and Hetionet DpS, with accuracy and the counterfactual rows "
           "of Table 6."),
    ("r", "<<RESP_M2B>>"),
    ("r", "On the damping: we kept 0.4, the value of Project Rephetio, for all path methods. Tuning it would add a "
          "hyperparameter to the baseline that DNFR learns per relation; since DWPC was already at least as accurate as "
          "DNFR, tuning could only strengthen that conclusion, and we say so in Section 4.1."),

    ("h", "M3. The treats-metapath finding: novelty and positioning"),
    ("rc", "Check what [1] and [27] reported about CtD-containing metapaths and state precisely what this paper adds."),
    ("r", "We checked Project Rephetio [1]. Its logistic model retained several metapaths through the treats relation "
          "among those with the largest coefficients (CbGbCtD 0.22, CiPCiCtD 0.16, CrCtD 0.15, CtDrD 0.13, CrCrCtD 0.12, "
          "CcSEcCtD 0.08, CpDpCtD 0.06), and the authors noted that for epilepsy 76% of the support came from the disease's "
          "known treatments; they did not ablate these metapaths or measure the share of evidence behind correct "
          "predictions. Hetnet connectivity search [27] makes metapath explanations searchable but does not evaluate "
          "prediction. Section 2.1 now says this and states our addition: a quantitative ablation across three path "
          "methods, the evidence share behind correct predictions, and the accuracy of the remaining evidence when whole "
          "diseases or compounds are held out."),
    ("rc", "Cite the literature on guilt by association and cold-start/zero-shot evaluation rather than presenting the "
           "issue as new."),
    ("r", "Section 2.1 now cites Kim et al. (CHIL 2023), who showed that drug indications are predictable from drug "
          "category and that random splits overestimate repurposing accuracy; Brière et al. (bioRxiv 2025), who found "
          "that random and cold-start splits both overestimate accuracy against an independent Orphanet test set; Bang et "
          "al. (Nat. Commun. 2023), who build guilt by association across drug, gene and disease layers into a "
          "repurposing model; and TxGNN's zero-shot motivation [8]. Section 5.2 no longer presents the issue as new."),
    ("rc", "Only 4 disease-centred splits; report the chance MRR for every task; more splits or leave-one-disease-out; add "
           "a compound-centred split."),
    ("r", "<<RESP_M3C>>"),
    ("rc", "Run the disease-centred split without treats metapaths."),
    ("r", "<<RESP_M3D>>"),

    ("h", "M4. Baselines for drug repurposing"),
    ("rc", "No current learned-operator baseline on the drug-repurposing graphs: run NBFNet/BioPathNet on a GPU, report "
           "TxGNN numbers as context, or narrow the RQ1 claims."),
    ("r", "We had no GPU, so we took option (c). Section 3.4 now states that our drug-repurposing conclusions concern "
          "path-weighting methods on a fixed path set, and Section 5.3 that BioPathNet, TxGNN and NBFNet were not run on "
          "these graphs. We considered option (b) but did not put TxGNN's published figures beside ours: they are "
          "reported on PrimeKG with disease-area splits and with metrics we did not compute, so the comparison would "
          "invite a reading it cannot support. We cite TxGNN's zero-shot protocol in Section 2.1 instead."),
    ("rc", "Untuned TransE and RotatE are uninformative: tune minimally or move them to the Supplement."),
    ("r", "They are removed from Table 2 and remain in Tables S1–S3, labelled untuned reference points."),
    ("rc", "AnyBURL had 100 s of rule learning; justify or add a longer run."),
    ("r", "<<RESP_M4C>>"),

    ("h", "M5. PrimeKG coverage and power"),
    ("rc", "PrimeKG has two splits and lacks most analyses but appears in the Highlights and the 'three of four' framing; "
           "add splits and the treats ablation or remove PrimeKG from those claims."),
    ("r", "We removed PrimeKG from the claims. Adding splits was not feasible on our machine (45.8 million paths per "
          "split). The Highlights, the Abstract and the Statement of significance now rest on the three tasks with ten "
          "splits; PrimeKG is described as a descriptive fourth task in the Introduction, Table 2 and Section 5.3, and the "
          "'three of four tasks' wording is gone."),
    ("rc", "Were path counts for the exclusion computed on the training graph of each split or on the full graph?"),
    ("r", "<<RESP_M5B>>"),

    ("h", "M6. Clinical assessment of explanations"),
    ("rc", "Primary rater is a co-author without medical training; second rater rated 40 pairs and knows the authors; "
           "κ = 0.40–0.46; AUC without CI; drug repurposing not rated; 'independent' in the acknowledgement conflicts with "
           "Section 3.6. Strengthen the design or present it as a pilot."),
    ("r", "<<RESP_M6>>"),

    ("h", "M7. Framing: method paper or controlled analysis?"),
    ("rc", "Make the framing consistent; report effect sizes in the Highlights."),
    ("r", "The Introduction now says that DNFR is built as an instrument for the question and is not proposed as a better "
          "predictor, the contribution paragraph describes a controlled analysis, and the versioned name 'DNFR 2.0' is "
          "replaced by the descriptive 'DNFR + R1–R4' throughout. The Highlights now give magnitudes (0.08–0.10 MRR for "
          "aggregation, 0.02–0.16 for hub discounting) and no longer contain a statement of sign consistency alone."),

    ("h", "Minor comments"),
    ("r", "1. Stale cross-references: the opening of Section S5 no longer refers to an earlier version; \"Section 4.8\" is "
          "now \"Section S6.3\", \"Section 5.5\" is \"Section 5.3\", and the caption of Table S13 no longer mentions a "
          "second version of the analysis."),
    ("r", "2. Penalty towards θ₀: Sections 3.2 and 4.4 now say that the penalty pulls p towards its start by construction "
          "and give the restarts without penalty (Table S11) in the main text."),
    ("r", "3. Nilpotent range: Section 3.4 now notes that the Łukasiewicz accuracy in Tables S1–S4 reflects the "
          "initialisation, not the operator (Section 4.4 gives its accuracy when started from the converged product)."),
    ("r", "4. Table 3: the cell is marked \"1/5 (5 splits)\" and the caption says that cells with fewer splits give "
          "the number run."),
    ("r", "<<RESP_MINOR5>>"),
    ("r", "6. Validation noise on CtD: Section 4.3 now notes that the CtD validation set has about 75 pairs and that the "
          "equivalent verdict for validation selection there should be read with this noise in mind."),
    ("r", "7. The sentence that the counterfactual measures the dependence of the trained model, not what a retrained "
          "model would learn, is now in Section 4.6."),
    ("r", "8. Section 4.6 now states that the fact-level counterfactual keeps degrees, and hence the memberships of "
          "neighbouring edges, fixed, so it matches the Shapley decomposition. We did not run a variant that recomputes "
          "degrees."),
    ("r", "9. HKG edge-weight leakage is now a limitation (Section 5.3, last item)."),
    ("r", "10. Section 7 now says plainly that the larger tasks were not re-run independently of the AI system; the "
          "independent re-run remains one split of the graded graph."),
    ("r", "11. Section S8.6 works through the decision rule for one comparison (validation-selected p against the "
          "product on the graded graph): both 90% intervals, the corrected t statistic, the Holm-adjusted p-value and "
          "the 95% bootstrap interval."),
    ("r", "12. The compute constraint (two-core CPU, no GPU) and what it ruled out are stated in Sections 3.6 and 5.3."),
    ("r", "13. Reference [25] and the sentence citing it are removed."),
    ("r", "<<RESP_MINOR14>>"),
    ("r", "15. Subscripts in Section 3.2 are now typeset (a_r, b_r, c_r, g_r, p_m, ω_m, κ_π)."),
]

CLOSING = [
    "We thank the Reviewer again; the controls requested under M1 and M2 changed what the paper claims, and we think "
    "for the better.",
    "Sincerely,",
    "Hai Bang Truong, on behalf of the authors",
]
