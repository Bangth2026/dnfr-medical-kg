# -*- coding: utf-8 -*-
"""Revised text of the JBI manuscript (revision 1 after review). Keys are paragraph indices of
Manuscript_JBI_DNFR_v2.docx; values are the new text with markup (**bold**, *italic*, _{sub}, ^{sup}).
Placeholders <<KEY>> are filled from build/numbers.json, which build/compute_numbers.py writes from the result files."""

HIGHLIGHTS = {
    9: "Noisy-OR over all paths beat the strongest path by 0.08–0.10 MRR on three tasks.",
    10: "Hub discounting added 0.02–0.16 MRR, with the same sign on all 30 splits.",
    11: "Training loss located the t-norm only loosely; its optimum moved across splits.",
    12: "<<HL4>>",
    13: "Paths via known indications carried 99% of evidence and hurt held-out diseases.",
}

ABSTRACT = {
    15: "**Objective:** Path-based reasoning over medical knowledge graphs explains its predictions but composes evidence "
        "with operators fixed in advance. We asked which components of a differentiable neuro-fuzzy path reasoner are worth "
        "learning, and whether the training objective determines a learned composition operator.",
    16: "**Methods:** Differentiable Neuro-Fuzzy Reasoning (DNFR), built as an instrument for this question, has "
        "degree-discounting edge memberships, a Schweizer–Sklar t-norm per metapath and a noisy-OR over paths, each of "
        "which can be fixed or removed. On drug repurposing and disease–symptom prediction (Hetionet; a graph learned "
        "from emergency-department records) we compared ablations, fixed and validation-selected t-norms and baselines "
        "under a pre-specified decision rule, profiled the training loss in the t-norm parameter to convergence, ablated "
        "the metapaths through the target relation, held out whole diseases or compounds, and attributed predictions to "
        "facts by exact Shapley values.",
    17: "**Results:** Aggregating all paths rather than the strongest raised mean reciprocal rank (MRR) by 0.08–0.10, and "
        "hub discounting added 0.02–0.16. Within a split the converged loss located the t-norm parameter only within wide "
        "or disjoint intervals, its minimiser moved between splits, and it missed the value that maximised validation MRR. "
        "Degree-weighted path counts were at least as accurate as DNFR<<ABS_M2>>. On Hetionet drug repurposing, 99% of "
        "DNFR's evidence ran through metapaths containing known indications; without them MRR fell from 0.599 to 0.289, "
        "and for held-out diseases it was <<ABS_TAIL>>.",
    18: "**Conclusion:** Path aggregation and hub discounting carry the accuracy; the composition operator should be "
        "selected on held-out data, not learned and interpreted. Exact, non-negative attribution to facts is a property "
        "of additive path models, not of fuzzy operators. On curated graphs, random-split accuracy largely reflects "
        "association with known indications, so claims of mechanistic reasoning need target-relation ablations and "
        "entity-centred splits.",
}

SIGNIF = {
    28: "**What is already known.** Degree-weighted path counts and rule learners explain their predictions but fix the "
        "composition operator, usually a product; neuro-fuzzy systems and differentiable fuzzy logic allow the operator to "
        "be learned, but whether medical graph data determine a learned operator has not been tested.",
    29: "**What this paper adds.** A controlled component analysis of a differentiable neuro-fuzzy path reasoner on three "
        "link-prediction tasks from two medical knowledge graphs with ten splits each, and a descriptive fourth task. "
        "Aggregating all paths and discounting hubs carried the gains; four plausible refinements did not help detectably. "
        "The training loss constrained the t-norm parameter only loosely, its optimum moved between splits and disagreed "
        "with held-out accuracy, so a learned operator should not be reported as a property of the domain. Degree-weighted "
        "path counts were at least as accurate, and a linear non-negative version of them has the same exact attribution "
        "to facts. On drug repurposing most of the evidence ran through known indications rather than through genes, and a "
        "model trained without those paths ranked drugs for held-out diseases about three times better.",
}

INTRO = {
    23: "In most methods the operator is fixed in advance: path counts and NeuralLP multiply, GNN-QE and FuzzQE use fixed "
        "fuzzy connectives [17,18], and clinical fuzzy systems have used max–min composition since Sanchez [19–21]. Flexible "
        "neuro-fuzzy systems learn parametric t-norms [22], and differentiable fuzzy logic has studied how operators shape "
        "learning [23,24], but whether such a parameter can be learned from medical graph data, rather than merely fitted, "
        "has not been asked.",
    24: "We study the question with a path-based link predictor built as an instrument for it, Differentiable Neuro-Fuzzy "
        "Reasoning (DNFR), in which each component can be switched off or fixed: an edge membership that discounts hubs and "
        "reads graded evidence; a Schweizer–Sklar t-norm learned per metapath, ranging from the minimum through the Hamacher "
        "product and the product to the Łukasiewicz t-norm; and a noisy-OR over paths, so that every score is a sum of path "
        "terms. Four refinements each replace one simplifying assumption with a standard alternative. DNFR is not proposed "
        "as a better predictor; the question is which of its parts deserve to be learned. The tasks are drug repurposing in "
        "Hetionet and disease–symptom prediction in Hetionet and in a graph learned from 273,174 emergency-department "
        "records [26], with PrimeKG [2] as a descriptive fourth task. We ask: (RQ1) how accurate is the learned reasoner "
        "against path-count, rule, completion, degree-only and chance baselines, including for diseases and compounds "
        "without known indications, and how much of its accuracy rests on paths through known instances of the target "
        "relation; (RQ2) which components carry the gain; (RQ3) does the training objective determine the t-norm "
        "parameter, within a split and across splits; and (RQ4) how concise and faithful are the explanations, at the level "
        "of paths and of facts, and which of their properties are specific to fuzzy composition?",
    25: "The contribution is a controlled analysis that applies to any learned operator in a path-based clinical reasoner: "
        "a component analysis under one protocol and decision rule; a converged profile of the training loss in the "
        "operator parameter, reported with the disagreement between optimisation chains and for single metapaths, which "
        "separates what the loss can resolve from what varies between splits; an ablation of the metapaths through the "
        "target relation together with disease- and compound-centred splits; and an exact edge-level Shapley attribution "
        "with fact-level counterfactuals, compared with degree-weighted path counts trained with the same objective and "
        "constrained to the same additive form. Code, tests and all result files are released.",
}

RELATED = {
    33: "Hetionet [1] and PrimeKG [2] are the most widely used open medical graphs for drug repurposing. Himmelstein et al. "
        "predicted treatment from degree-weighted path counts (DWPC) over 123 metapaths with a regularised logistic "
        "regression that also used degree features and a permutation prior, and later made metapath explanations "
        "searchable [1,27]. Their model already relied on known treatments: several of the metapaths with the largest "
        "coefficients ran through the treats relation (for example compound–binds–gene–binds–compound–treats–disease, "
        "compound–resembles–compound–treats–disease and compound–treats–disease–resembles–disease), and for epilepsy most of "
        "the support for its predictions came from the disease's known treatments [1]; they reported coefficients and "
        "case-level contributions but no ablation of these metapaths. Evaluations of repurposing models have since shown "
        "that indications are easy to predict from the category of a drug, so that random splits overestimate accuracy "
        "[47], that random and cold-start splits both overestimate accuracy on an independent test set [48], and that "
        "guilt by association across drugs and diseases can be a design principle rather than an artefact [49]; TxGNN was "
        "motivated by the zero-shot setting of diseases without treatments and evaluated with disease-area splits [8]. "
        "Embedding accuracy depends heavily on training configuration [28,29]. Rotmensch et al. fitted a noisy-OR model to "
        "emergency-department records, so that each disease–symptom edge carries an estimated probability [26]. Degree is "
        "a strong signal in these graphs: embeddings favour high-degree entities [3], and an edge prior computed from "
        "degrees alone reconstructs most Hetionet relations [4], so a learned component has to be measured against "
        "degree-only baselines. Resnik's information content [30] is the classical alternative to raw degree as a measure "
        "of specificity. Against this background, our analysis adds a quantitative ablation of the target-relation "
        "metapaths across three path methods, the share of the evidence behind correct predictions that runs through "
        "them, and the accuracy of the remaining mechanistic evidence when whole diseases or compounds are held out.",
}

METHODS = {
    47: "where σ is the logistic function, a_{r}, b_{r}, c_{r} and g_{r} are learned per relation and direction, and "
        "τ = 1 except in refinement R4. Positive b_{r} and c_{r} discount edges at hubs, as the damping exponent of DWPC "
        "does [1], but the strength of the discount is learned. Without graded weights the last term is absent.",
    48: "**Learned t-norm.** The strength of a path π of length L is the Schweizer–Sklar t-norm [16] with a parameter "
        "p_{m} learned for its metapath m:",
    50: "As p → −∞ the family tends to the minimum, p = −1 gives the Hamacher product, p → 0 the product and p = 1 the "
        "Łukasiewicz t-norm, so the fixed operators of Section 3.4 lie inside the family. For p > 0 a path whose memberships "
        "satisfy Σ_{i} μ_{i}^{p} ≤ L − 1 has strength exactly zero and passes no gradient. By default "
        "p_{m} = 6σ(θ_{m}) − 5, in (−5, 1), initialised at −0.1, close to the product. Equation 2 is evaluated in log space.",
    51: "**Noisy-OR aggregation.** With a reliability ω_{m} ∈ (0, 1) per metapath and a leak β_{0}, the paths Π(h, t) are "
        "combined as",
    53: "with κ_{π} = 1 in DNFR. Because −log(1 − P) is then a sum of one term per path, "
        "s_{π} = −κ_{π} log(1 − ω_{m} T(π)), each path's share is an exact decomposition of the model's score. It is not an "
        "exact account of the evidence, since paths that share an edge count it once per path; Section 3.6 removes this "
        "double counting at the level of edges.",
    54: "**Training.** For each training query we minimise the cross-entropy of a softmax over the logits of P(h, t) for "
        "all candidate tails, excluding the other known training tails of h (filtered softmax), computed exactly over all "
        "pairs with at least one path. Gradients are accumulated metapath by metapath in two passes, which bounds memory by "
        "the largest metapath. We use Adam (learning rate 0.1) for at most 400 full-batch epochs, evaluate validation MRR "
        "every five epochs, stop after ten evaluations without improvement and keep the best parameters. An L2 penalty "
        "λ‖θ − θ_{0}‖² with λ = 10^{−3} acts on the t-norm parameters; it pulls them towards their initial value, which is "
        "why Section 3.4 also trains without it. Degrees are counted on the training graph of each split. Algorithm S1 in "
        "the Supplementary material gives one training run in full.",
    56: "Each refinement replaces one element of Section 3.2, reduces to it for one parameter value, and is defined in "
        "Supplementary Section S5; together they give the variant we call DNFR + R1–R4. (R1) *Overlap-aware aggregation* "
        "discounts the weaker of two paths through the same interior node by a learned exponent β (Equation 3 at β = 0). "
        "(R2) *Discrete operator choice* selects the product, minimum or Hamacher product per metapath by Gumbel-softmax "
        "[41]. (R3) An *information-content discount* replaces the degree feature by log(1 + n(v)), with n(v) the number of "
        "training paths through v, which equals −IC(v) [30] up to a constant. (R4) *Membership annealing* lowers τ from 2 to "
        "0.5 over the first 100 epochs.",
    58: "**Fixed operators** (product, minimum, Hamacher, Łukasiewicz) use the membership, aggregation and training of "
        "DNFR. Because every Łukasiewicz path starts at strength zero from the default initialisation (Section 4.4), its "
        "accuracy in Supplementary Tables S1–S4 reflects the initialisation, not the operator. **Fixed p selected on "
        "validation**: one Schweizer–Sklar p shared by all metapaths is set to each value of a grid (−4, −2, −1, −0.5, 0.5 "
        "and the product; −2, −1, −0.5 and the product on Hetionet compound–treats–disease, for cost), and the value with "
        "the best validation MRR on each split is kept. **Component ablations** change one component: a single shared p; no "
        "degree terms (b_{r} = c_{r} = 0); degree terms fixed at 0.4 for every relation (with the product, DWPC-like); the "
        "strongest path only instead of noisy-OR (max-path); and, on the graded graph, binary edges (g_{r} = 0). Each "
        "refinement is added to DNFR alone and removed from DNFR + R1–R4.",
    59: "**Identifiability of the t-norm.** We restarted training from p_{0} = −2, −1, −0.1 and 0.5, with and without the "
        "penalty; used an unbounded parameter p_{m} = θ_{m}; trained for 400 epochs without early stopping; and froze the "
        "memberships, which tests whether memberships and operator substitute for each other, a structural property. On the "
        "graded graph and Hetionet disease–symptom we computed a converged profile, fixing a shared p on a grid from −5 to "
        "0 (−20 to 0.25 on Hetionet; extended to p > 0 with warm starts on the graded graph) and fitting all other "
        "parameters by L-BFGS until the gradient norm fell below 10^{−5} or the loss changed by less than 10^{−9}. Because "
        "the objective is not convex in the other parameters, each point was fitted by two to eight warm-start chains "
        "(sweeps up and down the grid, and chains started from the product or from the far end); the profile takes the "
        "lowest converged loss, and we call a point resolved when the two best chains agree within 10^{−4} nat, about a "
        "seventh of the profile-likelihood threshold χ²(0.95, 1)/(2N). On Hetionet disease–symptom we also profiled the "
        "parameters of the two metapaths that carry most of DNFR's evidence one at a time, with the parameters of the other "
        "15 metapaths re-fitted in the non-nilpotent range p ≤ 0. Following Raue et al. [42], a parameter is practically "
        "non-identified if its profile stays within the threshold over a wide range while the others compensate; a "
        "minimiser that moves between random splits is sampling variability, a separate matter.",
    60: "**Baselines.** A uniformly random ranker gives the chance level; under filtered ranking its expected MRR for a "
        "query with n candidates is H_{n}/n, with H_{n} the n-th harmonic number. Popularity ranks tails by training "
        "positives; the edge prior [4] uses 100 degree-preserving permutations. DWPC with damping 0.4 on the same metapaths "
        "feeds an L2-regularised logistic regression (DWPC-LR) [1], also with degree and prior features. Two further DWPC "
        "models separate the model from the objective (Section 4.1): the same asinh-transformed features trained with DNFR's "
        "exact filtered softmax, and a linear model in the untransformed DWPC with non-negative coefficients, trained with "
        "the same softmax; the penalty of both is chosen on validation data. AnyBURL [9] with 100 s of rule learning, "
        "truncated SVD and Soft-Impute [43] on the target relation alone complete the set; untuned TransE and RotatE [5,7] "
        "are reported in the Supplement only. NBFNet [35] was run as a compact CPU version (three layers, DistMult messages, "
        "dimension 32; also tuned over a 2 × 2 grid on the graded graph) on the two disease–symptom graphs; BioPathNet "
        "[13], TxGNN [8] and NBFNet on the drug-repurposing graphs need GPU training, which was not available, so our "
        "statements about drug repurposing concern path-weighting methods on a fixed path set. DNFR has 354, 80, 169 and 11 "
        "trainable parameters and DWPC-LR 131, 18, 52 and 2 (Table 1), so the comparison is between ways of weighting the "
        "same path evidence. Supplementary Section S5 gives every setting.",
    64: "**Hetionet** (v1.0) [1] has 47,031 nodes of 11 types and 2,250,197 edges of 24 types. **PrimeKG** [2] has 129,375 "
        "nodes; after removing the 1,538,090 anatomy–protein expression edges, which lie on no drug–disease path of length "
        "three or less, and duplicates, 2,511,974 edges remain. **Health knowledge graph (HKG).** Rotmensch et al. learned "
        "this graph from 273,174 emergency-department visits recorded between 2008 and 2013 [26]. Its 3,704 disease–symptom "
        "edges carry the noisy-OR importance w_{e} = 1 − f, where f is the estimated probability that the disease fails to "
        "produce the symptom (median 0.021). Its only metapath is disease–symptom–disease–symptom, so it tests the "
        "composition of graded evidence from records in a collaborative-filtering setting. Metapaths with more than five "
        "million paths were excluded for all path methods (4 of 134 on compound–treats–disease, 31 of 82 on PrimeKG, the "
        "most hub-dominated), so PrimeKG accuracies of all path methods are lower bounds. Path counts for this exclusion "
        "were computed once on the full graph, before splitting; <<EXCL_NOTE>>",
    66: "Positive pairs were split at random 80/10/10, with 10 splits on each Hetionet task and the graded graph and 2 on "
        "PrimeKG. Each test tail is ranked among all candidates of its type after removing the head's other known tails "
        "(filtered ranking, ties at the mean rank); we report mean reciprocal rank (MRR), Hits@k and the chance level. On "
        "compound–treats–disease a **disease-centred split** (10 splits) holds out every indication of a test disease [8], "
        "and a **compound-centred split** (10 splits) holds out every indication of a test compound, the setting of a drug "
        "without known indications. There, 33 of the 130 metapaths contain the treats relation itself (e.g. "
        "compound–resembles–compound–treats–disease); we retrained DNFR, the product and DWPC-LR without them on the random "
        "splits, then also without those through compound–palliates–disease, and retrained the product and DWPC-LR "
        "without them on the disease-centred splits.",
    69: "**Clinical assessment (pilot).** On each disease–symptom task, 25 held-out pairs that DNFR ranked in its top 10 "
        "were rated with their three leading paths by the second author, and 40 pairs, half of them incorrect top-10 "
        "predictions, by a physician who is not an author (Supplementary Section S7). The assessment was not "
        "pre-registered and is reported as a pilot.",
    70: "**Software and AI assistance.** Experiments ran on a shared two-core CPU with 8 GB of memory, without a GPU; this "
        "limited the number of PrimeKG splits and excluded GPU-trained baselines. Defaults were set during development for "
        "stable, feasible training, not by search; refinement settings were fixed before any was run on more than one "
        "split. The analysis code was written with the help of an AI system (Claude, Anthropic), which also ran the "
        "experiments under the authors' direction; the authors specified models, protocols and analyses (Section 7).",
}

RESULTS = {
    73: "Table 2 gives test MRR; methods are compared within a task, since the number of candidates ranges from 137 to "
        "17,080, and the expected MRR of a uniformly random ranker is given for reference (0.041, 0.016, 0.0006 and "
        "0.021). Degree alone explained little: popularity and the edge prior were far behind every path method on every "
        "task. The prior fell below popularity on compound–treats–disease because, under leave-dyad-out splits, many test "
        "compounds have no training indication left, so every candidate has a prior of zero (46% of test queries; "
        "Supplementary Table S9).",
    76: "<<P_41_DWPC>>",
}

RESULTS2 = {
    82: "**Learning the t-norm.** A parameter learned per metapath was not consistently better than fixed members of its "
        "own family: against the product the differences were −0.003, +0.003, −0.009 and +0.006, against the Hamacher "
        "product +0.024, −0.002, −0.005 and +0.010, all inconclusive. We claim no consistent gain, not equivalence. The "
        "+0.024 rests largely on one split in which the Hamacher run stopped in a poorer optimum (MRR 0.488 against 0.658, "
        "training loss 3.16 against 2.38); without it the difference was +0.007 (Supplementary Table S13). The t-norm "
        "mattered most when memberships could not move: with memberships frozen, learning p improved on the product by "
        "+0.056 [+0.049, +0.063] on the graded graph, and learning the memberships instead improved it by +0.121 [+0.111, "
        "+0.131]. Memberships and operator are largely substitutes, as weights and t-norm parameters are in flexible "
        "neuro-fuzzy systems [22].",
    86: "Of the 14 add-one comparisons in Table 4, none showed a gain under our rule, one was equivalent and 13 were "
        "inconclusive (range −0.046 to +0.019). Membership annealing (R4) had the most consistent positive estimates "
        "(+0.001, +0.013, +0.007), although saturation, its target, was rare (0.9% and 0.3% of memberships above 0.99), so "
        "we neither recommend nor rule it out. The exponent of R1 fell from 1 to a median of 0.014 on the graded graph, but "
        "the loss hardly constrains parameters that rescale paths (Section 4.4), so its value does not show whether such "
        "paths repeat evidence. R3 lost 0.046 on the graded graph because its feature had about half the spread of the "
        "degree feature; matched in standard deviation it changed MRR by −0.003 [−0.007, +0.001]. DNFR + R1–R4 differed "
        "from DNFR by −0.023, −0.020 and +0.005, and was less accurate under our rule on Hetionet disease–symptom.",
    87: "Selecting a fixed shared p on validation data, the procedure that Section 4.4 recommends, differed from the fixed "
        "product by +0.001 [−0.007, +0.008] on compound–treats–disease (equivalent), +0.016 [+0.011, +0.020] on Hetionet "
        "disease–symptom and +0.019 [+0.014, +0.025] on the graded graph, the last supported under our rule (a worked "
        "example of the rule for this comparison is in Supplementary Section S3); there the grid value chosen was −0.5 in 6 "
        "splits, −2 in 3 and −4 in 1. On Hetionet disease–symptom validation chose the product itself on four splits, and "
        "the gain came from splits on which the early-stopped product run ended low: fitted to convergence, the product "
        "reached 0.245, 0.269 and 0.210 on splits 0–2 against 0.208, 0.268 and 0.180 early-stopped (Supplementary Table "
        "S17). We therefore claim a benefit of validation selection for the graded graph only. On compound–treats–disease "
        "the validation set holds about 75 pairs, so validation MRR moves in steps that are large compared with the "
        "differences being selected (Supplementary Table S10); the equivalent verdict there should be read with this "
        "noise in mind.",
    89: "Section 4.2 answered the structural question: memberships can partly compensate for p. The practical question is "
        "whether the data, through the loss, determine p [42]. With early stopping and the default penalty, learned values "
        "stayed near their start, which the penalty towards θ_{0} favours by construction. Without the penalty, restarts "
        "still ended in different places at similar accuracy (Supplementary Table S11): on the graded graph from −2, −1, "
        "−0.1 and 0.5 they ended at medians of −3.51, −2.17, −0.46 and 0.91, moving away from the start in both directions. "
        "On compound–treats–disease the interquartile range of learned values was only [−0.06, 0.16] although a median of "
        "43 epochs could have moved p by close to one unit. Neither an unbounded parameterisation (graded graph: median "
        "−3.21 from −2, −0.46 from −0.1, 0.95 from 0.5) nor 400 epochs without early stopping made p converge "
        "(Supplementary Table S6). Runs that leave a value of −0.5 to −0.25 in opposite directions are what a ridge in the "
        "profile, with lower ground on either side, would produce; the converged profile shows such a ridge (Figure 2A).",
    91: "**Figure 2.** Converged profiles over a fixed shared t-norm parameter p, three splits each. (A) Graded graph and "
        "(B) Hetionet disease–symptom: training loss minus its minimum over the grid (×10⁻³ nat) for the lowest-loss chain "
        "at each p; the shaded band spans all warm-start chains at that p, and open markers flag points where the two best "
        "chains differ by more than 10⁻⁴ nat. The grey band is the profile-likelihood threshold χ²(0.95, 1)/(2N). "
        "(C) Validation MRR of the same fits, minus its maximum. All other parameters were fitted by L-BFGS to convergence. "
        "Dashed lines mark the Hamacher product (p = −1), the product (p = 0) and the Łukasiewicz t-norm (p = 1). The slope "
        "∂L/∂p and test MRR at every point are in Supplementary Tables S7 and S15.",
    92: "<<P_44_PROFILE>>",
    93: "<<P_44_HELDOUT>>",
}

RESULTS3 = {
    97: "<<P_45_TAIL>>",
    98: "<<P_45_TREATS>>",
}

EXPL = {
    103: "<<P_46_FACTS>>",
    104: "<<P_46_CLIN>>",
}

DISC = {
    107: "<<P_51>>",
    110: "**Check identifiability before interpreting a learned operator.** Restart from several values, profile over "
         "fixed values with the other parameters converged, report the disagreement between optimisation chains, and ask "
         "whether held-out accuracy and training loss agree. Where they disagree, as on our graded graph, select the "
         "operator on validation data as a hyperparameter, preferably among a few non-nilpotent t-norms, and do not report "
         "the learned value as a property of the medical domain.",
    111: "**Test what the evidence rests on.** On curated drug-repurposing graphs, metapaths through the target relation "
         "carried almost all of the evidence, for every path method, as the coefficients of earlier path-count models "
         "already suggested [1]. Random splits reward this guilt by association, useful for repositioning within a drug "
         "class but uninformative about mechanisms [47]; claims of multi-relational or mechanistic reasoning should be "
         "backed by an ablation of those metapaths and by disease- and compound-centred splits, with the chance level "
         "reported alongside. Where the use case is a disease without known treatments, train without those "
         "metapaths: on our held-out diseases this more than tripled MRR.",
    112: "**Attribute to facts and test at the level of facts.** Path shares double-count facts on shared paths; the "
         "edge-level Shapley value of any additive path model with non-negative terms is exact and cheap, and removing a "
         "fact from the graph and re-ranking tests what a clinician would question. Path-level deletion within a pair says "
         "little, because an additive score depends on how much evidence is removed.",
}

LIMIT = {
    115: "**Splits and power.** With ten splits, corrected 90% intervals still extend to about ±0.015, so equivalence "
         "within ±0.01 was rarely attainable and differences of 0.02 to 0.04 were often inconclusive despite a constant "
         "sign. PrimeKG, with two splits, is descriptive and does not enter the Highlights or the conclusions.",
    116: "**Partial coverage.** The refinements, validation selection, NBFNet, the entity-centred splits and the treats "
         "ablation were not run on PrimeKG, so the zero-shot results rest on Hetionet compound–treats–disease, with 137 "
         "candidate diseases and about 10 test diseases per disease-centred split; DNFR itself was run on four of the ten "
         "disease-centred splits, and the compound-centred and treats-free disease-centred splits used the fixed product, "
         "which did not differ from DNFR on this task. The converged profile covers the two "
         "disease–symptom tasks, with a shared p and, on Hetionet, two single metapaths; on drug repurposing we rely on "
         "restarts, the unbounded parameterisation and frozen memberships.",
    117: "**Learned-operator baselines and compute.** All experiments ran on a two-core CPU. NBFNet was a compact CPU "
         "implementation tuned on one graph; BioPathNet and TxGNN were not run, and TxGNN's published results use another "
         "graph and protocol [8], so they cannot be set beside ours. AnyBURL had 100 s of rule learning; ten times more "
         "changed nothing on the graded graph, but on compound–treats–disease its rule set may not have saturated. Metapaths with more than five million paths, the "
         "most hub-dominated, were excluded, so the effect of hub discounting may be underestimated.",
    118: "**Clinical evaluation.** The assessment is a pilot. The first rater, a co-author trained in clinical psychology "
         "rather than medicine, knew the aims of the study; the second, a physician, rated only 40 pairs and knew the "
         "authors; their agreement was moderate, and the protocol was not pre-registered. Drug repurposing was not rated. "
         "A study with several physicians who are not authors, a pre-specified protocol, blinding to the model and "
         "DNFR's and DWPC's explanations side by side is the next step.",
    119: "**Scope of the evidence.** Scores are rankings, not calibrated probabilities; edges were thinned uniformly "
         "(Supplementary Section S6.1), whereas less-studied entities lose more; and paths combine curated, text-mined and "
         "co-occurrence evidence, so a high-scoring path is a reason to look, not evidence of a causal mechanism. The edge "
         "weights of the graded graph were estimated by Rotmensch et al. from all records, so the weights of training "
         "edges were fitted with knowledge of the held-out edges; the effect is probably small, but it is a form of "
         "leakage that our splits cannot remove.",
}

CONCL = {
    121: "<<P_6>>",
}

DECL = {
    124: "**Data availability.** Hetionet v1.0 is available from its GitHub repository and PrimeKG from Harvard Dataverse "
         "(doi:10.7910/DVN/IXA7BM). The parsed health knowledge graph, the per-query ranks of every method on every split, "
         "the trained models, the path explanations, the clinical rating sheets (with the raters' names and affiliations "
         "removed) and the statistics files from which every number in this paper is taken are deposited at Zenodo "
         "(version <<VER>>: <<ZEN_VER_DOI>>; all versions: https://doi.org/10.5281/zenodo.23879352), with a script that "
         "downloads Hetionet and PrimeKG and checks them against the checksums of the files we used. The record and the "
         "code release are public.",
    125: "**Code availability.** The implementation of DNFR and its refinements, the compact NBFNet and all baselines, the "
         "tests, and the scripts that compute the statistics and figures are available under the MIT licence at "
         "https://github.com/Bangth2026/dnfr-medical-kg (release <<VER>>). The release includes check_numbers.py, which "
         "verifies that every number in this article and its supplementary material is one generated from the result "
         "files.",
    127: "**Acknowledgements.** We thank Dr Do Kim Phuc, MSc, MD (Pham Ngoc Thach University of Medicine, Ho Chi Minh "
         "City), who is not an author but is acquainted with the authors, for rating the explanation paths as the second "
         "clinical rater.",
    128: "**Independent verification.** Without the AI system, the corresponding author checked every result reported in "
         "this article and its supplementary material against the released result files. A re-run of DNFR, DNFR with the "
         "strongest path only and DNFR without degree terms on the first split of the graded graph from a clean clone of the "
         "repository, in a fresh environment (scripts/verify_clean.sh), reproduced the released test MRR to four decimals "
         "(0.3204, 0.2307, 0.2739). The larger tasks were not re-run independently of the AI system.",
}
