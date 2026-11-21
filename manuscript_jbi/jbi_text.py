# -*- coding: utf-8 -*-
"""Text of the JBI version. Markup: **bold**, *italic*, _{sub}, ^{sup}. Citations [n] use the numbering of the
v5.0 (AIM) manuscript and are renumbered by order of first appearance when the document is built.
Every number below is copied from the v5.0 manuscript or supplement, which were generated from the released
result files; check_numbers.py verifies this."""

TITLE = ("Which parts of a neuro-fuzzy path reasoner should be learned? A controlled component analysis on medical "
         "knowledge graphs")
RUNNING = "Neuro-fuzzy path reasoning on medical knowledge graphs"

HIGHLIGHTS = [
    "Noisy-OR over all paths beat the strongest path on all three tasks tested.",
    "Hub discounting had the same sign on every split of every task tested.",
    "The training loss barely constrained the t-norm parameter on two graphs.",
    "Degree-weighted path counts were at least as accurate as DNFR on three of four tasks.",
    "Drug-repurposing accuracy on Hetionet rested on paths through known indications.",
]

ABSTRACT = [
    ("Objective", "Path-based reasoning over medical knowledge graphs explains its predictions but composes evidence with "
     "fixed operators. We asked which parts of a differentiable neuro-fuzzy path reasoner are worth learning, and whether a "
     "learned composition operator is determined by the data."),
    ("Methods", "Differentiable Neuro-Fuzzy Reasoning (DNFR) learns degree-discounting edge memberships, a Schweizer–Sklar "
     "t-norm per metapath and a noisy-OR over paths. On drug repurposing (Hetionet, PrimeKG) and disease–symptom prediction "
     "(Hetionet; a graph learned from emergency-department records) we compared ablations, fixed and validation-selected "
     "t-norms, four refinements, NBFNet and simpler baselines under leave-dyad-out splits, profiled the t-norm parameter to "
     "convergence, removed the metapaths through the target relation and attributed predictions to facts with an exact "
     "Shapley decomposition."),
    ("Results", "Aggregating all paths rather than the strongest raised mean reciprocal rank (MRR) by 0.08–0.10, and hub "
     "discounting had the same sign on every split. No refinement improved MRR detectably. On the graded graph and on "
     "Hetionet disease–symptom, the training loss varied by at most 0.005 nats over t-norm parameters from −5 to 0, within "
     "the fits' precision, and its minimiser differed between splits; a validation-selected t-norm beat the fixed product by "
     "0.019 on the graded graph. Degree-weighted path counts were at least as accurate as DNFR on three of four tasks. On "
     "Hetionet drug repurposing, 99% of DNFR's evidence ran through metapaths that contain known indications, and without "
     "them MRR fell from 0.599 to 0.289."),
    ("Conclusion", "Path aggregation and hub discounting carry the accuracy, and the composition operator is better treated "
     "as a hyperparameter than learned. DNFR's advantage over degree-weighted path counts is an exactly additive "
     "attribution to facts, not accuracy. On curated graphs, random-split accuracy largely reflects association with known "
     "indications, so claims of mechanistic reasoning need ablations of the target relation or disease-centred splits."),
]
KEYWORDS = ["knowledge graph", "neuro-fuzzy reasoning", "t-norm", "identifiability", "explainable AI", "drug repurposing"]

SIGNIFICANCE = [
    ("Problem or issue", "Path-based reasoning over medical knowledge graphs is explainable, but it combines evidence along "
     "a path with an operator fixed in advance, and sparse graphs with hub nodes make that choice consequential."),
    ("What is already known", "Degree-weighted path counts and rule learners explain their predictions but fix the "
     "composition operator, usually a product; neuro-fuzzy systems and differentiable fuzzy logic allow the operator to be "
     "learned, but whether a learned operator is determined by medical graph data has not been tested."),
    ("What this paper adds", "A controlled component analysis of a differentiable neuro-fuzzy path reasoner on four "
     "link-prediction tasks from three medical knowledge graphs. Aggregating all paths and discounting hubs carried the "
     "gains; four plausible refinements did not help detectably. Once memberships are learned the training loss barely "
     "depends on the t-norm parameter, so a learned operator is not a property of the domain; selecting it on validation data "
     "helped on the graded graph. Degree-weighted path counts were at least as accurate on three of four tasks, and on drug "
     "repurposing most of the evidence ran through known indications rather than through genes."),
    ("Who would benefit", "Developers and evaluators of explainable clinical decision support built on knowledge graphs, "
     "and researchers who learn or report fuzzy operators for medical reasoning."),
]

# ---------------------------------------------------------------------------------------------------------------
# Body. ("h1"|"h2", text) headings; ("p", text) paragraph; ("eq", text) display equation; ("li", text) list item;
# ("sig",) statement of significance; ("table", k) original table index k; ("fig", k) original figure k.
BODY = [
("h1", "1. Introduction"),
("p", "A clinician asked to trust a predicted drug–disease link wants to know two things: how strongly the evidence "
 "supports it, and what that evidence is. Medical knowledge graphs can in principle answer both. Hetionet integrates 29 "
 "public resources into a graph of 47,031 nodes and 2,250,197 edges [1], and PrimeKG links 17,080 diseases to drugs, "
 "proteins, phenotypes and exposures [2]. In practice these graphs are sparse where it matters, so a new relation can only "
 "be inferred by chaining intermediate ones, for example from a compound to the gene it binds and on to a disease "
 "associated with that gene. They are also dominated by hubs, and node degree alone predicts much of what link predictors "
 "recover from them [3,4]."),
("p", "Knowledge graph embeddings [5–7] and graph neural networks such as TxGNN [8] score a candidate from learned vectors "
 "and are often accurate, but they give no account of why a pair scores high. Path-based methods score a pair from the "
 "paths that connect it: degree-weighted path counts [1], rule learners such as AnyBURL and NeuralLP [9,10], "
 "reinforcement-learning walkers [11,12] and path-representation networks such as BioPathNet [13] all relate a "
 "prediction to the paths behind it, which is the traceable evidence that reviews of explainable clinical AI ask "
 "decision support to provide [14]. Every such method must also decide how strengths combine along a path and across "
 "paths. Fuzzy logic makes the first decision explicit: the strength of a path is a triangular norm (t-norm) of the "
 "strengths of its edges [15,16]. The choice is not cosmetic. On a path of three edges of strength 0.3, the minimum "
 "returns 0.3, the product 0.027 and the Łukasiewicz t-norm 0."),
("p", "In most methods the operator is fixed in advance. Degree-weighted path counts multiply damped edge weights, "
 "NeuralLP composes relations by matrix products, GNN-QE and FuzzQE use fixed fuzzy connectives [17,18], and clinical "
 "fuzzy systems have used max–min composition since Sanchez's work on medical diagnosis [19–21]. Flexible neuro-fuzzy "
 "systems learn the parameters of parametric t-norms jointly with the other weights [22], and differentiable fuzzy logic "
 "has studied how the operator affects learning [23,24]. Whether such a parameter can be learned from medical graph "
 "data, rather than merely fitted, has not been asked. Our earlier, not yet peer-reviewed analysis of a population fuzzy "
 "graph from the US National Health and Nutrition Examination Survey found no gain from tuning a parametric t-norm over "
 "the product [25]; it motivated the question but is not part of the evidence here."),
("p", "We study the question with Differentiable Neuro-Fuzzy Reasoning (DNFR), a path-based link predictor built so that "
 "each component can be switched off or fixed: an edge membership that discounts high-degree nodes and reads graded "
 "evidence where it exists; a Schweizer–Sklar t-norm whose parameter is learned per metapath, ranging from the minimum "
 "through the Hamacher product and the product to the Łukasiewicz t-norm; and a noisy-OR over paths, so that every score "
 "is a sum of path terms. We also test four refinements, each replacing one simplifying assumption of the model with a "
 "standard alternative. The tasks are drug repurposing in Hetionet and PrimeKG and disease–symptom prediction in "
 "Hetionet and in a graph that Rotmensch et al. learned from 273,174 emergency-department records [28], whose edges "
 "carry graded noisy-OR weights. Four questions organise the study. (RQ1) How accurate is a learned neuro-fuzzy path "
 "reasoner against degree-weighted path counts, rule learning, low-rank completion and degree-only baselines, including "
 "for diseases without known indications, and how much of that accuracy rests on paths through known instances of the "
 "target relation? (RQ2) Which components carry the gain, and do the refinements add to them? (RQ3) Is the t-norm "
 "parameter identified by the training objective, and does selecting it on validation data recover what end-to-end "
 "learning misses? (RQ4) How concise and how faithful are the explanations, at the level of paths and of facts?"),
("p", "The contribution is methodological, and its lessons concern any learned operator in a path-based clinical "
 "reasoner. First, a controlled component analysis separates path aggregation, degree- and evidence-aware memberships and "
 "the t-norm, with fixed and validation-selected operators trained under one protocol and one decision rule for "
 "equivalence and difference. Second, an identifiability analysis, with restarts, an unbounded parameterisation and "
 "converged profiles, separates structural redundancy between memberships and operator from practical "
 "non-identifiability by the training loss. Third, removing the metapaths through the target relation measures how much "
 "drug-repurposing accuracy rests on known indications, for DNFR and for its baselines. Fourth, an exact edge-level "
 "Shapley attribution and fact-level counterfactuals evaluate the explanations against the same analysis of "
 "degree-weighted path counts. The implementation, tests and all result files are released."),
("sig",),

("h1", "2. Related work"),
("h2", "2.1 Medical knowledge graphs and degree bias"),
("p", "Hetionet [1] and PrimeKG [2] are the most widely used open medical knowledge graphs for drug repurposing. "
 "Himmelstein et al. predicted compound–disease treatment from degree-weighted path counts (DWPC) over 123 metapaths with a "
 "regularised logistic regression that also used degree features and a permutation-derived prior, and later made "
 "metapath explanations searchable [1,29]. Embedding methods score triples with learned vectors [5–7]; PyKEEN provides "
 "reference implementations [30], and their accuracy depends heavily on training configuration [31]. TxGNN trains a graph "
 "neural network on PrimeKG for zero-shot repurposing with a path-based explainer for clinicians [8]. Graphs learned from "
 "records add graded edges: Rotmensch et al. fitted a noisy-OR model to emergency-department notes and diagnosis codes, so "
 "that each disease–symptom edge carries an estimated probability that the disease produces the symptom [28]. Degree is a "
 "strong signal in these graphs. Embedding models trained on biomedical graphs favour high-degree entities [3], and an "
 "edge prior computed from node degrees alone reconstructs most Hetionet relations [4], so a claim that a learned "
 "component helps has to be measured against degree-only baselines. Resnik's information content [26] is the classical "
 "alternative to raw degree as a measure of how specific a node is."),
("h2", "2.2 Path-based and neuro-symbolic reasoning"),
("p", "The Path Ranking Algorithm scores a pair by a learned combination of path-constrained random walks [32], a direct "
 "ancestor of DWPC with logistic regression. AnyBURL mines path rules bottom-up [9]; NeuralLP and DRUM learn rule "
 "confidences by composing relation matrices [10,33]; MINERVA learns a walking policy [11], extended with metapath rules "
 "for repurposing on Hetionet [12], and KGML-xDTD combines a drug–disease classifier with reinforcement-learning path "
 "finding [34]. K-Paths retrieves diverse multi-hop paths from biomedical graphs as input to language models and graph "
 "networks [46]. Neural Bellman–Ford networks (NBFNet) learn both the operator that extends a path and the one that "
 "aggregates paths [35], and BioPathNet applies them to biomedical graphs, including drug repurposing [13]. These are the "
 "closest learned counterparts of our question, with neural instead of fuzzy operators, and they learn composition and "
 "aggregation jointly, which is the separation our ablations try to undo. Neural theorem provers compose rule bodies with "
 "the minimum [36]; GNN-QE, FuzzQE and continuous query decomposition use t-norms over learned projections or link "
 "predictors [17,18,37]; uncertain graph embeddings compose graded edges through embedding geometry [38]. In all of these "
 "the composition operator is a product, a t-norm chosen by the authors, or a neural function without a fuzzy reading."),
("h2", "2.3 Learnable fuzzy operators and identifiability"),
("p", "Hamacher, Frank and Schweizer–Sklar families span the classical t-norms with one parameter [16]. Rutkowski and "
 "Cpałka's flexible neuro-fuzzy systems learn such parameters with input and rule weights by gradient descent [22]; since "
 "a weight and a t-norm parameter can both raise or lower the firing strength of a rule, this line of work already "
 "suggests that the two can substitute for each other. Logic tensor networks ground formulas in configurable fuzzy "
 "connectives [39], van Krieken et al. analysed how operator choice affects gradients [23], Giannini et al. related "
 "t-norms to loss functions [24], and logical neural networks learn weights inside Łukasiewicz-type connectives [40]. "
 "Gumbel-softmax makes a discrete choice among operators differentiable [27]. Whether a learned parameter is determined "
 "by the data is the subject of identifiability analysis: Raue et al. distinguish structural non-identifiability, a "
 "redundancy that holds for any data, from practical non-identifiability, where the profile of the objective in one "
 "parameter, with all others re-optimised, is too flat for the data to fix it [41]. Among the methods reviewed here, none "
 "learns a t-norm parameter per metapath of a link predictor trained on a medical knowledge graph, and none reports a "
 "profile of the objective in that parameter. The review is not systematic, and the claim is limited to the work cited."),

("h1", "3. Methods"),
("h2", "3.1 Problem setting"),
("p", "A medical knowledge graph G = (V, E) has typed nodes and typed edges e = (u, r, v). Some edges carry a graded "
 "weight w_{e} ∈ (0, 1]; most carry none. A query asks for the tails of a target relation r* given a head h, for example "
 "which diseases a compound treats. A metapath m is a sequence of relation types, traversed in either direction, from the "
 "head type to the tail type; a path π instantiates m with distinct nodes. We use metapaths of length two and three. When "
 "a pair is held out for validation or testing, every edge between its two nodes is removed, whatever its relation type, "
 "so that, for example, a held-out compound–treats–disease pair cannot be recovered from a compound–palliates–disease edge "
 "between the same nodes. We call this protocol leave-dyad-out."),
("h2", "3.2 Differentiable Neuro-Fuzzy Reasoning"),
("p", "DNFR scores a pair (h, t) in four steps (Figure 1): it groups the paths between h and t by metapath, assigns each "
 "edge a membership, composes memberships along each path with a t-norm, and combines paths by a noisy-OR."),
("fig", 1),
("p", "**Edge membership.** Most edges in Hetionet and PrimeKG are assertions without strengths, and an edge into a hub "
 "says less about any one neighbour than an edge into a specific node. DNFR learns the membership of each traversed edge "
 "e = (u → v) of relation r from the degrees of its endpoints and, where present, its graded weight:"),
("eq", "μ(e) = σ( [a_{r} − b_{r} ℓ_{out}(u) − c_{r} ℓ_{in}(v) + g_{r} logit(w_{e})] / τ ),   ℓ = log(1 + degree in relation r)     (1)"),
("p", "where σ is the logistic function, a_{r}, b_{r}, c_{r} and g_{r} are learned per relation and direction, and τ = 1 "
 "except in refinement R4. Positive b_{r} and c_{r} discount edges at hubs, as the damping exponent of DWPC does [1], but "
 "the strength of the discount is learned. Without graded weights the last term is absent."),
("p", "**Learned t-norm.** The strength of a path π of length L is the Schweizer–Sklar t-norm [16] with a parameter p_{m} "
 "learned for its metapath m:"),
("eq", "T_{p}(μ_{1}, …, μ_{L}) = [ max(0, Σ_{i} μ_{i}^{p} − (L − 1)) ]^{1/p}  (p ≠ 0),     T_{0}(μ_{1}, …, μ_{L}) = Π_{i} μ_{i}     (2)"),
("p", "As p → −∞ the family tends to the minimum, p = −1 gives the Hamacher product, p → 0 the product and p = 1 the "
 "Łukasiewicz t-norm, so the fixed operators of Section 3.4 lie inside the family. For p > 0 a path whose memberships "
 "satisfy Σ_{i} μ_{i}^{p} ≤ L − 1 has strength exactly zero and passes no gradient. By default p_{m} = 6σ(θ_{m}) − 5, in "
 "(−5, 1), initialised at −0.1, close to the product. Equation 2 is evaluated in log space."),
("p", "**Noisy-OR aggregation.** With a reliability ω_{m} ∈ (0, 1) per metapath and a leak β_{0}, the paths Π(h, t) are "
 "combined as"),
("eq", "P(h, t) = 1 − (1 − β_{0}) Π_{π ∈ Π(h,t)} ( 1 − ω_{m(π)} T(π; p_{m(π)}) )^{κ_{π}}     (3)"),
("p", "with κ_{π} = 1 in DNFR. Because −log(1 − P) is then a sum of one term per path, s_{π} = −κ_{π} log(1 − ω_{m} T(π)), "
 "each path's share is an exact decomposition of the model's score. It is not an exact account of the evidence, since "
 "paths that share an edge count it once per path; Section 3.6 removes this double counting at the level of edges."),
("p", "**Training.** For each training query we minimise the cross-entropy of a softmax over the logits of P(h, t) for all "
 "candidate tails, excluding the other known training tails of h (filtered softmax), computed exactly over all pairs with "
 "at least one path. Gradients are accumulated metapath by metapath in two passes, which bounds memory by the largest "
 "metapath. We use Adam (learning rate 0.1) for at most 400 full-batch epochs, evaluate validation MRR every five epochs, "
 "stop after ten evaluations without improvement and keep the best parameters. An L2 penalty λ‖θ − θ_{0}‖² with λ = "
 "10^{−3} acts on the t-norm parameters. Degrees are counted on the training graph of each split. Algorithm S1 in the "
 "Supplementary material gives one training run in full."),
("h2", "3.3 Four refinements"),
("p", "Each refinement replaces one element of Section 3.2 and reduces to it for one parameter value; all four together "
 "form DNFR 2.0 (definitions in Supplementary Section S4). (R1) *Overlap-aware aggregation* discounts the weaker of two "
 "paths through the same interior node by a learned exponent β, which recovers Equation 3 at β = 0; for paths of length "
 "two and three, a shared interior node is equivalent to a node Jaccard index above 0.5, so the rule runs in time linear "
 "in the number of paths. (R2) *Discrete operator choice* lets each metapath choose among the product, the minimum and "
 "the Hamacher product through a Gumbel-softmax relaxation [27], in case a continuous parameter is hard to learn on a "
 "flat loss. (R3) An *information-content discount* replaces the degree feature by log(1 + n(v)), with n(v) the number of "
 "training paths through v, which equals −IC(v) [26] up to a constant per node type. (R4) *Membership annealing* lowers τ "
 "in Equation 1 from 2 to 0.5 over the first 100 epochs, against early saturation of the sigmoid."),
("h2", "3.4 Variants, ablations and baselines"),
("p", "**Fixed operators** (product, minimum, Hamacher, Łukasiewicz) use the membership, aggregation and training of "
 "DNFR. **Fixed p selected on validation**: one Schweizer–Sklar p shared by all metapaths is set to each value of a grid "
 "(−4, −2, −1, −0.5, 0.5 and the product; −2, −1, −0.5 and the product on Hetionet compound–treats–disease, for cost), "
 "and the value with the best validation MRR on each split is kept. **Component ablations** change one component: a "
 "single shared p; no degree terms (b_{r} = c_{r} = 0); degree terms fixed at 0.4 for every relation (with the product, "
 "DWPC-like); the strongest path only instead of noisy-OR (max-path); and, on the graded graph, binary edges (g_{r} = 0). "
 "Each refinement is added to DNFR alone and removed from DNFR 2.0."),
("p", "**Identifiability of the t-norm.** We (i) restarted training from p_{0} = −2, −1, −0.1 and 0.5, with and without "
 "the penalty; (ii) used an unbounded parameter p_{m} = θ_{m}; (iii) trained for 400 epochs without early stopping; (iv) "
 "computed a converged profile on the graded graph and on Hetionet disease–symptom, fixing a shared p on a grid from −5 to "
 "0 (extended into the nilpotent range with warm starts from the product) and fitting all other parameters by L-BFGS "
 "until the loss changed by less than 10^{−9}, with ∂L/∂p and the curvature at the minimum; and (v) froze the memberships "
 "at their initial values, which tests whether memberships and operator substitute for each other, a structural property "
 "of the model. A parameter whose profile has negligible curvature while the others compensate is practically "
 "non-identified in the sense of Raue et al. [41]."),
("p", "**Baselines.** Popularity ranks tails by their training positives; the edge prior of Zietz et al. [4] gives each "
 "pair its share of 100 degree-preserving permutations of the training target relation. DWPC with damping 0.4 on the same "
 "metapaths feeds an L2-regularised logistic regression (DWPC-LR) [1], also with head and tail degrees and the prior "
 "(DWPC-LR + degree + prior). AnyBURL [9] had 100 s of rule learning. A truncated SVD and a soft-thresholded SVD "
 "(Soft-Impute [42]) complete the target relation alone. TransE and RotatE [5,7], trained with PyKEEN [30] without "
 "tuning, are reference points only. NBFNet [35] was implemented as a compact CPU version (three generalised "
 "Bellman–Ford layers, DistMult messages, dimension 32) and, on the graded graph, also tuned over a 2 × 2 grid; it was run "
 "on the two disease–symptom graphs, while BioPathNet [13], TxGNN [8] and NBFNet on the drug-repurposing graphs need GPU "
 "training. DNFR has 354, 80, 169 and 11 trainable parameters on the four tasks and DWPC-LR 131, 18, 52 and 2 (Table 1), "
 "so the comparison is between ways of weighting the same path evidence. Supplementary Section S5 gives the settings of "
 "every baseline."),
("h2", "3.5 Data and tasks"),
("table", 2),
("p", "**Hetionet** (v1.0) [1] has 47,031 nodes of 11 types and 2,250,197 edges of 24 types; we predict "
 "compound–treats–disease and disease–presents–symptom. **PrimeKG** [2] has 129,375 nodes; after removing the 1,538,090 "
 "anatomy–protein expression edges, which lie on no drug–disease path of length three or less, and duplicates, 2,511,974 "
 "edges of 28 relation types remain, and we rank all 17,080 diseases for each drug. **Health knowledge graph (HKG).** "
 "Rotmensch et al. learned this graph from 273,174 emergency-department visits recorded between 2008 and 2013 and "
 "published it in full [28]. We parsed its 3,704 disease–symptom edges between 156 diseases and 330 symptoms; each carries "
 "the noisy-OR importance w_{e} = 1 − f, where f is the estimated probability that the disease fails to produce the "
 "symptom (median 0.021). With a single relation, its only metapath is disease–symptom–disease–symptom, so this task tests "
 "the composition of graded evidence from records in a collaborative-filtering setting, where low-rank completion is a "
 "natural competitor. To keep training feasible on a two-core CPU, metapaths with more than five million paths were "
 "excluded for all path methods: 4 of 134 on compound–treats–disease and 31 of 82 on PrimeKG, the most hub-dominated "
 "ones, so PrimeKG accuracies of all path methods are lower bounds."),
("h2", "3.6 Evaluation"),
("p", "Positive pairs were split at random into 80% training, 10% validation and 10% test, with 10 splits on each "
 "Hetionet task and on the graded graph and 2 on PrimeKG. Each test tail is ranked among all candidates of its type after "
 "removing the other known tails of the head (filtered ranking, ties at the mean rank); we report mean reciprocal rank "
 "(MRR) and Hits@k. On compound–treats–disease we also used a **disease-centred split** (4 splits) that holds out every "
 "indication of a test disease, as in zero-shot repurposing [8]. Of the 130 metapaths there, 33 contain the treats "
 "relation itself, for example compound–resembles–compound–treats–disease; to measure how much of the accuracy rests on "
 "known indications, we retrained DNFR, the fixed product and DWPC-LR without them, and then also without the metapaths "
 "through compound–palliates–disease."),
("p", "**Statistics and decision rule.** For each comparison we report the mean difference in MRR over splits, a paired "
 "bootstrap interval over test queries stratified by split, the number of splits with the same sign, and the corrected "
 "resampled t-test [44], which accounts for overlapping training sets. P-values are Holm-adjusted [45] within families "
 "defined by research question. Since most claims about t-norms are claims of equivalence or of no gain, one rule serves "
 "every comparison. With a margin of ±0.01 MRR, a difference is *equivalent* (≈) when both 90% intervals, corrected-t and "
 "bootstrap, lie inside the margin; *different* (†) when the adjusted corrected-t p-value is below 0.05 and the 95% "
 "bootstrap interval excludes zero on the same side; and *inconclusive* otherwise. A change of 0.01 corresponds to moving "
 "one correct answer in fifty from rank 1 to rank 2. With three or fewer splits no equivalence verdict is attainable, "
 "which affects PrimeKG and the disease-centred split. Supplementary Table S12 shows how the main verdicts depend on the "
 "rule."),
("p", "**Explanations.** On the seed-0 split we measure concision (paths, facts and metapaths covering 50% and 80% of "
 "the evidence −log(1 − P) above the leak) and faithfulness by deleting the leading paths, random paths, or random paths "
 "with the same total share. Because the score is additive in path terms, path shares double-count facts shared by "
 "several paths. With memberships fixed, the score is a sum of unanimity games, one per path, so the Shapley value of edge "
 "e is exactly φ_{e} = Σ_{π ∋ e} s_{π} / L_{π}, and edge shares sum to one without double counting. The fact-level "
 "counterfactual removes the fact with the largest Shapley share, or a random fact on the pair's paths, from the graph, "
 "re-scores every candidate of the head and re-ranks without retraining. For drug repurposing we classify the leading "
 "metapath of each correct top-10 prediction as mechanistic (through a gene that the compound binds or regulates and that "
 "is associated with the disease) or associative (through pharmacologic classes, similar compounds or similar diseases). "
 "DWPC-LR is attributed in the same way, exactly at the level of metapaths and proportionally within them."),
("p", "**Clinical assessment.** On each disease–symptom task we drew 25 held-out pairs of the seed-0 split that DNFR "
 "ranked in its top 10 and listed their three leading paths. The second author, a doctoral candidate in clinical "
 "psychology, rated them without seeing ranks, scores, memberships or shares: whether every relation on a path agrees "
 "with biomedical knowledge, whether the path offers a mechanistic, associative or no rationale, whether the pair is "
 "clinically plausible, and how useful the paths would be to a clinician (1 to 5). The sheet, key and ratings are "
 "released. We also thinned every relation to 50% and 25% of its edges at random over three splits (Supplementary Table "
 "S5)."),
("p", "**Software and use of AI assistance.** All experiments ran on a shared two-core CPU with 8 GB of memory. The "
 "defaults of DNFR and the path budget were set during development to values that made training stable and feasible, "
 "not by a search; the settings of the refinements were fixed before any of them was run on more than one split. The "
 "analysis code was written with the help of an AI system (Claude, Anthropic), which also ran the experiments under the "
 "authors' direction; the authors specified the models, protocols and analyses, and Section 7 describes independent "
 "verification. Released tests cover the leave-dyad-out removal, the t-norm limits, the overlap groups of R1 and the "
 "filtered ranking."),

("h1", "4. Results"),
("h2", "4.1 Accuracy against path, rule, completion and degree-only baselines (RQ1)"),
("p", "Table 2 gives test MRR; methods are compared within a task, since the number of candidates ranges from 137 to "
 "17,080. Degree alone explained little: popularity and the edge prior were far behind every path method on every task. "
 "The prior fell below popularity on compound–treats–disease because, under leave-dyad-out splits, many test compounds "
 "have no training indication left, so every candidate has a prior of zero (46% of test queries; Supplementary Table "
 "S9)."),
("table", 3),
("p", "Among path methods, the fixed weighting of DWPC-LR was at least as accurate as DNFR on three of the four tasks. On "
 "Hetionet drug repurposing DNFR minus DWPC-LR was +0.005 [−0.016, +0.025] (5/10 splits, inconclusive); on the two "
 "disease–symptom tasks DWPC-LR was ahead by 0.016 and 0.021, a difference under our rule on both. Only on PrimeKG was "
 "DNFR ahead (+0.070 [+0.056, +0.084], two splits, too few for a verdict). Degree and prior features did not help "
 "DWPC-LR on any task. Completion of the target relation alone was far behind on Hetionet drug repurposing (Soft-Impute "
 "0.262), came close on PrimeKG (0.627) and led on the dense graded graph (0.352 against 0.311 for DNFR), where a disease "
 "is linked to a symptom through diseases with similar profiles and a low-rank description should work well. AnyBURL "
 "(0.478, 0.153, 0.266) was below DNFR and DWPC-LR wherever it ran. The compact NBFNet, which learns both path extension "
 "and aggregation with about ten thousand parameters, did not differ from DNFR under our rule on either graph (graded "
 "graph −0.001 [−0.011, +0.009]; Hetionet disease–symptom +0.021 [+0.014, +0.029], ahead on 9 of 10 splits); tuned, it "
 "reached 0.315 on the graded graph, and it was behind DWPC-LR there (−0.022, different)."),
("h2", "4.2 What each learned component adds (RQ2)"),
("table", 4),
("p", "**Aggregating paths.** Keeping only the strongest path lowered MRR by 0.083 on compound–treats–disease, 0.096 on "
 "Hetionet disease–symptom and 0.083 on the graded graph, on 10/10 splits each and different under our rule (Table 3). "
 "The number of routes between two entities carries information that the best single route does not."),
("p", "**Discounting hubs.** Removing the degree terms lowered MRR by 0.162, 0.020, 0.064 and 0.072 on the four tasks, "
 "with the same sign on every split; the difference was supported on all but PrimeKG, where two splits are too few. "
 "Whether the discount must be learned per relation depended on the graph: learned coefficients beat a fixed 0.4 on the "
 "graded graph (+0.057 [+0.048, +0.066]), while elsewhere the fixed DWPC-like discount was at least as good (−0.004, "
 "−0.012 and −0.004). Reading graded edge weights added +0.011 [+0.004, +0.017]."),
("p", "**Learning the t-norm.** A Schweizer–Sklar parameter learned per metapath was not consistently better than fixed "
 "members of its own family. Against the fixed product the differences were −0.003, +0.003, −0.009 and +0.006; against "
 "the Hamacher product +0.024, −0.002, −0.005 and +0.010. All eight comparisons were inconclusive, so we claim no "
 "consistent gain from learning the t-norm, not equivalence. The +0.024 rests largely on one split of "
 "compound–treats–disease in which the Hamacher run stopped in a poorer optimum (MRR 0.488 against 0.658 for the product, "
 "training loss 3.16 against 2.38); without it the difference shrank to +0.007 (Supplementary Table S13). The t-norm "
 "mattered most when memberships could not move. With memberships frozen, learning p improved on the product by +0.056 "
 "[+0.049, +0.063] on the graded graph, and learning the memberships instead improved the product by +0.121 [+0.111, "
 "+0.131]. Memberships and operator are, to a large extent, substitutes: both rescale the strength of a path, as weights "
 "and t-norm parameters both rescale rule strength in flexible neuro-fuzzy systems [22]."),
("h2", "4.3 Four refinements and a validation-selected t-norm (RQ2)"),
("table", 5),
("p", "Of the 14 add-one comparisons in Table 4, none showed a gain under our rule, one was equivalent and 13 were "
 "inconclusive; point estimates ranged from −0.046 to +0.019. Membership annealing (R4) had the most consistent positive "
 "estimates (+0.001, +0.013, +0.007), although saturation, the problem it targets, was rare in trained models (0.9% and "
 "0.3% of memberships above 0.99), so we neither recommend nor rule it out. The learned exponent of overlap-aware "
 "aggregation (R1) fell from 1 to a median of 0.014 on the graded graph, but the loss hardly constrains parameters that "
 "rescale paths (Section 4.4), so the small value does not show whether such paths add or repeat evidence. The "
 "information-content discount (R3) lost 0.046 on the graded graph because its feature had about half the spread of the "
 "degree feature; matched in standard deviation as well as mean, it changed MRR by −0.003 [−0.007, +0.001]. Combined, "
 "DNFR 2.0 differed from DNFR by −0.023, −0.020 and +0.005, and was less accurate under our rule on Hetionet "
 "disease–symptom (0/10 splits in its favour)."),
("p", "Selecting a fixed shared p on validation data, the procedure that Section 4.4 recommends, differed from the fixed "
 "product by +0.001 [−0.007, +0.008] on compound–treats–disease (equivalent), +0.016 [+0.011, +0.020] on Hetionet "
 "disease–symptom and +0.019 [+0.014, +0.025] on the graded graph, the last supported under our rule; there the grid "
 "value chosen was −0.5 in 6 splits, −2 in 3 and −4 in 1. On Hetionet disease–symptom validation chose the product itself "
 "on four splits, and the gain came from splits on which the early-stopped product run ended low: fitted to convergence, "
 "the product reached 0.245, 0.269 and 0.210 on splits 0–2 against 0.208, 0.268 and 0.180 early-stopped (Supplementary "
 "Table S17). We therefore claim a benefit of validation selection for the graded graph only."),
("h2", "4.4 Is the t-norm parameter identified by the training objective? (RQ3)"),
("p", "Section 4.2 answered the structural question: the memberships can partly compensate for p. The practical question "
 "is whether the training data, through the loss, determine p within a useful range [41]. With the default early "
 "stopping, learned values stayed near their start and restarts from −2, −1 and 0.5 ended in different places at similar "
 "accuracy (Supplementary Table S11). On compound–treats–disease the best epoch had a median of 43, enough Adam steps to "
 "move p by close to one unit, yet the learned values had an interquartile range of only [−0.06, 0.16]: the gradient was "
 "small and inconsistent. An unbounded parameterisation did not converge either; on the graded graph the median learned "
 "value was −3.21 from −2, −0.46 from −0.1 and 0.95 from 0.5, and neither did 400 epochs without early stopping "
 "(Supplementary Table S6)."),
("fig", 2),
("p", "The converged profile settles the question for two graphs (Figure 2). On the graded graph, over p from −5 to 0 the "
 "converged training loss varied by 0.0047, 0.0020 and 0.0028 nats in three splits, against a loss of about 4.3, and "
 "|∂L/∂p| at the fits was at most 0.0170. The minimisers were p = 0.1, −0.25 and 0.5, and the sets of p within "
 "χ²(0.95, 1)/(2N) of each minimum, a profile-likelihood style interval used descriptively, were [−5, 0.25], [−0.5, "
 "−0.25] and {0.5}, which do not all overlap (Supplementary Tables S7 and S8). On Hetionet disease–symptom, with 17 "
 "metapaths, the loss varied by 0.0037, 0.0036 and 0.0010 nats over the same range, and test MRR by at most 0.016 "
 "(Supplementary Tables S15 and S16). These variations are of the same order as the optimisation error: L-BFGS fits at the "
 "same p from different starts differed by up to 0.0067 nats on the graded graph and 0.0321 nats on Hetionet "
 "disease–symptom. The training loss therefore does not locate p, either because the profile is flat within a split or "
 "because the minimiser moves between random splits of the same data."),
("p", "Held-out accuracy was more informative than the loss. On the graded graph, validation MRR at the converged fits "
 "peaked at p = −1.5, −2 and −1.5, and test MRR at the value chosen on validation exceeded that of the product by +0.014, "
 "+0.008 and +0.016. The data carry information about p, but the softmax loss, averaged over all candidates, hardly "
 "registers the reordering of the few top-ranked ones that decides MRR. The Łukasiewicz t-norm, which lost 0.080 when "
 "trained from the default initialisation because every path there starts at strength zero, reached 0.312, 0.333 and "
 "0.290 when started from the converged product (Supplementary Table S18): its failure is one of initialisation. "
 "Per-metapath values learned from different starts were at best weakly correlated (median Spearman correlation 0.44 on "
 "Hetionet disease–symptom), so no single learned value should be interpreted."),
("h2", "4.5 Held-out diseases and evidence through known indications (RQ1)"),
("table", 6),
("p", "When every indication of a test disease was held out (Table 5A), all methods fell sharply: DNFR reached 0.079 "
 "against 0.599 on random splits. Path methods stayed ahead of completion, popularity and the edge prior (0.015, 0.010 and "
 "0.012), which carry no information about a disease without training indications, and degree and prior features hurt "
 "DWPC-LR (0.028 against 0.074) because a held-out disease has degree zero. The variant without degree terms reached "
 "0.103 (+0.025 [+0.009, +0.041], 4/4 splits), a plausible reversal of the random-split result, since the routes left to "
 "an unseen disease often pass through hubs, but one the corrected interval [−0.003, +0.052] does not establish."),
("p", "On random splits, the 33 metapaths that contain the treats relation carried 99% of DNFR's evidence for its correct "
 "top-10 predictions on the seed-0 split, against 0.5% for metapaths through a gene that the compound targets. Without "
 "them (Table 5B), DNFR fell from 0.599 to 0.289 (−0.310 [−0.343, −0.277], different), DWPC-LR from 0.594 to 0.235 and "
 "the fixed product from 0.602 to 0.281, 60% of the way to the disease-centred accuracy. Removing the palliates metapaths "
 "as well (Table 5C; 60 of 130 removed) changed DNFR by a further −0.012 (inconclusive). With both relations removed, "
 "gene-only metapaths carried 46.5% of DNFR's evidence and associative ones 53.5%. On a curated graph, then, the "
 "random-split accuracy of path methods on drug repurposing comes largely from guilt by association: a compound ranks "
 "high for a disease because similar compounds, or compounds of the same pharmacologic class, are known to treat it. "
 "Mechanistic evidence is what the model falls back on when that association is unavailable, and it supports a much "
 "lower accuracy. This holds for DWPC-LR as much as for DNFR."),
("p", "Thinning every relation, including the training edges of the target relation, favoured path methods over "
 "completion (with 25% of edges, DNFR 0.342 against 0.082 for Soft-Impute on compound–treats–disease), but when the "
 "target relation was kept intact, completion was level with the best path method on Hetionet disease–symptom (0.228 "
 "against 0.226 for DWPC-LR; Supplementary Table S5 and Figure S4). Much of the robustness of path methods to missing "
 "edges thus comes from the loss of training edges, not from the robustness of path evidence."),
("h2", "4.6 Explanations (RQ4)"),
("table", 7),
("p", "At the level of paths the explanations were less concise than a single chain (Table 6): on Hetionet "
 "disease–symptom a test pair was supported by a median of 222 paths, 4 of which carried half of the evidence. Deleting "
 "the three leading paths left 40% of correct top-10 predictions in the top 10, but deleting random paths with the same "
 "total share left 34%, so within a pair an additive score depends on how much evidence is removed rather than on which "
 "paths carry it."),
("p", "At the level of facts the picture is clearer. Summed path shares credited the leading fact with 45%, 25% and 28% "
 "of the evidence on the three tasks; its Shapley value, which removes the double counting, was 16%, 12% and 9%, and half "
 "of the attribution lay on 7, 6 and 12 facts. The two criteria picked the same leading fact for 95%, 91% and 100% of "
 "pairs, so double counting changes how much a fact appears to carry far more than which fact leads. Removing the leading "
 "fact from the graph left 85%, 60% and 76% of correct top-10 predictions in the top 10, against 100%, 99% and 99% after "
 "removing a random fact on the same paths: a correct ranking depends more on the facts the explanation names than on "
 "other facts on its paths. DWPC-LR's fact-level counterfactuals were similarly faithful (87%, 55% and 82% against 100%, "
 "99% and 96%). The difference between the two explanations is one of kind: DNFR's attribution is exact and non-negative "
 "at the level of paths and facts, whereas on compound–treats–disease 31% of DWPC-LR's positive attribution was offset by "
 "metapaths with negative coefficients (median). On drug repurposing the leading metapath was associative for 100% of "
 "correct top-10 predictions, and mechanistic metapaths carried 0.5% of the evidence on average. Supplementary Table S19 "
 "shows three explanations in full, for example doxorubicin and breast cancer, supported by 22,225 paths that ran mainly "
 "through topoisomerase inhibitors and anthracyclines known to treat breast cancer."),
("p", "In the clinical assessment, every relation on a path agreed with biomedical knowledge for 53% of the 75 paths on "
 "Hetionet and 60% on the graded graph, at least one relation was judged wrong for 12% and 13%, and the rest (35% and "
 "27%) could not be judged with confidence. On Hetionet, 7 of the 9 paths judged wrong ran from the disease to an "
 "anatomical structure and back to another disease, and the model had already given that second edge a low membership "
 "(median 0.14, against 0.46 for the weakest edge of paths judged correct); with one rater this is an observation only. "
 "On Hetionet 43% of paths offered a mechanistic rationale, 44% an associative one and 13% none; on the graded graph 28%, "
 "47% and 25%. The rater judged 88% and 80% of the predicted pairs clinically plausible, and the median usefulness of the "
 "three leading paths was 3 of 5 on both tasks. The predictions were mostly plausible; the explanations were often a "
 "reason to look further rather than a reason to believe."),
("p", "One DNFR run took about 267 s (compound–treats–disease, splits 0–2), 190 s, 727 s and 90 s on the four tasks after "
 "a one-off path enumeration, against 8.6, 1.0, 15.5 and 0.4 s for DWPC-LR (Supplementary Table S14)."),

("h1", "5. Discussion"),
("h2", "5.1 Principal findings"),
("p", "Combining all paths by noisy-OR and discounting edges at hub nodes carried the gains that DNFR had over its "
 "ablations; both had the same sign on every split of every task on which they were tested. Learning the discount per "
 "relation helped only on the graph with graded clinical evidence. The t-norm, the most visibly neuro-fuzzy part of the "
 "model, gave no consistent gain when learned, and the training loss barely depends on it once memberships are learned. "
 "On the graded graph held-out accuracy did depend on it, and choosing it on validation data helped where learning it "
 "did not. As a predictor, DNFR was not more accurate than DWPC-LR on three of four tasks, trailed low-rank completion on "
 "the graded graph and did not differ from a compact NBFNet under our rule. For a developer who needs only accuracy on "
 "graphs like these, DWPC-LR is the better choice: simpler, faster and at least as accurate. What DNFR offers instead is an "
 "attribution that is exact and non-negative at the level of paths and facts. Our counterfactuals did not show its "
 "explanations to be more faithful, and whether their form helps a clinician more is a question for a user study."),
("h2", "5.2 Lessons for developers of biomedical informatics methods"),
("p", "Two mechanisms make the loss insensitive to the operator, and neither is specific to DNFR. First, memberships and "
 "metapath reliabilities already rescale every path: a t-norm with p < 0 lifts a path with one weak edge relative to the "
 "product, and raising the membership of that edge has much the same effect on the ranking. Second, the softmax loss "
 "averages over all candidates, so a change in p that reorders a few top-ranked candidates barely moves it while it can "
 "move MRR. The first was anticipated for flexible neuro-fuzzy systems [22]; the second applies to any operator learned "
 "through a ranking surrogate, including the learned aggregators of NBFNet-style models [13,35]. Three practices "
 "follow, and each is cheap compared with the experiments it qualifies."),
("li", "**Check identifiability before interpreting a learned operator.** Restart from several values, profile over "
 "fixed values with the other parameters converged, and ask whether held-out accuracy and training loss agree. Where they "
 "disagree, as on our graded graph, select the operator on validation data as a hyperparameter, preferably among a few "
 "non-nilpotent t-norms, and do not report the learned value as a property of the medical domain."),
("li", "**Test what the evidence rests on.** On curated drug-repurposing graphs, metapaths through the target relation "
 "carried almost all of the evidence, for every path method. Random splits reward this guilt by association, useful for "
 "repositioning within a drug class but uninformative about mechanisms; claims of multi-relational or mechanistic "
 "reasoning should be backed by an ablation of those metapaths or a disease-centred split."),
("li", "**Attribute to facts and test at the level of facts.** Path shares double-count facts on shared paths; the "
 "edge-level Shapley value of a noisy-OR path model is exact and cheap, and removing a fact from the graph and re-ranking "
 "tests what a clinician would question. Path-level deletion within a pair says little, because an additive score depends "
 "on how much evidence is removed."),
("p", "Three features of medical graphs shaped these results. Curated graphs mix evidence of very different strength "
 "under one relation type: Hetionet's disease–disease \"resembles\" edges come from text co-occurrence, while its "
 "compound–disease edges are curated indications [1], and a single membership per relation cannot tell them apart. Hubs "
 "are biologically meaningful but also reflect uneven research attention [3], so discounting them helped on random "
 "splits and possibly not for unseen diseases. And the graph learned from records is a single bipartite relation, on "
 "which composition of graded evidence competes with low-rank completion."),
("h2", "5.3 Limitations"),
("li", "**Splits and power.** With ten splits, corrected 90% intervals still extend to about ±0.015, so equivalence within "
 "±0.01 was rarely attainable and differences of 0.02 to 0.04 were often inconclusive despite a constant sign. PrimeKG "
 "results, with two splits, are descriptive."),
("li", "**Partial coverage.** The refinements, validation selection, NBFNet, the disease-centred split and the treats "
 "ablation were not run on PrimeKG, so the zero-shot result rests on Hetionet compound–treats–disease alone, with 4 "
 "splits and 137 candidate diseases. The converged profile, with one shared p, covers the two disease–symptom tasks; on "
 "drug repurposing we rely on restarts, the unbounded parameterisation and frozen memberships."),
("li", "**Learned-operator baselines.** NBFNet was a compact CPU implementation tuned on one graph; BioPathNet and TxGNN "
 "were not run, and the untuned embeddings are not evidence about embeddings. Metapaths with more than five million paths, "
 "the most hub-dominated, were excluded, so the effect of hub discounting may be underestimated."),
("li", "**Clinical evaluation.** One rater, a co-author trained in clinical psychology rather than medicine who knew the "
 "aims of the study, assessed 150 paths of 50 correct disease–symptom predictions; no agreement or discrimination between "
 "correct and incorrect predictions can be estimated, and drug repurposing was not rated. A blinded sheet that mixes "
 "correct and incorrect predictions is released for a rating study with several physicians."),
("li", "**Scope of the evidence.** Scores are rankings, not calibrated probabilities; edges were thinned uniformly, "
 "whereas less-studied entities lose more; and paths combine curated, text-mined and co-occurrence evidence, so a "
 "high-scoring path is a reason to look, not evidence of a causal mechanism."),

("h1", "6. Conclusion"),
("p", "In a controlled component analysis of differentiable neuro-fuzzy path reasoning on four tasks from three medical "
 "knowledge graphs, the accuracy came from aggregating all paths by noisy-OR and from discounting edges at hub nodes; "
 "learning the discount helped only where edges carried graded clinical evidence, and four plausible refinements did not "
 "improve the model. The training loss hardly constrained the t-norm parameter once memberships were learned, so the "
 "parameter should be selected on validation data and not interpreted when learned. Degree-weighted path counts were at "
 "least as accurate on three of four tasks, and on drug repurposing most of the evidence, for every path method, ran "
 "through known indications of similar compounds or related diseases. For explainable clinical decision support on "
 "knowledge graphs, effort is better spent on how much each fact counts, how paths are combined, attribution to facts and "
 "tests of what the evidence rests on than on learning the operator that composes it."),
]

DECLARATIONS = [
("Ethics", "The study uses only public, de-identified or aggregate data: Hetionet, PrimeKG and the disease–symptom graph "
 "published by Rotmensch et al. [28], whose underlying patient records, from the hospital behind MIMIC-IV [43], were not "
 "accessed. Saigon International University does not require ethics approval for secondary analysis of public "
 "anonymised data."),
("Data availability", "Hetionet v1.0 is available from its GitHub repository and PrimeKG from Harvard Dataverse "
 "(doi:10.7910/DVN/IXA7BM). The parsed health knowledge graph, the per-query ranks of every method on every split, the "
 "trained models, the path explanations, the clinical rating sheets and the statistics files from which every number in "
 "this paper is taken are deposited at Zenodo (version v6.0-jbi: https://doi.org/10.5281/zenodo.24414386; all versions: "
 "https://doi.org/10.5281/zenodo.23879352), with a script that downloads Hetionet and PrimeKG and checks them against the checksums of "
 "the files we used."),
("Code availability", "The implementation of DNFR and its refinements, the compact NBFNet and all baselines, the tests, "
 "and the scripts that compute the statistics and figures are available under the MIT licence at "
 "https://github.com/Bangth2026/dnfr-medical-kg (release v6.0-jbi). The release includes check_numbers.py, which verifies "
 "that every number in this article and its supplementary material is one generated from the result files."),
("CRediT authorship contribution statement", "Hai Bang Truong: Conceptualization; Methodology; Software; Formal analysis; "
 "Data curation; Visualization; Writing – original draft; Writing – review and editing. Van An Phan Thi: Conceptualization "
 "(clinical framing of the prediction tasks, of the disease-centred split and of what an explanation should let a "
 "clinician check); Validation (agreement of the relations on explanation paths with biomedical knowledge); Investigation "
 "(the clinical assessment of explanations); Writing – original draft (clinical interpretation in Section 5); Writing – "
 "review and editing. All authors approved the final manuscript."),
("Acknowledgements", "We thank Dr Do Kim Phuc, MSc, MD (Pham Ngoc Thach University of Medicine, Ho Chi Minh City), "
 "for rating the explanation paths as the second, independent clinical rater."),
("Independent verification", "Without the AI system, the corresponding author checked every result reported in this "
 "article and its supplementary material against the released result files. A re-run of DNFR, DNFR with the strongest "
 "path only and DNFR without degree terms on the first split of the graded graph from a clean clone of the repository, in a "
 "fresh environment (scripts/verify_clean.sh), reproduced the released test MRR to four decimals (0.3204, 0.2307, "
 "0.2739)."),
("Declaration of generative AI and AI-assisted technologies in the manuscript preparation process", "During the "
 "preparation of this work the authors used Claude (Anthropic) to draft, shorten and edit the manuscript text and, as "
 "described in Section 3.6, to write and run the analysis code under the authors' direction. After using this tool, the "
 "authors reviewed and edited the content as needed and take full responsibility for the content of the publication."),
("Funding", "This research did not receive any specific grant from funding agencies in the public, commercial, or "
 "not-for-profit sectors."),
("Declaration of competing interest", "The authors declare that they have no known competing financial interests or "
 "personal relationships that could have appeared to influence the work reported in this paper."),
]

NEW_REFS = {46: "T. Abdullahi, I. Gemou, N.V. Nayak, G. Murtaza, S.H. Bach, C. Eickhoff, R. Singh, K-Paths: reasoning "
                "over graph paths for drug repurposing and drug interaction prediction, arXiv:2502.13344 (2025)."}

CAPTIONS = {
 "fig1": "**Figure 1.** Differentiable Neuro-Fuzzy Reasoning in four steps. (1) Paths of length two and three between a "
         "query pair are grouped by metapath, after every edge between the two nodes has been removed. (2) Each edge "
         "receives a learned membership that discounts high-degree endpoints and reads graded evidence when the graph has "
         "it. (3) A Schweizer–Sklar t-norm with a parameter learned per metapath composes memberships along each path; the "
         "maximum with 0 is active only for p > 0. (4) A noisy-OR weighted by metapath reliability combines paths, so that "
         "the score is a sum of path terms. The refinements of Section 3.3 act on steps 2–4.",
 "fig2": "**Figure 2.** Converged profile over a fixed shared t-norm parameter p on the graded graph, three splits; p > 0 "
         "fitted with warm starts from the product. Left: training loss minus its minimum over the grid (×10⁻³ nat). "
         "Centre: ∂L/∂p at the converged fit. Right: test MRR of the converged fit. All other parameters were fitted by "
         "L-BFGS to convergence. Dashed lines mark the Hamacher product (p = −1), the product (p = 0) and the Łukasiewicz "
         "t-norm (p = 1).",
}
