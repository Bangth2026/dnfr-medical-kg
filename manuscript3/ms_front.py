"""Introduction, related work and methods (third revision: four refinements of DNFR, reframed identifiability)."""
from ms_common import *
from docx_helpers import para, heading, bullets, table, boxed, figure


def P(d, t, **k):
    return para(d, C(t), **k)


def introduction(d, contributions):
    heading(d, "1. Introduction", 1)
    P(d, "A clinician asked to trust a predicted drug–disease link wants to know two things: how strongly the evidence "
         "supports it, and what that evidence is. Medical knowledge graphs can in principle answer both. Hetionet integrates "
         f"29 public resources into a graph of {DS['het']['nodes']:,} nodes and {DS['het']['edges']:,} edges [@himmelstein2017], and "
         "PrimeKG links 17,080 diseases to drugs, proteins, phenotypes and exposures [@chandak2023]. In practice these graphs are "
         "sparse where it matters: most drug–disease and disease–phenotype pairs are simply absent, so a new relation can only "
         "be inferred by chaining several intermediate relations, for example from a compound to the gene it binds and on to a "
         "disease associated with that gene. They are also dominated by hubs, and node degree alone predicts much of what "
         "link predictors recover from them [@bonner2022;zietz2024].")
    P(d, "Two families of methods address such multi-hop inference, and they trade accuracy against explanation. Knowledge graph "
         "embeddings such as TransE, ComplEx and RotatE [@bordes2013;trouillon2016;sun2019], and graph neural networks such as TxGNN "
         "[@huang2024], score a candidate relation from learned vectors; they are often accurate but give no account of why a pair "
         "scores high. Path-based methods instead score a pair from the paths that connect it. The degree-weighted path counts of "
         "Rephetio [@himmelstein2017], rule learners such as AnyBURL and NeuralLP [@meilicke2019;yang2017], reinforcement-learning "
         "walkers such as MINERVA and its rule-guided extension [@das2018;liu2020], and path-representation networks such as BioPathNet "
         "[@hu2026] all relate a prediction to the paths behind it. Such traceable evidence is what multidisciplinary reviews of "
         "explainable clinical AI ask decision-support systems to provide [@amann2020].")
    P(d, "Every path-based method also has to decide how strengths combine along a path and across paths. Fuzzy logic makes the first "
         "decision explicit: the strength of a path is a triangular norm (t-norm) of the strengths of its edges [@zadeh1965;klement2000]. "
         "The choice is consequential. On a path of three edges of strength 0.3, the minimum returns 0.3, the product 0.027 and the "
         "Łukasiewicz t-norm 0. Yet in most methods the operator is fixed in advance: degree-weighted path counts multiply damped edge "
         "weights, NeuralLP composes relations by matrix products, GNN-QE and FuzzQE answer queries with fixed fuzzy connectives "
         "[@zhu2022;chen2022fuzzqe], and clinical fuzzy systems have used max–min composition since Sanchez's work on medical diagnosis "
         "[@sanchez1979;vanpham2023;apostolopoulos2024]. Flexible neuro-fuzzy systems learn the parameters of parametric t-norms jointly "
         "with the other weights [@rutkowski2003], and differentiable fuzzy logic has studied how the operator affects learning "
         "[@vankrieken2022;giannini2023]. Whether such a parameter can be learned from medical graph data, rather than merely fitted, "
         "has not been asked. Our earlier, not yet peer-reviewed analysis of a population fuzzy graph from the US National Health and "
         "Nutrition Examination Survey found no gain from tuning a parametric t-norm over the product [@truong2026]; it motivated the "
         "question but is not part of the evidence here.")
    P(d, "We study the question with Differentiable Neuro-Fuzzy Reasoning (DNFR), a path-based link predictor built so that each of its "
         "components can be switched off or fixed: an edge membership that discounts high-degree nodes and, when available, reads graded "
         "evidence; a Schweizer–Sklar t-norm whose parameter is learned for each metapath, so that the operator can range from the "
         "minimum through the Hamacher product and the product to the Łukasiewicz t-norm; and a noisy-OR combination of paths weighted "
         "by metapath reliability. The score of every prediction is a sum of path terms, so it decomposes into path contributions. "
         "Four parts of this design rest on simplifying assumptions that have standard alternatives. The noisy-OR treats "
         "paths as independent and counts a node shared by several paths once per path. A continuous t-norm parameter may sit on a "
         "flat stretch of the loss, so a discrete choice among named operators might be easier to learn. Raw degree is a crude measure "
         "of how specific a node is; information content [@resnik1995] is the classical alternative. And sigmoid memberships may "
         "saturate early in training. We implemented one refinement for each (overlap-aware aggregation, Gumbel-softmax operator "
         "selection [@jang2017], an information-content discount and membership-temperature annealing), tested each under the "
         "protocol used for every other comparison, and call their combination DNFR 2.0.")
    P(d, "We evaluate on four medical link-prediction tasks: drug repurposing in Hetionet and PrimeKG, disease–symptom prediction in "
         "Hetionet, and disease–symptom prediction on a graph that Rotmensch et al. learned from 273,174 emergency-department records "
         "[@rotmensch2017], whose edges carry graded noisy-OR weights. The study asks four questions. (RQ1) How accurate is a learned "
         "neuro-fuzzy path reasoner against degree-weighted path counts, rule learning, low-rank completion and degree-only "
         "baselines, including when test diseases have no known indications, and how much of that accuracy rests on paths through "
         "known instances of the target relation? (RQ2) Which of its components carry the gain, and do "
         "the four refinements add to them? (RQ3) Is the t-norm parameter identified by the training objective, and does selecting "
         "it on validation data recover what end-to-end learning misses? (RQ4) How concise and how faithful are the path "
         "explanations, at the level of paths and at the level of individual facts? The contributions are:")
    bullets(d, [C(c) for c in contributions], numbered=True)


def related(d):
    heading(d, "2. Related work", 1)
    heading(d, "2.1 Medical knowledge graphs, link prediction and degree bias", 2)
    P(d, "Hetionet [@himmelstein2017] and PrimeKG [@chandak2023] are the two most widely used open medical knowledge graphs for drug "
         "repurposing. Himmelstein et al. predicted compound–disease treatment from degree-weighted path counts (DWPC) over 123 "
         "metapaths with a regularised logistic regression that also used degree features and a permutation-derived prior, and later "
         "made metapath-based explanations searchable [@himmelstein2017;himmelstein2023]. Embedding methods treat link prediction as "
         "scoring triples with learned vectors [@bordes2013;trouillon2016;sun2019]; PyKEEN provides reference implementations "
         "[@ali2021], and their accuracy depends heavily on training configuration [@ruffinelli2020]. TxGNN trains a graph neural "
         "network on PrimeKG for zero-shot drug repurposing and adds a path-based explainer for clinicians [@huang2024]. Graphs learned "
         "from clinical records add a different kind of edge: Rotmensch et al. learned a disease–symptom graph from emergency-department "
         "notes and diagnosis codes with a noisy-OR model, so that each edge carries an estimated probability that the disease produces "
         "the symptom [@rotmensch2017].")
    P(d, "Degree is a strong signal in these graphs. Bonner et al. showed that embedding models trained on biomedical knowledge graphs "
         "favour high-degree entities [@bonner2022]. Zietz et al. computed the probability of an edge from node degrees alone, by "
         "degree-preserving permutation of the network and by an analytic approximation, and found that this edge prior reconstructs "
         "most Hetionet relations and accounts for much of the performance of some link-prediction features [@zietz2024]. Claims that a "
         "learned component helps on such graphs have to be measured against degree-only baselines. Information content, the negative "
         "log probability of encountering a concept, was introduced by Resnik to weight the specificity of concepts in a taxonomy "
         "[@resnik1995]; it is a natural candidate for a hub discount that depends on how often a node occurs as evidence rather than "
         "on how many edges it has.")
    heading(d, "2.2 Path-based and neuro-symbolic reasoning", 2)
    P(d, "Rule learners and walkers return explicit evidence. The Path Ranking Algorithm scores a pair by a learned linear combination "
         "of path-constrained random-walk probabilities [@lao2010], a direct ancestor of DWPC with logistic regression; AnyBURL mines "
         "path rules bottom-up [@meilicke2019]; NeuralLP and DRUM learn rule confidences end to end by composing relation matrices "
         "[@yang2017;sadeghian2019]; MINERVA learns a policy that walks from the query entity to an answer [@das2018], Liu et al. added "
         "metapath rules to its reward for drug repurposing on Hetionet [@liu2020], and KGML-xDTD combines a drug–disease classifier "
         "with reinforcement-learning path finding to describe treatment mechanisms [@ma2023]. Neural Bellman–Ford networks (NBFNet) "
         "learn both the operator that extends a path and the operator that aggregates paths, and read explanations from the paths that "
         "contribute most [@zhu2021]. BioPathNet applies this path-representation learning to biomedical knowledge graphs, including "
         "drug repurposing, and reports gains over embedding baselines together with path-based explanations [@hu2026]. These are the "
         "closest learned counterparts of the question we ask, with neural instead of fuzzy operators: they learn composition and "
         "aggregation jointly, which is exactly the separation our ablations try to undo. Neural theorem provers compose rule bodies "
         "with the minimum over soft unification scores [@rocktaschel2017]. In complex query answering, GNN-QE and FuzzQE combine "
         "learned relation projections with fixed fuzzy connectives [@zhu2022;chen2022fuzzqe], and continuous query decomposition uses "
         "t-norms over neural link predictors [@arakelyan2021]. In all of these the composition operator is either a product, implicit "
         "in matrix multiplication and in path counting, a t-norm chosen by the authors, or a neural function without a fuzzy reading. "
         "Uncertain knowledge graph embeddings model graded edges but compose them through embedding geometry [@chen2019].")
    heading(d, "2.3 Learnable fuzzy operators and identifiability", 2)
    P(d, "Parametric families such as Hamacher, Frank and Schweizer–Sklar span the classical t-norms with one parameter [@klement2000]. "
         "Rutkowski and Cpałka's flexible neuro-fuzzy systems learn the parameters of such families, together with weights on the "
         "inputs and rules, by gradient descent within a fuzzy inference system [@rutkowski2003]. Because a weight and a t-norm "
         "parameter can both raise or lower the firing strength of a rule, this line of work already suggests that learned operator "
         "parameters and learned weights can substitute for each other. Logic tensor networks ground first-order formulas in "
         "differentiable fuzzy semantics with configurable connectives [@badreddine2022]; van Krieken et al. analysed how operator "
         "choice affects gradients [@vankrieken2022], Giannini et al. related t-norms to loss functions [@giannini2023], and logical "
         "neural networks learn weights inside Łukasiewicz-type connectives [@riegel2020]. A discrete choice among operators can be "
         "made differentiable with the Gumbel-softmax relaxation [@jang2017], which we use for one of the refinements. Whether a "
         "learned parameter is determined by the data is the subject of identifiability analysis. In systems biology, Raue et al. "
         "distinguish structural non-identifiability, a redundancy in the model that holds for any data, from practical "
         "non-identifiability, where the profile of the objective in one parameter, with all others re-optimised, is too flat for the "
         "available data to fix it [@raue2009]. We use this vocabulary in Section 4.4. Among the methods reviewed here, none learns the "
         "parameter of a t-norm family per metapath of a link predictor trained on a medical knowledge graph, and none reports a "
         "profile of the objective in that parameter. This review is not systematic, and the claim is limited to the work cited.")


def methods(d):
    heading(d, "3. Methods", 1)
    heading(d, "3.1 Problem setting", 2)
    P(d, "A medical knowledge graph G = (V, E) has typed nodes and typed edges e = (u, r, v), where r is a relation such as "
         "compound–binds–gene. Some edges carry a graded weight w~e~ ∈ (0, 1]; most carry none. A query asks for the tail of a target "
         "relation r* given a head h, for example which diseases a compound treats. A metapath m is a sequence of relation types, "
         "traversed in either direction, that leads from the head type to the tail type; a path π instantiates m with distinct nodes. "
         "We consider metapaths of length two and three.")
    P(d, "The direct relation between h and t is never used to score (h, t). When a pair is held out for validation or testing, every "
         "edge between h and t is removed from the graph, whatever its relation type, so that, for example, a held-out "
         "compound–treats–disease pair cannot be recovered from a compound–palliates–disease edge between the same two nodes. We call "
         "this protocol leave-dyad-out.")
    heading(d, "3.2 Differentiable Neuro-Fuzzy Reasoning", 2)
    P(d, "DNFR scores a pair (h, t) in four steps (Figure 1): it extracts the paths between h and t and groups them by metapath, assigns "
         "each traversed edge a membership, composes the memberships along each path with a t-norm, and combines the paths by a noisy-OR.")
    figure(d, f"{ROOT}/figures/fig1_dnfr.png", 6.3,
           "**Figure 1.** Differentiable Neuro-Fuzzy Reasoning in four steps. (1) Paths of length two and three between a query pair "
           "are grouped by metapath, after every edge between the two nodes has been removed. (2) Each edge receives a learned "
           "membership that discounts high-degree endpoints and reads graded evidence when the graph has it. (3) A Schweizer–Sklar "
           "t-norm with a parameter learned per metapath composes memberships along each path; the maximum with 0 is active only for "
           "p > 0. (4) A noisy-OR weighted by metapath reliability combines paths, so that the score is a sum of path terms. The "
           "refinements of Section 3.3 act on steps 2–4.")
    P(d, "**Edge membership.** Most edges in Hetionet and PrimeKG are assertions without strengths, and an edge into a hub, such as a "
         "gene associated with hundreds of diseases, says less about any one of them than an edge into a specific node. DNFR therefore "
         "learns the membership of each traversed edge e = (u → v) of relation r from the degrees of its endpoints and, where present, "
         "from its graded weight:")
    P(d, "μ(e) = σ( [a~r~ − b~r~ ℓ~out~(u) − c~r~ ℓ~in~(v) + g~r~ logit(w~e~)] / τ ),   ℓ = log(1 + degree in relation r)          (1)",
      align="center")
    P(d, "where σ is the logistic function, a~r~, b~r~, c~r~ and g~r~ are learned for each relation and direction of traversal, and the "
         "temperature τ is 1 except in the annealing refinement of Section 3.3. Positive b~r~ and c~r~ discount edges at high-degree "
         "nodes, as the damping exponent of DWPC does [@himmelstein2017], but the strength of the discount is learned per relation. "
         "When edges carry no weight, the last term is absent; weights are clipped to [10^−4^, 1 − 10^−4^] before the logit is taken.")
    P(d, "**Learned t-norm.** The strength of a path π of length L with memberships μ~1~, ..., μ~L~ is the Schweizer–Sklar t-norm "
         "[@klement2000] with a parameter p~m~ learned for its metapath m:")
    P(d, "T~p~(μ~1~, ..., μ~L~) = [ max(0, Σ~i~ μ~i~^p^ − (L − 1)) ]^1/p^  (p ≠ 0),     T~0~(μ~1~, ..., μ~L~) = Π~i~ μ~i~          (2)",
      align="center")
    P(d, "For p < 0 the sum is at least L, so the maximum is inactive; it matters only for p > 0. As p → −∞ the family tends to the "
         "minimum, p = −1 gives exactly the Hamacher product xy/(x + y − xy), p → 0 gives the product and p = 1 the Łukasiewicz "
         "t-norm, so the fixed operators of Section 3.4 lie inside the family. For p > 0 a path whose memberships satisfy "
         "Σ~i~ μ~i~^p^ ≤ L − 1 has strength exactly zero and passes no gradient to any parameter. By default p~m~ = 6 σ(θ~m~) − 5, "
         "which keeps p~m~ in (−5, 1), initialised at −0.1, close to the product; Section 3.4 describes an unbounded alternative. "
         "Equation 2 is evaluated in log space with expm1, log1p and, where μ^p^ would overflow, a log-sum-exp.")
    P(d, "**Noisy-OR aggregation.** A pair is typically connected by many paths. DNFR combines them by a noisy-OR with a reliability "
         "ω~m~ ∈ (0, 1) per metapath and a leak probability β~0~:")
    P(d, "P(h, t) = 1 − (1 − β~0~) Π~π ∈ Π(h,t)~ ( 1 − ω~m~ T(π; p~m~) )^κ~π~,   m = m(π)          (3)", align="center")
    P(d, "with κ~π~ = 1 in DNFR. Because −log(1 − P) is then a sum of one term per path, s~π~ = −κ~π~ log(1 − ω~m~ T(π)), each path's "
         "share of it is an exact decomposition of the model's score. It is not an exact account of the evidence: paths that share an "
         "edge count that edge once per path, and Section 3.6 describes an edge-level attribution that removes this double counting. "
         "Pairs with no connecting path receive P = β~0~.")
    P(d, "**Training.** For each training query (h, r*, ?) we minimise the cross-entropy of a softmax over the logits of P(h, t) for all "
         "candidate tails of the right type, excluding the other known training tails of h (filtered softmax). The loss is computed "
         "exactly over all pairs that have at least one path, and the remaining tails share the logit of β~0~. Gradients are accumulated "
         "metapath by metapath in two passes, which bounds memory by the largest metapath rather than by the whole path set. We use Adam "
         "with learning rate 0.1 for at most 400 full-batch epochs, evaluate validation MRR every five epochs and stop after ten "
         "evaluations (50 epochs) without improvement, keeping the parameters with the best validation MRR. The default model adds an L2 "
         "penalty λ‖θ − θ~0~‖² with λ = 10^−3^ on the t-norm parameters and a penalty of 10^−5^ on the metapath reliabilities. "
         "Degrees in Equation 1 are counted on the training graph of each split, after the edges of all validation and test pairs have "
         "been removed. Algorithm 1 summarises one training run.")
    boxed(d, [
        "**Algorithm 1.** Training DNFR on one split and explaining a prediction.",
        "**Input:** graph G; target relation r*; training, validation and test pairs Q~tr~, Q~va~, Q~te~; metapaths ℳ of length 2–3; path budget B = 5 × 10^6^",
        "**Output:** parameters Θ = {a~r~, b~r~, c~r~, g~r~, θ~m~, ω~m~, β~0~}; scores P(h, t); path and edge attributions",
        "1: G′ ← G without every edge, of any relation, between the two nodes of a pair in Q~va~ ∪ Q~te~   *▷ leave-dyad-out*",
        "2: Π ← simple paths from the query heads that instantiate some m ∈ ℳ; drop metapaths with more than B paths and paths that use an edge removed in line 1",
        "3: ℓ~out~, ℓ~in~ ← log(1 + degree) of every node in every relation and traversal direction of G′",
        "4: initialise a~r~ = 1, b~r~ = c~r~ = 0.3, g~r~ = 1, p~m~ = −0.1, ω~m~ = σ(−3), β~0~ = σ(−6)",
        "5: **for** epoch = 1, ..., 400 **do**",
        "  6: S(h, t) ← 0 for every pair connected by a path",
        "  7: **for each** metapath m **do**   *▷ first pass, no gradient; one metapath in memory at a time*",
        "    8: **for each** path π ∈ Π~m~ with pair (h, t): log μ~i~ ← Eq. 1 for each edge; log T ← Eq. 2 in log space; s~π~ ← −κ~π~ log(1 − ω~m~ T); S(h, t) ← S(h, t) + s~π~",
        "  9: z(h, t) ← logit(1 − (1 − β~0~) e^−S(h,t)^); tails without a path share z~0~ = logit(β~0~)",
        "  10: L ← mean over (h, t) ∈ Q~tr~ of −z(h, t) + log(e^z(h,t)^ + Σ~t′~ e^z(h,t′)^), t′ ranging over tails that are not training positives of h; add λ‖θ − θ~0~‖²",
        "  11: γ(h, t) ← ∂L/∂S(h, t)",
        "  12: **for each** metapath m **do** recompute s~π~ with gradients and back-propagate γ(h, t) to Θ   *▷ second pass*",
        "  13: Adam step with learning rate 0.1",
        "  14: **every** 5 epochs: filtered MRR on Q~va~; keep the best Θ; **stop** after 10 evaluations without improvement",
        "15: **return** best Θ; P(h, t) = 1 − (1 − β~0~) e^−S(h,t)^ for pairs in Q~te~",
        "**Explanation of a pair (h, t):** path share s~π~ / Σ~π′~ s~π′~; edge share φ~e~ = Σ~π ∋ e~ (s~π~ / L~π~) / Σ~π′~ s~π′~ (Section 3.6)",
    ])

    heading(d, "3.3 Four refinements", 2)
    P(d, "Each refinement replaces one element of Section 3.2 and reduces to it for a particular parameter value, so that it can be "
         "added to DNFR alone or removed from the combination. All four together form DNFR 2.0.")
    P(d, "**(R1) Overlap-aware aggregation.** Two paths that pass through the same intermediate node rest partly on the same evidence, "
         "but Equation 3 with κ~π~ = 1 counts them as independent causes. For the paths of one pair (h, t), all of length two or three, "
         "two distinct paths share more than half of their nodes (Jaccard index of the node sets above 0.5) if and only if they share an "
         "interior node: a shared interior node gives a Jaccard index of 3/4, 3/5 or 1, no shared interior node at most 1/2. A rule "
         "that discounts the weaker of two strongly overlapping paths can therefore be computed per interior node, in time linear in "
         "the number of paths. We use a smooth version of that rule. With v~π~ = ω~m~ T(π) and v*~x~ the largest v~π′~ among the "
         "paths of the same pair through interior node x,")
    P(d, "κ~π~ = Π~x ∈ I(π)~ ( v~π~ / v*~x~ )^β,   β = softplus(β′) ≥ 0          (4)", align="center")
    P(d, "where I(π) are the interior nodes of π. The strongest path through each node keeps its full weight, weaker paths through "
         "the same node are discounted the more the weaker they are, and β is learned (initial value 1). With β = 0 Equation 3 is "
         "recovered. The maxima v*~x~ are computed without gradient in the first pass of Algorithm 1 and held fixed in the second.")
    P(d, "**(R2) Discrete operator choice.** Instead of a continuous p~m~, each metapath chooses among the product, the minimum and the "
         "Hamacher product. During training the path strength is a relaxed mixture, T~m~ = Σ~k~ y~mk~ T~k~ with "
         "y~m~ = softmax((α~m~ + g~m~)/τ~g~), where α~m~ are learned logits, g~m~ is Gumbel noise drawn once per epoch [@jang2017] and "
         "τ~g~ = max(0.2, e^−0.02·epoch^); the mixture is computed in log space. At evaluation each metapath uses the operator with the "
         "largest α~mk~. If a continuous parameter is hard to learn because the loss is flat between operators, a discrete choice "
         "should at least be reproducible across restarts; we test that directly.")
    P(d, "**(R3) Information-content discount.** The degree feature ℓ in Equation 1 counts edges. Information content counts how often "
         "a node occurs as evidence: with n(v) the number of training-graph paths of the task on which node v occurs, "
         "P(v) = n(v)/N~type(v)~ and IC(v) = −log P(v) [@resnik1995]. We replace ℓ(v) by log(1 + n(v)), which equals −IC(v) up to a "
         "constant per node type that a~r~ absorbs, rescaled so that its mean over the paths of each relation equals that of the degree "
         "feature it replaces, so that the initial b~r~ = c~r~ = 0.3 keep their meaning.")
    P(d, "**(R4) Membership annealing.** A sigmoid membership saturates when its argument is large, and its gradient then vanishes. "
         "Annealing starts with soft memberships and sharpens them: τ in Equation 1 decreases as τ = max(0.5, 2 · 0.25^epoch/100^), "
         "from 2 at the start to 0.5 from epoch 100. Because the loss uses log μ, the gradient vanishes only as μ → 1; we therefore also "
         "report how many memberships of the trained default model are close to 1 (Section 4.3).")
    P(d, "Overlap-aware aggregation needs an extra pass over the paths of each pair and two to three times the time per run on the "
         "graded graph (Section 4.8); the other three add little cost.")

    heading(d, "3.4 Variants, ablations and baselines", 2)
    np_ = RS["n_params"]
    P(d, "**Fixed operators.** Every fixed-operator variant uses the same membership, aggregation and training as DNFR: the product, the "
         "minimum, the Hamacher product and the Łukasiewicz t-norm. **Fixed p selected on validation.** A single Schweizer–Sklar p, "
         "shared by all metapaths and not learned, is set to each value of a grid (−4, −2, −1, −0.5, 0.5 and the product, p = 0; on "
         "Hetionet compound–treats–disease −2, −1, −0.5 and the product, for cost) and the model is trained with the default protocol; "
         "on each split the value with the best validation MRR is kept, and its test MRR is reported as *fixed p (validation)*.")
    P(d, "**Component ablations.** Each ablation changes one component of DNFR: a single p shared by all metapaths (shared-p); memberships "
         "without degree terms (no-degree, b~r~ = c~r~ = 0); degree terms fixed at b~r~ = c~r~ = 0.4 for every relation (fixed-degree; "
         "with the product, DWPC-like), with 0.2 and 0.6 as a sensitivity check on the graded graph; the strongest path only instead of "
         "noisy-OR (max-path); and, on the graded graph, edge weights ignored (binary edges, g~r~ = 0). **Refinements.** Each of R1–R4 "
         "is added to DNFR alone (add-one) and removed from DNFR 2.0 (leave-one-out); R1 is also added to the fixed product.")
    P(d, "**Identifiability of the t-norm.** (i) *Restarts* with the default early stopping from p~0~ = −2, −1, −0.1 and 0.5 without "
         "the penalty, and with stronger penalties (λ = 10^−2^, 10^−1^). (ii) *Unbounded parameter:* p~m~ = θ~m~, clipped to "
         "[−20, 0.95] only to keep Equation 2 finite, restarted from −2, −0.1 and 0.5 without penalty. (iii) *Training without early "
         "stopping* for 400 epochs from p~0~ = −2, −0.1 and 0.5, recording p~m~ and the training loss every five epochs. (iv) "
         "*Converged profile* on the graded graph and on Hetionet disease–symptom: a shared p is fixed on a grid from −5 to 0, "
         "extended to 0.1, 0.25 and 0.5 with warm starts from the fit at the product (the nilpotent range cannot be entered from "
         "the default initialisation, where every path starts at strength zero), and all other parameters are fitted by "
         "L-BFGS with strong Wolfe line search on the unpenalised training loss until the loss changes by less than 10^−9^ between "
         "iterations; at each converged point we compute ∂L/∂p by automatic differentiation and, from the converged losses, the "
         "curvature ∂²L/∂p² at the minimum by central differences (by a one-sided difference when the minimum lies at an end of "
         "the grid). A parameter whose profile has negligible curvature, while the other "
         "parameters can compensate for changes in it, is practically non-identified by the objective in the sense of Raue et al. "
         "[@raue2009]. (v) *Frozen memberships:* a~r~ = 1, b~r~ = c~r~ = 0.3, g~r~ = 1 fixed, with a learned p or the fixed product, "
         "tests whether the memberships and the operator substitute for each other, a structural property of the model. (vi) For R2, "
         "two further restarts with different random seeds test whether the chosen operators agree. (vii) For R3, a version whose "
         "feature is matched to the degree feature in standard deviation as well as in mean tests whether the loss of R3 on the "
         "graded graph comes from the scale of the feature.")
    P(d, "**Baselines.** Two degree-only baselines measure how much of the accuracy node degree alone explains: popularity ranks every "
         "tail by its number of training positives, and the edge prior of Zietz et al. [@zietz2024] gives each (head, tail) pair the "
         "share of 100 degree-preserving permutations of the training target relation (XSwap, 10 × |edges| attempted swaps each) in "
         "which the pair is an edge. The path baselines are DWPC with damping 0.4 on the same metapaths, arcsinh-transformed after "
         "division by the mean of the non-zero values of each metapath and fed to an L2-regularised logistic regression (DWPC-LR) "
         "[@himmelstein2017], a version that adds the logarithms of the head's and tail's degrees in the training target relation and "
         "the logit of the edge prior (DWPC-LR + degree + prior), and AnyBURL [@meilicke2019] (version 23-1, 100 s of rule learning, "
         "top 200 candidates per query). DWPC-LR is trained on the training positives and 20 random non-positive tails of the same head "
         "per positive, with balanced class weights and a penalty chosen from {0.01, 0.1, 1, 10} by validation MRR. Two completions use "
         "the target relation alone: a rank-k truncated SVD and a soft-thresholded SVD in the spirit of Soft-Impute [@mazumder2010], "
         "with k and the threshold chosen on validation data. TransE and RotatE [@bordes2013;sun2019], trained with PyKEEN [@ali2021] "
         "for 25 epochs without tuning on the Hetionet and PrimeKG tasks, are untuned reference points only. **NBFNet** [@zhu2021] "
         "learns how a path is extended and how paths are aggregated. We implemented a compact CPU version: three layers of "
         "generalised Bellman–Ford iteration with DistMult messages, sum aggregation, layer normalisation, residual connections and "
         "dimension 32, on the training graph of each split restricted to the relations on the task's metapaths, with the head "
         "query vector as boundary condition and an MLP on the final node state; in each epoch the target edges of a random half of "
         "each head's training tails are masked and serve as positives of the same filtered softmax loss, with the other half kept as "
         "context, and training uses Adam (learning rate 5 × 10^−3^) with the early stopping of DNFR and at most 300 epochs (150 on "
         "Hetionet disease–symptom, for cost). On the graded graph we also ran a 2 × 2 grid of learning rate (5 × 10^−3^, 10^−2^) "
         "and dimension (32, 64) and kept, on each split, the configuration with the best validation MRR (*tuned NBFNet*). It was "
         "run on the two disease–symptom graphs; BioPathNet [@hu2026] and TxGNN [@huang2024], and NBFNet on the drug-repurposing graphs, need GPU "
         "training and were not run (Section 5.5).")
    P(d, "**Model size.** DNFR learns three membership coefficients per relation and direction (four on the graded graph), a reliability "
         "and a t-norm parameter per metapath, and the leak: "
         f"{np_['het_CtD']['dnfr']}, {np_['het_DpS']['dnfr']}, {np_['pkg_ind']['dnfr']} and {np_['hkg_DpS']['dnfr']} trainable parameters "
         "on the four tasks (Table 1). DNFR 2.0 replaces the t-norm parameter by three operator logits per metapath and adds β. DWPC-LR "
         f"has one coefficient per metapath and an intercept ({np_['het_CtD']['dwpc_lr']}, {np_['het_DpS']['dwpc_lr']}, "
         f"{np_['pkg_ind']['dwpc_lr']} and {np_['hkg_DpS']['dwpc_lr']}). All path methods are small models; the comparison is between "
         "ways of weighting the same path evidence.")

    heading(d, "3.5 Data and tasks", 2)
    rows = []
    info = {t: S[t]["info"] for t in TASKS}
    lab = {"het_CtD": ("Drug repurposing", f"Hetionet ({DS['het']['nodes']:,} / {DS['het']['edges']:,})", "Compound–treats–Disease"),
           "het_DpS": ("Disease–symptom", "Hetionet", "Disease–presents–Symptom"),
           "pkg_ind": ("Drug repurposing", f"PrimeKG ({DS['pkg']['nodes']:,} / {DS['pkg']['edges']:,})", "Drug–indication–Disease"),
           "hkg_DpS": ("Graded disease–symptom (EHR)", f"HKG ({DS['hkg']['nodes']} / {DS['hkg']['edges']:,})", "Disease–Symptom, noisy-OR weight")}
    for t in TASKS:
        i = info[t]; dd = DS[t]
        rows.append([lab[t][0], lab[t][1], lab[t][2], f"{i['n_pos']:,}", f"{i['n_sources']:,}", f"{i['n_tails']:,}",
                     f"{dd['metapaths_kept']} of {dd['metapaths_all']}", f"{dd['paths']:,}",
                     f"{np_[t]['dnfr']} / {np_[t]['dwpc_lr']}", f"{nseeds(t, 'dnfr')}"])
    table(d, "**Table 1.** Tasks, graphs and path sets. Positives are the known pairs of the target relation; candidates are all nodes "
             "of the tail type, ranked for every test query. Parameters: trainable parameters of DNFR / DWPC-LR. Splits: random splits "
             "used for the main comparisons.",
          ["Task", "Graph (nodes / edges)", "Target relation", "Positives", "Query heads", "Candidates", "Metapaths used", "Paths",
           "Parameters", "Splits"], rows, widths=[0.8, 0.95, 0.9, 0.55, 0.5, 0.6, 0.6, 0.72, 0.6, 0.45], size=7.5,
          note=C("HKG, health knowledge graph learned from emergency-department records [@rotmensch2017]. Edge counts are unique "
                 "undirected edges; PrimeKG excludes anatomy–protein expression edges (Section 3.5)."))
    pk = DS["pkg"]; und = pk["raw_rows"] // 2; extra = und - pk["dropped_edges"] - pk["edges"]
    P(d, f"**Hetionet** (v1.0) [@himmelstein2017] has {DS['het']['nodes']:,} nodes of {DS['het']['node_types']} types and "
         f"{DS['het']['edges']:,} edges of {DS['het']['rel_types']} types. We predict compound–treats–disease ({info['het_CtD']['n_pos']} "
         f"pairs; {info['het_CtD']['n_sources']} compounds, ranked against all {info['het_CtD']['n_tails']} diseases) and "
         f"disease–presents–symptom ({info['het_DpS']['n_pos']:,} pairs, ranked against {info['het_DpS']['n_tails']} symptoms).")
    P(d, f"**PrimeKG** [@chandak2023] has {DS['pkg']['nodes']:,} nodes. Its edge file has {pk['raw_rows']:,} rows and stores each "
         f"undirected edge in both directions ({und:,} edges). We removed the {pk['dropped_edges']:,} anatomy–protein expression edges, "
         "which do not lie on drug–disease paths of length three or less but dominate the edge count; keeping one orientation per "
         f"relation and dropping repeated pairs and self-loops removed another {extra:,}, leaving {pk['edges']:,} edges of "
         f"{pk['rel_types']} relation types. All methods use the same graph. We predict drug–indication–disease "
         f"({info['pkg_ind']['n_pos']:,} pairs from {info['pkg_ind']['n_sources']:,} drugs), ranking every one of the "
         f"{info['pkg_ind']['n_tails']:,} diseases.")
    P(d, "**Health knowledge graph (HKG).** Rotmensch et al. learned this graph from 273,174 emergency-department visits recorded "
         "between 2008 and 2013 and published it in full [@rotmensch2017]. We parsed the "
         f"{DS['hkg']['edges']:,} disease–symptom edges between {info['hkg_DpS']['n_sources']} diseases and {info['hkg_DpS']['n_tails']} "
         "symptoms from their supplementary appendix. Each edge carries the noisy-OR importance w~e~ = 1 − f, where f is the estimated "
         f"probability that the disease fails to produce the symptom; w~e~ has median {f3(DS['hkg']['w_median'])} (interquartile range "
         f"{f3(DS['hkg']['w_q25'])} to {f3(DS['hkg']['w_q75'])}). HKG has a single relation, so its only metapath is "
         "disease–symptom–disease–symptom: a disease is linked to a new symptom through other diseases that share its known symptoms. "
         "This task therefore tests the composition of graded evidence learned from patient records in a collaborative-filtering "
         "setting, not multi-relational reasoning, and low-rank completion is a natural competitor on it. We use the published graph "
         "rather than records because the underlying records, from the hospital behind MIMIC-IV [@johnson2023], require credentialed "
         "access that we did not hold.")
    P(d, "**Metapaths and computation.** For each task we enumerated all metapaths of length two and three between the head and tail "
         "types and all simple paths that instantiate them from the query heads. To keep training on a two-core CPU feasible, metapaths "
         "with more than five million paths over the query heads were excluded for all path methods: "
         f"{len(DS['het_CtD']['skipped'])} of {DS['het_CtD']['metapaths_all']} metapaths for compound–treats–disease (those through side "
         f"effects and anatomy expression), none for disease–symptom, and {len(DS['pkg_ind']['skipped'])} of "
         f"{DS['pkg_ind']['metapaths_all']} for PrimeKG, mostly through drug–drug interactions and contraindications. The remaining "
         "sets contain " + ", ".join(f"{DS[t]['paths']:,}" for t in TASKS[:3]) + f" and {DS['hkg_DpS']['paths']:,} paths. Because the "
         "excluded metapaths are the most hub-dominated, PrimeKG accuracies of all path methods are lower bounds.")

    heading(d, "3.6 Evaluation", 2)
    nd = nseeds("het_DpS", "dnfr"); nh = nseeds("hkg_DpS", "dnfr"); nc = nseeds("het_CtD", "dnfr")
    nt = nseeds("het_CtD", "dnfr", "_tail"); npk = nseeds("pkg_ind", "dnfr")
    P(d, "Positive pairs of each target relation were split at random into 80% training, 10% validation and 10% test. The main "
         f"comparisons use {nc} splits on Hetionet compound–treats–disease, {nd} on Hetionet disease–symptom, {nh} on the graded graph "
         f"and {npk} on PrimeKG; the refinements were run on the same splits, except where the tables give a smaller number. For each test "
         "pair (h, t) we rank t among all candidate tails of its type after removing the other known tails of h (filtered ranking), "
         "break ties by the mean rank, and report mean reciprocal rank (MRR) and Hits@1, @3 and @10. On compound–treats–disease the "
         "validation set has about 75 pairs, so model selection there is noisy; we report the spread of validation MRR across "
         "checkpoints in the supplementary material.")
    P(d, f"**Disease-centred split.** For compound–treats–disease we also split the diseases instead of the pairs ({nt} splits; 80/10/10 "
         "of the diseases with at least one indication), so that every indication of a test disease is held out together, as in "
         "zero-shot repurposing [@huang2024]. **Metapaths through the target relation.** On compound–treats–disease, "
         f"{R4['explain_dwpc']['het_CtD']['dnfr']['n_target_metapaths']} of the {R4['explain_dwpc']['het_CtD']['dnfr']['n_metapaths']} "
         "metapaths contain the treats relation itself, "
         "for example compound–resembles–compound–treats–disease. They let a prediction rest on known indications of similar "
         "compounds or of related diseases. To measure how much of the accuracy comes from them, we trained DNFR, the fixed product "
         "and DWPC-LR on the same random splits with these metapaths removed, so that a prediction can rest only on the other "
         "relations of the graph.")
    P(d, "**Statistics and decision rule.** For a comparison between two methods we report the difference in MRR averaged over the "
         "splits on which both were run, a paired bootstrap interval over test queries stratified by split, the number of splits on "
         "which the difference has the same sign, and the corrected resampled t-test across splits [@nadeau2003], which accounts for "
         "the overlap between training sets and for variation between splits. Within each task, p-values are Holm-adjusted "
         "[@holm1979] within families defined by research question: the baselines against DNFR (RQ1), the learned-operator "
         "baselines against the path methods (RQ1), the components of DNFR (RQ2; Table 3, all panels), the refinements and the "
         "validation-selected t-norm (RQ2; Table 4), and the ablation of metapaths through the target relation (RQ1). Restarts and "
         "other checks are descriptive and not tested. Most claims about t-norms and refinements are claims of equivalence or of "
         "absence of a gain, so we use one decision rule for every comparison. With "
         "a margin of ±0.01 in MRR, a difference is *equivalent* (≈) when both 90% intervals, corrected-t and bootstrap, lie inside the "
         "margin (two one-sided tests at the 5% level); *different* (†) when the Holm-adjusted corrected-t p-value is below 0.05 and "
         "the 95% bootstrap interval excludes zero on the same side; and *inconclusive* otherwise. Requiring both intervals guards "
         "against either one being too narrow: the bootstrap ignores variation between splits, and with few splits the between-split "
         "variance in the corrected test is itself poorly estimated and can make that interval narrower than the bootstrap. An "
         "earlier version of this analysis called a difference supported when the unadjusted 95% corrected-t interval excluded zero; "
         "we replaced that rule because, with three splits, it accepted differences whose bootstrap interval contained zero and it "
         "did not adjust for multiplicity. Supplementary Table S12 gives the verdicts under both rules for the comparisons on which "
         "the Highlights and the Conclusion rest. A change of "
         "0.01 in MRR corresponds to moving one correct answer in fifty from rank 1 to rank 2, or about one in seventeen from rank 2 "
         "to rank 3; we regard smaller differences as too small to change which candidates a reviewer of a ranked list would examine. With three or fewer splits no equivalence verdict is attainable, which affects PrimeKG and "
         "the disease-centred split. Test queries are never pooled across splits for testing, since the same pair can be a test query "
         "in several random splits.")
    P(d, "**Explanations.** We measure concision (the number of paths and of metapaths that account for 50% and 80% of the evidence "
         "−log(1 − P) above the leak) and faithfulness on the test queries of the seed-0 split that have at least one path (at most 400 "
         "sampled at random). Path deletion removes the k paths with the largest shares (k = 1, 3, 10), k random paths of the same pair, "
         "random paths with the same total share (share-matched), or the leading metapath, and records the new filtered rank with all "
         "other scores fixed. Because the score is additive in path terms, path shares double-count facts that lie on several paths. "
         "We therefore also attribute the score to edges. With all memberships fixed, s~π~ counts only if every edge of π is present, "
         "so the score is a sum of unanimity games, one per path, and the Shapley value of edge e is exactly "
         "φ~e~ = Σ~π ∋ e~ s~π~ / L~π~: each path's term is split equally among its edges. Edge shares φ~e~ / Σ~e′~ φ~e′~ sum to one "
         "without double counting. The edge-level counterfactual removes the fact with the largest Shapley share, or a random fact on "
         "the pair's paths, from the graph, drops every path from the query's head that uses it, so that competing tails lose evidence "
         "too, and re-ranks without retraining. For drug repurposing we also classify the leading metapath of each correct top-10 "
         "prediction as mechanistic (through a gene the compound binds, up- or down-regulates and that is associated with, or regulated "
         "in, the disease) or associative (through pharmacologic classes, similar compounds or similar diseases), and we report the "
         "share of the evidence on each class and on metapaths that contain the target relation. Equal division credits a weak and a "
         "strong edge of the same path alike: φ~e~ measures how much the score depends on the presence of a fact, while the "
         "strength the model assigns to the fact is its membership μ, which the explanations show next to it (Table 7). For a "
         "clinician checking a prediction, the first question is which facts it rests on, and the second how much weight each is "
         "given; the two quantities answer them separately. For comparison we attribute DWPC-LR in the same way. Its score is "
         "z = b + Σ~m~ β~m~ asinh(D~m~ / d̄~m~), where D~m~ is the degree-weighted path count of metapath m and d̄~m~ its mean. The "
         "metapath terms add up exactly; we split each term over the paths of its metapath in proportion to their degree-weighted "
         "weights, which is exact within a metapath and a proportional rather than marginal allocation across the asinh, and split "
         "each path's share equally among its facts. The same fact-level counterfactual then re-scores every candidate of the head "
         "with the DWPC features recomputed without the removed fact. The score P orders candidates but is not calibrated as a "
         "probability.")
    P(d, "**Sparsity analysis.** We removed 50% and 75% of all edges of every relation type at random, including training edges of the "
         "target relation, and repeated training and evaluation of the path methods and of completion over three splits with the same "
         "test pairs. This favours path methods by construction, since completion uses no other relation.")
    P(d, "**Choice of defaults.** The defaults of DNFR (learning rate 0.1, initial b~r~ = c~r~ = 0.3, p~0~ = −0.1, λ = 10^−3^, the "
         "patience of early stopping) and the path budget were set during development to values that made training stable and "
         "feasible on a two-core CPU; they were not chosen by a search over values, and development runs used the same graphs. The "
         "settings of the refinements (initial β, the temperature schedules, the operator set) were fixed before any of them was run "
         "on more than one split. Sensitivity to the defaults that matter for the conclusions is reported in Sections 4.2–4.4.")
    P(d, "**Software, cost and use of AI assistance.** All experiments ran on a shared two-core CPU with 8 GB of memory. The analysis "
         "code was written with the help of an AI system (Claude, Anthropic), which also ran the experiments under the authors' "
         "direction. The authors specified the models, protocols and analyses. The released tests cover the leave-dyad-out removal, "
         "the Schweizer–Sklar t-norm against its product, Hamacher, Łukasiewicz and minimum limits, the overlap groups of R1 against a "
         "direct pairwise Jaccard computation, and the filtered ranking with ties. Independent human verification is described in "
         "Section 7. Every number reported below is read from the released result files by the script that builds this manuscript.")
