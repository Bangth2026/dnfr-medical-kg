"""Introduction, related work and methods (second revision)."""
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
         "Rephetio [@himmelstein2017], rule learners such as AnyBURL and NeuralLP [@meilicke2019;yang2017], and reinforcement-learning "
         "walkers such as MINERVA and its rule-guided extension [@das2018;liu2020] all return the paths behind a prediction. Such "
         "traceable evidence is what multidisciplinary reviews of explainable clinical AI ask decision-support systems to provide [@amann2020].")
    P(d, "Every path-based method also has to decide how strengths combine along a path. Fuzzy logic makes this decision explicit: "
         "the strength of a path is a triangular norm (t-norm) of the strengths of its edges [@zadeh1965;klement2000]. The choice is "
         "consequential. On a path of three edges of strength 0.3, the minimum returns 0.3, the product 0.027 and the Łukasiewicz "
         "t-norm 0. Yet in the methods above the operator is fixed in advance. Degree-weighted path counts multiply damped edge weights; "
         "NeuralLP composes relations by matrix products; GNN-QE models conjunction with product fuzzy logic [@zhu2022]; clinical fuzzy "
         "systems have used max–min composition since Sanchez's work on medical diagnosis [@sanchez1979] and still fix the operator "
         "[@vanpham2023;apostolopoulos2024]. Learning the operator is not new in neuro-fuzzy systems: flexible neuro-fuzzy systems "
         "learn the parameters of parametric t-norms jointly with the other weights [@rutkowski2003], differentiable fuzzy logic has "
         "studied how the choice of operator affects learning [@vankrieken2022;giannini2023], logical neural networks learn weighted "
         "Łukasiewicz connectives [@riegel2020], and neural Bellman–Ford networks learn neural composition and aggregation operators "
         "[@zhu2021]. Among the work we review (Section 2), however, none learns a t-norm per kind of path for medical link prediction, "
         "and none asks whether the learned parameter is determined by the data.")
    P(d, "A motivating observation comes from our earlier analysis of a population fuzzy graph of depressive symptoms and their "
         "correlates, built from the US National Health and Nutrition Examination Survey (NHANES); that analysis is not yet peer "
         "reviewed, and its code and results are deposited openly [@truong2026]. The product ranked held-out relations best among "
         "fixed t-norms, a tuned parametric t-norm was no better, and the raw relations had to be corrected for the base rates of the "
         "concepts they linked before composition was meaningful. These observations raise a question for large knowledge graphs, "
         "where edges carry no strengths at all and node degrees range over several orders of magnitude: when composition is learned "
         "end to end, which parts can actually be learned from the data, the t-norm or the edge memberships it is applied to?")
    P(d, "We address this question with Differentiable Neuro-Fuzzy Reasoning (DNFR), a path-based link predictor built so that each "
         "of its components can be switched off or fixed: an edge membership that discounts high-degree nodes and, when available, "
         "reads graded evidence; a Schweizer–Sklar t-norm whose parameter is learned separately for each metapath, so that the operator "
         "can range from the minimum through the Hamacher product and the product to the Łukasiewicz t-norm; and a noisy-OR "
         "combination of paths weighted by metapath reliability. Every prediction decomposes exactly into the contributions of "
         "individual paths. We use DNFR as an instrument for a controlled component analysis on four medical link-prediction tasks: "
         "drug repurposing in Hetionet and PrimeKG, disease–symptom prediction in Hetionet, and disease–symptom prediction in a graph "
         "learned from 273,174 emergency-department records [@rotmensch2017], whose edges carry graded noisy-OR weights.")
    P(d, "The study asks three questions. (RQ1) How accurate is a learned neuro-fuzzy path reasoner against degree-weighted path "
         "counts, rule learning, low-rank completion and degree-only baselines on sparse medical graphs, including when test diseases "
         "have no known indications? (RQ2) Which of its components carry the gain, and is the learned t-norm parameter identified by "
         "the data? (RQ3) How does it behave when the graph is thinned, and how concise and how faithful are its path explanations? "
         "The contributions are:")
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
         "favour high-degree entities, so that a gene's degree predicts how highly it is ranked [@bonner2022]. Zietz et al. computed the "
         "probability of an edge from node degrees alone, by degree-preserving permutation of the network, and found that this edge "
         "prior reconstructs most Hetionet relations with high accuracy and accounts for much of the performance of some link-prediction "
         "features [@zietz2024]. Any claim that a learned component helps on such graphs therefore has to be measured against a "
         "degree-only baseline that accounts for the degrees of both endpoints; we use their edge prior for this purpose.")
    heading(d, "2.2 Path-based and neuro-symbolic reasoning", 2)
    P(d, "Rule learners and walkers return explicit evidence. The Path Ranking Algorithm scores a pair by a learned linear combination "
         "of path-constrained random-walk probabilities [@lao2010], a direct ancestor of DWPC with logistic regression; AnyBURL mines "
         "path rules bottom-up [@meilicke2019]; NeuralLP and DRUM learn rule confidences end to end by composing relation matrices "
         "[@yang2017;sadeghian2019]; MINERVA learns a policy that walks from the query entity to an answer [@das2018], Liu et al. added "
         "metapath rules to its reward for drug repurposing on Hetionet [@liu2020], and KGML-xDTD combines a drug–disease classifier "
         "with reinforcement-learning path finding guided by demonstration paths to describe treatment mechanisms [@ma2023]. Neural "
         "Bellman–Ford networks generalise path-based link prediction by learning both the operator that extends a path and the "
         "operator that aggregates paths, and read explanations from the paths that contribute most [@zhu2021]; they are the closest "
         "learned counterpart of the question we ask, with neural instead of fuzzy operators. Neural theorem provers compose rule "
         "bodies with the minimum over soft unification scores [@rocktaschel2017]. Complex query answering combines link predictors "
         "with t-norms for conjunction [@zhu2022;arakelyan2021]. In all of these the composition operator is either a product, implicit "
         "in matrix multiplication and in path counting, or a t-norm chosen by the authors. Uncertain knowledge graph embeddings model "
         "graded edges but compose them through embedding geometry rather than through a t-norm [@chen2019].")
    heading(d, "2.3 Learnable fuzzy operators", 2)
    P(d, "Parametric families such as Hamacher, Frank and Schweizer–Sklar span the classical t-norms with one parameter [@klement2000]. "
         "Rutkowski and Cpałka's flexible neuro-fuzzy systems learn the parameters of such families, together with weights on the "
         "inputs and rules and a parameter that blends t-norms and t-conorms, by gradient descent within a fuzzy inference system "
         "[@rutkowski2003]. Because a weight and a t-norm parameter can both raise or lower the firing strength of a rule, this line of "
         "work already suggests that learned operator parameters and learned weights can trade off against each other, which is the "
         "redundancy we test in Section 4.2. Logic tensor networks ground first-order formulas in differentiable fuzzy semantics with "
         "configurable connectives and p-mean aggregators whose exponent is set by the modeller [@badreddine2022]; van Krieken et al. "
         "analysed how operator choice affects gradients in differentiable fuzzy logic [@vankrieken2022], Giannini et al. related "
         "t-norms to loss functions [@giannini2023], and logical neural networks learn weights inside Łukasiewicz-type connectives "
         "[@riegel2020]. Clinical fuzzy systems, from Sanchez's max–min diagnosis [@sanchez1979] to fuzzy cognitive maps "
         "[@apostolopoulos2024] and fuzzy knowledge graph pairs [@vanpham2023], fix the operator. In our earlier NHANES study, tuning one "
         "parameter of a parametric family on development data gave no meaningful improvement over the product [@truong2026]. Among the "
         "methods reviewed in this section, none learns the parameter of a t-norm family separately for each metapath of a link "
         "predictor trained on a medical knowledge graph, and none reports whether a learned operator parameter is identified: flexible "
         "neuro-fuzzy systems learn t-norm parameters per rule in rule-based classifiers and approximators, NBFNet learns its operators "
         "but not within a t-norm family, and logic tensor networks fix the exponent of their aggregators. This review is not "
         "systematic, and the claim is limited to the work cited here.")


def methods(d):
    heading(d, "3. Methods", 1)
    heading(d, "3.1 Problem setting", 2)
    P(d, "A medical knowledge graph G = (V, E) has typed nodes and typed edges e = (u, r, v), where r is a relation such as "
         "compound–binds–gene. Some edges carry a graded weight w~e~ ∈ (0, 1]; most carry none. A query asks for the tail of a target "
         "relation r\\* given a head h, for example which diseases a compound treats. A metapath m is a sequence of relation types, "
         "traversed in either direction, that leads from the head type to the tail type; a path π instantiates m with distinct nodes. "
         "We consider metapaths of length two and three.".replace("\\*", "*"))
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
           "p > 0. (4) A noisy-OR weighted by metapath reliability combines paths, so that every prediction decomposes into path "
           "contributions.")
    P(d, "**Edge membership.** Most edges in Hetionet and PrimeKG are assertions without strengths, and an edge into a hub, such as a "
         "gene associated with hundreds of diseases, says less about any one of them than an edge into a specific node. DNFR therefore "
         "learns the membership of each traversed edge e = (u → v) of relation r from the degrees of its endpoints and, where present, "
         "from its graded weight:")
    P(d, "μ(e) = σ( a~r~ − b~r~ ℓ~out~(u) − c~r~ ℓ~in~(v) + g~r~ logit(w~e~) ),   ℓ = log(1 + degree in relation r)          (1)",
      align="center")
    P(d, "where σ is the logistic function and a~r~, b~r~, c~r~ and g~r~ are learned for each relation and direction of traversal. "
         "Positive b~r~ and c~r~ discount edges at high-degree nodes, as the damping exponent of DWPC does [@himmelstein2017], but the "
         "strength of the discount is learned per relation. When edges carry no weight, the last term is absent. Because logit(w~e~) "
         "is infinite at w~e~ = 1 (and at 0), weights are clipped to [10^−4^, 1 − 10^−4^] before the logit is taken.")
    P(d, "**Learned t-norm.** The strength of a path π of length L with memberships μ~1~, ..., μ~L~ is the Schweizer–Sklar t-norm "
         "[@klement2000] with a parameter p~m~ learned for its metapath m:")
    P(d, "T~p~(μ~1~, ..., μ~L~) = [ max(0, Σ~i~ μ~i~^p^ − (L − 1)) ]^1/p^  (p ≠ 0),     T~0~(μ~1~, ..., μ~L~) = Π~i~ μ~i~          (2)",
      align="center")
    P(d, "For p < 0 the sum is at least L, so the maximum is inactive; it matters only for p > 0. As p → −∞ the family tends to the "
         "minimum, p = −1 gives exactly the Hamacher product xy/(x + y − xy) (λ = 0), p → 0 gives the product and p = 1 the Łukasiewicz "
         "t-norm. The fixed Hamacher and product variants of Section 3.3 therefore lie inside the family that DNFR can learn. For p > 0 "
         "a path whose memberships are small enough that Σ~i~ μ~i~^p^ ≤ L − 1 has strength exactly zero; in log space its strength is "
         "−∞, it contributes nothing to the score and it passes no gradient to any parameter. We parameterise p~m~ = 6 σ(θ~m~) − 5, "
         "which keeps p~m~ in (−5, 1), and by default initialise every p~m~ at −0.1, close to the product. Equation 2 is evaluated in "
         "log space with expm1 and log1p, so that gradients with respect to p~m~ remain finite near p~m~ = 0; when p~m~ < 0 and an edge "
         "membership is so small that μ^p^ would overflow, the logarithm of the sum is computed with a log-sum-exp.")
    P(d, "**Noisy-OR aggregation.** A pair is typically connected by many paths. DNFR combines them by a noisy-OR with a reliability "
         "ω~m~ ∈ (0, 1) per metapath and a leak probability β~0~:")
    P(d, "P(h, t) = 1 − (1 − β~0~) Π~π ∈ Π(h,t)~ ( 1 − ω~m~ T(π; p~m~) ),   m = m(π)          (3)", align="center")
    P(d, "The noisy-OR treats paths as independent causes, which they are not: paths that share an edge or pass through the same hub "
         "count the shared evidence once per path. The degree terms of Equation 1 partly compensate, since an edge at a hub lies on many "
         "paths and is discounted, but they do not remove the double counting, and we quantify it for the explanations in Section 4.6. "
         "Because −log(1 − P) is a sum over paths, each path's share of it is an exact attribution of the model's score, and the paths "
         "with the largest shares are the explanation shown for a pair. Pairs with no connecting path receive P = β~0~.")
    P(d, "**Training.** For each training query (h, r*, ?) we minimise the cross-entropy of a softmax over the logits of P(h, t) for all "
         "candidate tails of the right type, excluding the other known training tails of h (filtered softmax). The loss is computed "
         "exactly over all pairs that have at least one path, and the remaining tails share the logit of β~0~. Gradients are accumulated "
         "metapath by metapath in two passes, which bounds memory by the largest metapath rather than by the whole path set. We use Adam "
         "with learning rate 0.1 for at most 400 full-batch epochs, evaluate validation MRR every five epochs and stop after ten "
         "evaluations (50 epochs) without improvement, keeping the parameters with the best validation MRR. The default model adds an L2 "
         "penalty λ‖θ − θ~0~‖² with λ = 10^−3^ that pulls the unconstrained t-norm parameters θ~m~ towards their initial value θ~0~, and "
         "a penalty of 10^−5^ on the metapath reliabilities. Degrees in Equation 1 are counted on the training graph of each split, that "
         "is, after the edges of all validation and test pairs have been removed; they include the training edges of the target "
         "relation. Algorithm 1 summarises one training run.")
    boxed(d, [
        "**Algorithm 1.** Training DNFR on one split and explaining a prediction.",
        "**Input:** graph G; target relation r*; training, validation and test pairs Q~tr~, Q~va~, Q~te~; metapaths ℳ of length 2–3; path budget B = 5 × 10^6^",
        "**Output:** parameters Θ = {a~r~, b~r~, c~r~, g~r~, θ~m~, ω~m~, β~0~}; scores P(h, t); path attributions",
        "1: G′ ← G without every edge, of any relation, between the two nodes of a pair in Q~va~ ∪ Q~te~   *▷ leave-dyad-out*",
        "2: Π ← simple paths from the query heads that instantiate some m ∈ ℳ; drop metapaths with more than B paths and paths that use an edge removed in line 1",
        "3: ℓ~out~, ℓ~in~ ← log(1 + degree) of every node in every relation and traversal direction of G′",
        "4: initialise a~r~ = 1, b~r~ = c~r~ = 0.3, g~r~ = 1, p~m~ = −0.1, ω~m~ = σ(−3), β~0~ = σ(−6)",
        "5: **for** epoch = 1, ..., 400 **do**",
        "  6: S(h, t) ← 0 for every pair connected by a path",
        "  7: **for each** metapath m **do**   *▷ first pass, no gradient; one metapath in memory at a time*",
        "    8: **for each** path π ∈ Π~m~ with pair (h, t): log μ~i~ ← Eq. 1 for each edge; log T ← Eq. 2 in log space; s~π~ ← −log(1 − ω~m~ T); S(h, t) ← S(h, t) + s~π~",
        "  9: z(h, t) ← logit(1 − (1 − β~0~) e^−S(h,t)^); tails without a path share z~0~ = logit(β~0~)",
        "  10: L ← mean over (h, t) ∈ Q~tr~ of −z(h, t) + log(e^z(h,t)^ + Σ~t′~ e^z(h,t′)^), t′ ranging over tails that are not training positives of h; add λ‖θ − θ~0~‖²",
        "  11: γ(h, t) ← ∂L/∂S(h, t)",
        "  12: **for each** metapath m **do** recompute s~π~ with gradients and back-propagate γ(h, t) to Θ   *▷ second pass*",
        "  13: Adam step with learning rate 0.1",
        "  14: **every** 5 epochs: filtered MRR on Q~va~; keep the best Θ; **stop** after 10 evaluations without improvement (default)",
        "15: **return** best Θ; P(h, t) = 1 − (1 − β~0~) e^−S(h,t)^ for pairs in Q~te~",
        "**Explanation of a pair (h, t):** sort its paths by s~π~; the share of path π is s~π~ / Σ~π′~ s~π′~, its exact share of −log(1 − P) above the leak",
    ])

    heading(d, "3.3 Variants, ablations and baselines", 2)
    np_ = RS["n_params"]
    P(d, "**Fixed operators.** Every fixed-operator variant uses the same membership, aggregation and training as DNFR: the product, the "
         "minimum, the Hamacher product (λ = 0) and the Łukasiewicz t-norm.")
    P(d, "**Component ablations.** Each ablation changes one component of the full model: a single p shared by all metapaths "
         "(shared-p); memberships without degree terms (no-degree, b~r~ = c~r~ = 0); degree terms fixed at b~r~ = c~r~ = 0.4 for every "
         "relation instead of learned (fixed-degree; with the product, DWPC-like), with 0.2 and 0.6 as a sensitivity check on the graded "
         "graph; the strongest path only instead of noisy-OR (max-path); and, on the graded graph, edge weights ignored (binary edges, "
         "g~r~ = 0). Hetionet disease–symptom edges carry no weight, so the binary-edge ablation applies only to the graded graph.")
    P(d, "**Identifiability of the t-norm.** Because the product (p → 0) and the Hamacher product (p = −1) are members of the learned "
         "family, a learned p that stays near its initial value and loses to a fixed member of its own family would point to an "
         "optimisation or regularisation effect rather than to a property of the data. We used three analyses. (i) *Restarts:* DNFR "
         "trained with the default early stopping from p~0~ = −2, −1, −0.1 and 0.5 without the penalty, and with stronger penalties "
         "(λ = 10^−2^ and 10^−1^). (ii) *Training without early stopping:* from p~0~ = −2, −0.1 and 0.5 without the penalty, for the "
         "full 400 epochs, recording every p~m~, the training loss and the norm of ∂L/∂θ every five epochs. (iii) *Profile:* p fixed, "
         "shared by all metapaths, on a grid (−4, −2, −1, −0.5, 0, 0.25, 0.5, 0.75 on the graded graph; −4, −1, 0, 0.5, 0.75 on "
         "Hetionet disease–symptom), with every other parameter trained without early stopping for up to 300 epochs (250 on Hetionet "
         "disease–symptom) or until the training loss decreased by less than 10^−4^ over 50 epochs; we record "
         "the final training loss and the test MRR of the parameters with the best validation MRR. If the data identified p, the profile "
         "of the training loss would have a clear minimum and runs without early stopping would converge to it from every start. "
         "Analyses (ii) and (iii) were run on the two disease–symptom tasks, where a run is cheap enough (Section 4.7). To test whether "
         "learned memberships make the t-norm redundant, we also froze the memberships at their initial values (a~r~ = 1, "
         "b~r~ = c~r~ = 0.3, g~r~ = 1) and compared a learned p with the fixed product under the frozen memberships (frozen-membership).")
    P(d, "**Baselines.** Two degree-only baselines measure how much of the accuracy node degree alone explains: popularity ranks every "
         "tail by its number of training positives, and the edge prior of Zietz et al. [@zietz2024] gives each (head, tail) pair the "
         "share of 100 degree-preserving permutations of the training target relation (XSwap, 10 × |edges| attempted swaps each) in "
         "which the pair is an edge, so that it accounts for the degrees of both endpoints. The path baselines are DWPC with damping 0.4 "
         "on the same metapaths, arcsinh-transformed after division by the mean of the non-zero values of each metapath and fed to an "
         "L2-regularised logistic regression (DWPC-LR) [@himmelstein2017], and AnyBURL [@meilicke2019], which learns path rules for the "
         "target relation from the same leave-dyad-out training graph (version 23-1, 100 s of rule learning on one CPU thread, top 200 "
         "candidates per query; tails outside the candidate list share the remaining ranks). DWPC-LR is trained on the training "
         "positives and, for each of them, 20 tails of the same head drawn uniformly at random among those that are not training "
         "positives, with classes weighted to balance; its penalty is chosen from {0.01, 0.1, 1, 10} by validation MRR. Rephetio also "
         "used degree features and a permutation prior [@himmelstein2017], so we report a second version, DWPC-LR + degree + prior, that "
         "adds the logarithms of the head's and the tail's degrees in the training target relation and the logit of the edge prior as "
         "features and scores every (head, tail) pair. Two completions use the target relation alone: a rank-k truncated singular value "
         "decomposition, with k chosen on validation data, and a soft-thresholded SVD in the spirit of Soft-Impute [@mazumder2010], with "
         "the threshold chosen on validation data. As references without paths we report TransE and RotatE [@bordes2013;sun2019], "
         "trained with PyKEEN [@ali2021] on the whole training graph for 25 epochs with embedding dimension 128 (64 complex dimensions "
         "for RotatE), four negative samples per positive, the self-adversarial negative-sampling loss and learning rate 0.005 (0.001 "
         "for TransE). This budget was fixed by our CPU-only setting and not tuned; ComplEx [@trouillon2016] stayed near chance level "
         "under it. We therefore treat the embeddings as untuned reference points and draw no conclusion from the gap between them and "
         "the path methods. Graph neural networks that learn path operators, such as NBFNet [@zhu2021] and TxGNN [@huang2024], need GPU "
         "training on graphs of this size and were not run.")
    P(d, "**Model size.** DNFR learns three membership coefficients per relation and direction (four on the graded graph), a reliability "
         "and a t-norm parameter per metapath, and the leak: "
         f"{np_['het_CtD']['dnfr']}, {np_['het_DpS']['dnfr']}, {np_['pkg_ind']['dnfr']} and {np_['hkg_DpS']['dnfr']} trainable parameters "
         "on the four tasks (Table 1). The fixed-t-norm variants have one parameter fewer per metapath. DWPC-LR has one coefficient per "
         f"metapath and an intercept ({np_['het_CtD']['dwpc_lr']}, {np_['het_DpS']['dwpc_lr']}, {np_['pkg_ind']['dwpc_lr']} and "
         f"{np_['hkg_DpS']['dwpc_lr']}), and three more with degree and prior features. All path methods are thus small models; the "
         "comparison is between ways of weighting the same path evidence, not between models of very different capacity.")

    heading(d, "3.4 Data and tasks", 2)
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
                     f"{np_[t]['dnfr']} / {np_[t]['dwpc_lr']}"])
    table(d, "**Table 1.** Tasks, graphs and path sets. Positives are the known pairs of the target relation; candidates are all nodes "
             "of the tail type, ranked for every test query. Parameters: trainable parameters of DNFR / DWPC-LR.",
          ["Task", "Graph (nodes / edges)", "Target relation", "Positives", "Query heads", "Candidates", "Metapaths used", "Paths",
           "Parameters"], rows, widths=[0.85, 1.0, 0.95, 0.55, 0.5, 0.6, 0.6, 0.75, 0.6], size=7.5,
          note=C("HKG, health knowledge graph learned from emergency-department records [@rotmensch2017]. Edge counts are unique "
                 "undirected edges; PrimeKG excludes anatomy–protein expression edges (Section 3.4)."))
    P(d, f"**Hetionet** (v1.0) [@himmelstein2017] has {DS['het']['nodes']:,} nodes of {DS['het']['node_types']} types and "
         f"{DS['het']['edges']:,} edges of {DS['het']['rel_types']} types. We predict compound–treats–disease ({info['het_CtD']['n_pos']} "
         f"pairs; {info['het_CtD']['n_sources']} compounds with at least one indication, ranked against all {info['het_CtD']['n_tails']} "
         f"diseases) and disease–presents–symptom ({info['het_DpS']['n_pos']:,} pairs, ranked against {info['het_DpS']['n_tails']} symptoms).")
    P(d, f"**PrimeKG** [@chandak2023] has {DS['pkg']['nodes']:,} nodes. We removed the {DS['pkg']['dropped_edges']:,} anatomy–protein "
         "expression edges, which do not lie on drug–disease paths of length three or less but dominate the edge count, leaving "
         f"{DS['pkg']['edges']:,} edges of {DS['pkg']['rel_types']} relation types; all methods use the same graph. We predict "
         f"drug–indication–disease ({info['pkg_ind']['n_pos']:,} pairs from {info['pkg_ind']['n_sources']:,} drugs), ranking every one of "
         f"the {info['pkg_ind']['n_tails']:,} diseases.")
    mu = RS.get("mu_hkg", {})
    P(d, "**Health knowledge graph (HKG).** Rotmensch et al. learned this graph from 273,174 emergency-department visits recorded "
         "between 2008 and 2013 and published it in full [@rotmensch2017]. We parsed the "
         f"{DS['hkg']['edges']:,} disease–symptom edges between {info['hkg_DpS']['n_sources']} diseases and {info['hkg_DpS']['n_tails']} "
         "symptoms from their supplementary appendix. Each edge carries the noisy-OR importance w~e~ = 1 − f, where f is the estimated "
         f"probability that the disease fails to produce the symptom; w~e~ has median {f3(DS['hkg']['w_median'])} (interquartile range "
         f"{f3(DS['hkg']['w_q25'])} to {f3(DS['hkg']['w_q75'])}). Paths between a disease and a symptom run through shared symptoms and "
         "diseases (disease–symptom–disease–symptom), so this task tests composition of graded evidence learned from patient records. "
         "The records come from the emergency department of Beth Israel Deaconess Medical Center, the hospital behind MIMIC-IV "
         "[@johnson2023]; we use the published graph rather than records because the full MIMIC-IV release requires credentialed "
         "access, which we did not hold.")
    P(d, "**Metapaths and computation.** For each task we enumerated all metapaths of length two and three between the head and tail "
         "types and all simple paths that instantiate them from the query heads. To keep training on a two-core CPU feasible, metapaths "
         "with more than five million paths over the query heads were excluded for all path methods: "
         f"{len(DS['het_CtD']['skipped'])} of {DS['het_CtD']['metapaths_all']} metapaths for compound–treats–disease (those through side "
         f"effects and anatomy expression), none for disease–symptom, and {len(DS['pkg_ind']['skipped'])} of "
         f"{DS['pkg_ind']['metapaths_all']} for PrimeKG, mostly through the dense drug–drug interaction and contraindication relations. "
         "The remaining sets contain "
         + ", ".join(f"{DS[t]['paths']:,}" for t in TASKS[:3]) + f" and {DS['hkg_DpS']['paths']:,} paths. A secondary test on a "
         "population relation from NHANES [@truong2026] is described and reported in Appendix B.")

    heading(d, "3.5 Evaluation", 2)
    nd = nseeds("het_DpS", "dnfr"); nh = nseeds("hkg_DpS", "dnfr"); nt = nseeds("het_CtD", "dnfr", "_tail")
    P(d, "Positive pairs of each target relation were split at random into 80% training, 10% validation and 10% test. For the "
         f"comparisons of DNFR with the fixed product and Hamacher product on the two disease–symptom tasks (and of the restarts on the "
         f"graded graph) we used {nd} splits (seeds 0–{nd - 1}); the remaining variants and "
         "ablations of these tasks, which are not the subject of an equivalence claim, use the first five. Within the computing budget "
         f"of a two-core CPU we used three splits for Hetionet compound–treats–disease and two for PrimeKG; embedding models and some "
         "variants were run on fewer splits, as marked in the tables. For each test pair (h, t) we rank t among all candidate tails of "
         "its type after removing the other known tails of h (filtered ranking), break ties by the mean rank, and report mean reciprocal "
         "rank (MRR) and Hits@1, @3 and @10.")
    P(d, "**Disease-centred split.** Random splits let a held-out disease keep other indications in training. For compound–treats–disease "
         "we also split the diseases instead of the pairs (80/10/10 of the diseases with at least one indication), so that every "
         f"indication of a test disease is held out together, as in zero-shot repurposing [@huang2024]; we used {nt} such splits.")
    P(d, "**Statistics and decision rule.** For a comparison between two methods we report the difference in MRR averaged over the "
         "splits on which both were run, a 95% confidence interval from a paired bootstrap over test queries stratified by split (2,000 "
         "resamples), the number of splits on which the difference has the same sign, and the corrected resampled t-test across splits "
         "[@nadeau2003], which accounts for the overlap between training sets of different splits and, unlike the bootstrap, for "
         "variation between splits. Within each task the p-values of all comparisons with DNFR are adjusted with Holm's procedure "
         "[@holm1979]. Most of our claims about t-norms are claims of equivalence, and a non-significant difference is not evidence of "
         "equivalence. Before the analyses of this revision we therefore fixed one rule and apply it to every comparison of t-norms and "
         "restarts: with an equivalence margin of ±0.01 in MRR, a difference is called *equivalent* (≈ in the tables) when the 90% "
         "corrected-t interval lies inside the margin, which is the two one-sided tests procedure at the 5% level; *different* (†) when "
         "the 95% corrected-t interval excludes zero; and *inconclusive* otherwise. The margin is about one tenth of the gain from "
         "hub discounting on Hetionet compound–treats–disease. The verdicts use unadjusted corrected intervals; the Holm-adjusted "
         "p-values in Tables A1–A4 account for multiplicity and are more conservative. "
         "With two or three splits the corrected interval is wide, and most comparisons on Hetionet compound–treats–disease and PrimeKG "
         "are inconclusive by construction. A test query is a (head, tail) pair of one split; the same pair can be a test query in "
         "several random splits, so queries are never pooled across splits for testing.")
    P(d, "**Explanations.** We measure how concise the explanations are (the number of paths and of metapaths that account for 50% and "
         "80% of the evidence −log(1 − P) above the leak) and how much a correct ranking depends on the evidence shown. For all test "
         "queries of the seed-0 split that have at least one path (at most 400 sampled at random) we delete the k paths with the largest "
         "shares (k = 1, 3, 10), k random paths of the same pair, or the leading metapath, and record the new filtered rank with all other "
         "scores fixed. Because the score is additive in path contributions, deleting paths from one pair tests only how much of its "
         "evidence is removed; we therefore also delete random paths until their total share matches that of the top-k paths "
         "(share-matched), and record the rank against the share of evidence removed. Closer to what a clinician would question, an "
         "edge-level counterfactual removes one fact from the graph, the edge that carries the largest total share of the pair's "
         "evidence or, as a control, a random edge used by the pair's paths, drops every path from the query's head that uses it, so "
         "that competing tails lose evidence too, and re-ranks without retraining and with degrees fixed. We also record how often the "
         "three leading paths share an edge, which measures the double counting of Section 3.2. For DWPC-LR the corresponding "
         "attribution is the contribution coef~m~ × x~m~ of each metapath, and we compare the leading metapaths of the two methods. The "
         "score P is the output of a model trained with a softmax ranking loss; it orders candidates but is not calibrated as a "
         "probability, and we do not interpret it as one.")
    P(d, "**Sparsity analysis.** To test robustness to missing knowledge, we removed 50% and 75% of all edges of every relation type at "
         "random, including training edges of the target relation, and repeated training and evaluation of the path methods and of "
         "completion on the thinned graphs, over three splits, with the same held-out test pairs. Removing training edges of the target "
         "relation affects completion more than path methods by construction, since completion uses no other relation. On the thinned "
         "graphs no metapath exceeded the path budget.")
    P(d, "**Choice of defaults and analysis history.** The defaults of DNFR (learning rate 0.1, initial b~r~ = c~r~ = 0.3, p~0~ = −0.1, "
         "λ = 10^−3^, the patience of early stopping) and the path budget of 5 × 10^6^ were set during development, before the "
         "experiments reported here, to values that made training stable and feasible on a two-core CPU; they were not chosen by a "
         "search over values. Development runs used the same graphs, so we cannot exclude that early results on the first split "
         "influenced them, and we report the sensitivity of the conclusions to the defaults that matter for them: the initial value "
         "of p and its penalty (Section 4.3), the stopping rule (Section 4.3) and the fixed degree discount (Section 4.2). The "
         "fixed-degree value 0.4 is the damping exponent of DWPC [@himmelstein2017]. The first version of this "
         "manuscript reported the default model and its ablations. The first revision added the restarts of the t-norm parameter, the "
         "stronger penalties, the fixed-degree and frozen-membership variants, the popularity baseline, AnyBURL, the disease-centred "
         "split and the deletion tests; it also corrected our reading of the default result, which we had first taken as evidence for "
         "the product. This revision adds the profile analysis, training without early stopping, splits 5–9 of the two disease–symptom "
         "tasks for the t-norm comparisons, a fourth split of the disease-centred split, the equivalence rule above, the edge prior and the "
         "DWPC-LR model with degree and prior features, the sensitivity of the fixed degree discount, and the share-matched and "
         "edge-level explanation tests. All result files of every version are released (Section 7).")
    P(d, "**Software, cost and use of AI assistance.** All experiments ran on a shared two-core CPU with 8 GB of memory; we report the "
         "wall-clock time of one training run. The analysis code was written with the help of an AI system (Claude, Anthropic), which "
         "also ran the experiments under the authors' direction. The authors specified the models, protocols and analyses, and checked "
         "the components on which the conclusions rest; the released code includes tests of the leave-dyad-out removal (no held-out "
         "dyad remains in the training graph or on any training path), of the Schweizer–Sklar t-norm against its product, Hamacher, "
         "Łukasiewicz and minimum limits, and of the filtered ranking with ties against a direct computation. Every number reported "
         "below is read from the released result files by the script that builds this manuscript.")
