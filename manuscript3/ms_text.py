"""Front matter, results, discussion, conclusion and declarations (third revision, DNFR 2.0)."""
import numpy as np
from ms_common import *
from docx_helpers import para, heading, bullets, figure
import ms_results as R
from ms_results import r3p, r3sum, mean_any, R3T

P_ = lambda d, t, **k: para(d, C(t), **k)
E = XS["explain_eval"]; E2 = RS["explain2"]; L = RS.get("long", {})
E3 = R3.get("explain3", {}); PS = R3.get("profile_summary") or {}
ADD = [("dnfr_overlap", "R1"), ("dnfr_gumbel", "R2"), ("dnfr_ic", "R3"), ("dnfr_anneal", "R4")]
LOO = [("dnfr2_no_overlap", "R1"), ("dnfr2_no_gumbel", "R2"), ("dnfr2_no_ic", "R3"), ("dnfr2_no_anneal", "R4")]
SHORT = {"het_CtD": "compound–treats–disease", "het_DpS": "Hetionet disease–symptom", "pkg_ind": "PrimeKG",
         "hkg_DpS": "graded graph"}


def SH(t):
    return ("the " if t == "hkg_DpS" else "") + SHORT[t]


def m(t, v, tag=""):
    if tag:
        s = summ(t, v, tag)
        return f3(s["MRR"]["mean"]) if s else "—"
    x = mean_any(t, v)
    return f3(x) if x is not None else "—"


def dci3(p):
    return "—" if p is None else f"{sg(p['delta'])} [{sg(p['bs95'][0])}, {sg(p['bs95'][1])}]"


def nbci(p):
    return f"[{sg(p['nb_ci95_lo'])}, {sg(p['nb_ci95_hi'])}]"


def num(k):
    return ["none", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten", "eleven", "twelve"][k] if k < 13 else str(k)


def Num(k):
    s = num(k)
    return s[0].upper() + s[1:]


def joinand(xs):
    xs = list(xs)
    return xs[0] if len(xs) == 1 else ", ".join(xs[:-1]) + " and " + xs[-1]


def tasks_with(f):
    return [t for t in TASKS if f(t)]


def verdict_of(p):
    return None if p is None else p.get("verdict")


def comp_support(a, b):
    """Tasks on which a minus b is positive, and on which it is 'different' under the rule."""
    run = [t for t in TASKS if pr(t, a, b)]
    pos = [t for t in run if pr(t, a, b)["delta_MRR"] > 0 and pr(t, a, b)["seed_wins"] == pr(t, a, b)["seeds"]]
    dif = [t for t in run if verdict_of(pr(t, a, b)) == "different" and pr(t, a, b)["delta_MRR"] > 0]
    return run, pos, dif


def refine_counts():
    """Verdicts of the add-one comparisons over the three tasks."""
    out = dict(pos_diff=[], neg_diff=[], equivalent=[], inconclusive=[], n=0, deltas=[])
    for t in R3T:
        for v, lab in ADD + [("product_overlap", "R1 on product")]:
            b = "product" if v == "product_overlap" else "dnfr"
            p = r3p(t, v, b)
            if p is None:
                continue
            out["n"] += 1; out["deltas"].append(p["delta"])
            vd = p.get("verdict")
            if vd == "different":
                out["pos_diff" if p["delta"] > 0 else "neg_diff"].append((t, lab))
            elif vd == "equivalent":
                out["equivalent"].append((t, lab))
            else:
                out["inconclusive"].append((t, lab))
    return out


def best_tail():
    return max(summ("het_CtD", x, "_tail")["MRR"]["mean"] for x in ("dnfr", "product", "hamacher", "dnfr_nodeg", "dwpc_lr")
               if summ("het_CtD", x, "_tail"))


def ps_by(p, k):
    return PS["by_p"][f"{p:g}"][k]


# ------------------------------------------------------------------ front matter
TITLE = ("Which parts of a neuro-fuzzy path reasoner should be learned? A controlled component analysis on medical knowledge "
         "graphs, with four refinements (DNFR 2.0)")


def title_block(dc):
    para(dc, f"**{TITLE}**", align="left", size=14, line=1.3, space_after=10)
    para(dc, "**Article type:** Research Paper", align="left")
    para(dc, "**Running title:** Neuro-fuzzy path reasoning on medical knowledge graphs", align="left")
    para(dc, "**Authors**", align="left", space_after=2)
    para(dc, "Truong Hai Bang^1^*, Van An Phan Thi^2^", align="left")
    para(dc, "^1^ Faculty of Engineering and Computer Science, Saigon International University, 8C Tong Huu Dinh Street, An Khanh Ward, "
             "Ho Chi Minh City, Vietnam. ORCID: https://orcid.org/0000-0001-5094-3128", align="left", size=11)
    para(dc, "^2^ VNU University of Education, Vietnam National University, Hanoi, G7 Building, No. 144 Xuan Thuy Street, Cau Giay "
             "District, Hanoi, Vietnam. Email: phanthivanantlls@gmail.com. ORCID: https://orcid.org/0009-0005-4135-5689",
         align="left", size=11)
    para(dc, "* Corresponding author: Truong Hai Bang, truonghaibang@siu.edu.vn", align="left", size=11)


def highlights_list():
    rc = refine_counts()
    run, pos, dif = comp_support("dnfr", "dnfr_max")
    h = [f"Noisy-OR over all paths beat the strongest path on all {num(len(run))} tasks tested."
         if len(pos) == len(run) else "Noisy-OR over all paths beat the strongest single path on most tasks.",
         "Hub discounting raised accuracy on every task and every split." if len(comp_support("dnfr", "dnfr_nodeg")[1]) == len(comp_support("dnfr", "dnfr_nodeg")[0])
         else "Hub discounting helped on most tasks tested.",
         ("None of four refinements (DNFR 2.0) improved accuracy detectably." if not rc["pos_diff"]
          else f"{Num(len(rc['pos_diff']))} of {rc['n']} refinement tests showed a detectable gain."),
         "The t-norm parameter is practically non-identified by the training loss.",
         "Degree-weighted path counts and NBFNet matched or beat the learned operator."]
    for x in h:
        assert len(x) <= 85, (len(x), x)
    return h


def highlights(dc, H):
    para(dc, "**Highlights**", align="left", space_after=2)
    bullets(dc, H)


def abstract(dc):
    rc = refine_counts()
    lr = PS.get("loss_range") or [float("nan")]
    fv = r3p("hkg_DpS", "fixp_val", "product")
    para(dc, "**Abstract**", align="left", space_after=2)
    P_(dc, "**Objective:** Path-based reasoning over medical knowledge graphs explains its predictions but composes evidence "
           "with fixed operators. We asked which parts of a differentiable neuro-fuzzy path reasoner are worth learning, "
           "and whether four refinements improve it.")
    P_(dc, "**Methods:** Differentiable Neuro-Fuzzy Reasoning (DNFR) learns degree-discounting edge memberships, a Schweizer–Sklar "
           "t-norm per metapath and a noisy-OR over paths. DNFR 2.0 adds overlap-aware aggregation, a discrete operator choice, an "
           "information-content hub discount and membership annealing. On drug repurposing (Hetionet, PrimeKG) and disease–symptom "
           "prediction (Hetionet; a graph learned from emergency-department records) we compared ablations, fixed and "
           "validation-selected t-norms, the refinements, NBFNet and simpler baselines under leave-dyad-out splits, and profiled "
           "the t-norm parameter to convergence.")
    r_max = [pr(t, "dnfr", "dnfr_max")["delta_MRR"] for t in TASKS if pr(t, "dnfr", "dnfr_max")]
    txt = (f"**Results:** Aggregating all paths instead of the strongest one raised mean reciprocal rank (MRR) by "
           f"{min(r_max):.2f}–{max(r_max):.2f}, and hub discounting helped on every task. "
           + ("No refinement improved MRR detectably" if not rc["pos_diff"] else
              f"{Num(len(rc['pos_diff']))} of {rc['n']} refinement tests showed a gain")
           + (", and two lowered it on the graded graph" if sum(1 for t, l in rc["neg_diff"] if t == "hkg_DpS") >= 2 else "")
           + ("; combined, they lowered MRR by " + f"{abs(r3p('het_DpS', 'dnfr2', 'dnfr')['delta']):.3f} on Hetionet disease–symptom"
              if (r3p("het_DpS", "dnfr2", "dnfr") or {}).get("verdict") == "different" else "")
           + ". " + f"With everything else fitted to convergence, the training loss varied by at most {max(lr):.3f} nats over "
           "t-norm parameters from −5 to 0, with split-dependent minimisers; a t-norm selected on validation data beat the "
           f"fixed product by {fv['delta']:.3f} on the graded graph. Degree-weighted path counts beat DNFR on two tasks and "
           "were not detectably worse on a third; a compact NBFNet came within "
           f"{abs(r3p('hkg_DpS', 'nbfnet', 'dnfr')['delta']):.3f} of DNFR on the graded graph.")
    P_(dc, txt)
    P_(dc, "**Conclusion:** Path aggregation and hub discounting carry the accuracy. The training loss barely constrains the "
           "composition operator, so a learned operator should not be read as a property of the domain; it is better chosen on "
           "validation data.")


def keywords(dc):
    P_(dc, "**Keywords:** knowledge graph; neuro-fuzzy reasoning; t-norm; identifiability; explainable AI; drug repurposing", align="left")


def significance(dc):
    from docx_helpers import table
    table(dc, "**Statement of significance**", ["Heading", "Content"], [
        ["Problem or issue", "Path-based reasoning over medical knowledge graphs is explainable, but it combines evidence along a path "
                             "with an operator fixed in advance, and sparse graphs with hub nodes make that choice consequential."],
        ["What is already known", "Path methods such as degree-weighted path counts and rule learners explain their predictions but fix "
                                  "the composition operator, usually a product; neuro-fuzzy systems and differentiable fuzzy logic allow "
                                  "the operator to be learned, but whether a learned operator is determined by medical graph data has not "
                                  "been tested."],
        ["What this paper adds", "A controlled component analysis of a differentiable neuro-fuzzy path reasoner, and of four "
                                 "refinements of it, on four link-prediction tasks from three medical knowledge graphs. Aggregating all "
                                 "paths and discounting hub nodes carried the gains; none of the refinements helped detectably. The "
                                 "training loss barely depends on the t-norm parameter once memberships are learned, so a learned "
                                 "operator is not a property of the domain, while selecting it on validation data does help. "
                                 "Degree-weighted path counts were as accurate on three of four tasks."],
        ["Who would benefit", "Developers and evaluators of explainable clinical decision support built on knowledge graphs, and "
                              "researchers who learn or report fuzzy operators for medical reasoning."]],
        widths=[1.5, 4.77], size=9)


# ------------------------------------------------------------------ results
def results(dc):
    heading(dc, "4. Results", 1)
    # ---------------- 4.1
    heading(dc, "4.1 Accuracy against path, rule, completion and degree-only baselines (RQ1)", 2)
    PSm = R3.get("prior_summary", {})
    P_(dc, "Table 2 summarises test MRR. Because the tasks rank different numbers of candidates, from 137 diseases to 17,080, we "
           "compare methods within a task and not across tasks. Degree alone explained little. Popularity, which ranks every tail "
           "by its number of training positives, and the edge prior of Zietz et al. [@zietz2024], which uses the degrees of both "
           "endpoints, were far behind every path method on every task (Supplementary Tables S1–S4).")
    if PSm:
        c = PSm["het_CtD"]
        P_(dc, "The edge prior was below popularity, which looks wrong at first sight, because for a fixed head the prior is close "
               "to monotone in the degree of the tail. We checked it (Supplementary Table S9). On the two disease–symptom tasks the "
               "gap was a matter of resolution: with 1000 permutations instead of 100, or with the analytical approximation of the "
               "prior, which ranks tails exactly as popularity does, the prior reached "
               f"{f3(PSm['het_DpS']['prior1000_MRR'])} and {f3(PSm['hkg_DpS']['prior1000_MRR'])} against "
               f"{f3(PSm['het_DpS']['popularity_MRR'])} and {f3(PSm['hkg_DpS']['popularity_MRR'])} for popularity. On "
               "compound–treats–disease the gap remained "
               f"({f3(c['analytic_MRR'])} analytic, {f3(c['popularity_MRR'])} popularity). There, for "
               f"{pct(c['share_zero_true'])} of test queries the true pair had a prior of zero: most compounds have one or two "
               "indications, so under leave-dyad-out splits many test compounds have no training indication left, the prior of "
               "every candidate is zero, and all candidates tie. Popularity still orders the diseases. The prior is therefore "
               "correct, and weak for a reason that follows from the split rather than from the implementation.")
    P_(dc, "Among path methods, the fixed weighting of DWPC with logistic regression was at least as accurate as DNFR on three of the "
           f"four tasks. On Hetionet drug repurposing DNFR minus DWPC-LR was {dci(pr('het_CtD','dnfr','dwpc_lr'))} "
           f"({pr('het_CtD','dnfr','dwpc_lr')['seed_wins']}/{pr('het_CtD','dnfr','dwpc_lr')['seeds']} splits in favour of DNFR, "
           f"inconclusive); on the two disease–symptom tasks DWPC-LR was ahead by {sg(-pr('het_DpS','dnfr','dwpc_lr')['delta_MRR'])[1:]} "
           f"and {sg(-pr('hkg_DpS','dnfr','dwpc_lr')['delta_MRR'])[1:]}"
           + (", a difference under our rule on both" if all(verdict_of(pr(t, 'dnfr', 'dwpc_lr')) == 'different' for t in ('het_DpS', 'hkg_DpS')) else "")
           + f". Only on PrimeKG was DNFR ahead ({dci(pr('pkg_ind','dnfr','dwpc_lr'))}, two splits, which is too few for a "
           "verdict). Adding degree and prior features to DWPC-LR, as in Rephetio, did not help on any task. Completion of the "
           f"target relation alone was far behind on Hetionet drug repurposing (Soft-Impute {m('het_CtD','softimpute')}), which "
           f"gives it no access to the rest of the graph, came close on PrimeKG ({m('pkg_ind','softimpute')}), where drugs have "
           f"more indications, and led on the dense graded graph ({m('hkg_DpS','softimpute')} against {m('hkg_DpS','dnfr')} for "
           "DNFR), where a disease is linked to a symptom through diseases with similar symptom profiles, the setting in which a "
           f"low-rank description should work well. AnyBURL, given 100 s of rule learning, reached {m('het_CtD','anyburl')}, "
           f"{m('het_DpS','anyburl')} and {m('hkg_DpS','anyburl')} on the three tasks on which it ran, below DNFR and DWPC-LR on "
           "each. The untuned embeddings were far below every path method; since their training was not tuned, we do not "
           "interpret that gap. We did not train embeddings on the graded graph: with a single bipartite relation an embedding "
           "reduces to a low-rank factorisation of that relation, which Soft-Impute provides. PrimeKG accuracies of all path methods are lower bounds, because the metapaths with more than five "
           "million paths, the most hub-dominated ones, were excluded (Section 3.5).")
    R.table2(dc)
    nb = {t: r3p(t, "nbfnet", "dnfr") for t in R3T}
    nbs = {t: r3sum(t, "nbfnet") for t in R3T}
    P_(dc, "NBFNet, which learns both how a path is extended and how paths are aggregated, is the closest learned counterpart to the "
           "question we ask. Our compact CPU version (three layers of dimension 32, Section 3.4) reached "
           f"{m('hkg_DpS','nbfnet')} on the graded graph over {nbs['hkg_DpS']['n']} splits, against {m('hkg_DpS','dnfr')} for DNFR "
           f"(NBFNet minus DNFR {dci3(nb['hkg_DpS'])}) and {m('hkg_DpS','dwpc_lr')} for DWPC-LR, and "
           + (f"{f3(nbs['het_DpS']['mean'])} on Hetionet disease–symptom over {num(nbs['het_DpS']['n'])} "
              f"split{'s' if nbs['het_DpS']['n'] > 1 else ''} (DNFR minus NBFNet on the same "
              f"split{'s' if nbs['het_DpS']['n'] > 1 else ''} {sg(-nb['het_DpS']['delta'])})" if nbs.get("het_DpS") else "was not run on Hetionet disease–symptom")
           + ". A learned operator for path extension and aggregation, with about "
           "ten thousand parameters, thus did not do detectably better than DNFR on these tasks (inconclusive on both under our "
           "rule). It was "
           "not run on the drug-repurposing graphs, whose size exceeds what our CPU implementation can train in reasonable time.")

    # ---------------- 4.2
    heading(dc, "4.2 What each learned component adds (RQ2)", 2)
    P_(dc, "Table 3 isolates the components of DNFR, all trained with the same protocol.")
    R.table3(dc)
    rn, pn, dn = comp_support("dnfr", "dnfr_max")
    P_(dc, "**Aggregating paths.** Keeping only the strongest path instead of the noisy-OR of all paths lowered MRR by "
           + joinand([f"{sg(pr(t,'dnfr','dnfr_max')['delta_MRR'])[1:]} on {SH(t)} ({pr(t,'dnfr','dnfr_max')['seed_wins']}/{pr(t,'dnfr','dnfr_max')['seeds']} splits)" for t in rn])
           + f". The difference was supported under our rule on {joinand([SH(t) for t in dn]) if dn else 'none of the tasks'}"
           + (f" and inconclusive on {joinand([SH(t) for t in rn if t not in dn])}" if len(dn) < len(rn) else "")
           + ". The number of routes between two entities carries information that the best single route does not. The ablation was "
           "not run on PrimeKG.")
    rd, pd_, dd_ = comp_support("dnfr", "dnfr_nodeg")
    fd = {t: pr(t, "dnfr", "dnfr_fixdeg") for t in TASKS}
    s02, s06 = pr("hkg_DpS", "product_fixdeg", "product_fixdeg02"), pr("hkg_DpS", "product_fixdeg", "product_fixdeg06")
    P_(dc, "**Discounting hubs.** Removing the degree terms of Equation 1 lowered MRR by "
           + joinand([f"{sg(pr(t,'dnfr','dnfr_nodeg')['delta_MRR'])[1:]} on {SH(t)} ({pr(t,'dnfr','dnfr_nodeg')['seed_wins']}/{pr(t,'dnfr','dnfr_nodeg')['seeds']})" for t in rd])
           + f", with the same sign on every split; under our rule the difference was supported on "
           f"{joinand([SH(t) for t in dd_]) if dd_ else 'none of the tasks'}"
           + (f", while on {joinand([SH(t) for t in rd if t not in dd_])} the corrected test did not reach significance after "
              "adjustment" if len(dd_) < len(rd) else "") + ". "
           "Whether the discount has to be learned per relation depended on the graph. Learned coefficients were better than a "
           f"fixed b~r~ = c~r~ = 0.4 on the graded graph ({dci(fd['hkg_DpS'])}); elsewhere the fixed discount was at least as good "
           f"(learned minus fixed {dci(fd['het_CtD'])} on compound–treats–disease, {dci(fd['het_DpS'])} on Hetionet disease–symptom "
           f"and {dci(fd['pkg_ind'])} on PrimeKG). On the graded graph the fixed product with discount 0.4 was "
           f"{sg(s02['delta_MRR'])} against a discount of 0.2 and {sg(s06['delta_MRR'])} against 0.6, so the DWPC value is not a "
           f"tuned optimum. Reading the graded weights added {dci(pr('hkg_DpS','dnfr','dnfr_binary'))} on that graph.")
    tp = {t: pr(t, "dnfr", "product") for t in TASKS}; th = {t: pr(t, "dnfr", "hamacher") for t in TASKS}
    tn = [(t, b) for t in TASKS for b in ("product", "hamacher") if pr(t, "dnfr", b)]
    vc = {k: sum(1 for t, b in tn if verdict_of(pr(t, "dnfr", b)) == k) for k in ("different", "equivalent", "inconclusive")}
    mx = max(abs(pr(t, "dnfr", b)["delta_MRR"]) for t, b in tn)
    P_(dc, "**Learning the t-norm.** A Schweizer–Sklar parameter learned per metapath was not consistently better than fixed members "
           "of its own family. Against the fixed product the differences were " + joinand([f"{sg(tp[t]['delta_MRR'])} on {SH(t)}" for t in TASKS])
           + "; against the fixed Hamacher product " + joinand([sg(th[t]['delta_MRR']) for t in TASKS]) + f". Of these {num(len(tn))} "
           f"comparisons, {num(vc['different'])} {'was' if vc['different'] <= 1 else 'were'} different under our rule, "
           f"{num(vc['equivalent'])} equivalent and {num(vc['inconclusive'])} inconclusive; the largest difference in either "
           f"direction was {mx:.3f}. We therefore claim that learning the t-norm gave no consistent gain, not that fixed and "
           "learned t-norms are equivalent. One p per metapath differed from a single shared p by "
           + joinand([sg(pr(t, 'dnfr', 'dnfr_shared')['delta_MRR']) for t in TASKS if pr(t, 'dnfr', 'dnfr_shared')])
           + " where the shared variant was run.")
    fr = {t: pr(t, "dnfr_frozen", "product_frozen") for t in TASKS}; lm = {t: pr(t, "product", "product_frozen") for t in TASKS}
    P_(dc, "The t-norm mattered most when the memberships could not be learned. With memberships frozen at their initial values, "
           f"learning p improved on the product by {dci(fr['hkg_DpS'])} on the graded graph and by {dci(fr['het_DpS'])} on Hetionet "
           f"disease–symptom, and learning the memberships instead of the t-norm improved the product by {dci(lm['hkg_DpS'])} on the "
           "graded graph. Memberships and operator are, to a large extent, substitutes in this model: both rescale the strength of "
           "a path, as in flexible neuro-fuzzy systems in which weights and t-norm parameters scale the strength of a rule "
           "[@rutkowski2003]. This is a structural property of the model, and Section 4.4 asks what it implies for learning p.")

    # ---------------- 4.3 refinements
    heading(dc, "4.3 Four refinements (DNFR 2.0) and a validation-selected t-norm (RQ2)", 2)
    rc = refine_counts()
    P_(dc, "Table 4 and Figure 3 report the four refinements of Section 3.3, each added to DNFR alone and removed from DNFR 2.0, "
           f"on the three tasks on which we ran them. Of the {num(rc['n'])} add-one comparisons, "
           + (f"none showed a gain under our rule" if not rc["pos_diff"] else f"{num(len(rc['pos_diff']))} showed a gain under our rule")
           + (f", {num(len(rc['neg_diff']))} showed a loss" if rc["neg_diff"] else "")
           + (f", {num(len(rc['equivalent']))} {'was' if len(rc['equivalent']) == 1 else 'were'} equivalent within ±0.01" if rc["equivalent"] else "")
           + f" and {num(len(rc['inconclusive']))} were inconclusive; the point estimates ranged from {sg(min(rc['deltas']))} to "
           f"{sg(max(rc['deltas']))}. The refinements were plausible on paper, and we expected some of them to help. They did not, "
           "and the reasons differ between them.")
    R.table4(dc)
    figure(dc, f"{ROOT}/figures/fig3_refinements.png", 6.3,
           "**Figure 3.** The refinements of Section 3.3 and the validation-selected fixed t-norm: difference in test MRR with its "
           "95% paired bootstrap interval, per task. Grey band, the equivalence margin ±0.01. Colour: blue, equivalent; red, "
           "different; grey, inconclusive (rule of Section 3.6). Numbers at the right give the splits run.")
    bt = R3.get("hkg_DpS", {}).get("beta", {})
    med = lambda x: float(np.median(x)) if x else float("nan")
    ov = {t: r3p(t, "dnfr_overlap", "dnfr") for t in R3T}; po = {t: r3p(t, "product_overlap", "product") for t in R3T}
    P_(dc, "**R1, overlap-aware aggregation.** Added to DNFR, the overlap penalty changed MRR by "
           + joinand([f"{sg(ov[t]['delta'])} on {SH(t)}" for t in R3T if ov[t]])
           + ". The learned exponent β, initialised at 1, fell to a median of "
           f"{f3(med(bt.get('dnfr_overlap')), 3)} on the graded graph (and {f3(med(bt.get('dnfr2')), 3)} within DNFR 2.0): when the "
           "memberships are learned, training switches the penalty off. With memberships tied to the fixed product, β stayed "
           f"larger (median {f3(med(bt.get('product_overlap')), 2)}) and the penalty cost "
           + joinand([f"{sg(po[t]['delta'])[1:]} on {SH(t)}" for t in R3T if po[t]])
           + ". Evidence that the independence assumption of the noisy-OR hurts would have shown up as a gain here; instead, paths "
           "through a shared node appear to carry information that a discount removes, consistent with the gain from aggregation "
           "itself.")
    gu = {t: r3p(t, "dnfr_gumbel", "dnfr") for t in R3T}
    ops = R3.get("hkg_DpS", {}).get("ops_single") or {}
    ra = {t: R3.get(t, {}).get("restarts", {}).get("dnfr2") for t in R3T}
    txt = ("**R2, discrete operator choice.** Choosing among the product, the minimum and the Hamacher product instead of "
           "learning a continuous p changed MRR by " + joinand([f"{sg(gu[t]['delta'])} on {SH(t)}" for t in R3T if gu[t]])
           + (" (equivalent within ±0.01 on " + joinand([SH(t) for t in R3T if gu[t] and gu[t]["verdict"] == "equivalent"]) + ")"
              if any(gu[t] and gu[t]["verdict"] == "equivalent" for t in R3T) else "") + ". ")
    if ops:
        n = sum(ops.values())
        txt += ("On the graded graph, with its single metapath, DNFR 2.0 chose the Hamacher product in "
                f"{ops.get('hamacher', 0)} of {n} splits and the minimum in {ops.get('min', 0)}, never the product, in line with the "
                "profile below, which favours t-norms stronger than the product there. ")
    if any(ra.values()):
        txt += ("Restarted with different random seeds, the operator chosen for a metapath was the same in all three restarts for "
                + joinand([f"{pct(ra[t]['all_same_mean'])} of metapaths on {SH(t)}" for t in R3T if ra[t]]) + ". ")
    txt += ("A discrete choice is easier to report than a continuous parameter, but it does not make the operator more "
            "accurate.")
    P_(dc, txt)
    ic = {t: r3p(t, "dnfr_ic", "dnfr") for t in R3T}; icl = {t: r3p(t, "dnfr2", "dnfr2_no_ic") for t in R3T}
    gl = {t: r3p(t, "dnfr2", "dnfr2_no_gumbel") for t in R3T}
    P_(dc, "**R3, information-content discount.** Replacing the degree feature by the logarithm of the number of paths through a node "
           "changed MRR by " + joinand([f"{sg(ic[t]['delta'])} on {SH(t)} ({ic[t]['wins']}/{ic[t]['J']} splits)" for t in R3T if ic[t]])
           + ". Within DNFR 2.0 the same feature was harmless (removing it changed MRR by "
           + joinand([sg(-icl[t]["delta"]) for t in R3T if icl[t]]) + "), and DNFR 2.0 without the discrete operator choice lost "
           + joinand([sg(-gl[t]["delta"])[1:] for t in R3T if gl[t]]) + ": the information-content feature hurt with a continuous "
           "Schweizer–Sklar parameter and not with a choice among fixed operators. We did not investigate this interaction further. "
           "Path occurrence and degree are strongly correlated in these graphs, so there is little that the information content "
           "could add to the degree discount.")
    sat = R3.get("saturation", {})
    an = {t: r3p(t, "dnfr_anneal", "dnfr") for t in R3T}
    tb = R3.get("hkg_DpS", {}).get("tau_best", {})
    txt = ("**R4, membership annealing.** Annealing the temperature of the memberships changed MRR by "
           + joinand([f"{sg(an[t]['delta'])} on {SH(t)} ({an[t]['wins']}/{an[t]['J']} splits)" for t in R3T if an[t]])
           + (" (not run on compound–treats–disease)" if not an.get("het_CtD") else "") + ". ")
    if sat:
        h, dps = sat["hkg_DpS"][0], sat["het_DpS"][0]
        txt += (f"It addresses a problem that the trained models barely have: in the default model of the first split, "
                f"{100 * h['mu_gt_099']:.1f}% of the memberships on paths of the graded graph and {100 * dps['mu_gt_099']:.1f}% on "
                f"Hetionet disease–symptom exceeded 0.99, the region where the gradient of log μ vanishes, whereas "
                f"{pct(h['mu_lt_001'])} and {pct(dps['mu_lt_001'])} were below 0.01, where it does not. ")
    if tb.get("dnfr_anneal"):
        txt += (f"The best validation epoch fell at a temperature of {f3(med(tb['dnfr_anneal']), 2)} (median) on the graded "
                "graph, after most of the annealing, so the schedule mainly changed the final temperature. ")
    P_(dc, txt)
    v2 = {t: r3p(t, "dnfr2", "dnfr") for t in R3T}; v2p = {t: r3p(t, "dnfr2", "product") for t in R3T}
    P_(dc, "**DNFR 2.0.** With all four refinements, DNFR 2.0 differed from DNFR by "
           + joinand([f"{dci3(v2[t])} on {SH(t)}" for t in R3T if v2[t]]) + ", and from the fixed product by "
           + joinand([sg(v2p[t]["delta"]) for t in R3T if v2p[t]])
           + ". None of these differences was supported under our rule"
           * (not any(v2[t] and v2[t]["verdict"] == "different" for t in R3T))
           + "".join(f". On {SH(t)} DNFR 2.0 was less accurate than DNFR under our rule ({v2[t]['wins']}/{v2[t]['J']} splits "
                     "in its favour)" for t in R3T if v2[t] and v2[t]["verdict"] == "different" and v2[t]["delta"] < 0)
           + ". We keep the name for the combined model so that the four modifications can be cited, but we do not recommend it "
           "over DNFR or over the fixed product.")
    fv = {t: r3p(t, "fixp_val", "dnfr") for t in R3T}; fvp = {t: r3p(t, "fixp_val", "product") for t in R3T}
    ch = R3.get("hkg_DpS", {}).get("fixp_val_choice") or {}
    P_(dc, "**A fixed t-norm selected on validation data.** Training with a fixed shared p from a grid and keeping, on each split, "
           "the value with the best validation MRR is the procedure that the identifiability analysis below recommends; it uses the "
           "same early stopping as every other variant. It differed from DNFR by "
           + joinand([f"{dci3(fv[t])} on {SH(t)}" for t in R3T if fv[t]]) + " and from the fixed product by "
           + joinand([f"{dci3(fvp[t])} on {SH(t)}" for t in R3T if fvp[t]])
           + (f"; the gain over the product on the graded graph was supported under our rule" if fvp.get("hkg_DpS") and fvp["hkg_DpS"]["verdict"] == "different" else "")
           + (f". On the graded graph the grid value chosen was −0.5 in {ch.get('fixp_m05', 0)} splits, −2 in {ch.get('fixp_m2', 0)} "
              f"and −4 in {ch.get('fixp_m4', 0)}. A fixed p = −2 trained with early stopping reached {m('hkg_DpS','fixp_m2')}, below "
              "the 0.338 that the first, unconverged profile reported without early stopping, so part of that earlier gain was due "
              "to longer training rather than to the value of p" if ch else "") + ".")

    # ---------------- 4.4 identifiability
    heading(dc, "4.4 Is the t-norm parameter identified by the training objective? (RQ3)", 2)
    pv = S["het_CtD"]["p_by_variant"]["dnfr"]
    bt_ = {t: RS["stopping"][t]["dnfr"]["best_epoch"]["median"] for t in TASKS if t in RS["stopping"]}
    P_(dc, "Two questions need to be kept apart. One is structural: whether the model can produce the same scores with different "
           "values of p, because the memberships compensate; Section 4.2 showed that they partly can. The other is practical: "
           "whether the training data, through the training loss, determine p within a useful range [@raue2009]. Table 5 and "
           "Figure 2 address the second.")
    P_(dc, "**Restarts.** With the default early stopping, the learned p ended near its starting value (Table 5). On "
           f"compound–treats–disease the best validation epoch had a median of {ri(bt_['het_CtD'])}; with learning rate 0.1 and "
           "dp/dθ ≈ 0.9 near p~0~ = −0.1, that many Adam steps could move p by close to one unit, yet the learned values had an "
           f"interquartile range of only [{f3(pv['q25'], 2)}, {f3(pv['q75'], 2)}]. The small spread is an empirical outcome, a "
           "small and inconsistent gradient, not a mechanical consequence of early stopping. Restarts from p~0~ = −2, −1 and 0.5 "
           "ended in different places and reached similar accuracy, except that starts in the nilpotent range (p~0~ = 0.5) "
           "failed on the graded graph.")
    F = {t: R3.get(t, {}).get("free_p") for t in TASKS}
    if any(F.values()):
        ft = [t for t in TASKS if F[t]]
        P_(dc, "**Unbounded parameter.** The bounded parameterisation p = 6σ(θ) − 5 has a small derivative near its bounds, and some "
               "learned values sat close to them. With p = θ, clipped only to keep Equation 2 finite, the learned values still "
               "depended on the start: " + joinand([
                   f"on {SH(t)}, median {f3(np.median([r['median_by_init']['dnfr_free_pm2'] for r in F[t]]), 2)} from −2, "
                   f"{f3(np.median([r['median_by_init']['dnfr_free'] for r in F[t]]), 2)} from −0.1 and "
                   f"{f3(np.median([r['median_by_init']['dnfr_free_pp05'] for r in F[t]]), 2)} from 0.5" for t in ft])
               + ", with a median spread of " + joinand([f"{f3(np.median([r['spread_median'] for r in F[t]]), 2)}" for t in ft])
               + " between starts for the same metapath. The lack of convergence is not an artefact of the parameterisation.")
    if PS:
        iv = PS["intervals"]
        P_(dc, "**Converged profile.** On the graded graph we fixed a shared p on a grid from −5 to 0 and fitted every other "
               "parameter by L-BFGS until the loss changed by less than 10^−9^ (Section 3.4). Over the whole range the converged "
               "training loss varied by " + joinand([f"{x:.4f}" for x in PS["loss_range"]]) + " nats in the "
               f"{num(len(PS['seeds']))} splits, against a loss of about 4.3, and |∂L/∂p| at the converged fits was at most "
               f"{PS['max_abs_dLdp']:.4f}. The minimiser was not reproducible: the loss was lowest at p = "
               + joinand([f"{r['p_argmin']:g}".replace("-", "−") for r in iv]) + " in the three splits, and the sets of p within "
               "χ²(0.95, 1)/(2N) of each split's minimum, a profile-likelihood style interval that we use descriptively, were "
               + joinand([f"[{r['p_lo']:g}, {r['p_hi']:g}]".replace("-", "−") if r["p_lo"] != r["p_hi"] else f"{{{r['p_lo']:g}}}".replace("-", "−") for r in iv])
               + (", which do not all overlap" if max(r["p_lo"] for r in iv) > min(r["p_hi"] for r in iv) else "")
               + " (Figure 2; Supplementary Tables S7 and S8). Only one split had an interior minimum, with curvature "
               + (f"∂²L/∂p² ≈ {[c for c in PS['curvature'] if c][0]:.3f}" if any(PS["curvature"]) else "too small to estimate")
               + ". In the terminology of Raue et al. [@raue2009], p is practically non-identified by the training objective once "
               "the memberships are learned.")
        P_(dc, "Held-out accuracy was more informative than the loss. Validation MRR at the converged fits was highest at p = "
               + joinand([f"{r['p_best_valid']:g}".replace("-", "−") for r in iv]) + ", and test MRR at the value chosen on "
               "validation exceeded that of the product by " + joinand([sg(r["test_at_valid_choice"] - r["test_product"]) for r in iv])
               + ". The data therefore carry information about p, but the softmax loss, averaged over all candidates, hardly "
               "registers the reordering of the few top-ranked ones that decides MRR. That is why selecting p on validation data "
               "helps (Section 4.3) while learning it by gradient descent does not.")
    figure(dc, f"{ROOT}/figures/fig2_profile_lbfgs.png", 6.3,
           "**Figure 2.** Converged profile over a fixed shared t-norm parameter p on the graded graph, three splits. Left: training "
           "loss minus its minimum over the grid (×10⁻³ nat). Centre: ∂L/∂p at the converged fit. Right: test MRR of the converged "
           "fit. All other parameters were fitted by L-BFGS to convergence. Dashed lines mark the Hamacher product (p = −1) and the "
           "product (p = 0).")
    R.table5(dc)
    P_(dc, "Two further observations complete the picture. Training for 400 epochs without early stopping did not make the learned "
           "p converge either (Supplementary Table S6), and on the graded graph the nilpotent end of the family is excluded by the "
           f"data: the Łukasiewicz t-norm lost {sg(-pr('hkg_DpS','dnfr','luk')['delta_MRR'])[1:]} in MRR, because every path there "
           "contains a symptom-to-disease edge with a small membership, and for p > 0 such a path has strength exactly zero. The "
           "per-metapath values learned from different starts were at best weakly correlated (median Spearman correlation "
           f"{f3(S['het_DpS']['p_agreement']['spearman_median'], 2)} on Hetionet disease–symptom), so no single learned value, "
           "of any metapath, should be interpreted.")

    # ---------------- 4.5 held-out diseases
    heading(dc, "4.5 Held-out diseases", 2)
    tg = "_tail"; nt = nseeds("het_CtD", "dnfr", tg)
    pn = pr("het_CtD", "dnfr_nodeg", "dnfr", tg)
    P_(dc, "A random split lets a test disease keep other indications in training. When every indication of a test disease was held "
           f"out together (Table 6, {nt} splits), all methods fell sharply: DNFR reached {m('het_CtD','dnfr',tg)}, against "
           f"{m('het_CtD','dnfr')} with random splits. Path methods remained ahead of completion, popularity and the edge prior, which "
           f"carry no information about a disease without training indications ({m('het_CtD','softimpute',tg)}, "
           f"{m('het_CtD','popularity',tg)} and {m('het_CtD','prior',tg)}). Degree and prior features hurt DWPC-LR here "
           f"({m('het_CtD','dwpc_lr_full',tg)} against {m('het_CtD','dwpc_lr',tg)}), because a held-out disease has degree zero. "
           f"The variant without degree terms reached {m('het_CtD','dnfr_nodeg',tg)}, {dci(pn)} relative to DNFR, ahead on "
           f"{pn['seed_wins']} of {pn['seeds']} splits. A reversal of the random-split result is plausible, since for a disease "
           "without indications the remaining routes often pass through well-connected genes and drugs that the degree terms "
           f"discount, but the corrected interval {nbci(pn)} does not establish it, and we ran this split on one task only.")
    R.table_tail(dc)

    # ---------------- 4.6 sparsity
    heading(dc, "4.6 Robustness to missing edges", 2)
    P_(dc, "Removing edges at random lowered every method, and path methods stayed ahead of completion (Figure 4; Supplementary "
           f"Table S5). With a quarter of the edges left, DNFR reached {m('het_CtD','dnfr','_sp0.25')} on compound–treats–disease, "
           f"against {m('het_CtD','softimpute','_sp0.25')} for Soft-Impute. This comparison favours path methods by construction, "
           "because completion loses the very edges it is fitted to. The advantage of the degree terms vanished on the thinnest "
           f"graphs ({m('het_CtD','dnfr_nodeg','_sp0.25')} without them at 25% of the edges), consistent with their role: once most "
           "edges are gone, few nodes remain hubs."
           + (" To separate the robustness of path evidence from the loss of training edges, we also thinned every relation of "
              "Hetionet disease–symptom except the target relation to 25%: DNFR reached "
              f"{m('het_DpS','dnfr','_spt0.25')}, the product {m('het_DpS','product','_spt0.25')}, DWPC-LR "
              f"{m('het_DpS','dwpc_lr','_spt0.25')} and Soft-Impute, which now keeps all its training edges, "
              f"{m('het_DpS','softimpute','_spt0.25')} (Supplementary Table S5)." if summ("het_DpS", "dnfr", "_spt0.25") else ""))
    figure(dc, f"{ROOT}/figures/fig4_sparsity.png", 6.0,
           "**Figure 4.** MRR with all edges and with 50% and 25% of the edges of every relation kept after random removal. Points are "
           "means and error bars standard deviations over splits (three splits on the thinned graphs; the all-edges points use the "
           "random splits of Table 2). The two panels have different y ranges.")

    # ---------------- 4.7 explanations
    heading(dc, "4.7 Explanations (RQ4)", 2)
    e = lambda t, k: E[t][k]
    P_(dc, "Because each score is a noisy-OR, deleting a path changes −log(1 − P) by exactly its share; the attribution is exact for "
           "the model's additive score, not for the underlying evidence. Paths that share a fact count it once per path: the three "
           "leading paths shared at least one fact for " + joinand([f"{pct(E2[t]['top3_share_edge'])} of pairs on {SH(t)}" for t in ("het_CtD", "het_DpS", "hkg_DpS") if t in E2])
           + ". Table 7 therefore reports the attribution both to paths and to facts.")
    R.table_expl(dc)
    P_(dc, "At the level of paths the explanations were less concise than a single chain: a test pair on Hetionet disease–symptom "
           f"was supported by a median of {e('het_DpS','n_paths_median'):.0f} paths, {e('het_DpS','k50_median'):.0f} of which "
           f"carried half of the evidence. Deleting the three leading paths left {pct(e('het_DpS','del_top3_keep10'))} of correct "
           "top-10 predictions in the top 10 there, but deleting random paths with the same total share left "
           f"{pct(E2['het_DpS']['stay_top10']['matched3'])}, "
           "so within a pair an additive score depends on how much evidence is removed, not on which paths carry it.")
    if E3:
        ts = [t for t in ("het_CtD", "het_DpS", "hkg_DpS") if t in E3]
        P_(dc, "At the level of facts the picture is clearer. Summing path shares, the leading fact appeared to carry "
               + joinand([f"{pct(E3[t]['top_edge_double_median'])} of the evidence on {SH(t)}" for t in ts])
               + "; its Shapley value, which splits each path term equally among its facts and sums to the total without double "
               "counting, was " + joinand([pct(E3[t]["top_edge_shapley_median"]) for t in ts]) + ". Half of the attribution lay on "
               + joinand([f"{E3[t]['k50_edges_median']:.0f}" for t in ts]) + " facts. Removing the fact with the largest Shapley "
               "share from the graph, so that competing candidates lost its evidence too, left "
               + joinand([f"{pct(E3[t]['stay_top10_edge_top_sh'])} of correct top-10 predictions on {SH(t)}" for t in ts])
               + " in the top 10, against " + joinand([pct(E2[t]["stay_top10"]["edge_rand"]) for t in ts if t in E2])
               + " after removing a random fact on the same paths. A correct ranking depends more on the facts the explanation "
               "names than on other facts on its paths, which is what a clinician checking an explanation would want to know, "
               "although most rankings survived the loss of their leading fact. This test was run without retraining, so it "
               "measures the dependence of the trained model on a fact, not what a model trained without it would predict.")
    if "het_CtD" in E3 and E3["het_CtD"].get("lead_class"):
        lc = E3["het_CtD"]["lead_class"]
        P_(dc, "For drug repurposing we classified the leading metapath of each correct top-10 prediction. It ran through a gene "
               f"that the compound binds or regulates and that is associated with the disease for {pct(lc['gene'])} of predictions, "
               f"and through pharmacologic classes, similar compounds or similar diseases for {pct(lc['associative'])}. Most "
               "explanations of correct predictions are therefore evidence by association, drugs like this one treat the disease, "
               "rather than a mechanism of action.")
    ex = {x["task"]: x for x in XS["explanations"]}
    P_(dc, "Table 8 shows the leading paths for one correctly top-ranked pair from each of three tasks. For doxorubicin and breast "
           f"cancer, {ex['het_CtD']['n_paths']:,} paths supported the prediction, and the strongest ran through the drug's "
           "pharmacologic classes, topoisomerase inhibitors and anthracyclines, to other drugs of those classes that Hetionet "
           "records as treating breast cancer. For multiple sclerosis and muscle weakness, the leading path ran through amyotrophic "
           "lateral sclerosis, which Hetionet links to multiple sclerosis by a \"resembles\" edge derived from text co-occurrence "
           "[@himmelstein2017], a weaker kind of evidence than a curated association. On the graded emergency-department graph, "
           "bronchitis was linked to fatigue through persistent cough, sore throat and congestion shared with pneumonia, "
           "streptococcal pharyngitis, mononucleosis and sinusitis. These examples illustrate the form of the explanations; they "
           "were not rated by clinicians, and we make no claim about their clinical plausibility or usefulness.")
    R.table_examples(dc)

    # ---------------- 4.8 cost
    heading(dc, "4.8 Computational cost", 2)
    ts_ = {t: S[t]["time_s"] for t in TASKS}
    nbh = r3sum("hkg_DpS", "nbfnet"); nbd = r3sum("het_DpS", "nbfnet")
    P_(dc, "One training run of DNFR took about " + ", ".join(f"{ts_[t]['dnfr']:.0f} s" for t in TASKS[:3])
           + f" and {ts_['hkg_DpS']['dnfr']:.0f} s on the four tasks with the default early stopping, after a one-off path "
           "enumeration; DWPC with logistic regression took " + ", ".join(f"{ts_[t]['dwpc_lr']:.1f}" for t in TASKS)
           + " s on the same path sets. Overlap-aware aggregation nearly tripled the time per run on the graded graph "
           f"({r3sum('hkg_DpS','dnfr_overlap')['time_s']:.0f} s against {r3sum('hkg_DpS','dnfr')['time_s']:.0f} s). NBFNet took "
           + (f"{nbh['time_s']:.0f} s per run on the graded graph" if nbh and nbh.get("time_s") else "")
           + (f" and {nbd['time_s'] / 60:.0f} min on Hetionet disease–symptom" if nbd and nbd.get("time_s") else "")
           + ". Timings were measured on a shared two-core machine with one thread per run and are indicative only.")


# ------------------------------------------------------------------ discussion
def discussion(dc):
    rc = refine_counts()
    heading(dc, "5. Discussion", 1)
    heading(dc, "5.1 Principal findings", 2)
    _, _, dmax = comp_support("dnfr", "dnfr_max"); _, _, ddeg = comp_support("dnfr", "dnfr_nodeg")
    P_(dc, "We asked which learned components of a neuro-fuzzy path reasoner matter for link prediction on medical knowledge graphs. "
           "Combining all paths by noisy-OR rather than keeping the best one, and discounting edges at hub nodes, carried the gains "
           "that the model had over its ablations. Both had the same sign on every split of every task on which they were tested; "
           f"after adjustment for multiple comparisons, aggregation was supported on {num(len(dmax))} and hub discounting on "
           f"{num(len(ddeg))} of the tasks, and the remaining differences, though large, were inconclusive with the splits we had. "
           "Learning the discount per relation "
           "helped on the graph with graded clinical evidence and not elsewhere. The t-norm, the most visibly neuro-fuzzy part of "
           "the model, gave no consistent gain when learned, and the training loss barely depends on it once the memberships are "
           "learned: with every other parameter fitted to convergence, the loss changed by thousandths of a nat over p from −5 to "
           "0, and its minimiser differed between splits. Held-out accuracy did depend on it, which is why choosing it on "
           "validation data helped where learning it did not.")
    P_(dc, "Four refinements that addressed apparent weaknesses of the model did not improve it"
           + (", and two made it worse on the graded graph" if sum(1 for t, l in rc["neg_diff"] if t == "hkg_DpS") >= 2 else "")
           + ". Each failure is informative. The overlap penalty was switched off by training whenever the memberships were "
           "learned, and cost accuracy when it could not be, so paths through a shared node add evidence rather than repeat it, "
           "at least as these graphs encode it. A discrete operator choice was not more accurate than a continuous one. "
           "Information content added nothing to degree. Membership annealing addressed saturation near 1, which the trained "
           "models hardly show. Combined, the four did not help either"
           + (", and on Hetionet disease–symptom they lowered accuracy" if (r3p("het_DpS", "dnfr2", "dnfr") or {}).get("verdict") == "different" else "")
           + ". As a predictor, DNFR was matched or beaten by degree-weighted path counts with logistic regression "
           "on three of four tasks and by low-rank completion on the graded graph, and a compact NBFNet came close to it; its distinctive "
           "contribution is the resolution of its explanations, path by path and fact by fact, rather than accuracy.")
    heading(dc, "5.2 Why the training objective does not identify the t-norm", 2)
    P_(dc, "Two mechanisms make the loss insensitive to p. First, the memberships and metapath reliabilities already rescale every "
           "path. A t-norm with p < 0 raises the strength of a path with one weak edge relative to the product; raising the "
           "membership of weak edges, or the reliability of the metapath, has much the same effect on the ranking. When all three "
           "are learned together, many combinations rank candidates in nearly the same order. The frozen-membership experiment "
           "shows this structural redundancy directly: once the memberships could not move, learning p recovered part of what "
           "learning them gave. Second, the softmax loss averages over all candidate tails, and a change in p that reorders a few "
           "top-ranked candidates barely changes it, while it can change MRR. Neither mechanism is specific to our model; the first "
           "was anticipated for flexible neuro-fuzzy systems [@rutkowski2003], and the second applies to any operator learned "
           "through a ranking surrogate.")
    P_(dc, "The practical consequence is narrow and, we think, general. A learned t-norm parameter, or a learned choice among "
           "operators, should not be read as a property of the medical domain unless its identifiability has been checked: by "
           "restarting from several values, by a profile over fixed values with the other parameters converged, and by asking "
           "whether held-out accuracy and training loss agree. Where they disagree, as on our graded graph, the operator should be "
           "selected on validation data as a hyperparameter. These checks are cheap compared with the experiments they qualify.")
    heading(dc, "5.3 What is specific to medical knowledge graphs", 2)
    P_(dc, "Most of these findings would hold for any sparse, hub-dominated graph, and we do not claim otherwise. Three features of the "
           "graphs studied here shaped them. First, curated biomedical graphs mix evidence of very different strength under the "
           "same relation types: Hetionet's disease–disease \"resembles\" edges come from text co-occurrence, while its "
           "compound–disease edges are curated indications [@himmelstein2017]. A single membership per relation cannot tell these "
           "apart, which is one reason why the explanations in Table 8 include weak links and why learned memberships helped only "
           "where edges carried their own graded weight. Second, hubs in these graphs are biologically meaningful and are also the "
           "product of uneven research attention [@bonner2022]. Discounting them helped on random splits but perhaps not for "
           "diseases with no known indications (Section 4.5), where the remaining evidence runs through exactly these hubs. Third, "
           "the graded graph learned from patient records is a single bipartite relation, so its only metapath, "
           "disease–symptom–disease–symptom, links a disease to a symptom through diseases with similar symptom profiles. It tests "
           "the composition of graded collaborative evidence from records, not multi-relational reasoning, and it is the task on "
           "which low-rank completion led and on which the operator mattered most for held-out accuracy.")
    heading(dc, "5.4 Implications for explainable clinical decision support", 2)
    bullets(dc, [C(x) for x in [
        "**Learn or fix memberships before operators.** A path reasoner over a medical graph should discount hubs, with a fixed "
        "DWPC-like discount as a strong default and a learned one where edges carry graded evidence. The operator can then be chosen "
        "among a few fixed, non-nilpotent t-norms on validation data; learning it end to end adds parameters without adding accuracy, "
        "and the learned value is not interpretable.",
        "**Aggregate evidence, do not select it.** Noisy-OR over all paths was consistently better than the strongest path, and "
        "discounting overlapping paths did not help, so an explanation should show the leading paths together with the share of the "
        "evidence they carry rather than a single \"reason\".",
        "**Attribute to facts, and test at the level of facts.** Path shares double-count facts shared by several paths; an edge-level "
        "Shapley attribution does not, and removing a fact from the graph and re-ranking tests what a clinician would question.",
        "**Expect explanations to be by association.** On the split we analysed, the leading paths of correct drug-repurposing predictions all ran through pharmacologic classes, similar compounds or similar diseases, none through genes, "
        "and some disease–symptom paths ran through text-mined \"resembles\" links. Path shares make such evidence "
        "visible and discountable, but they do not turn it into a mechanism.",
        "**Report the strongest simple baselines and harder splits.** Degree-weighted path counts matched or beat every learned path "
        "model on three of four tasks, low-rank completion led on the record-derived graph, and a disease-centred split reduced every "
        "method to a small fraction of its accuracy on random splits."]], numbered=True, start=1)
    heading(dc, "5.5 Limitations", 2)
    nc = nseeds("het_CtD", "dnfr"); nca = nseeds("het_CtD", "dnfr_nodeg")
    bullets(dc, [C(x) for x in [
        f"**Splits and power.** The main comparisons used {num(nc)} splits on compound–treats–disease, ten on the two disease–symptom "
        f"tasks and two on PrimeKG; the component ablations on compound–treats–disease used {nca}, and several variants fewer, as "
        "the tables mark. With Holm adjustment over the comparisons of each table, differences of about 0.02 to 0.04 MRR were often "
        "not supported even when their sign was the same on every split, and with ten splits the corrected 90% intervals still extend "
        "to about ±0.015, so equivalence within ±0.01 was rarely attainable. With two splits no equivalence verdict is possible, and "
        "PrimeKG results are descriptive.",
        "**PrimeKG not covered by the third-revision experiments.** The refinements, the validation-selected t-norm, NBFNet and the "
        "disease-centred split were not run on PrimeKG, whose path sets take several hours per run on our hardware. The zero-shot "
        "result, the most clinically relevant one, therefore rests on Hetionet compound–treats–disease alone, with "
        f"{nseeds('het_CtD','dnfr','_tail')} splits and 137 candidate diseases; whether removing the degree terms helps for unseen "
        "diseases on a realistic candidate set of 17,080 diseases remains open.",
        "**Identifiability analyses on the smallest graph.** The converged profile was run on the graded graph only, with one shared "
        "p; it does not test whether the data identify differences between metapaths, which the restarts suggest they do only "
        "weakly. On the other tasks we rely on restarts, the unbounded parameterisation and the frozen-membership experiment.",
        "**Learned-operator baselines.** Our NBFNet is a compact CPU implementation (three layers, dimension 32, DistMult messages, "
        "sum aggregation) and was run on the two disease–symptom graphs only; the original model with larger dimensions, and "
        "BioPathNet [@hu2026] and TxGNN [@huang2024] on the drug-repurposing graphs, require GPU training. TransE and RotatE were "
        "untuned and ComplEx did not train beyond chance under the same budget; embedding accuracy depends strongly on training "
        "configuration [@ruffinelli2020], so the gap is not a finding of this study.",
        "**Metapath budget.** Metapaths with more than five million paths were excluded for all path methods, including the "
        "side-effect and anatomy-expression metapaths on Hetionet and "
        f"{len(DS['pkg_ind']['skipped'])} of {DS['pkg_ind']['metapaths_all']} metapaths on PrimeKG, mostly through drug–drug "
        "interactions and contraindications. These are the most hub-dominated metapaths, so the effect of hub discounting may be "
        "underestimated.",
        "**No clinical evaluation of explanations.** The explanations were not rated by clinicians; Table 7 measures concision, "
        "attribution and the dependence of the ranking on the facts shown, and the mechanistic classification is a structural check "
        "against Hetionet's own gene relations, not a clinical judgement. A rating study with several clinicians on correct and "
        "incorrect predictions is the next step.",
        "**Scores are not probabilities,** and DNFR's score P orders candidates without being calibrated.",
        "**Missing data are not random.** The sparsity analysis removed edges uniformly at random; in practice, less-studied genes and "
        "diseases lose more edges.",
        "**Associations, not mechanisms.** Paths in these graphs combine curated, text-mined and co-occurrence evidence; a high-scoring "
        "path is a reason to look, not evidence of a causal mechanism. The graded graph was learned by others from records we did not "
        "access [@rotmensch2017;johnson2023]."]])


def conclusion(dc):
    heading(dc, "6. Conclusion", 1)
    P_(dc, "In a controlled component analysis of differentiable neuro-fuzzy path reasoning on four tasks from three medical knowledge "
           "graphs, the accuracy came from aggregating all paths by noisy-OR and from discounting edges at hub nodes; learning the "
           "discount per relation helped only where edges carried graded clinical evidence. Four refinements that addressed "
           "apparent weaknesses of the model, overlap-aware aggregation, a discrete operator choice, an information-content discount "
           "and membership annealing, did not improve it. The t-norm parameter is practically non-identified by the training loss "
           "once memberships are learned, while held-out accuracy depends on it, so it should be selected on validation data and "
           "not interpreted when learned. Degree-weighted path counts were not detectably less accurate on three of four tasks. For explainable clinical "
           "decision support on knowledge graphs, effort is better spent on how much each fact counts, on how paths are combined "
           "and on attributing predictions to facts than on learning the operator that composes them.")


# ------------------------------------------------------------------ declarations
def declarations(dc):
    heading(dc, "7. Declarations", 1)
    P_(dc, "**Ethics.** The study uses only public, de-identified or aggregate data: Hetionet and PrimeKG, the disease–symptom graph "
           "published by Rotmensch et al. [@rotmensch2017], whose underlying patient records were not accessed, and relation matrices "
           "derived from the public NHANES files [@truong2026]. Saigon International University does not require ethics approval for "
           "secondary analysis of public anonymised data.")
    P_(dc, "**Data availability.** Hetionet v1.0 is available from its GitHub repository and PrimeKG from Harvard Dataverse "
           "(doi:10.7910/DVN/IXA7BM). The parsed health knowledge graph, the per-query ranks of every method on every split, the "
           "trained models, the path explanations and the statistics files from which every number in this paper is generated are "
           "deposited at Zenodo (https://doi.org/10.5281/zenodo.23879352, which resolves to the latest version; the version "
           "for this article is the deposit of code release v3.0-rev3), together with a script that downloads Hetionet and PrimeKG and checks them against the "
           "checksums of the files we used.")
    P_(dc, "**Code availability.** The implementation of DNFR, DNFR 2.0, the compact NBFNet and all baselines, the tests, and the "
           "scripts that compute the statistics and figures and build this manuscript from the result files are available under the "
           "MIT licence at https://github.com/Bangth2026/dnfr-medical-kg (release tag v3.0-rev3, branch revision3).")
    P_(dc, "**CRediT author statement.** Truong Hai Bang: Conceptualization; Methodology; Software; Formal analysis; Data curation; "
           "Visualization; Writing – original draft; Writing – review and editing. Van An Phan Thi: Validation (a check that the "
           "nodes and edges of the path explanations in Table 8 exist in the source graphs with the stated relations; this was a "
           "check of graph correctness, not a clinical judgement); Writing – review and editing. All authors approved the final "
           "manuscript and agree to be accountable for all aspects of the work.")
    P_(dc, "**Independent verification.** Without the AI system, the corresponding author (Truong Hai Bang) checked every result "
           "reported in this article, in the text, the tables, the figures and the supplementary material, against the released "
           "result files.")
    P_(dc, "**Declaration of generative AI and AI-assisted technologies in the manuscript preparation process.** During the preparation "
           "of this work the authors used Claude (Anthropic) to draft and edit the manuscript text and, as described in Section 3.6, "
           "to write and run the analysis code under the authors' direction. After using this tool, the authors reviewed and edited "
           "the content as needed and take full responsibility for the content of the publication.")
    P_(dc, "**Funding.** This research did not receive any specific grant from funding agencies in the public, commercial, or "
           "not-for-profit sectors.")
    P_(dc, "**Declaration of competing interest.** The authors declare no known competing financial interests or personal relationships "
           "that could have appeared to influence the work reported in this paper.")


def references(dc):
    from references import REFS
    heading(dc, "References", 1)
    for i, k in enumerate(ORDER):
        para(dc, f"[{i + 1}] {REFS[k]}", align="left", size=10, line=1.15, space_after=3)


# ------------------------------------------------------------------ supplementary material
def supplement(dc):
    para(dc, f"**Supplementary material for: {TITLE}**", align="left", size=13, line=1.3, space_after=8)
    heading(dc, "S1. Full results by task", 1)
    R.supp_task(dc, 1, "het_CtD", "Hetionet compound–treats–disease.")
    R.supp_task(dc, 2, "het_DpS", "Hetionet disease–presents–symptom.")
    R.supp_task(dc, 3, "pkg_ind", "PrimeKG drug–indication–disease, ranked against all diseases (two splits; descriptive).")
    R.supp_task(dc, 4, "hkg_DpS", "Graded disease–symptom graph learned from emergency-department records.")
    heading(dc, "S2. Sparsity, long training and converged profile", 1)
    R.supp_sparsity(dc, 5)
    R.supp_long(dc, 6)
    R.supp_profile(dc, 7)
    figure(dc, f"{ROOT}/figures/figA1_restarts.png", 6.3,
           "**Figure S1.** Learned p at the default early stopping, by initial value p₀ (rows) and task (panels). Each point is one "
           "metapath in one split; | marks p₀. The graded graph has a single metapath.")
    figure(dc, f"{ROOT}/figures/fig3_trajectories.png", 6.3,
           "**Figure S2.** Learned p during 400 epochs without early stopping and without penalty, by initial value p₀. Graded graph: "
           "one line per split. Hetionet disease–symptom: thin lines are the metapaths of the first split, thick lines the median "
           "over metapaths per split. Dotted vertical lines mark the epoch with the best validation MRR.")
    heading(dc, "S3. Checks requested in review", 1)
    R.supp_prior(dc, 9)
    R.supp_valid(dc, 10)
    heading(dc, "S4. Secondary test on a population relation (NHANES)", 1)
    N = XS["nhanes"]; J = XS["jack"]
    P_(dc, "A secondary test uses the 24 × 24 relation between nine PHQ-9 symptoms and 15 socioeconomic, lifestyle and health concepts "
           "from our earlier, not peer-reviewed analysis [@truong2026]. DNFR is fitted to the relation measured in 2013–2018, with "
           "every pair predicted from its indirect paths only, and evaluated by the Spearman correlation of its predictions with the "
           "relation measured independently in 2007–2012. With 24 concepts this test has little power.")
    P_(dc, f"The fixed rows of Table S11 reproduce the earlier analysis, with the product ahead of the other fixed t-norms (ρ = "
           f"{f3(N['product']['rho_hold'])}). A single learned Schweizer–Sklar parameter added nothing ({f3(N['ss_shared']['rho_hold'])}). "
           f"Calibrating the membership per relation type raised agreement to {f3(N['calib_product']['rho_hold'])} with the product "
           f"and {f3(N['dnfr']['rho_hold'])} with learned p per metapath; a jackknife over concepts gave a standard error of "
           f"{f3(J['dnfr_minus_calibproduct']['jack_se'])} for the difference, as large as the difference itself.")
    R.supp_nhanes(dc, 11)
    heading(dc, "References", 1)
    from references import REFS
    for i, k in enumerate(ORDER):
        para(dc, f"[{i + 1}] {REFS[k]}", align="left", size=10, line=1.15, space_after=3)
