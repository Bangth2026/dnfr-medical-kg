"""Front matter, results, discussion, conclusion and declarations (third revision, DNFR 2.0)."""
import collections
import numpy as np
from ms_common import *
from docx_helpers import para, heading, bullets, figure
import ms_results as R
from ms_results import r3p, r3sum, mean_any, R3T

P_ = lambda d, t, **k: para(d, C(t), **k)
RELEASE = "v5.0"                      # code release for this version of the article (to be published with it)
ZENODO_VERSION = ""                   # "version v5.0: https://doi.org/...; " once the Zenodo version exists
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
         "graphs")


def title_block(dc):
    para(dc, f"**{TITLE}**", align="left", size=14, line=1.3, space_after=10)
    para(dc, "**Article type:** Research Paper", align="left")
    para(dc, "**Running title:** Neuro-fuzzy path reasoning on medical knowledge graphs", align="left")
    para(dc, "**Authors**", align="left", space_after=2)
    para(dc, "Hai Bang Truong^1^*, Van An Phan Thi^2^", align="left")
    para(dc, "^1^ Faculty of Engineering and Computer Science, Saigon International University, 8C Tong Huu Dinh Street, An Khanh Ward, "
             "Ho Chi Minh City, Vietnam. ORCID: https://orcid.org/0000-0001-5094-3128", align="left", size=11)
    para(dc, "^2^ VNU University of Education, Vietnam National University, Hanoi, G7 Building, No. 144 Xuan Thuy Street, Cau Giay "
             "District, Hanoi, Vietnam. Email: phanthivanantlls@gmail.com. ORCID: https://orcid.org/0009-0005-4135-5689",
         align="left", size=11)
    para(dc, "* Corresponding author: Truong Hai Bang, truonghaibang@siu.edu.vn", align="left", size=11)


def profile_flat(key):
    """Largest loss range over p in [-5, 0] across the splits of a converged profile, or None."""
    P = R3.get(key) or {}
    return max(P["loss_range_nonpos"]) if P.get("loss_range_nonpos") else None


def notgt_drop():
    NT = R4.get("notgt") or {}
    return (NT.get("paired") or {}).get("dnfr_notgt__dnfr")


def ctd_split_note():
    """Parenthetical on the variants that have fewer compound–treats–disease splits than DNFR (Section 5.5)."""
    nc = nseeds("het_CtD", "dnfr")
    small = sorted(x for x in ({nseeds("het_CtD", v) for v in ("hamacher", "dnfr_fixdeg", "dnfr_shared", "dnfr_frozen",
                                                                 "product_frozen") if nseeds("het_CtD", v)}
                               | {r3p("het_CtD", v, "dnfr")["J"] for v in ("dnfr2", "fixp_val", "dnfr_overlap", "dnfr_gumbel",
                                                                          "dnfr_ic", "dnfr_anneal") if r3p("het_CtD", v, "dnfr")})
                   if x < nc)
    if not small:
        return ""
    k = num(small[0]) if small[0] == small[-1] else f"between {num(small[0])} and {num(small[-1])}"
    return f" ({k} for some variants, as the tables mark)"


def dwpc_at_least():
    """Tasks on which DNFR minus DWPC-LR is not positive under the rule (DWPC-LR at least as accurate)."""
    return [t for t in TASKS if pr(t, "dnfr", "dwpc_lr") and not (
        pr(t, "dnfr", "dwpc_lr")["delta_MRR"] > 0 and verdict_of(pr(t, "dnfr", "dwpc_lr")) == "different")
        and not (t == "pkg_ind" and pr(t, "dnfr", "dwpc_lr")["delta_MRR"] > 0)]


def highlights_list():
    run, pos, dif = comp_support("dnfr", "dnfr_max")
    rd, pd_, _ = comp_support("dnfr", "dnfr_nodeg")
    fd = profile_flat("profile_summary_dps")
    nd = notgt_drop()
    h = [f"Noisy-OR over all paths beat the strongest path on all {num(len(run))} tasks tested."
         if len(pos) == len(run) else "Noisy-OR over all paths beat the strongest single path on most tasks.",
         "Hub discounting had the same sign on every split of every task tested." if len(pd_) == len(rd)
         else "Hub discounting helped on most tasks tested.",
         ("The training loss barely constrained the t-norm parameter on two graphs." if fd is not None and fd < 0.01
          else "On the graded graph the training loss barely constrained the t-norm parameter."),
         f"Degree-weighted path counts were at least as accurate as DNFR on {num(len(dwpc_at_least()))} of four tasks.",
         ("Drug-repurposing accuracy on Hetionet rested on paths through known indications." if nd and nd["delta"] < 0 and nd["verdict"] == "different"
          else "None of four plausible refinements improved accuracy detectably.")]
    for x in h:
        assert len(x) <= 85, (len(x), x)
    return h


def highlights(dc, H):
    para(dc, "**Highlights**", align="left", space_after=2)
    bullets(dc, H)


def abstract(dc):
    rc = refine_counts()
    PSh = R3.get("profile_summary") or {}; PSd = R3.get("profile_summary_dps") or {}
    fv = r3p("hkg_DpS", "fixp_val", "product")
    nd = notgt_drop(); NT = R4.get("notgt") or {}
    XC = (R4.get("explain_dwpc") or {}).get("het_CtD", {}).get("dnfr", {})
    para(dc, "**Abstract**", align="left", space_after=2)
    P_(dc, "**Objective:** Path-based reasoning over medical knowledge graphs explains its predictions but composes evidence "
           "with fixed operators. We asked which parts of a differentiable neuro-fuzzy path reasoner are worth learning.")
    P_(dc, "**Methods:** Differentiable Neuro-Fuzzy Reasoning (DNFR) learns degree-discounting edge memberships, a Schweizer–Sklar "
           "t-norm per metapath and a noisy-OR over paths. On drug repurposing (Hetionet, PrimeKG) and disease–symptom prediction "
           "(Hetionet; a graph learned from emergency-department records) we compared ablations, fixed and validation-selected "
           "t-norms, four refinements, NBFNet and simpler baselines under leave-dyad-out splits, profiled the t-norm parameter to "
           "convergence, and attributed predictions to individual facts.")
    r_max = [pr(t, "dnfr", "dnfr_max")["delta_MRR"] for t in TASKS if pr(t, "dnfr", "dnfr_max")]
    two = PSd and max(PSd["loss_range_nonpos"]) < 0.01
    lr = max(PSh["loss_range_nonpos"] + (PSd["loss_range_nonpos"] if two else []))
    txt = (f"**Results:** Aggregating all paths rather than the strongest raised mean reciprocal rank (MRR) by "
           f"{min(r_max):.2f}–{max(r_max):.2f}, and hub discounting had the same sign on every split. "
           + ("No refinement improved MRR detectably. " if not rc["pos_diff"] else
              f"{Num(len(rc['pos_diff']))} of {rc['n']} refinement tests showed a gain. ")
           + ("On the graded graph and on Hetionet disease–symptom" if two else "On the graded graph")
           + f", the converged training loss varied by at most {lr:.3f} nats over t-norm "
           "parameters from −5 to 0, and its minimiser differed between splits; a t-norm selected on validation data beat the fixed "
           f"product by {fv['delta']:.3f} on the graded graph. Degree-weighted path counts were at least as accurate as DNFR on "
           f"{num(len(dwpc_at_least()))} of four tasks.")
    if nd and XC:
        txt += (f" On Hetionet drug repurposing, {pct(XC['share_target_mean'])} of DNFR's evidence ran through metapaths that "
                "contain known indications, and without them MRR fell from "
                f"{f3(NT['means']['dnfr']['mean'])} to {f3(NT['means']['dnfr_notgt']['mean'])}.")
    P_(dc, txt)
    P_(dc, "**Conclusion:** Path aggregation and hub discounting carry the accuracy, and the composition operator is better treated as a "
           "hyperparameter than learned. DNFR's advantage over degree-weighted path counts is an exactly additive attribution to "
           "facts, not accuracy. On curated graphs, random-split accuracy largely reflects association with known indications.")


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
        ["What this paper adds", "A controlled component analysis of a differentiable neuro-fuzzy path reasoner on four "
                                 "link-prediction tasks from three medical knowledge graphs. Aggregating all paths and discounting "
                                 "hub nodes carried the gains; four plausible refinements did not help detectably. The training "
                                 "loss barely depends on the t-norm parameter once memberships are learned, so a learned operator "
                                 "is not a property of the domain; selecting it on validation data helped on the graded graph. "
                                 "Degree-weighted path counts were at least as accurate on three of four tasks, and on drug "
                                 "repurposing most of the evidence ran through known indications rather than through genes."],
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
    nb = {t: r3p(t, "nbfnet", "dnfr") for t in R3T}; nw = {t: r3p(t, "nbfnet", "dwpc_lr") for t in R3T}
    nbs = {t: r3sum(t, "nbfnet") for t in R3T}
    tu = r3p("hkg_DpS", "nbfnet_tuned", "dnfr"); tuw = r3p("hkg_DpS", "nbfnet_tuned", "dwpc_lr")
    ch = collections.Counter(R3.get("hkg_DpS", {}).get("nbf_tuned_choice") or {})
    nbt = [t for t in ("hkg_DpS", "het_DpS") if nb.get(t)]
    def vtxt(p):
        return {"different": "different under our rule", "equivalent": "equivalent within ±0.01",
                "inconclusive": "inconclusive"}.get(p.get("verdict"), "inconclusive")
    txt = ("NBFNet, which learns both how a path is extended and how paths are aggregated, is the closest learned counterpart to "
           "the question we ask. Our compact CPU version (Section 3.4) reached "
           + joinand([f"{f3(nbs[t]['mean'])} on {SH(t)} over {nbs[t]['n']} splits (NBFNet minus DNFR {dci3(nb[t])}, {vtxt(nb[t])}; "
                      f"minus DWPC-LR {sg(nw[t]['delta'])}, {vtxt(nw[t])})" for t in nbt]) + ". ")
    if tu:
        txt += (f"The configuration tuned on validation data among four reached {m('hkg_DpS','nbfnet_tuned')} on the graded graph "
                f"(minus DNFR {dci3(tu)}, {vtxt(tu)}; minus DWPC-LR {sg(tuw['delta'])}, {vtxt(tuw)}). ")
    nb_dif = [t for t in nbt if nb[t].get("verdict") == "different"]
    nb_ahead = [t for t in nbt if t not in nb_dif and nb[t]["delta"] > 0 and nb[t]["wins"] >= 0.8 * nb[t]["J"]]
    w_below = [t for t in nbt if nw[t]["delta"] < 0 and nw[t].get("verdict") == "different"]
    w_other = [t for t in nbt if t not in w_below]
    txt += ("A learned operator for path extension and aggregation, with about ten thousand parameters, thus "
            + ("did not differ from DNFR under our rule on either graph" if not nb_dif else
               "differed from DNFR on " + joinand([SH(t) for t in nb_dif]))
            + ("" if not nb_ahead else
               "; on " + joinand([SH(t) for t in nb_ahead]) + " it was ahead on "
               + joinand([f"{nb[t]['wins']} of {nb[t]['J']} splits" for t in nb_ahead])
               + ", but the corrected test did not reach significance")
            + ". Against DWPC-LR it was "
            + joinand(([f"behind on {joinand([SH(t) for t in w_below])}"] if w_below else [])
                      + ([f"not detectably different on {joinand([SH(t) for t in w_other])}"] if w_other else []))
            + ". It was not run on the drug-repurposing graphs, whose size exceeds what our CPU implementation can train in "
            "reasonable time.")
    P_(dc, txt)

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
    H = R4.get("hamacher_ctd") or []
    if H:
        hm = min(H, key=lambda h: h["hamacher"] - h["product"])
        low = [h for h in H if h["hamacher"] - h["product"] < -0.05]
        hi = [h for h in H if h not in low]
        dh = lambda S, a, b: sum(h[a] - h[b] for h in S) / len(S)
        txt = (f"On compound–treats–disease {num(len(low))} of {num(len(H))} splits {'weighs' if len(low) == 1 else 'weigh'} heavily in the comparisons with the "
               f"fixed Hamacher product (Supplementary Table S13). On split {hm['seed']} the Hamacher run reached an MRR of "
               f"{f3(hm['hamacher'])} against {f3(hm['product'])} for the product")
        if hm.get("loss_hamacher") is not None and hm.get("loss_product") is not None:
            txt += (f", with a final training loss of {hm['loss_hamacher']:.2f} against {hm['loss_product']:.2f}: training did not "
                    "fail numerically, it stopped in a poorer optimum")
        txt += ". We report the run as it is."
        HR = [h for h in (R4.get("hamacher_restarts") or []) if h["seed"] == hm["seed"]]
        if HR:
            txt += (f" Restarted from other initial memberships (degree discount b = c = "
                    + joinand([f"{h['b_init']:g}" for h in HR]) + " instead of 0.3), the Hamacher run on that split reached "
                    + joinand([f3(h["mrr"]) for h in HR])
                    + ("" if any(h.get("loss") is None for h in HR) else
                       " with final training losses of " + joinand([f"{h['loss']:.2f}" for h in HR]))
                    + (", so the poorer optimum is not an accident of one initialisation: on this split the Hamacher product "
                       "settles in a worse region of the loss than the product does."
                       if all(h["mrr"] < hm["product"] - 0.05 for h in HR) else
                       ", so the poorer optimum depended on the initialisation." if all(h["mrr"] >= hm["product"] - 0.05 for h in HR) else
                       ", so whether the run ends in the poorer optimum depends on the initialisation."))
        if low and hi:
            txt += (f" Over all splits the Hamacher product was {abs(dh(H, 'hamacher', 'product')):.3f} below the product; without "
                    + ("that split" if len(low) == 1 else "those splits")
                    + f" it was {abs(dh(hi, 'hamacher', 'product')):.3f} below")
            if all(h.get("dnfr") is not None for h in H):
                txt += (f", and the difference between the learned p and the Hamacher product shrank from "
                        f"{dh(H, 'dnfr', 'hamacher'):+.3f} to {dh(hi, 'dnfr', 'hamacher'):+.3f}")
            txt += "."
        P_(dc, txt)
    fr = {t: pr(t, "dnfr_frozen", "product_frozen") for t in TASKS}; lm = {t: pr(t, "product", "product_frozen") for t in TASKS}
    P_(dc, "The t-norm mattered most when the memberships could not be learned. With memberships frozen at their initial values, "
           f"learning p improved on the product by {dci(fr['hkg_DpS'])} on the graded graph and by {dci(fr['het_DpS'])} on Hetionet "
           f"disease–symptom, and learning the memberships instead of the t-norm improved the product by {dci(lm['hkg_DpS'])} on the "
           "graded graph. Memberships and operator are, to a large extent, substitutes in this model: both rescale the strength of "
           "a path, as in flexible neuro-fuzzy systems in which weights and t-norm parameters scale the strength of a rule "
           "[@rutkowski2003]. This is a structural property of the model, and Section 4.4 asks what it implies for learning p.")

    # ---------------- 4.3 refinements
    heading(dc, "4.3 Four refinements and a validation-selected t-norm (RQ2)", 2)
    rc = refine_counts()
    P_(dc, "Table 4 reports the four refinements of Section 3.3, each added to DNFR alone and removed from their combination, "
           f"DNFR 2.0, on the three tasks on which we ran them. Of the {num(rc['n'])} add-one comparisons, "
           + (f"none showed a gain under our rule" if not rc["pos_diff"] else f"{num(len(rc['pos_diff']))} showed a gain under our rule")
           + (f", {num(len(rc['neg_diff']))} showed a loss" if rc["neg_diff"] else "")
           + (f", {num(len(rc['equivalent']))} {'was' if len(rc['equivalent']) == 1 else 'were'} equivalent within ±0.01" if rc["equivalent"] else "")
           + f" and {num(len(rc['inconclusive']))} were inconclusive; the point estimates ranged from {sg(min(rc['deltas']))} to "
           f"{sg(max(rc['deltas']))}. Supplementary Section S4 and Figure S3 give the details of each refinement; here we report "
           "the three results that bear on the conclusions.")
    R.table4(dc)
    an = {t: r3p(t, "dnfr_anneal", "dnfr") for t in R3T}; anl = {t: r3p(t, "dnfr2", "dnfr2_no_anneal") for t in R3T}
    sat = R3.get("saturation", {})
    txt = ("**Membership annealing (R4)** had the most consistent positive point estimates, so we ran it on every split. Added to "
           "DNFR, it changed MRR by " + joinand([f"{dci3(an[t])} on {SH(t)} ({an[t]['wins']}/{an[t]['J']} splits, "
                                                   f"{an[t]['verdict']})" for t in R3T if an[t]]) + ". ")
    if any(anl.values()):
        txt += ("Removing it from DNFR 2.0 changed MRR by " + joinand([f"{sg(-anl[t]['delta'])} on {SH(t)} "
                                                                       f"({anl[t]['J']} splits)" for t in R3T if anl[t]]) + ". ")
    if sat:
        h, dps = sat["hkg_DpS"][0], sat["het_DpS"][0]
        txt += (f"Saturation, the problem R4 was meant to address, was rare in the trained default models ({100 * h['mu_gt_099']:.1f}% "
                f"and {100 * dps['mu_gt_099']:.1f}% of memberships above 0.99 on the graded graph and on Hetionet disease–symptom), "
                "so whatever R4 does, it does through the schedule of the temperature rather than by removing saturation. ")
    P_(dc, txt + "We do not recommend it on this evidence, and we do not rule it out either.")
    bt = R3.get("hkg_DpS", {}).get("beta", {})
    med = lambda x: float(np.median(x)) if x else float("nan")
    po = {t: r3p(t, "product_overlap", "product") for t in R3T}
    P_(dc, "**Overlap-aware aggregation (R1)** did not improve accuracy. Its learned exponent β fell from 1 to a median of "
           f"{f3(med(bt.get('dnfr_overlap')), 3)} on the graded graph when the memberships were learned, and stayed larger (median "
           f"{f3(med(bt.get('product_overlap')), 2)}) when they were tied to the fixed product. Added to the fixed product, R1 "
           "changed MRR by "
           + joinand([f"{dci3(po[t])} on {SH(t)} ({po[t]['wins']}/{po[t]['J']} splits, {po[t]['verdict']})" for t in R3T if po[t]])
           + ". We read the small β with caution: Section 4.4 shows that the training loss hardly constrains parameters that "
           "rescale paths, and β is one of them. What the data do show is that discounting paths through a shared node did not "
           "improve the held-out accuracy of DNFR"
           + (lambda up, dn: "" if not (up or dn) else
              "; with the fixed product its estimate was "
              + joinand((["negative on " + joinand([SH(t) for t in dn])] if dn else [])
                        + (["positive on " + joinand([f"{SH(t)} ({po[t]['J']} splits)" for t in up])] if up else []))
              + ("" if any(po[t]["verdict"] == "different" for t in up + dn) else
                 ", and neither difference was supported under our rule"))(
               [t for t in R3T if po[t] and po[t]["delta"] > 0], [t for t in R3T if po[t] and po[t]["delta"] < 0])
           + ". Whether such paths add evidence or repeat it cannot be decided from the value of β.")
    ic = {t: r3p(t, "dnfr_ic", "dnfr") for t in R3T}; isd = R3.get("hkg_DpS", {}).get("paired", {})
    sd1, sd2 = isd.get("dnfr_ic_sd__dnfr"), isd.get("dnfr_ic_sd__dnfr_ic")
    txt = ("**The information-content discount (R3)** lost "
           f"{sg(-ic['hkg_DpS']['delta'])[1:]} on the graded graph ({ic['hkg_DpS']['wins']}/{ic['hkg_DpS']['J']} splits positive). "
           "We checked the implementation. The feature log(1 + n(v)) is matched to the degree feature it replaces in mean, so that "
           "the initial coefficients keep their meaning, but on the first split of that graph it had about half the spread of the "
           "degree feature (correlations of 0.85 to 1.00 along the hops of the metapath), so the same coefficients discount hubs "
           "about half as much. ")
    if sd1:
        txt += ("Matched in standard deviation as well, the feature changed MRR by "
                f"{dci3(sd1)} relative to DNFR ({sd1['wins']}/{sd1['J']} splits) and by {sg(sd2['delta'])} relative to the "
                "mean-matched version. ")
        txt += ("The loss of R3 therefore came from the scale of the feature, not from information content as such."
                if abs(sd1["delta"]) < abs(ic["hkg_DpS"]["delta"]) / 2 else
                "Rescaling did not remove the loss, so it does not come from the scale of the feature alone.")
    P_(dc, txt)
    v2 = {t: r3p(t, "dnfr2", "dnfr") for t in R3T}; v2p = {t: r3p(t, "dnfr2", "product") for t in R3T}
    P_(dc, "**Combined.** DNFR 2.0 differed from DNFR by "
           + joinand([f"{dci3(v2[t])} on {SH(t)} ({v2[t]['J']} splits)" for t in R3T if v2[t]]) + ", and from the fixed product by "
           + joinand([sg(v2p[t]["delta"]) for t in R3T if v2p[t]])
           + ". None of these differences was supported under our rule" * (not any(v2[t] and v2[t]["verdict"] == "different" for t in R3T))
           + "".join(f". On {SH(t)} DNFR 2.0 was less accurate than DNFR under our rule ({v2[t]['wins']}/{v2[t]['J']} splits "
                     "in its favour)" for t in R3T if v2[t] and v2[t]["verdict"] == "different" and v2[t]["delta"] < 0)
           + ". We do not recommend it over DNFR or over the fixed product.")
    fv = {t: r3p(t, "fixp_val", "dnfr") for t in R3T}; fvp = {t: r3p(t, "fixp_val", "product") for t in R3T}
    ch = R3.get("hkg_DpS", {}).get("fixp_val_choice") or {}
    P_(dc, "**A fixed t-norm selected on validation data.** Training with a fixed shared p from a grid and keeping, on each split, "
           "the value with the best validation MRR is the procedure that the identifiability analysis below recommends; it uses the "
           "same early stopping as every other variant. It differed from DNFR by "
           + joinand([f"{dci3(fv[t])} on {SH(t)}" for t in R3T if fv[t]]) + " and from the fixed product by "
           + joinand([f"{dci3(fvp[t])} on {SH(t)}" for t in R3T if fvp[t]])
           + "; " + joinand([f"{fvp[t]['J']}" for t in R3T if fvp[t]]) + " splits"
           + (f"; the gain over the product on the graded graph was supported under our rule" if fvp.get("hkg_DpS") and fvp["hkg_DpS"]["verdict"] == "different" else "")
           + (f". On the graded graph the grid value chosen was −0.5 in {ch.get('fixp_m05', 0)} splits, −2 in {ch.get('fixp_m2', 0)} "
              f"and −4 in {ch.get('fixp_m4', 0)}. A fixed p = −2 trained with early stopping reached {m('hkg_DpS','fixp_m2')}, below "
              "the 0.338 that an earlier, unconverged profile reported without early stopping, so part of that earlier gain was due "
              "to longer training rather than to the value of p" if ch else "") + ".")
    dps = fvp.get("het_DpS"); PSd_ = R3.get("profile_summary_dps") or {}
    if dps:
        ps_ = dps["per_split"]; chd = R3.get("het_DpS", {}).get("fixp_val_choice") or {}
        ties = sum(1 for x in ps_ if abs(x) < 1e-9)
        sd_p = (R3.get("het_DpS", {}).get("summary", {}).get("product") or {}).get("sd")
        sd_f = (R3.get("het_DpS", {}).get("summary", {}).get("fixp_val") or {}).get("sd")
        txt = (f"On Hetionet disease–symptom the gain over the product ({sg(dps['delta'])}, {dps['wins']}/{dps['J']} splits, "
               f"{dps['verdict']}) has a different source. The per-split differences were "
               + ", ".join(sg(x) for x in ps_) + f" (Supplementary Table S17): validation chose the product itself on "
               f"{num(chd.get('product', 0))} splits, so {num(ties)} differences are exactly zero, and the gain came from the splits "
               "on which the early-stopped product run ended low")
        if sd_p and sd_f:
            txt += f" (SD of test MRR over splits {sd_p:.3f} for the product against {sd_f:.3f} for the selected p)"
        txt += ". "
        # converged fits at p = 0 on the profiled splits, against the early-stopped product run of the same split
        prow = {sd_: next((r for r in rows_ if r["p"] == 0), None) for sd_, rows_ in zip(PSd_.get("seeds", []), PSd_.get("rows", []))}
        es = {r["seed"]: r["methods"]["product"]["MRR"] for r in json.load(open(f"{RES}/het_DpS.json"))["runs"]}
        cmp_ = [(s, es[s], prow[s]["test_MRR"]) for s in sorted(prow) if prow[s] and s in es]
        if cmp_:
            txt += ("Fitted to convergence, the product reached " + joinand([f"{c:.3f}" for _, _, c in cmp_])
                    + " on splits " + joinand([str(s) for s, _, _ in cmp_]) + ", against " + joinand([f"{e:.3f}" for _, e, _ in cmp_])
                    + " for the early-stopped run, and at convergence test MRR varied by at most "
                    f"{max(r['test_range'] for r in PSd_.get('intervals', [])):.3f} over p (Section 4.4). On this graph the gain "
                    "of the selected p therefore reflects the variability of early-stopped training, not a better value of p; "
                    "we claim a benefit of validation selection for the graded graph only.")
        P_(dc, txt)

    # ---------------- 4.4 identifiability
    heading(dc, "4.4 Is the t-norm parameter identified by the training objective? (RQ3)", 2)
    pv = S["het_CtD"]["p_by_variant"]["dnfr"]
    bt_ = {t: RS["stopping"][t]["dnfr"]["best_epoch"]["median"] for t in TASKS if t in RS["stopping"]}
    P_(dc, "Two questions need to be kept apart. One is structural: whether the model can produce the same scores with different "
           "values of p, because the memberships compensate; Section 4.2 showed that they partly can. The other is practical: "
           "whether the training data, through the training loss, determine p within a useful range [@raue2009]. Figure 2 and "
           "Supplementary Table S11 address the second.")
    P_(dc, "**Restarts.** With the default early stopping, the learned p ended near its starting value (Supplementary Table S11). On "
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
    PSd = R3.get("profile_summary_dps") or {}
    if PS:
        iv = PS["intervals"]; cur = PS["curvature"]; inter = PS["interior"]
        def ptxt(x):
            return f"{x:g}".replace("-", "−")
        def itv(r):
            return (f"[{ptxt(r['p_lo'])}, {ptxt(r['p_hi'])}]" if r["p_lo"] != r["p_hi"] else f"{{{ptxt(r['p_lo'])}}}")
        wide = [k for k, r in enumerate(iv) if r["p_hi"] - r["p_lo"] >= 2]
        g_ = sorted({x for gg in PS["grid"] for x in gg})
        P_(dc, f"**Converged profile.** On the graded graph we fixed a shared p on a grid from {ptxt(g_[0])} to {ptxt(g_[-1])} and fitted every other "
               "parameter by L-BFGS until the loss changed by less than 10^−9^ (Section 3.4). Over p from −5 to 0 the converged "
               "training loss varied by " + joinand([f"{x:.4f}" for x in PS["loss_range_nonpos"]]) + f" nats in the "
               f"{num(len(PS['seeds']))} splits, against a loss of about 4.3, and |∂L/∂p| at the converged fits was at most "
               f"{PS['max_abs_dLdp']:.4f}. The loss was lowest at p = " + joinand([ptxt(r["p_argmin"]) for r in iv])
               + (lambda ii, ee: (f"; the curvature at the interior minim{'um' if len(ii) == 1 else 'a'} was "
                                  + joinand([f"{cur[k]:.3f}" for k in ii]) if ii else "")
                  + (f"; in split{'s' if len(ee) > 1 else ''} " + joinand([str(iv[k]['seed']) for k in ee])
                     + " the minimum lay at an end of the grid, where a curvature is not defined"
                     + " and we report only the slope (Supplementary Table S8)" if ee else ""))(
                  [k for k in range(len(iv)) if inter[k]], [k for k in range(len(iv)) if not inter[k]])
               + ". The sets of p within χ²(0.95, 1)/(2N) of each split's minimum, a profile-likelihood style interval that we use "
               "descriptively, were " + joinand([itv(r) for r in iv])
               + (", which do not all overlap" if max(r["p_lo"] for r in iv) > min(r["p_hi"] for r in iv) else "")
               + " (Figure 2; Supplementary Tables S7 and S8). The two kinds of evidence should be kept apart. Within a split, "
               + (f"the profile was flat over most of the grid in {num(len(wide))} of the three splits"
                  if wide else "no split had a profile flat over most of the grid")
               + (f" and narrow in the other {num(3 - len(wide))}, though the threshold behind these intervals treats the training "
                  "queries as independent, which they are not, so narrow intervals are, if anything, too narrow"
                  if 0 < len(wide) < 3 else "")
               + "; between splits, the minimisers disagreed. In the sense of Raue et al. "
               "[@raue2009], p is practically non-identified within a split only where the profile is flat; elsewhere the "
               "non-identifiability is the instability of the minimiser across random splits of the same data. Both mean that the "
               "training loss does not fix p to a value that could be read as a property of the data.")
        if PSd:
            ivd = PSd["intervals"]
            P_(dc, "On Hetionet disease–symptom, with 17 metapaths and 80 parameters, the converged loss varied by "
                   + joinand([f"{x:.4f}" for x in PSd["loss_range_nonpos"]]) + " nats over p from −5 to 0 in the "
                   f"{num(len(PSd['seeds']))} splits (loss about {np.mean([min(r['loss'] for r in rr) for rr in PSd['rows']]):.1f}); "
                   "the loss was lowest at p = " + joinand([ptxt(r["p_argmin"]) for r in ivd]) + ", with intervals "
                   + joinand([itv(r) for r in ivd]) + " (Supplementary Tables S15 and S16). "
                   + ("The same flatness therefore holds on a multi-relational graph." if max(PSd["loss_range_nonpos"]) < 0.01 else
                      "The loss there constrained p more than on the graded graph.")
                   + (lambda e: "" if not e else
                      f" In {num(len(e))} of the {num(len(ivd))} splits the minimum lay at the lower end of the grid, so the "
                      "minimiser may lie further towards the minimum t-norm; the narrow intervals there reflect the small threshold "
                      "rather than a well-located optimum.")([r for r in ivd if r["p_argmin"] <= min(PSd["grid"][0])])
                   + " Test MRR varied by at most " + f"{max(r['test_range'] for r in ivd):.3f}" + " across the grid in any split.")
        P_(dc, "Held-out accuracy was more informative than the loss. On the graded graph, validation MRR at the converged fits was "
               "highest at p = " + joinand([ptxt(r["p_best_valid"]) for r in iv]) + ", and test MRR at the value chosen on "
               "validation exceeded that of the product by " + joinand([sg(r["test_at_valid_choice"] - r["test_product"]) for r in iv])
               + ". The data therefore carry information about p, but the softmax loss, averaged over all candidates, hardly "
               "registers the reordering of the few top-ranked ones that decides MRR. That is consistent with what Section 4.3 "
               "found on this graph: selecting p on validation data helped, learning it by gradient descent did not.")
    figure(dc, f"{ROOT}/figures/fig2_profile_lbfgs.png", 6.3,
           "**Figure 2.** Converged profile over a fixed shared t-norm parameter p on the graded graph, three splits; p > 0 fitted "
           "with warm starts from the product. Left: training "
           "loss minus its minimum over the grid (×10⁻³ nat). Centre: ∂L/∂p at the converged fit. Right: test MRR of the converged "
           "fit. All other parameters were fitted by L-BFGS to convergence. Dashed lines mark the Hamacher product (p = −1), the "
           "product (p = 0) and the Łukasiewicz t-norm (p = 1).")
    nil = []
    for sd_, rows_ in zip((PS or {}).get("seeds", []), (PS or {}).get("rows", [])):
        r0 = next((r for r in rows_ if r["p"] == 0), None); r1 = next((r for r in rows_ if r["p"] == 1), None)
        if r0 and r1:
            nil.append((sd_, r0, r1))
    txt = ("Two further observations complete the picture. Training for 400 epochs without early stopping did not make the learned "
           "p converge either (Supplementary Table S6). On the graded graph the Łukasiewicz t-norm, trained from the default "
           f"initialisation, lost {sg(-pr('hkg_DpS','dnfr','luk')['delta_MRR'])[1:]} in MRR: every path there contains a "
           "symptom-to-disease edge with a small initial membership, and for p > 0 such a path has strength exactly zero and passes "
           "no gradient. ")
    if nil:
        txt += ("Started instead from the converged fit at the product, the fits at p = 1 reached a test MRR of "
                + joinand([f3(r1["test_MRR"]) for _, _, r1 in nil]) + " against " + joinand([f3(r0["test_MRR"]) for _, r0, _ in nil])
                + " for the product in the " + num(len(nil)) + " splits (validation MRR "
                + joinand([f3(r1["valid_MRR"]) for _, _, r1 in nil]) + " against " + joinand([f3(r0["valid_MRR"]) for _, r0, _ in nil])
                + "; Supplementary Table S18). The failure of the trained Łukasiewicz model is therefore a matter of initialisation, "
                "not a property of the data. In the nilpotent range the loss and held-out accuracy again pointed in different "
                "directions: the loss was lowest at p > 0 in "
                + num(sum(1 for r in PS["intervals"] if r["p_argmin"] > 0)) + " of the splits, while test MRR at p > 0 was "
                + ("never higher than at the product." if all(max(r["test_MRR"] for r in rows_ if r["p"] > 0) <= r0["test_MRR"] + 1e-9
                                                            for (_, r0, _), rows_ in zip(nil, PS["rows"]))
                   else "higher than at the product in some splits and lower in others.")
                + " ")
    P_(dc, txt + "The "
           "per-metapath values learned from different starts were at best weakly correlated (median Spearman correlation "
           f"{f3(S['het_DpS']['p_agreement']['spearman_median'], 2)} on Hetionet disease–symptom), so no single learned value, "
           "of any metapath, should be interpreted.")

    # ---------------- 4.5 held-out diseases
    heading(dc, "4.5 Held-out diseases and evidence through known indications", 2)
    tg = "_tail"; nt = nseeds("het_CtD", "dnfr", tg)
    pn = pr("het_CtD", "dnfr_nodeg", "dnfr", tg)
    P_(dc, "A random split lets a test disease keep other indications in training. When every indication of a test disease was held "
           f"out together (Table 5A, {nt} splits), all methods fell sharply: DNFR reached {m('het_CtD','dnfr',tg)}, against "
           f"{m('het_CtD','dnfr')} with random splits. Path methods remained ahead of completion, popularity and the edge prior, which "
           f"carry no information about a disease without training indications ({m('het_CtD','softimpute',tg)}, "
           f"{m('het_CtD','popularity',tg)} and {m('het_CtD','prior',tg)}). Degree and prior features hurt DWPC-LR here "
           f"({m('het_CtD','dwpc_lr_full',tg)} against {m('het_CtD','dwpc_lr',tg)}), because a held-out disease has degree zero. "
           f"The variant without degree terms reached {m('het_CtD','dnfr_nodeg',tg)}, {dci(pn)} relative to DNFR, ahead on "
           f"{pn['seed_wins']} of {pn['seeds']} splits. A reversal of the random-split result is plausible, since for a disease "
           "without indications the remaining routes often pass through well-connected genes and drugs that the degree terms "
           f"discount, but the corrected interval {nbci(pn)} does not establish it, and we ran this split on one task only.")
    NT = R4.get("notgt"); XC = (R4.get("explain_dwpc") or {}).get("het_CtD", {}).get("dnfr")
    TL_DNFR = summ("het_CtD", "dnfr", "_tail")["MRR"]["mean"]
    if NT:
        q = NT["paired"]; mn = NT["means"]
        qd, qw, qp = q.get("dnfr_notgt__dnfr"), q.get("dwpc_lr_notgt__dwpc_lr"), q.get("product_notgt__product")
        txt = ("The disease-centred split removes every indication of a test disease. A second question is how much of the "
               "random-split accuracy rests on the indications that remain. On compound–treats–disease, "
               f"{R4['explain_dwpc']['het_CtD']['dnfr']['n_target_metapaths']} of the "
               f"{R4['explain_dwpc']['het_CtD']['dnfr']['n_metapaths']} metapaths contain the treats relation itself")
        if XC:
            txt += (f", and on the seed-0 split they carried {pct(XC['share_target_mean'])} of DNFR's evidence for its correct top-10 "
                    f"predictions (mean), against {100 * XC['share_mean']['gene']:.1f}% on metapaths through a gene the compound "
                    "targets (Table 6)")
        txt += (f". Without these metapaths (Table 5B, {qd['J']} random splits), DNFR fell from {f3(mn['dnfr']['mean'])} to "
                f"{f3(mn['dnfr_notgt']['mean'])} ({dci3(qd)}, {qd['verdict']})")
        if qw:
            txt += f", DWPC-LR from {f3(mn['dwpc_lr']['mean'])} to {f3(mn['dwpc_lr_notgt']['mean'])} ({qw['verdict']})"
        if qp:
            txt += f" and the fixed product from {f3(mn['product']['mean'])} to {f3(mn['product_notgt']['mean'])}"
        drop_share = (mn["dnfr"]["mean"] - mn["dnfr_notgt"]["mean"]) / max(mn["dnfr"]["mean"] - TL_DNFR, 1e-9)
        txt += (f". That is {pct(min(max(drop_share, 0), 1))} of the way from the random-split accuracy to the "
                f"disease-centred accuracy ({m('het_CtD','dnfr','_tail')}). ")
        txt += ("On a curated graph, then, the random-split accuracy of a path method on drug repurposing comes largely from "
                "guilt by association: a compound is ranked high for a disease because similar compounds, or compounds of the same "
                "pharmacologic class, are known to treat it or related diseases. The remaining relations, genes among them, carry "
                "much less of the signal. This holds for DWPC-LR as much as for DNFR, and it qualifies what any result on "
                "compound–treats–disease says about multi-relational reasoning."
                if qd["delta"] < 0 and qd["verdict"] == "different" else
                "The remaining relations carry most of the signal, so the random-split accuracy is not a matter of association "
                "with known indications alone.")
        P_(dc, txt)
    NP = R4.get("notgtp"); EV = R4.get("evidence_ablation") or {}
    if NP or EV:
        e1, e2 = EV.get("_notgt"), EV.get("_notgtp")
        txt = ("Hetionet also records compound–palliates–disease (CpD), a related clinical use of the same compound–disease kind. ")
        if e1 and e1.get("share_mean"):
            txt += (f"After the treats ablation, metapaths through CpD carried {pct1(e1['share_mean']['cpd'])} of DNFR's remaining "
                    f"evidence for its correct top-10 predictions on the seed-0 split (mean; median {pct1(e1['share_median']['cpd'])}, "
                    f"{e1['n_hits']} pairs), and gene-only metapaths {pct1(e1['share_mean']['gene'])}. ")
        if NP:
            q2, mn2 = NP["paired"], NP["means"]
            qa, qb = q2.get("dnfr_notgtp__dnfr_notgt"), q2.get("dnfr_notgtp__dnfr")
            nb_ = (e2 or {}).get("n_dropped_metapaths")
            txt += ("Without the metapaths that contain treats or palliates"
                    + (f" ({nb_} of {e2['n_metapaths']})" if nb_ else "")
                    + f" (Table 5C, {qa['J'] if qa else qb['J']} random splits), DNFR reached {f3(mn2['dnfr_notgtp']['mean'])}"
                    + (f", {dci3(qa)} relative to the treats ablation ({qa['verdict']})" if qa else ""))
            if mn2.get("dwpc_lr_notgtp"):
                txt += f"; DWPC-LR reached {f3(mn2['dwpc_lr_notgtp']['mean'])} and the fixed product {f3(mn2['product_notgtp']['mean'])}"
            ds2 = (mn2["dnfr"]["mean"] - mn2["dnfr_notgtp"]["mean"]) / max(mn2["dnfr"]["mean"] - TL_DNFR, 1e-9)
            txt += (f". That is {pct(min(max(ds2, 0), 1))} of the way from the random-split to the disease-centred accuracy. ")
        if e2 and e2.get("share_mean"):
            txt += (f"With both relations removed, gene-only metapaths carried {pct1(e2['share_mean']['gene'])} of DNFR's evidence "
                    f"(median {pct1(e2['share_median']['gene'])}). ")
        if NP and qa:
            close = ds2 > 0.8
            gsmall = bool(e2 and e2.get("share_mean") and e2["share_mean"]["gene"] < 0.05)
            txt += (("Removing palliates as well brought the accuracy close to the disease-centred level, which sharpens the "
                     "reading above: on this graph most of what a path method recovers is association with known clinical uses. "
                     if close else
                     "Removing palliates as well lowered the accuracy further"
                     + ("" if qa["verdict"] == "different" else " in its point estimate")
                     + ", but not to the disease-centred level, so association through palliates explains part, not all, of what "
                     "remains after the treats ablation. ")
                    + ("Even then the mechanistic metapaths carried only a small share of the evidence, so the conclusion that "
                       "genes carry much less of the signal holds after both relations are removed." if gsmall else
                       "With both relations removed, the mechanistic metapaths carried a larger share of the evidence."))
        P_(dc, txt)
    R.table_tail(dc)

    # ---------------- 4.6 sparsity
    heading(dc, "4.6 Robustness to missing edges", 2)
    P_(dc, "Removing edges of every relation at random, including the training edges of the target relation, lowered every method, "
           "and path methods stayed ahead of completion in that setting (Figure 3; Supplementary "
           f"Table S5). With a quarter of the edges left, DNFR reached {m('het_CtD','dnfr','_sp0.25')} on compound–treats–disease, "
           f"against {m('het_CtD','softimpute','_sp0.25')} for Soft-Impute. This comparison favours path methods by construction, "
           "because completion loses the very edges it is fitted to. The advantage of the degree terms vanished on the thinnest "
           f"graphs ({m('het_CtD','dnfr_nodeg','_sp0.25')} without them at 25% of the edges), consistent with their role: once most "
           "edges are gone, few nodes remain hubs."
           + (" To separate the robustness of path evidence from the loss of training edges, we also thinned every relation of "
              "Hetionet disease–symptom except the target relation to 25%: DNFR reached "
              f"{m('het_DpS','dnfr','_spt0.25')}, the product {m('het_DpS','product','_spt0.25')}, DWPC-LR "
              f"{m('het_DpS','dwpc_lr','_spt0.25')} and Soft-Impute, which now keeps all its training edges, "
              f"{m('het_DpS','softimpute','_spt0.25')} (Supplementary Table S5). With its training edges intact, completion was "
              "level with the best path method, so much of the advantage of path methods in the first setting comes from the "
              "loss of training edges rather than from the robustness of path evidence." if summ("het_DpS", "dnfr", "_spt0.25") else ""))
    figure(dc, f"{ROOT}/figures/fig4_sparsity.png", 6.0,
           "**Figure 3.** MRR with all edges and with 50% and 25% of the edges of every relation kept after random removal. Points are "
           "means and error bars standard deviations over splits (three splits on the thinned graphs; the all-edges points use the "
           "random splits of Table 2). The two panels have different y ranges.")

    # ---------------- 4.7 explanations
    heading(dc, "4.7 Explanations (RQ4)", 2)
    e = lambda t, k: E[t][k]
    XR = XREV
    P_(dc, "Because each score is a noisy-OR, deleting a path changes −log(1 − P) by exactly its share; the attribution is exact for "
           "the model's additive score, not for the underlying evidence. Paths that share a fact count it once per path: the three "
           "leading paths shared at least one fact for " + joinand([f"{pct(E2[t]['top3_share_edge'])} of pairs on {SH(t)}" for t in ("het_CtD", "het_DpS", "hkg_DpS") if t in E2])
           + ". Table 6 therefore reports the attribution both to paths and to facts.")
    R.table_expl(dc)
    P_(dc, "At the level of paths the explanations were less concise than a single chain: a test pair on Hetionet disease–symptom "
           f"was supported by a median of {e('het_DpS','n_paths_median'):.0f} paths, {e('het_DpS','k50_median'):.0f} of which "
           f"carried half of the evidence. Deleting the three leading paths left {pct(e('het_DpS','del_top3_keep10'))} of correct "
           "top-10 predictions in the top 10 there, but deleting random paths with the same total share left "
           f"{pct(E2['het_DpS']['stay_top10']['matched3'])}, "
           "so within a pair an additive score depends on how much evidence is removed, not on which paths carry it.")
    XD = R4.get("explain_dwpc", {})
    if E3:
        ts = [t for t in ("het_CtD", "het_DpS", "hkg_DpS") if t in E3]
        P_(dc, "At the level of facts the picture is clearer. Summing path shares, the leading fact appeared to carry "
               + joinand([f"{pct(E3[t]['top_edge_double_median'])} of the evidence on {SH(t)}" for t in ts])
               + "; its Shapley value, which splits each path term equally among its facts and sums to the total without double "
               "counting, was " + joinand([pct(E3[t]["top_edge_shapley_median"]) for t in ts]) + ". Half of the attribution lay on "
               + joinand([f"{E3[t]['k50_edges_median']:.0f}" for t in ts]) + " facts. "
               + ("The two criteria picked the same leading fact for " + joinand([pct(XD[t]["dnfr"]["same_shapley_summed"]) for t in ts if t in XD])
                  + " of the pairs, so double counting changes how much a fact appears to carry far more than which fact leads. "
                  if all(t in XD for t in ts) else "")
               + "Removing the fact with the largest Shapley share from the graph, so that competing candidates lost its evidence too, "
               "left " + joinand([f"{pct(E3[t]['stay_top10_edge_top_sh'])} of correct top-10 predictions on {SH(t)}" for t in ts])
               + " in the top 10, against " + joinand([pct(E2[t]["stay_top10"]["edge_rand"]) for t in ts if t in E2])
               + " after removing a random fact on the same paths. A correct ranking depends more on the facts the explanation "
               "names than on other facts on its paths, which is what a clinician checking an explanation would want to know, "
               "although most rankings survived the loss of their leading fact. This test was run without retraining, so it "
               "measures the dependence of the trained model on a fact, not what a model trained without it would predict.")
    if XD and all(t in XD for t in ("het_CtD", "het_DpS", "hkg_DpS")):
        ts = ("het_CtD", "het_DpS", "hkg_DpS"); w = lambda t, k: XD[t]["dwpc"][k]
        P_(dc, "DWPC-LR can be attributed to facts in the same way (Section 3.6), and the comparison shows what DNFR's exact "
               "decomposition adds. DWPC-LR's leading fact carried "
               + joinand([f"{pct(w(t, 'top_edge_share_median'))} of its positive attribution on {SH(t)}" for t in ts])
               + ", and half of that attribution lay on " + joinand([f"{w(t, 'k50_edges_median'):.0f}" for t in ts])
               + " facts. Removing its leading fact left " + joinand([pct(w(t, "stay_top10_cf_top")) for t in ts])
               + " of its correct top-10 predictions in the top 10, against " + joinand([pct(w(t, "stay_top10_cf_random")) for t in ts])
               + " for a random fact; for pairs in the top 10 of both models, DWPC-LR and DNFR named the same leading fact for "
               + joinand([f"{pct(w(t, 'same_lead_as_dnfr'))} ({w(t, 'n_both')} pairs)" for t in ts]) + ". "
               + (f"On compound–treats–disease, {pct(w('het_CtD', 'neg_share_median'))} of DWPC-LR's positive attribution was "
                  "offset by metapaths with negative coefficients (median), a cancellation that a noisy-OR of non-negative path "
                  "terms cannot produce. " if w("het_CtD", "neg_share_median") > 0.05 else "")
               + "The fact-level counterfactuals of the two models were similarly faithful. DNFR's attribution is exact and "
               "non-negative at the level of paths and facts; DWPC-LR's is exact at the level of metapaths, proportional within them, "
               "and mixes signs across them. That is the difference in kind between the two explanations; we found no evidence "
               "that DNFR's are more faithful in the sense tested here.")
    if "het_CtD" in E3 and E3["het_CtD"].get("lead_class"):
        lc = E3["het_CtD"]["lead_class"]; XC = (XD.get("het_CtD") or {}).get("dnfr")
        P_(dc, "For drug repurposing we classified the metapaths of each correct top-10 prediction. The leading metapath ran through "
               f"a gene that the compound binds or regulates and that is associated with the disease for {pct(lc['gene'])} of "
               f"predictions, and through pharmacologic classes, similar compounds or similar diseases for {pct(lc['associative'])}"
               + (f"; across all paths, the mechanistic metapaths carried {100 * XC['share_mean']['gene']:.1f}% of the evidence on "
                  f"average (median {100 * XC['share_median']['gene']:.2f}%)" if XC else "")
               + ". A mechanistic signal is present but small. Explanations of correct predictions are evidence by association, "
               "drugs like this one treat the disease, rather than a mechanism of action, in line with Section 4.5.")
    ex = {x["task"]: x for x in XS["explanations"]}
    P_(dc, "Table 7 shows the leading paths for one correctly top-ranked pair from each of three tasks. For doxorubicin and breast "
           f"cancer, {ex['het_CtD']['n_paths']:,} paths supported the prediction, and the strongest ran through the drug's "
           "pharmacologic classes, topoisomerase inhibitors and anthracyclines, to other drugs of those classes that Hetionet "
           "records as treating breast cancer. For multiple sclerosis and muscle weakness, the leading path ran through amyotrophic "
           "lateral sclerosis, which Hetionet links to multiple sclerosis by a \"resembles\" edge derived from text co-occurrence "
           "[@himmelstein2017], a weaker kind of evidence than a curated association. On the graded emergency-department graph, "
           "bronchitis was linked to fatigue through persistent cough, sore throat and congestion shared with pneumonia, "
           "streptococcal pharyngitis, mononucleosis and sinusitis. These examples illustrate the form of the explanations"
           + ("; a clinical assessment of a larger sample follows." if XR else
              "; they were not rated by clinicians, and we make no claim about their clinical plausibility or usefulness."))
    R.table_examples(dc)
    if XR and all(t in XR for t in ("het_DpS", "hkg_DpS")):
        def r(t, *k):
            v = XR[t]
            for x in k:
                v = v[x]
            return v
        H, G = "het_DpS", "hkg_DpS"
        XRN = XR[H].get("wrong_paths")
        sh = lambda t, *k: pct(r(t, *k, "share"))
        uns = lambda t: pct(1 - r(t, "relations_correct", "share") - r(t, "relations_wrong", "share"))
        P_(dc, f"The second author, a doctoral candidate in clinical psychology, rated {XR[H]['n_paths']} paths of "
               f"{XR[H]['n_pairs']} correctly ranked pairs on Hetionet disease–symptom and {XR[G]['n_paths']} paths of "
               f"{XR[G]['n_pairs']} pairs on the graded graph without seeing the model's scores (Section 3.6). Every relation on the "
               f"path agreed with biomedical knowledge for {sh(H, 'relations_correct')} of the paths on Hetionet and "
               f"{sh(G, 'relations_correct')} on the graded graph; at least one relation was judged wrong for "
               f"{sh(H, 'relations_wrong')} and {sh(G, 'relations_wrong')}, and the rest ({uns(H)} and {uns(G)}) could not be judged "
               "with confidence. "
               + (f"On Hetionet, {XRN['n_dla']} of the {XRN['n_no']} paths judged wrong ran from the disease to an anatomical structure and "
                  "back to another disease that localises there, and the model had already given the second edge a low "
                  f"membership (median {XRN['mu_mid']:.2f}, against {XRN['mu_yes']:.2f} for the weakest edge of paths judged "
                  "correct). With one rater and so few paths this is an observation, not evidence that low memberships flag "
                  "wrong edges. " if XRN else "")
               + f"Across all rated paths on Hetionet, {sh(H, 'rationale', 'M')} of the paths offered a mechanistic rationale, {sh(H, 'rationale', 'A')} an "
               f"associative one only and {sh(H, 'rationale', 'I')} none; the leading path was mechanistic more often than the second "
               f"and third ({sh(H, 'mechanistic_by_position', '1')} against {sh(H, 'mechanistic_by_position', '2')} and "
               f"{sh(H, 'mechanistic_by_position', '3')}). On the graded graph, where every path runs from the disease through a "
               f"symptom it shares with another disease, the shares were {sh(G, 'rationale', 'M')}, {sh(G, 'rationale', 'A')} and "
               f"{sh(G, 'rationale', 'I')}, and the leading path was mechanistic for only {sh(G, 'mechanistic_by_position', '1')}. "
               f"The rater judged {sh(H, 'pair_plausible')} and {sh(G, 'pair_plausible')} of the predicted pairs clinically "
               "plausible, and the median usefulness of the three leading paths to a clinician judging a prediction was "
               + (f"{r(H, 'usefulness', 'median')} of 5 on both tasks" if r(H, 'usefulness', 'median') == r(G, 'usefulness', 'median')
                  else f"{r(H, 'usefulness', 'median')} and {r(G, 'usefulness', 'median')} of 5")
               + f" (means {r(H, 'usefulness', 'mean'):.1f} and {r(G, 'usefulness', 'mean'):.1f}). The predictions were mostly "
               "plausible; the explanations were often a reason to look further rather than a reason to believe. Three limits "
               "apply. The rater is trained in clinical psychology, not in medicine, and judging whether a disease–anatomy–disease "
               "or disease–disease–symptom path agrees with biomedical knowledge calls for medical training; the share of paths "
               f"she could not judge ({uns(H)} and {uns(G)}) reflects this. Only correct predictions were rated, so this part of "
               "the assessment cannot show whether the explanations help to separate correct from incorrect predictions. And the "
               "rater, an author, knew the aims of the study.")
        X2 = XREV2
        if X2:
            a2, g1, g0 = X2["agreement"], X2["correct"], X2["incorrect"]
            kp = lambda k: "—" if a2[k]["kappa"] is None else f"{a2[k]['kappa']:.2f}"
            P_(dc, f"A second rater, {X2['rater_desc']}, not an author and blinded to the hypothesis, rated a further sheet "
                   f"without AI tools (Section 3.6): {g1['n_pairs']} correct pairs drawn from those of the first rater and "
                   f"{g0['n_pairs']} incorrect top-10 predictions, shuffled. On the shared pairs, agreement between the two raters "
                   f"was κ = {kp('relations_correct')} for whether the relations on a path are correct ({a2['relations_correct']['n']} "
                   f"paths), κ = {kp('rationale')} for the kind of rationale and κ = {kp('pair_plausible')} for whether the pair is "
                   f"plausible ({a2['pair_plausible']['n']} pairs). The second rater judged {pct(g1['plausible'])} of the correct "
                   f"and {pct(g0['plausible'])} of the incorrect predictions plausible, and rated the usefulness of their "
                   f"explanations at {g1['usefulness_mean']:.1f} and {g0['usefulness_mean']:.1f} of 5 on average"
                   + ("" if X2.get("auc_usefulness") is None else
                      f" (probability that a correct prediction's explanation was rated more useful than an incorrect one's, "
                      f"{X2['auc_usefulness']:.2f})")
                   + ". Some incorrect predictions may be true links that the graph lacks, so what this measures is how well the "
                   "explanations separate the graph's links from its non-links, not clinically true from false links.")

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
           + ". Timings were measured on a shared two-core machine and are indicative only. On compound–treats–disease, splits 0–2 "
           "took about "
           + f"{np.mean([e['s_per_epoch'] for e in R4.get('time_per_epoch_ctd', []) if e['seed'] < 3]):.1f}"
           + " s per epoch and the other splits about "
           + f"{np.mean([e['s_per_epoch'] for e in R4.get('time_per_epoch_ctd', []) if e['seed'] >= 3]):.1f}"
           + " s: the latter ran with one thread while a second job shared the machine, and the number of epochs did not differ "
           "systematically (Supplementary Table S14). The compound–treats–disease run time above mixes the two settings; on "
           "splits 0–2 alone a run took about "
           + f"{np.mean([e['dnfr_s'] for e in R4.get('time_per_epoch_ctd', []) if e['seed'] < 3]):.0f}"
           + " s, which is the better estimate of the cost of the method.")


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
           "0, and its minimiser differed between splits. On the graded graph held-out accuracy did depend on it, and choosing it "
           "on validation data helped where learning it did not; on Hetionet disease–symptom neither made a supported difference.")
    P_(dc, "Four refinements of the model did not improve it"
           + (", and combined they lowered accuracy on Hetionet disease–symptom" if (r3p("het_DpS", "dnfr2", "dnfr") or {}).get("verdict") == "different" else "")
           + ". Membership annealing had positive point estimates and remains untested at the precision needed to recommend it; the "
           "loss of the information-content discount on the graded graph came from the scale of its feature; and the overlap "
           "penalty did not raise held-out accuracy whatever the value its exponent took.")
    P_(dc, "As a predictor, DNFR was not more accurate than degree-weighted path counts with logistic regression on three of four "
           "tasks, was behind low-rank completion on the graded graph, and did not differ from a compact NBFNet under our rule, "
           + (lambda q: "" if not q else f"though NBFNet was ahead on {q['wins']} of {q['J']} splits on Hetionet disease–symptom and "
              f"had a higher mean MRR there than DWPC-LR ({f3(r3sum('het_DpS', 'nbfnet')['mean'])} against "
              f"{f3(summ('het_DpS', 'dwpc_lr')['MRR']['mean'])}); NBFNet was tuned on the graded graph only, and "
              "tuning it on Hetionet might widen that gap. ")(r3p("het_DpS", "nbfnet", "dnfr") if (r3p("het_DpS", "nbfnet", "dnfr") or {}).get("delta", 0) > 0 else None)
           + "For a "
           "developer who needs only accuracy on graphs like these, DWPC-LR is the better choice: it is simpler, faster and at "
           "least as accurate. What DNFR offers instead is an attribution that is exact and non-negative at the level of paths and "
           "facts, where DWPC-LR's is exact at the level of metapaths and proportional, and of mixed sign, below it. Our "
           "fact-level counterfactuals did not show DNFR's explanations to be more faithful. Whether the exact, non-negative form "
           "helps a clinician more than DWPC-LR's is a question for a user study, not one this paper answers.")
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
           "apart, which is one reason why the explanations in Table 7 include weak links and why learned memberships helped only "
           "where edges carried their own graded weight. Second, hubs in these graphs are biologically meaningful and are also the "
           "product of uneven research attention [@bonner2022]. Discounting them helped on random splits but perhaps not for "
           "diseases with no known indications (Section 4.5), where the remaining evidence runs through exactly these hubs. Third, "
           "the graded graph learned from patient records is a single bipartite relation, so its only metapath, "
           "disease–symptom–disease–symptom, links a disease to a symptom through diseases with similar symptom profiles. It tests "
           "the composition of graded collaborative evidence from records, not multi-relational reasoning, and it is the task on "
           "which low-rank completion led and on which the operator mattered most for held-out accuracy. Fourth, on curated "
           "drug-repurposing graphs most of the evidence for a correct prediction runs through known indications of similar "
           "compounds or related diseases (Section 4.5). Random splits reward that guilt by association, which is useful for "
           "repositioning within a drug class but says little about reasoning through mechanisms; claims about multi-relational "
           "reasoning on these graphs should be tested without the metapaths that contain the target relation, or on "
           "disease-centred splits.")
    heading(dc, "5.4 Implications for explainable clinical decision support", 2)
    bullets(dc, [C(x) for x in [
        "**Learn or fix memberships before operators.** A path reasoner over a medical graph should discount hubs, with a fixed "
        "DWPC-like discount as a strong default and a learned one where edges carry graded evidence. The operator can then be chosen "
        "among a few fixed, non-nilpotent t-norms on validation data; learning it end to end adds parameters without adding accuracy, "
        "and the learned value is not interpretable.",
        "**Aggregate evidence, do not select it.** Noisy-OR over all paths was consistently better than the strongest path, so an "
        "explanation should show the leading paths together with the share of the evidence they carry rather than a single "
        "\"reason\".",
        "**Attribute to facts, and test at the level of facts.** Path shares double-count facts shared by several paths; an edge-level "
        "Shapley attribution does not, and removing a fact from the graph and re-ranking tests what a clinician would question.",
        "**Choose the model by what the explanation must do.** Where accuracy is the goal, degree-weighted path counts are at least "
        "as good; where an exactly additive, non-negative attribution to facts is required, a noisy-OR path model provides it at "
        "similar accuracy.",
        "**Expect explanations to be by association, and test for it.** On drug repurposing, the evidence for correct predictions ran "
        "almost entirely through known indications of similar compounds or related diseases, and some disease–symptom paths through "
        "text-mined \"resembles\" links. Path shares make such evidence visible, and an ablation of the metapaths through the target "
        "relation measures how much of the accuracy depends on it."]], numbered=True, start=1)
    heading(dc, "5.5 Limitations", 2)
    nc = nseeds("het_CtD", "dnfr")
    bullets(dc, [C(x) for x in [
        f"**Splits and power.** The comparisons of Tables 2–4 used {num(nc)} splits on compound–treats–disease"
        + ctd_split_note()
        + ", ten on the two disease–symptom tasks and two on PrimeKG. With Holm adjustment within each family, differences of about "
        "0.02 to 0.04 MRR were often not supported even when their sign was the same on every split, and with ten splits the "
        "corrected 90% intervals still extend to about ±0.015, so equivalence within ±0.01 was rarely attainable. With two splits "
        "no equivalence verdict is possible, and PrimeKG results are descriptive.",
        "**PrimeKG covered only partly.** The refinements, the validation-selected t-norm, NBFNet, the disease-centred split and the "
        "ablation of metapaths through the target relation were not run on PrimeKG, whose path sets take several hours per run on "
        "our hardware. The zero-shot result, the most clinically relevant one, therefore rests on Hetionet compound–treats–disease "
        f"alone, with {nseeds('het_CtD','dnfr','_tail')} splits and 137 candidate diseases; whether removing the degree terms helps "
        "for unseen diseases on a realistic candidate set of 17,080 diseases remains open.",
        "**Identifiability on two graphs with one shared p.** The converged profile was run on the graded graph and on Hetionet "
        "disease–symptom, with one shared p; it does not test whether the data identify differences between metapaths, which the "
        "restarts suggest they do only weakly. The identifiability conclusion is therefore shown for the two disease–symptom tasks. "
        "On drug repurposing we rely on restarts, the unbounded parameterisation and the frozen-membership experiment, and on "
        "compound–treats–disease the loss has at least one poor optimum into which a fixed operator can fall (Section 4.2): a "
        "landscape that is flat in p at convergence on one graph need not be benign for every optimiser state on another.",
        "**Learned-operator baselines.** Our NBFNet is a compact CPU implementation (three layers, DistMult messages, sum "
        "aggregation), tuned over a 2 × 2 grid on the graded graph only and run on the two disease–symptom graphs; the original "
        "model with larger dimensions, and BioPathNet [@hu2026] and TxGNN [@huang2024] on the drug-repurposing graphs, require GPU "
        "training. TransE and RotatE were untuned and ComplEx did not train beyond chance under the same budget; embedding accuracy "
        "depends strongly on training configuration [@ruffinelli2020], so the gap is not a finding of this study.",
        "**Metapath budget.** Metapaths with more than five million paths were excluded for all path methods, including the "
        "side-effect and anatomy-expression metapaths on Hetionet and "
        f"{len(DS['pkg_ind']['skipped'])} of {DS['pkg_ind']['metapaths_all']} metapaths on PrimeKG, mostly through drug–drug "
        "interactions and contraindications. These are the most hub-dominated metapaths, so the effect of hub discounting may be "
        "underestimated.",
        (("**A limited clinical evaluation of explanations.** The first rater, a co-author trained in clinical psychology rather "
          "than medicine who knew the aims of the study, assessed 150 paths of 50 correct disease–symptom predictions. "
          + (f"A second, medically trained rater assessed 40 pairs, half of them incorrect; two raters on a few dozen pairs give "
             "only a rough estimate of agreement and discrimination. " if XREV2 else
             "Without a second rater no agreement can be estimated, and with correct predictions only no discrimination. ")
          + "Drug repurposing was not rated, and the mechanistic classification there is a structural check against Hetionet's own "
          "gene relations, not a clinical judgement. A rating study with several physicians, blinded to the hypothesis, on correct "
          "and incorrect predictions and comparing DNFR's and DWPC-LR's explanations, is the next step.")
         if XREV else
         "**No clinical evaluation of explanations.** The explanations were not rated by clinicians; Table 6 measures concision, "
         "attribution and the dependence of the ranking on the facts shown, and the mechanistic classification is a structural check "
         "against Hetionet's own gene relations, not a clinical judgement. A rating study with several clinicians on correct and "
         "incorrect predictions, comparing DNFR's and DWPC-LR's explanations, is the next step."),
        "**Scores are not probabilities,** and DNFR's score P orders candidates without being calibrated.",
        "**Missing data are not random.** The sparsity analysis removed edges uniformly at random; in practice, less-studied genes and "
        "diseases lose more edges.",
        "**Associations, not mechanisms.** Paths in these graphs combine curated, text-mined and co-occurrence evidence; a high-scoring "
        "path is a reason to look, not evidence of a causal mechanism. The graded graph was learned by others from records we did not "
        "access [@rotmensch2017;johnson2023]."]])


def conclusion(dc):
    heading(dc, "6. Conclusion", 1)
    two = (R3.get("profile_summary_dps") or {}).get("loss_range_nonpos") and max(R3["profile_summary_dps"]["loss_range_nonpos"]) < 0.01
    P_(dc, "In a controlled component analysis of differentiable neuro-fuzzy path reasoning on four tasks from three medical knowledge "
           "graphs, the accuracy came from aggregating all paths by noisy-OR and from discounting edges at hub nodes; learning the "
           "discount per relation helped only where edges carried graded clinical evidence. Four plausible refinements did not "
           "improve the model. "
           + ("On the graded graph and on Hetionet disease–symptom" if two else "On the graded graph")
           + ", the training loss hardly constrained the t-norm parameter once the memberships were learned; on the graded graph "
           "held-out accuracy depended on it, so the parameter should be selected on validation data and not interpreted when learned. "
           "Degree-weighted path counts were at least as accurate on three of four tasks, and on drug repurposing most of the "
           "evidence, for every path method, ran through known indications of similar compounds or related diseases. For "
           "explainable clinical decision support on knowledge graphs, effort is better spent on how much each fact counts, on how "
           "paths are combined, on attributing predictions to facts and on testing what the evidence rests on than on learning the "
           "operator that composes it.")


# ------------------------------------------------------------------ declarations
def declarations(dc):
    heading(dc, "7. Declarations", 1)
    P_(dc, "**Ethics.** The study uses only public, de-identified or aggregate data: Hetionet, PrimeKG, and the disease–symptom "
           "graph published by Rotmensch et al. [@rotmensch2017], whose underlying patient records were not accessed. Saigon "
           "International University does not require ethics approval for secondary analysis of public anonymised data.")
    P_(dc, "**Data availability.** Hetionet v1.0 is available from its GitHub repository and PrimeKG from Harvard Dataverse "
           "(doi:10.7910/DVN/IXA7BM). The parsed health knowledge graph, the per-query ranks of every method on every split, the "
           "trained models, the path explanations and the statistics files from which every number in this paper is generated are "
           f"deposited at Zenodo ({ZENODO_VERSION}all versions: "
           "https://doi.org/10.5281/zenodo.23879352), together with a script that downloads Hetionet and PrimeKG and checks them against the "
           "checksums of the files we used.")
    P_(dc, "**Code availability.** The implementation of DNFR and its refinements, the compact NBFNet and all baselines, the tests, "
           "and the scripts that compute the statistics and figures and build this manuscript from the result files are available "
           f"under the MIT licence at https://github.com/Bangth2026/dnfr-medical-kg (release {RELEASE}).")
    P_(dc, "**CRediT author statement.** Hai Bang Truong: Conceptualization; Methodology; Software; Formal analysis; Data curation; "
           "Visualization; Writing – original draft; Writing – review and editing. Van An Phan Thi (doctoral candidate in clinical psychology): "
           "Conceptualization (clinical framing of the prediction tasks, of the disease-centred split and of what an explanation "
           "should let a clinician check); Validation (a check that the relations on the explanation paths agree with biomedical "
           "knowledge, not only with the source graphs); "
           + ("Investigation (the clinical assessment of explanations in Section 4.7); " if XREV else "")
           + "Writing – original draft (the clinical interpretation in Sections 5.3–5.5); Writing – review and editing. All "
           "authors approved the final manuscript and agree to be accountable for all aspects of the work.")
    P_(dc, "**Independent verification.** Without the AI system, the corresponding author (Hai Bang Truong) checked every result "
           "reported in this article, in the text, the tables, the figures and the supplementary material, against the released "
           "result files."
           + (" A re-run of DNFR, DNFR with the strongest path only and DNFR without degree terms on the first split of the graded "
              "graph from a clean clone of the repository, in a fresh environment (scripts/verify_clean.sh), reproduced the released "
              "test MRR to four decimals (" + ", ".join(f"{VC['rows'][v]['rerun']:.4f}" for v in ("dnfr", "dnfr_max", "dnfr_nodeg"))
              + ")." if VC and all(abs(VC["rows"][v]["rerun"] - VC["rows"][v]["released"]) < 5e-5 for v in VC["rows"]) else ""))
    P_(dc, "**Declaration of generative AI and AI-assisted technologies in the manuscript preparation process.** During the preparation "
           "of this work the authors used Claude (Anthropic) to draft and edit the manuscript text and, as described in Section 3.6, "
           "to write and run the analysis code under the authors' direction. After using this tool, the authors reviewed and edited "
           "the content as needed and take full responsibility for the content of the publication.")
    P_(dc, "**Funding.** This research did not receive any specific grant from funding agencies in the public, commercial, or "
           "not-for-profit sectors.")
    P_(dc, "**Declaration of competing interest.** The authors declare no known competing financial interests or personal relationships "
           "that could have appeared to influence the work reported in this paper.")
    if XREV2:
        nm = {k.split(" /")[0].strip().lower(): v for k, v in (XREV2.get("rater") or {}).items()}
        who = ", ".join(str(nm[k]) for k in ("name", "qualification", "specialty") if nm.get(k))
        P_(dc, f"**Acknowledgements.** We thank {who or 'the physician who acted as second rater'} for rating the explanations "
               "in the second clinical assessment (Section 4.7).")


def supp_refinements(dc):
    """Per-refinement details moved from Section 4.3 (reviewer M5, minor 13)."""
    heading(dc, "S4. The four refinements in detail", 1)
    figure(dc, f"{ROOT}/figures/fig3_refinements.png", 6.3,
           "**Figure S3.** The refinements of Section 3.3 and the validation-selected fixed t-norm: difference in test MRR with its "
           "95% paired bootstrap interval, per task. Grey band, the equivalence margin ±0.01. Colour: blue, equivalent; red, "
           "different; grey, inconclusive (rule of Section 3.6). Numbers at the right give the splits run.")
    bt = R3.get("hkg_DpS", {}).get("beta", {})
    med = lambda x: float(np.median(x)) if x else float("nan")
    ov = {t: r3p(t, "dnfr_overlap", "dnfr") for t in R3T}
    P_(dc, "**R1, overlap-aware aggregation.** Added to DNFR, the overlap penalty changed MRR by "
           + joinand([f"{sg(ov[t]['delta'])} on {SH(t)} ({ov[t]['J']} splits)" for t in R3T if ov[t]])
           + f". The learned exponent β was {f3(med(bt.get('dnfr_overlap')), 3)} (median) on the graded graph alone and "
           f"{f3(med(bt.get('dnfr2')), 3)} within DNFR 2.0.")
    gu = {t: r3p(t, "dnfr_gumbel", "dnfr") for t in R3T}
    ops = R3.get("hkg_DpS", {}).get("ops_single") or {}
    ra = {t: R3.get(t, {}).get("restarts", {}).get("dnfr2") for t in R3T}
    txt = ("**R2, discrete operator choice.** Choosing among the product, the minimum and the Hamacher product instead of "
           "learning a continuous p changed MRR by " + joinand([f"{sg(gu[t]['delta'])} on {SH(t)} ({gu[t]['J']} splits)" for t in R3T if gu[t]])
           + ". ")
    if ops:
        n = sum(ops.values())
        txt += ("On the graded graph, which has a single metapath, DNFR 2.0 chose the Hamacher product on "
                f"{ops.get('hamacher', 0)} of {n} splits and the minimum on {ops.get('min', 0)}, never the product. ")
    if any(ra.values()):
        parts = []
        for t in R3T:
            r = ra[t]
            if not r:
                continue
            units = round(sum(r["op_counts"].values()) / 3)
            same = round(sum(r["all_same_per_split"][i] * units / r["n_splits"] for i in range(r["n_splits"])))
            parts.append(f"on {same} of {r['n_splits']} splits on {SH(t)}" if t == "hkg_DpS" else
                         f"for {same} of {units} metapath–split pairs on {SH(t)} ({r['n_splits']} splits)")
        txt += ("Restarted with two further random seeds, the three runs chose the same operator " + joinand(parts) + ". ")
    P_(dc, txt + "A discrete choice is easier to report than a continuous parameter, but it was neither more accurate nor more "
                 "reproducible.")
    icl = {t: r3p(t, "dnfr2", "dnfr2_no_ic") for t in R3T}; gl = {t: r3p(t, "dnfr2", "dnfr2_no_gumbel") for t in R3T}
    ic = {t: r3p(t, "dnfr_ic", "dnfr") for t in R3T}
    P_(dc, "**R3, information-content discount.** Added to DNFR, it changed MRR by "
           + joinand([f"{sg(ic[t]['delta'])} on {SH(t)} ({ic[t]['wins']}/{ic[t]['J']} splits)" for t in R3T if ic[t]])
           + ". Removed from DNFR 2.0, it changed MRR by " + joinand([f"{sg(-icl[t]['delta'])} on {SH(t)}" for t in R3T if icl[t]])
           + "; removing the discrete operator choice from DNFR 2.0 changed it by "
           + joinand([f"{sg(-gl[t]['delta'])} on {SH(t)}" for t in R3T if gl[t]]) + ".")
    tb = R3.get("hkg_DpS", {}).get("tau_best", {})
    if tb.get("dnfr_anneal"):
        P_(dc, f"**R4, membership annealing.** The best validation epoch fell at a temperature of {f3(med(tb['dnfr_anneal']), 2)} "
               "(median) on the graded graph, after most of the annealing, so the schedule mainly changed the final temperature.")


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
    heading(dc, "S2. Sparsity, long training and converged profiles", 1)
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
    heading(dc, "S3. Checks", 1)
    R.supp_prior(dc, 9)
    R.supp_valid(dc, 10)
    R.supp_identifiability(dc, 11)
    R.supp_rules(dc, 12)
    R.supp_hamacher(dc, 13)
    R.supp_profile(dc, 15, "profile_summary_dps", "Hetionet disease–symptom")
    R.supp_fixp_dps(dc, 17)
    R.supp_nilpotent(dc, 18)
    supp_refinements(dc)
    heading(dc, "References", 1)
    from references import REFS
    for i, k in enumerate(ORDER):
        para(dc, f"[{i + 1}] {REFS[k]}", align="left", size=10, line=1.15, space_after=3)
