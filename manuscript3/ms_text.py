"""Front matter, results text, discussion, conclusion, appendix B and declarations (second revision)."""
import numpy as np
from ms_common import *
from docx_helpers import para, heading, bullets, figure
import ms_results as R

P_ = lambda d, t, **k: para(d, C(t), **k)
E = XS["explain_eval"]; E2 = RS["explain2"]; L = RS.get("long", {}); PF = RS.get("profile", {})


def m(t, v, tag=""):
    s = summ(t, v, tag)
    return f3(s["MRR"]["mean"]) if s else "[NA]"


def d(t, a, b, tag=""):
    return pr(t, a, b, tag)


def dd(t, a, b, tag="", ci=True):
    p = pr(t, a, b, tag)
    return dci(p) if ci else sg(p["delta_MRR"])


def nbci(p):
    return f"[{sg(p['nb_ci95_lo'])}, {sg(p['nb_ci95_hi'])}]"


def nb90(p):
    return f"[{sg(p['nb_ci90_lo'])}, {sg(p['nb_ci90_hi'])}]"


def verdict_count(pairs):
    out = {"equivalent": 0, "different": 0, "inconclusive": 0}
    for t, a, b in pairs:
        p = pr(t, a, b)
        if p and p.get("verdict"):
            out[p["verdict"]] += 1
    return out


TN_PAIRS = [(t, "dnfr", b) for t in TASKS for b in ("product", "hamacher")]


def prof_row(t, p):
    for r in PF[t]["rows"]:
        if abs(r["p"] - p) < 1e-9:
            return r
    return None


def prof_summary(t):
    rows = PF[t]["rows"]; nonpos = [r for r in rows if r["p"] <= 0]; pos = [r for r in rows if r["p"] > 0]
    lo = min(nonpos, key=lambda r: r["final_loss"]); hi = max(nonpos, key=lambda r: r["final_loss"])
    return dict(nonpos_loss_range=hi["final_loss"] - lo["final_loss"], best_p=lo["p"],
                nonpos_test=(min(r["test_best"] for r in nonpos), max(r["test_best"] for r in nonpos)),
                pos=pos, rows=rows, argmin=PF[t].get("argmin_final_loss_by_seed"), seeds=PF[t]["seeds"])


# ------------------------------------------------------------------ front matter
def title_block(dc):
    from docx_helpers import heading as H
    para(dc, "**Path aggregation and hub discounting, not the learned t-norm: a controlled component analysis of differentiable "
             "neuro-fuzzy reasoning on medical knowledge graphs**", align="left", size=14, line=1.3, space_after=10)
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


def highlights(dc, H):
    para(dc, "**Highlights**", align="left", space_after=2)
    for h in H:
        assert len(h) <= 85, (len(h), h)
    bullets(dc, H)


# ------------------------------------------------------------------ abstract etc.
def abstract(dc):
    pt = {t: prof_summary(t) for t in ("hkg_DpS", "het_DpS") if t in PF}
    lh = L["hkg_DpS"]["by_init"]
    para(dc, "**Abstract**", align="left", space_after=2)
    P_(dc, "**Objective:** Path-based reasoning over medical knowledge graphs is explainable but composes evidence with an "
           "operator fixed in advance. We asked which parts of a neuro-fuzzy path reasoner are worth learning: t-norm, edge memberships or path aggregation.")
    P_(dc, "**Methods:** Differentiable Neuro-Fuzzy Reasoning (DNFR) aggregates paths by noisy-OR, composes each "
           "path with a Schweizer–Sklar t-norm learned per metapath, and learns degree-discounting edge memberships. On drug repurposing "
           "(Hetionet, PrimeKG) and disease–symptom prediction (Hetionet; a graph learned from emergency-department records) we "
           "compared ablations, fixed t-norms, restarts, a profile over fixed t-norm parameters and path, rule, completion and "
           "degree-only baselines under leave-dyad-out splits.")
    P_(dc, "**Results:** Keeping only the strongest path lowered mean reciprocal rank (MRR) by "
           f"{dd('het_CtD','dnfr','dnfr_max',ci=False)[1:]}–{dd('het_DpS','dnfr','dnfr_max',ci=False)[1:]}, and removing the degree "
           f"discount by {dd('het_DpS','dnfr','dnfr_nodeg',ci=False)[1:]}–{dd('het_CtD','dnfr','dnfr_nodeg',ci=False)[1:]}. On the "
           f"record-derived graph, training loss varied on average by {prof_row('hkg_DpS',-4)['final_loss_minus_seed_min']:.3f} or "
           f"less over fixed parameters from −4 to 0, and learned parameters did not converge, even without early stopping (median {f3(lh['-2']['p_final']['median'],1)} from −2, {f3(lh['-0.1']['p_final']['median'],2)} from −0.1). "
           f"Learned t-norms were not "
           f"consistently better "
f"than fixed product or Hamacher t-norms (differences ≤ {maxabs_tn():.3f}), though equivalence within ±0.01 was not shown. "
           f"Degree-weighted path counts did not differ detectably from DNFR on Hetionet drug repurposing ({m('het_CtD','dwpc_lr')} vs {m('het_CtD','dnfr')}), "
           f"trailed it on PrimeKG ({m('pkg_ind','dwpc_lr')} vs {m('pkg_ind','dnfr')}) and led on disease–symptom prediction. With all "
           f"indications of test diseases held out, MRR fell to about {f3(max(summ('het_CtD',x,'_tail')['MRR']['mean'] for x in ('dnfr','product','hamacher','dnfr_nodeg','dwpc_lr')),2)} "
           "or less.")
    P_(dc, "**Conclusion:** Path aggregation and hub discounting carry the gains. The loss is too shallow in the t-norm parameter for "
           "end-to-end learning to settle it; a learned value reflects initialisation and stopping, not the domain.")


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
        ["What this paper adds", "A controlled component analysis of a differentiable neuro-fuzzy path reasoner on four link-prediction "
                                 "tasks from three medical knowledge graphs. Aggregating all paths by noisy-OR and discounting hub nodes carried the gains. The training loss "
                                 "is so shallow in the t-norm parameter that its learned value depends on where it starts and when training "
                                 "stops, not on the data; degree-weighted path counts were better on two of four tasks and not detectably worse on a third."],
        ["Who would benefit", "Developers and evaluators of explainable clinical decision support built on knowledge graphs, and "
                              "researchers who learn or report fuzzy operators for medical reasoning."]],
        widths=[1.5, 4.77], size=9)


def long_minutes(t):
    import json
    d = json.load(open(f"{RES}/long_{t}.json"))
    return float(np.mean([r["time_s"] for r in d.values()])) / 60


SHORT = {"het_CtD": "compound–treats–disease", "het_DpS": "Hetionet disease–symptom", "hkg_DpS": "graded graph"}


def by_task(f):
    """'x on compound–treats–disease, y on Hetionet disease–symptom and z on the graded graph' for the explanation tasks."""
    ts = [t for t in ("het_CtD", "het_DpS", "hkg_DpS") if t in E2]
    parts = [f"{f(t)} on {'the ' if t == 'hkg_DpS' else ''}{SHORT[t]}" for t in ts]
    return ", ".join(parts[:-1]) + (" and " if len(parts) > 1 else "") + parts[-1]


def long_vs_default():
    import json
    var = {-2.0: "dnfr_pm2", -0.1: "dnfr_l2_0", 0.5: "dnfr_pp05"}; diffs = []
    for t in ("hkg_DpS", "het_DpS"):
        runs = {r["seed"]: r for r in json.load(open(f"{RES}/{t}.json"))["runs"]}
        for r in json.load(open(f"{RES}/long_{t}.json")).values():
            v = var[r["value"]]
            if v in runs[r["seed"]]["methods"]:
                diffs.append(r["best_valid"]["test_MRR"] - runs[r["seed"]]["methods"][v]["MRR"])
    return sg(min(diffs)), sg(max(diffs))


def num(k):
    return ["none", "one", "two", "three", "four", "five", "six", "seven", "eight"][k]


def ahead(p, a="ahead of", b="behind"):
    return a if p["delta_MRR"] > 0 else b


def maxabs_tn():
    return max(abs(pr(t, "dnfr", b)["delta_MRR"]) for t, _, b in TN_PAIRS if pr(t, "dnfr", b))


# ------------------------------------------------------------------ results
def results(dc):
    heading(dc, "4. Results", 1)
    heading(dc, "4.1 Accuracy against path, rule, completion and degree-only baselines (RQ1)", 2)
    P_(dc, "Table 2 summarises test MRR on the four tasks. Degree alone explained little of it. The edge prior of Zietz et al. "
           f"[@zietz2024], which uses the degrees of both endpoints, reached {m('het_CtD','prior')} on Hetionet compound–treats–disease, "
           f"{m('het_DpS','prior')} on disease–symptom and {m('hkg_DpS','prior')} on the graded graph, and popularity "
           f"{m('het_CtD','popularity')}, {m('het_DpS','popularity')}, {m('pkg_ind','popularity')} and {m('hkg_DpS','popularity')}, against "
           f"{m('het_CtD','dnfr')}, {m('het_DpS','dnfr')}, {m('pkg_ind','dnfr')} and {m('hkg_DpS','dnfr')} for DNFR. Under leave-dyad-out "
           "splits the target pair contributes nothing to its own degree, which is one reason why degree-only baselines are weaker here "
           "than when edges are reconstructed in the graph they came from [@zietz2024].")
    R.table2(dc)
    p1, p2, p3 = pr("het_CtD", "dnfr", "dwpc_lr"), pr("pkg_ind", "dnfr", "dwpc_lr"), pr("pkg_ind", "dnfr", "softimpute")
    q1, q2, q3 = pr("het_DpS", "dnfr", "dwpc_lr"), pr("het_DpS", "dnfr", "softimpute"), pr("hkg_DpS", "dnfr", "softimpute")
    P_(dc, "On Hetionet drug repurposing, methods that use paths were far ahead of completion of the treatment relation alone, "
           f"which has no access to the rest of the graph (Soft-Impute {m('het_CtD','softimpute')}); on PrimeKG, where a drug has more "
           f"indications on average, Soft-Impute ({m('pkg_ind','softimpute')}) came close to the path methods and exceeded DWPC-LR "
           f"({m('pkg_ind','dwpc_lr')}). Among path methods, DNFR "
           f"and DWPC with logistic regression did not differ detectably on Hetionet (DNFR minus DWPC-LR {dci(p1)}, corrected 95% interval "
           f"{nbci(p1)}), while on PrimeKG DNFR was ahead by {dci(p2)} on both splits and ahead of Soft-Impute by {dci(p3)}, a "
           f"difference that the corrected interval {nbci(p3)} does not establish with two splits. On "
           f"disease–symptom prediction DNFR was not the most accurate method: on Hetionet DWPC-LR was ahead (DNFR minus DWPC-LR "
           f"{dci(q1)}, {q1['seed_wins']}/{q1['seeds']} splits in favour of DNFR) and Soft-Impute was {sg(-q2['delta_MRR'])} ahead; "
           f"on the dense graded graph learned from emergency-department records Soft-Impute led ({m('hkg_DpS','softimpute')} against "
           f"{m('hkg_DpS','dnfr')}), the setting in which a low-rank description is expected to work well. Adding degree and prior "
           "features to DWPC-LR, as in Rephetio, did not help: its difference from DWPC-LR was within the equivalence margin on the "
           f"two disease–symptom tasks ({sg(pr('het_DpS','dwpc_lr_full','dwpc_lr')['delta_MRR'])} and "
           f"{sg(pr('hkg_DpS','dwpc_lr_full','dwpc_lr')['delta_MRR'])}) and {dci(pr('het_CtD','dwpc_lr_full','dwpc_lr'))} on "
           "compound–treats–disease. AnyBURL, given 100 s "
           f"of rule learning, reached {m('het_CtD','anyburl')}, {m('het_DpS','anyburl')} and {m('hkg_DpS','anyburl')}, below DNFR and "
           "DWPC-LR on each. The untuned embeddings were far below every path method; since their training was not tuned, we do not "
           "interpret that gap.")

    heading(dc, "4.2 What each learned component adds (RQ2)", 2)
    P_(dc, "Table 3 isolates the components. Two of them helped on every task on which they were tested, with the same sign on every "
           "split; the t-norm rows are judged by the decision rule of Section 3.5.")
    R.table3(dc)
    P_(dc, "**Aggregating paths.** Keeping only the strongest path instead of the noisy-OR of all paths lowered MRR by "
           f"{dd('het_CtD','dnfr','dnfr_max',ci=False)[1:]} on Hetionet compound–treats–disease, {dd('het_DpS','dnfr','dnfr_max',ci=False)[1:]} "
           f"on disease–symptom and {dd('hkg_DpS','dnfr','dnfr_max',ci=False)[1:]} on the graded graph. The number of routes between two "
           "entities carries information that the best single route does not. The max-path ablation was not run on PrimeKG.")
    fd = {t: pr(t, "dnfr", "dnfr_fixdeg") for t in TASKS}
    s02, s06 = pr("hkg_DpS", "product_fixdeg", "product_fixdeg02"), pr("hkg_DpS", "product_fixdeg", "product_fixdeg06")
    P_(dc, "**Discounting hubs.** Removing the degree terms of Equation 1 lowered MRR by "
           + ", ".join(dd(t, 'dnfr', 'dnfr_nodeg', ci=False)[1:] for t in TASKS[:3]) + f" and {dd('hkg_DpS','dnfr','dnfr_nodeg',ci=False)[1:]} "
           "on the four tasks. Whether the discount has to be learned per relation depended on the graph. Learned coefficients were "
           f"better than a fixed b~r~ = c~r~ = 0.4 on the graded graph ({dci(fd['hkg_DpS'])}). Elsewhere the fixed discount was at "
           f"least as good: learned minus fixed was {dci(fd['het_CtD'])} on Hetionet compound–treats–disease (different under our "
           f"rule, corrected interval {nbci(fd['het_CtD'])}), {dci(fd['het_DpS'])} on Hetionet disease–symptom (inconclusive) and "
           f"{dci(fd['pkg_ind'])} on PrimeKG. On the graded graph the fixed product with discount 0.4 was "
           f"{sg(s02['delta_MRR'])} [{sg(s02['ci_lo'])}, {sg(s02['ci_hi'])}] against a discount of 0.2 and {sg(s06['delta_MRR'])} "
           f"[{sg(s06['ci_lo'])}, {sg(s06['ci_hi'])}] against 0.6, so the value we took from DWPC is not a tuned optimum. Discounting "
           "hubs matters everywhere; learning the discount per relation matters only on the graded graph. Reading the graded weights "
           f"added {dci(pr('hkg_DpS','dnfr','dnfr_binary'))} on that graph.")
    tp = {t: pr(t, "dnfr", "product") for t in TASKS}; th = {t: pr(t, "dnfr", "hamacher") for t in TASKS}
    vc = verdict_count(TN_PAIRS)
    P_(dc, "**Learning the t-norm.** A Schweizer–Sklar parameter learned per metapath was not consistently better than fixed members of "
           "its own family. Against the fixed product the differences were " + ", ".join(sg(tp[t]['delta_MRR']) for t in TASKS)
           + " on the four tasks, and against the fixed Hamacher product " + ", ".join(sg(th[t]['delta_MRR']) for t in TASKS) + ". "
           f"Under our rule {num(vc['different'])} of these eight comparisons {'was' if vc['different'] == 1 else 'were'} different, "
           f"{num(vc['equivalent'])} equivalent and {num(vc['inconclusive'])} inconclusive. Judged by bootstrap intervals alone, as in "
           "the first version of this paper, the learned t-norm is ahead of the product and of the Hamacher product on the graded "
           f"graph ({dci(tp['hkg_DpS'])} and {dci(th['hkg_DpS'])}) and behind the product on PrimeKG ({dci(tp['pkg_ind'])}, two "
           "splits); the gain over the product on Hetionet disease–symptom that the first version reported with five splits is "
           f"{dci(tp['het_DpS'])} with ten. The corrected intervals, which include variation between splits, are wider: on the graded graph "
           f"{nbci(tp['hkg_DpS'])} against the product and {nbci(th['hkg_DpS'])} against the Hamacher product. The largest difference "
           f"in either direction was {maxabs_tn():.3f}, and with ten splits the 90% corrected intervals still extend to about ±0.015, so "
           "we cannot claim that fixed and learned t-norms are equivalent within ±0.01 either; we claim only that learning the t-norm "
           "gave no consistent gain. "
           f"One p per metapath differed from a single shared p by {sg(pr('het_CtD','dnfr','dnfr_shared')['delta_MRR'])}, "
           f"{sg(pr('het_DpS','dnfr','dnfr_shared')['delta_MRR'])} and {sg(pr('pkg_ind','dnfr','dnfr_shared')['delta_MRR'])} where the "
           "shared variant was run.")
    fr = {t: pr(t, "dnfr_frozen", "product_frozen") for t in TASKS}; lm = {t: pr(t, "product", "product_frozen") for t in TASKS}
    P_(dc, "The t-norm mattered most when the memberships could not be learned. With memberships frozen at their initial values, "
           f"learning p improved on the product by {dci(fr['hkg_DpS'])} on the graded graph and by {dci(fr['het_DpS'])} on Hetionet "
           f"disease–symptom, and learning the memberships instead of the t-norm improved the product by {dci(lm['hkg_DpS'])} on the "
           "graded graph. Learned memberships and a learned operator are, to a large extent, substitutes, as one would expect from "
           "flexible neuro-fuzzy systems in which weights and t-norm parameters both scale the strength of a rule [@rutkowski2003].")

    heading(dc, "4.3 Is the t-norm parameter identified?", 2)
    st = RS["stopping"]
    bt = {t: st[t]["dnfr"]["best_epoch"]["median"] for t in TASKS if t in st}
    P_(dc, "With the default early stopping, the learned p ended close to where it started (Table 4). The default runs reached their "
           "best validation MRR at a median epoch of " + ", ".join(f"{ri(bt[t])}" for t in TASKS) + " on the four tasks. On "
           f"compound–treats–disease, {ri(bt['het_CtD'])} Adam steps of at most about 0.1 each in θ~m~ leave p near p~0~ whatever the "
           "data say; on the other tasks training ran longer, and the penalty and the shallow loss described below kept p close to its "
           "start. Table 4 is therefore a statement about the default training procedure, not about the objective. We separated the two "
           "with training without early stopping (Table 5, Figure 3) and a profile over fixed values of p (Figure 2).")
    R.table4(dc)
    lh, ld = L["hkg_DpS"], L.get("het_DpS")
    P_(dc, f"**Without early stopping** (Table 5, Figure 3), p still did not converge to a common value. On the graded graph, which has a "
           f"single metapath, p ended at a median of {f3(lh['by_init']['-2']['p_final']['median'],2)} from p~0~ = −2 "
           f"(range {f3(lh['by_init']['-2']['p_final']['min'],2)} to {f3(lh['by_init']['-2']['p_final']['max'],2)} over splits), "
           f"{f3(lh['by_init']['-0.1']['p_final']['median'],2)} from −0.1 and {f3(lh['by_init']['0.5']['p_final']['median'],2)} from 0.5. "
           "Started at −2, p moved between about −4.2 and −1.4 before settling (Figure 3), with a gradient norm of the order of "
           f"10^−4^; started at −0.1, it reached about −1, the Hamacher product, within 100 epochs and then returned to the product as the memberships adapted. "
           + (f"On Hetionet disease–symptom ({len(ld['seeds'])} splits) the per-metapath values spread from "
              f"{f3(min(ld['by_init'][k]['p_final']['min'] for k in ld['by_init']),1)} to "
              f"{f3(max(ld['by_init'][k]['p_final']['max'] for k in ld['by_init']),1)}, with medians of "
              + ", ".join(f"{f3(ld['by_init'][k]['p_final']['median'],2)} from p~0~ = {k.replace('-', '−')}" for k in ld['by_init'])
              + "; the per-metapath vectors from different starts had a median Spearman correlation of "
              f"{f3(ld['agreement']['spearman']['median'],2)}, and only {pct(ld['agreement']['sign_agree'])} of metapaths had the same "
              "sign from every start. " if ld else "")
           + "On the same splits, training for 400 epochs (with the parameters of the best validation epoch) changed test MRR relative "
           "to the corresponding early-stopped runs by "
           f"{long_vs_default()[0]} to {long_vs_default()[1]}, so longer training did not consistently help.")
    ph_ = prof_summary("hkg_DpS")
    hrow = lambda p: prof_row("hkg_DpS", p)
    ex = lambda p: hrow(p)["final_loss_minus_seed_min"]
    am = ph_["argmin"] or []
    same = len(set(am)) == 1
    txt = ("**Profile.** With p fixed and every other parameter trained for up to 300 epochs, the training loss on the graded "
           "graph after training was lowest at "
           + (f"p = {am[0]:g} in every split".replace("-", "−") if same else
              "different values in different splits (" + ", ".join(f"{x:g}".replace("-", "−") for x in am) + ")")
           + f" (Figure 2), but the differences were small: the loss in excess of each split's minimum averaged {ex(-1):.3f} at "
           f"p = −1, {ex(-0.5):.3f} at p = −0.5, {ex(-4):.3f} at p = −4 and {ex(0):.3f} at the product, on a loss of about 4.3. None "
           "of these runs met the convergence criterion; the loss was still falling by "
           f"{min(r['loss_drop_last50'] for r in ph_['rows'] if r['p'] <= 0):.3f} to "
           f"{max(r['loss_drop_last50'] for r in ph_['rows'] if r['p'] <= 0):.3f} per 50 epochs at the epoch limit, as much as the "
           "differences between values of p, so the profile locates no minimum within the non-positive range. For p ≥ 0.5 the loss "
           "was far higher; at p = 0.5 every path started with strength zero, no gradient reached any parameter, and the "
           f"model did not train (test MRR {f3(hrow(0.5)['test_best'])}). Held-out accuracy was clearer than the loss: validation MRR "
           "was highest at p ≤ −0.5 in every split (at " + ", ".join(f"{x:g}".replace("-", "−") for x in PF['hkg_DpS']['argmax_valid_by_seed'])
           + f"), and test MRR was {f3(hrow(-2)['test_best'])}, {f3(hrow(-1)['test_best'])} and {f3(hrow(-0.5)['test_best'])} at "
           f"p = −2, −1 and −0.5 against {f3(hrow(0)['test_best'])} at the product, higher in every split and a larger difference "
           "than any between t-norms trained with early stopping (Table 3). ")
    if "het_DpS" in PF:
        pd_ = prof_summary("het_DpS"); drow = lambda p: prof_row("het_DpS", p)
        dam = pd_["argmin"] or []
        dx = lambda p: drow(p)["final_loss_minus_seed_min"]
        dsame = len(set(dam)) == 1
        txt += ("On Hetionet disease–symptom, with one p for all metapaths and a coarser grid (−4, −1, 0, 0.5, 0.75), the loss was lowest at "
                + (f"p = {dam[0]:g} in both splits".replace("p = 0 in", "the product in").replace("-", "−") if dsame else
                   "different values in the two splits")
                + f", validation MRR also peaked there, and the excess loss was {dx(-1):.3f} at p = −1 and {dx(-4):.3f} at p = −4; "
                f"test MRR was {f3(drow(0)['test_best'])} at the product, {f3(drow(-1)['test_best'])} at p = −1 and "
                f"{f3(drow(-4)['test_best'])} at p = −4. ")
    ldm = L["het_DpS"]["by_init"]["-0.1"]["p_final"]["median"] if "het_DpS" in L else None
    txt += ("Held-out accuracy thus prefers a region of the family on each task, stronger-than-product t-norms on the graded graph "
            "and the neighbourhood of the product on Hetionet disease–symptom, while the training loss varies across that region by "
            "little more than its residual decrease, and gradient-based training does not reach the preferred region reliably. From p~0~ = −0.1 the learned p on the graded graph passed through −1 and returned "
            "to the product as the memberships adapted, and from −2 it overshot to about −4"
            + (f"; on Hetionet disease–symptom, learned per metapath from −0.1, it ended at a median of {f3(ldm,2)}, away from the "
               "product" if ldm is not None else "")
            + " (Table 5, Figure 3). Which value training reports depends on where it starts, on the penalty and on when it stops.")
    P_(dc, txt)
    figure(dc, f"{ROOT}/figures/fig2_profile.png", 6.3,
           "**Figure 2.** Profile over a fixed t-norm parameter p (shared by all metapaths). Top: final training loss minus its minimum "
           "over the grid, per split. Bottom: test MRR of the parameters with the best validation MRR. Every other parameter was trained "
           "without early stopping until the loss stopped decreasing or the epoch limit was reached. Dashed lines mark the Hamacher "
           "product (p = −1), the product (p = 0) and the Łukasiewicz t-norm (p = 1). At p = 0.5 on the graded graph the model did not "
           "train (all paths have strength zero); its loss lies far above the plotted range and is omitted from the top panel.")
    R.table5(dc)
    figure(dc, f"{ROOT}/figures/fig3_trajectories.png", 6.3,
           "**Figure 3.** Learned p during 400 epochs without early stopping and without penalty, by initial value p₀. Graded graph: one "
           "line per split (single metapath). Hetionet disease–symptom: thin lines are the metapaths of the first split, thick lines the "
           "median over metapaths for each split. Dotted vertical lines mark the epoch with the best validation MRR.")
    P_(dc, "We conclude that the learned t-norm parameter is set by optimisation, that is by initialisation, penalty and stopping, "
           "rather than by the data: the training loss is so shallow in p that its gradient does not move p to the region that "
           "held-out accuracy prefers. The data determine two things. They exclude the nilpotent end of the family on graded evidence, where the "
           f"Łukasiewicz t-norm and starts at p~0~ = 0.5 lost {dd('hkg_DpS','dnfr','luk',ci=False)[1:]} and "
           f"{sg(-pr('hkg_DpS','dnfr_pp05','dnfr')['delta_MRR'])[1:]} in MRR; and on the graded graph they favour a broad region of "
           "stronger-than-product t-norms (and on Hetionet disease–symptom the neighbourhood of the product), which a profile or a "
           "search on validation data finds and end-to-end learning misses. In the "
           "restarts with early stopping, the per-metapath values from different starts had a median Spearman correlation of "
           f"{f3(S['het_DpS']['p_agreement']['spearman_median'],2)} on Hetionet disease–symptom and "
           f"{f3(S['het_CtD']['p_agreement']['spearman_median'],2)} on compound–treats–disease, and the same sign for "
           f"{pct(S['het_DpS']['p_agreement']['sign_agree_mean'])} and {pct(S['het_CtD']['p_agreement']['sign_agree_mean'])} of "
           "metapaths, so the relative ordering of metapaths is at best weakly reproducible. The DrD–DpS metapath of Table 8, with "
           "p = −2.78 under the default penalty, is one of the few that moved far from its start.")

    heading(dc, "4.4 Held-out diseases", 2)
    tg = "_tail"; nt = nseeds("het_CtD", "dnfr", tg)
    pn = pr("het_CtD", "dnfr_nodeg", "dnfr", tg)
    P_(dc, "A random split lets a test disease keep other indications in training. When every indication of a test disease was held "
           f"out together (Table 6, {nt} splits), all methods fell sharply: DNFR reached {m('het_CtD','dnfr',tg)}, against "
           f"{m('het_CtD','dnfr')} with random splits. Path methods remained ahead of completion, popularity and the edge prior, which "
           f"carry no information about a disease without training indications ({m('het_CtD','softimpute',tg)}, "
           f"{m('het_CtD','popularity',tg)} and {m('het_CtD','prior',tg)}), and at this level of accuracy the differences among the DNFR "
           "variants and DWPC-LR are small. Degree and prior features, harmless on random splits, hurt DWPC-LR here "
           f"({m('het_CtD','dwpc_lr_full',tg)} against {m('het_CtD','dwpc_lr',tg)}), because a held-out disease has no training "
           "indications, so its degree is zero and its prior near zero. The variant without degree terms reached "
           f"{m('het_CtD','dnfr_nodeg',tg)}, "
           f"{dci(pn)} relative to DNFR, ahead on {pn['seed_wins']} of {pn['seeds']} splits; a reversal of the random-split result is "
           "plausible, since for a disease without indications the remaining routes often pass through well-connected genes and drugs "
           "that the degree terms discount, but the corrected interval "
           f"{nbci(pn) if 'nb_ci95_lo' in pn else ''} does not establish it.")
    R.table_tail(dc)

    heading(dc, "4.5 Robustness to missing edges (RQ3)", 2)
    P_(dc, "Removing edges at random lowered every method, but path methods stayed ahead of completion (Table A5, Figure 4). With a "
           f"quarter of the edges left, DNFR still reached an MRR of {m('het_CtD','dnfr','_sp0.25')} on compound–treats–disease, against "
           f"{m('het_CtD','softimpute','_sp0.25')} for Soft-Impute. This comparison is structurally unfavourable to completion, which "
           "loses the very treatment edges it is fitted to as the graph thins, whereas path methods can route around them. The "
           f"advantage of the degree terms, large on the full graph, vanished on the thinnest graphs ({m('het_CtD','dnfr_nodeg','_sp0.25')} "
           "without them at 25% of the edges), consistent with their role: once most edges are gone, few nodes remain hubs. DNFR and "
           "the fixed product were within 0.01 of each other on both thinned graphs.")
    figure(dc, f"{ROOT}/figures/fig4_sparsity.png", 6.0,
           "**Figure 4.** MRR with all edges and with 50% and 25% of the edges of every relation kept after random removal. Points are "
           "means and error bars standard deviations over splits (three splits on the thinned graphs; the all-edges points use all "
           "random splits of Table 2). The two panels have different y ranges.")

    heading(dc, "4.6 Explanations (RQ3)", 2)
    P_(dc, "Because each score is a noisy-OR, deleting a path changes −log(1 − P) by exactly its share, so the attribution is exact for "
           "the additive score with all other scores fixed. Two qualifications matter for how a reader should interpret a share. "
           "First, paths that share an edge count that edge once per path, so the shares of overlapping paths partly double-count the "
           "same fact; the three leading paths shared at least one edge for "
           + by_task(lambda t: f"{pct(E2[t]['top3_share_edge'])} of pairs") + ". Second, deleting paths from one pair tests only how much of its evidence is removed. Table 7 reports both kinds of test.")
    R.table_expl(dc)
    e = lambda t, k: E[t][k]
    P_(dc, "The explanations were less concise than a single chain. A test pair on Hetionet disease–symptom was supported by a median of "
           f"{e('het_DpS','n_paths_median'):.0f} paths, and {e('het_DpS','k50_median'):.0f} of them carried half of the evidence; on "
           f"compound–treats–disease the corresponding numbers were {e('het_CtD','n_paths_median'):.0f} and {e('het_CtD','k50_median'):.0f}. "
           "At the level of metapaths they were concise: a single metapath carried at least half of the evidence for the median pair on "
           "every task. Deleting the three leading paths left "
           f"{pct(e('het_DpS','del_top3_keep10'))} of correct top-10 predictions on Hetionet disease–symptom in the top 10, "
           f"{pct(e('hkg_DpS','del_top3_keep10'))} on the graded graph and {pct(e('het_CtD','del_top3_keep10'))} on "
           "compound–treats–disease, whereas deleting three random paths of the same pair left almost all of them. The share-matched "
           "control shows what this contrast measures: deleting random paths until their share equalled that of the top three left "
           + by_task(lambda t: pct(E2[t]['stay_top10']['matched3'])) + " of them in the top 10, close to the top-three deletion, and the curves of rank against the share removed were nearly the same for "
           "leading and random paths. Within a pair, an additive score depends on how much evidence is removed, not on which paths "
           "carry it; the top-3 against random-3 contrast mainly confirms that shares are skewed.")
    P_(dc, "The edge-level counterfactual is more informative. Removing the single fact that carried the largest share of a pair's "
           "evidence, and with it every path from the same head that used it, left "
           + by_task(lambda t: f"{pct(E2[t]['stay_top10']['edge_top'])} (random fact: {pct(E2[t]['stay_top10']['edge_rand'])})")
           + " of correct top-10 predictions in the top 10; the control removed a random fact on the pair's paths. The leading fact "
           "carried a median of " + by_task(lambda t: pct(E2[t]['top_edge_share'])) + " of the evidence. A correct ranking thus depends more on the facts the explanation names than on other facts on the same "
           "paths, which is what a clinician checking an explanation would want to know, although most correct rankings survived the "
           "loss of their leading fact; this test was run without retraining, so it measures the dependence of the "
           "trained model on a fact, not what the model would have learned without it. DWPC with logistic regression gives a comparable "
           "attribution by metapath; its leading metapath coincided with that of DNFR for "
           f"{pct(e('het_DpS','same_top_mp'))} of the pairs on Hetionet disease–symptom and {pct(e('het_CtD','same_top_mp'))} on "
           "compound–treats–disease, and deleting it removed a similar share of correct predictions. The advantage of DNFR for "
           "explanation is therefore resolution, individual paths and facts with memberships rather than metapath totals, not "
           "faithfulness, which both additive models have.")
    ex = {x["task"]: x for x in XS["explanations"]}
    P_(dc, "Table 8 shows the leading paths for one correctly top-ranked pair from each of three tasks. For doxorubicin and breast cancer, "
           f"{ex['het_CtD']['n_paths']:,} paths supported the prediction, and the strongest ran through the drug's pharmacologic classes, "
           "topoisomerase inhibitors and anthracyclines, to other drugs of those classes that Hetionet records as treating breast cancer. "
           "This is evidence by association, drugs like this one treat the disease, not a mechanism of action, and the three leading "
           f"paths carried only {pct(ex['het_CtD']['top3_share'])} of the evidence. For multiple sclerosis and muscle weakness, the "
           "leading path ran through amyotrophic lateral sclerosis, which Hetionet links to multiple sclerosis by a \"resembles\" edge "
           "derived from text co-occurrence [@himmelstein2017], a weaker kind of evidence than a curated association. On the graded "
           "emergency-department graph, bronchitis was linked to fatigue most strongly through persistent cough, sore throat and congestion "
           "shared with pneumonia, streptococcal pharyngitis, mononucleosis and sinusitis. These examples illustrate the form of the "
           "explanations; they were not rated by clinicians, and we make no claim about their clinical plausibility or usefulness.")
    R.table_examples(dc)

    heading(dc, "4.7 Computational cost", 2)
    ts = {t: S[t]["time_s"] for t in TASKS}
    P_(dc, "One training run of DNFR took about " + ", ".join(f"{ts[t]['dnfr']:.0f} s" for t in TASKS[:3])
           + f" and {ts['hkg_DpS']['dnfr']:.0f} s on the four tasks with the default early stopping, after a one-off path enumeration; "
           "DWPC with logistic regression took " + ", ".join(f"{ts[t]['dwpc_lr']:.1f}" for t in TASKS) + " s on the same path sets. "
           + f"Training for 400 epochs without early stopping took about {long_minutes('hkg_DpS'):.0f} min per run on the graded graph and {long_minutes('het_DpS'):.0f} min on Hetionet disease–symptom. "
           + "Timings were measured on a shared two-core machine with one or "
           "two threads per run and are indicative only.")


# ------------------------------------------------------------------ discussion
def discussion(dc):
    heading(dc, "5. Discussion", 1)
    heading(dc, "5.1 Principal findings", 2)
    P_(dc, "We set out to find which learned components of a neuro-fuzzy path reasoner matter for multi-hop reasoning on medical "
           "knowledge graphs. Across four tasks the answer was consistent. Combining all paths by noisy-OR rather than keeping the best "
           "one, and discounting edges at hub nodes, carried the gains that the model had over its ablations. Learning the discount per "
           "relation helped on the graph with graded clinical evidence and not elsewhere, where a fixed DWPC-like discount was as good "
           "or better. The t-norm itself, the most visibly neuro-fuzzy part of the model, was set by optimisation rather than by the data: "
           "the training loss was shallow in its parameter over the non-positive range, the learned value ended wherever initialisation, "
           "penalty and stopping left it, also after 400 epochs without early stopping, and learning it gave no consistent gain over "
           "the fixed product or Hamacher product. The data did reject the nilpotent end of the family on graded evidence, and on the "
           "graded graph held-out accuracy favoured a broad region of stronger-than-product t-norms that end-to-end learning did not reach. It mattered most when "
           "the memberships were frozen, which shows that learned memberships and a learned operator do much the same work.")
    P_(dc, f"As a predictor, DNFR did not differ detectably from degree-weighted path counts on Hetionet drug repurposing, was more "
           "accurate on PrimeKG, and was less accurate than DWPC with logistic regression on both disease–symptom tasks and than "
           "low-rank completion on the graded graph. Degree-only "
           "baselines, including an edge prior that accounts for the degrees of both endpoints, were far behind every path method, so "
           "the accuracy of the path methods is not an artefact of degree. When test diseases had no indications in training, every "
           f"method fell to an MRR of about {f3(max(summ('het_CtD',x,'_tail')['MRR']['mean'] for x in ('dnfr','product','hamacher','dnfr_nodeg','dwpc_lr')),2)} "
           "or less. The distinctive contribution of DNFR is the resolution of its explanations, path by path and fact by fact with "
           "edge memberships, rather than accuracy or faithfulness.")
    heading(dc, "5.2 Why end-to-end learning does not identify the t-norm parameter", 2)
    P_(dc, "The profile gives a direct answer: over p ∈ [−4, 0] the training loss differs by at most a few hundredths of a nat, "
           "on the graded graph about as much as it still decreases per 50 epochs late in training, so its gradient with respect to p "
           "is small and its sign changes as the other parameters move (Figure 3). Two mechanisms make the "
           "loss insensitive. First, the degree terms and the metapath reliabilities already rescale every path. A t-norm with p < 0 raises "
           "the strength of a path with one weak edge relative to the product; the same change in ranking can be produced by raising "
           "the membership of weak edges, which the membership parameters do per relation, or by raising the reliability of the "
           "metapath. When all three are learned together, many combinations rank the candidates in nearly the same order. The "
           "frozen-membership experiment supports this reading: once the memberships could not move, learning p recovered part of what "
           "learning the memberships gave. Second, the softmax loss averages over all candidate tails, and changes in p that reorder a "
           "few top-ranked candidates barely change it. That is why, on the graded graph, test MRR differed by "
           f"{f3(prof_row('hkg_DpS', -1)['test_best'] - prof_row('hkg_DpS', 0)['test_best'], 2)} between p = −1 and the product while "
           f"the training loss differed by only {prof_row('hkg_DpS',0)['final_loss_minus_seed_min'] - prof_row('hkg_DpS',-1)['final_loss_minus_seed_min']:.3f}. Neither mechanism is specific to our model, and the first was anticipated in flexible "
           "neuro-fuzzy systems, where t-norm parameters are learned jointly with weights that can play the same role [@rutkowski2003].")
    mu = RS["mu_hkg"]
    P_(dc, "Only the nilpotent end of the family behaves differently, and the reason is visible in the memberships the model learned. "
           "On the graded graph the published weights are small (median w~e~ = "
           f"{f3(mu['raw_weight']['median'])}), the learned memberships of disease-to-symptom edges tracked them (median μ "
           f"{f3(mu['DpS']['median'],2)}, Spearman ρ = {f3(mu['DpS_spearman_with_weight'],2)} with w~e~), and the memberships of "
           f"symptom-to-disease edges, which pass through symptom hubs, were much smaller (median μ {f3(mu['DpS^-1']['median'],4)}). "
           "Every path of the graded task contains one such edge. For p > 0 such a path has strength exactly zero and passes no "
           "gradient (Section 3.2), so the Łukasiewicz t-norm discards almost all evidence and a start at p = 0.5 cannot recover. "
           "Large knowledge graphs are also close to block-structured, and our earlier analysis of the NHANES relation suggested that "
           "under such structure the product and its neighbours in the family order pairs in nearly the same way [@truong2026].")
    P_(dc, "For fuzzy-logic practice the implication is that a learned t-norm parameter should not be read as a property of the domain "
           "unless its identifiability has been checked: by restarting from several values, by training without early stopping, and, "
           "most directly, by a profile over fixed values. Each check is cheap relative to the experiments it qualifies. Where the "
           "profile shows that held-out accuracy depends on p more clearly than the training loss does, as on our graded graph, p should be "
           "selected on validation data as a hyperparameter rather than learned.")
    heading(dc, "5.3 What is specific to medical knowledge graphs", 2)
    P_(dc, "Most of these findings would hold for any sparse, hub-dominated graph, and we do not claim otherwise. Three features of the "
           "medical graphs studied here shaped them. First, curated biomedical graphs mix evidence of very different strength under the "
           "same relation types: Hetionet's disease–disease \"resembles\" edges come from text co-occurrence, while its compound–disease "
           "edges are curated indications [@himmelstein2017]. A single membership per relation cannot tell these apart, which is one "
           "reason why the explanations in Table 8 include weak links and why learned memberships helped only where edges carried their "
           "own graded weight. Second, hubs in these graphs are biologically meaningful and well studied, such as genes with hundreds "
           "of disease associations or symptoms such as fatigue and pain shared by many diseases, and they are also the product of "
           "uneven research attention [@bonner2022]. Discounting them helped on random splits but not, apparently, for diseases with no "
           "known indications (Section 4.4), where the remaining evidence runs through exactly these hubs. Third, the graded graph "
           "learned from patient records encodes how often a disease produces a symptom; its weights are small and its symptom nodes "
           "highly connected, which is what made the nilpotent t-norms fail and the learned discount useful. A developer building on "
           "a curated graph should therefore discount hubs with a fixed damping, a developer building on a graph learned from records "
           "should learn the discount, and both should check zero-shot performance separately.")
    heading(dc, "5.4 Implications for explainable clinical decision support", 2)
    bullets(dc, [C(x) for x in [
        "**Learn or fix memberships before operators.** A path reasoner over a medical graph should discount hubs, with a fixed DWPC-like "
        "discount as a strong default and a learned one where edges carry graded evidence. The t-norm can then be chosen among a few "
        "fixed, non-nilpotent operators, on validation data if it matters; learning it end to end adds parameters without adding "
        "accuracy, and the learned value is not interpretable.",
        "**Aggregate evidence, do not select it.** Noisy-OR over all paths was consistently better than the strongest path, so an "
        "explanation should show the leading paths together with the share of the evidence they carry (Tables 7 and 8) rather than a "
        "single \"reason\", and should flag paths that share a fact.",
        "**Test explanations at the level of facts.** Deleting paths within one pair confirms only that shares are skewed; removing a "
        "fact from the graph and re-ranking tests what a clinician would question.",
        "**Expect explanations to be by association.** The leading paths for drug repurposing ran through pharmacologic classes and "
        "similar drugs, and some disease–symptom paths through text-mined \"resembles\" links. Path shares make such evidence visible "
        "and discountable, but they do not turn it into a mechanism.",
        "**Report the strongest simple baselines and harder splits.** DWPC with logistic regression was better than DNFR on two of "
        "four tasks and not detectably worse on a third, low-rank completion led on the dense graph, and a disease-centred split reduced every method to a small fraction "
        "of its accuracy on random splits. Claims for new medical link predictors should be made against these, under leave-dyad-out "
        "splits and, for repurposing, with held-out diseases."]], numbered=True)
    heading(dc, "5.5 Limitations", 2)
    bullets(dc, [C(x) for x in [
        f"**Splits and power.** The comparisons with the fixed product and Hamacher product on the two disease–symptom tasks used "
        f"{nseeds('het_DpS','dnfr')} splits, but "
        "Hetionet compound–treats–disease used three, PrimeKG two and the disease-centred split four, and other variants were run on "
        "five or fewer splits, as the tables mark. Under the corrected resampled t-test even ten splits leave 90% intervals of about "
        "±0.015, so equivalence within ±0.01 could not be shown; differences of this size are not interpreted.",
        f"**Identifiability analyses on two tasks, with limited convergence.** The profile and the training without early stopping were run on the two "
        f"disease–symptom tasks ({len(PF.get('hkg_DpS',{}).get('seeds',[]))} and {len(PF.get('het_DpS',{}).get('seeds',[]))} splits for "
        "the profile), where a run is cheap; on the drug-repurposing tasks we rely on the restarts with early stopping. The profile "
        "fixes one p for all metapaths and so does not test whether the data identify the differences between metapaths, which the "
        "restarts suggest they do only weakly. The profile runs reached the epoch limit before the loss had stopped decreasing.",
        "**Early stopping.** The default stopping rule ended training early on some tasks; training for 400 epochs did not "
        "consistently change accuracy where we tried it, but we did not re-run every variant and baseline without early stopping. "
        "The comparisons in Tables 2 and 3 are between variants trained with the same rule.",
        "**Untuned embeddings, no graph neural networks.** TransE and RotatE were trained for 25 epochs without hyperparameter search, "
        f"and ComplEx did not train beyond chance level under the same budget (MRR {m('het_CtD','kge_ComplEx') if summ('het_CtD','kge_ComplEx') else '—'} "
        "on compound–treats–disease). Embedding accuracy depends strongly on training configuration [@ruffinelli2020], so the gap "
        "between them and the path methods is not a finding of this study. NBFNet [@zhu2021] and TxGNN [@huang2024], which learn path "
        "operators and are the natural competitors of a learned neuro-fuzzy reasoner, require GPU training on graphs of this size and "
        "were not run. AnyBURL was given 100 s of rule learning and might improve with more.",
        "**Metapath budget.** Metapaths with more than five million paths were excluded for tractability, including the side-effect and "
        "anatomy-expression metapaths on Hetionet and "
        f"{len(DS['pkg_ind']['skipped'])} of {DS['pkg_ind']['metapaths_all']} metapaths on PrimeKG, mostly through drug–drug "
        "interactions and contraindications. These are the densest and most hub-dominated metapaths, so the size of the effect of hub "
        "discounting may be underestimated relative to a model that uses them: the discount would matter more where paths through "
        "hubs dominate. All path methods used the same set, so the comparisons between them are unaffected, but the absolute accuracy "
        "of path methods on PrimeKG is a lower bound. A sampled-path version of the excluded metapaths was beyond our computing budget.",
        "**No clinical evaluation of explanations.** The explanations in Table 8 were not rated by clinicians; Table 7 measures "
        "concision and the dependence of the ranking on the paths and facts shown, not clinical usefulness. A rating study with "
        "several clinicians on a sample of correct and incorrect predictions is the next step.",
        "**Scores are not probabilities.** DNFR is trained with a ranking loss, and its score P orders candidates without being "
        "calibrated.",
        "**Missing data are not random.** The sparsity analysis removed edges uniformly at random; in practice, less-studied genes and "
        "diseases lose more edges. The disease-centred split addresses one form of this for drug repurposing only.",
        "**Secondary evidence from an unreviewed study.** The NHANES relation and the earlier observations that motivated this work come "
        "from our own analysis, deposited openly but not yet peer reviewed [@truong2026]; the conclusions of this paper do not depend "
        "on it, and the secondary test is reported in Appendix B only.",
        "**Associations, not mechanisms.** Paths in these graphs combine curated, text-mined and co-occurrence evidence; a high-scoring "
        "path is a reason to look, not evidence of a causal mechanism. The graded graph was learned by others from records we did not "
        "access, and the full MIMIC-IV release, which would allow building such graphs directly, requires credentialed access we did "
        "not hold [@johnson2023]."]])


def conclusion(dc):
    heading(dc, "6. Conclusion", 1)
    P_(dc, "In a controlled component analysis of differentiable neuro-fuzzy reasoning on four tasks from three medical knowledge graphs, the gains came "
           "from aggregating all paths by noisy-OR and from discounting edges at hub nodes; learning the discount per relation helped "
           "only where edges carried graded clinical evidence. The parameter of a learned t-norm was not identified by end-to-end "
           "loss: the loss was shallow in it, its learned value depended on where it started and when training stopped, and learning "
           "it gave no consistent gain over a fixed product or Hamacher product. The data excluded the nilpotent t-norms on graded "
           "evidence, and held-out accuracy favoured stronger-than-product t-norms there, a preference that a profile over fixed values "
           "found and end-to-end learning missed. Path explanations from such a model are exact and fine-grained, and correct rankings "
           "depend more on the facts they name than on other facts, but they mostly express evidence by association. For explainable clinical decision "
           "support on knowledge graphs, effort is better spent on how much each edge counts and on how paths are combined than on "
           "learning the operator that composes them.")


# ------------------------------------------------------------------ appendices
def appendix(dc):
    heading(dc, "Appendix A. Supplementary tables and figure", 1)
    R.appendix_task(dc, 1, "het_CtD", "Hetionet compound–treats–disease.")
    R.appendix_task(dc, 2, "het_DpS", "Hetionet disease–presents–symptom. The t-norm comparisons (fixed product and Hamacher product) "
                                      "use ten splits, other variants five.")
    R.appendix_task(dc, 3, "pkg_ind", "PrimeKG drug–indication–disease, ranked against all diseases; no p-value is computed for "
                                      "comparisons run on one split.")
    R.appendix_task(dc, 4, "hkg_DpS", "Graded disease–symptom graph learned from emergency-department records. The t-norm comparisons "
                                      "and restarts use ten splits, other variants five.")
    R.table_sparsity(dc)
    figure(dc, f"{ROOT}/figures/figA1_restarts.png", 6.3,
           "**Figure A1.** Learned p at the default early stopping, by initial value p₀ (rows) and task (panels). Each point is one "
           "metapath in one split; | marks p₀. Grey lines join the values of the same metapath in the first split across the four "
           "restarts without penalty. The graded graph has a single metapath. Accuracies of the same runs are in Table 4.")
    heading(dc, "Appendix B. Secondary test on a population relation (NHANES)", 1)
    N = XS["nhanes"]; J = XS["jack"]
    P_(dc, "A secondary test uses the 24 × 24 relation between nine PHQ-9 symptoms and 15 socioeconomic, lifestyle and health concepts "
           "from our earlier analysis [@truong2026]. Each concept is a survey-weighted fuzzy membership a~n,c~ ∈ [0, 1] of respondent n. "
           "The raw relation from concept i to concept j is the weighted fuzzy subsethood S(i → j) = Σ~n~ w~n~ min(a~n,i~, a~n,j~) / Σ~n~ "
           "w~n~ a~n,i~, and the relation used here corrects it for chance: μ(i → j) = max(0, (S − S~0~)/(1 − S~0~)), where S~0~ is the "
           "value S would take if j were independent of i. DNFR is fitted to the relation measured in 2013–2018, with every pair "
           "predicted from its indirect paths only, and evaluated by the Spearman correlation of its predictions with the relation "
           "measured independently in 2007–2012. Membership here is a calibration of the given degree, σ(a~r~ + g~r~ logit μ), with one "
           "(a~r~, g~r~) per pair of concept types. With 24 concepts this test has little power, and it measures agreement between two "
           "survey periods rather than link prediction.")
    P_(dc, f"The fixed rows of Table B1 reproduce our earlier analysis, with the product ahead of the other fixed t-norms (Spearman ρ = "
           f"{f3(N['product']['rho_hold'])}). A single learned Schweizer–Sklar parameter added nothing ({f3(N['ss_shared']['rho_hold'])}), "
           f"and a parameter per metapath without calibration was slightly worse ({f3(N['ss_permp']['rho_hold'])}). Calibrating the "
           f"membership per relation type raised agreement with the held-out relation to {f3(N['calib_product']['rho_hold'])} with the "
           f"product and to {f3(N['dnfr']['rho_hold'])} with learned p per metapath. Dropping each of the 24 concepts in turn, DNFR "
           f"exceeded the calibrated product in {J['dnfr_minus_calibproduct']['n_positive']} of 24 runs by "
           f"{f3(J['dnfr_minus_calibproduct']['min'])} to {f3(J['dnfr_minus_calibproduct']['max'])}, with a jackknife standard error "
           f"of {f3(J['dnfr_minus_calibproduct']['jack_se'])}, as large as the differences; we do not treat this as evidence for "
           "learning the operator. Most of the improvement comes from calibrating memberships, in line with the main results.")
    R.table_nhanes(dc)


def declarations(dc):
    heading(dc, "7. Declarations", 1)
    P_(dc, "**Ethics.** The study uses only public, de-identified or aggregate data: Hetionet and PrimeKG, the disease–symptom graph "
           "published by Rotmensch et al. [@rotmensch2017], whose underlying patient records were not accessed, and relation matrices "
           "derived from the public NHANES files [@truong2026]. Saigon International University does not require ethics approval for "
           "secondary analysis of public anonymised data.")
    P_(dc, "**Data availability.** Hetionet v1.0 is available from its GitHub repository and PrimeKG from Harvard Dataverse "
           "(doi:10.7910/DVN/IXA7BM). The parsed health knowledge graph, the NHANES relations used in Appendix B, the per-query ranks of "
           "every method on every split, the trained models, the path explanations and the statistics files from which every number in "
           "this paper is generated are deposited at Zenodo (version 1.1.0; all versions: https://doi.org/10.5281/zenodo.23879352), "
           "together with a script that downloads Hetionet and PrimeKG and checks them against the checksums of the files we used.")
    P_(dc, "**Code availability.** The implementation of DNFR and of all baselines, the tests, and the scripts that compute the statistics "
           "and figures and build this manuscript from the result files are available under the MIT licence at "
           "https://github.com/Bangth2026/dnfr-medical-kg (release v1.1.0, archived at https://doi.org/10.5281/zenodo.23879352).")
    P_(dc, "**CRediT author statement.** Truong Hai Bang: Conceptualization; Methodology; Software; Formal analysis; Data curation; "
           "Visualization; Writing – original draft; Writing – review and editing. Van An Phan Thi: Validation (review of the path "
           "explanations in Table 8 against the source graphs); Writing – review and editing. All authors approved the final manuscript "
           "and agree to be accountable for all aspects of the work.")
    P_(dc, "**Declaration of generative AI and AI-assisted technologies in the manuscript preparation process.** During the preparation "
           "of this work the authors used Claude (Anthropic) to draft and edit the manuscript text. Its use in writing and running the "
           "analysis code is described in Section 3.5. After using this tool, the authors reviewed and edited the content as needed and "
           "take full responsibility for the content of the publication.")
    P_(dc, "**Funding.** This research did not receive any specific grant from funding agencies in the public, commercial, or "
           "not-for-profit sectors.")
    P_(dc, "**Declaration of competing interest.** The authors declare no known competing financial interests or personal relationships "
           "that could have appeared to influence the work reported in this paper.")


def references(dc):
    from references import REFS
    heading(dc, "References", 1)
    for i, k in enumerate(ORDER):
        para(dc, f"[{i + 1}] {REFS[k]}", align="left", size=10, line=1.15, space_after=3)
