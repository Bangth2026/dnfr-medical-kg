# -*- coding: utf-8 -*-
"""Collect every new number of the revision from the result files and format the texts/table cells that depend on
them. Writes build/numbers.json (read by build_ms.py, build_supp.py and build_response.py)."""
import json, os, glob, ast, collections
import numpy as np
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
R0, R6 = f"{ROOT}/results", f"{ROOT}/results_rev6"
C = json.load(open(f"{R6}/rev6_compare.json"))
S = json.load(open(f"{R6}/rev6_stats.json"))
VER = "v7.0-jbi"
N = {"VER": VER, "ZEN_VER_DOI": os.environ.get("ZEN_VER_DOI", "https://doi.org/10.5281/zenodo.23879352 (concept DOI)")}


def f3(x, sign=False):
    s = f"{x:+.3f}" if sign else f"{x:.3f}"
    return s.replace("-", "−")


def ci(v):
    return f"[{f3(v['ci_lo'], True)}, {f3(v['ci_hi'], True)}]"


def mark(v):
    return {"equivalent": "≈", "different": "†"}.get(v.get("verdict"), "")


def cell_delta(v):
    return f"{f3(v['delta_MRR'], True)} {ci(v)} {v['seed_wins']}/{v['seeds']}{mark(v)}"


def word(v):
    return {"equivalent": "equivalent", "different": "different", "inconclusive": "inconclusive"}.get(v.get("verdict"), "inconclusive")


def ms(m):
    return f"{f3(m['MRR'])} ± {f3(m['sd'])}"


# ---------------------------------------------------------------- chance
ch = S["chance"]
N["CHANCE"] = {k: v["chance_MRR"] for k, v in ch.items()}

# ---------------------------------------------------------------- M2
H, D = C["M2_hkg_DpS"], C["M2_het_DpS"]
FD = D["family"]
sm_lr, nn_lr, nnlr_lr = FD["dwpc_sm__dwpc_lr"], FD["dwpc_nn_sm__dwpc_lr"], FD["dwpc_nn_lr__dwpc_lr"]
dn_sm, dn_nn = FD["dnfr_rel__dwpc_sm"], FD["dnfr_rel__dwpc_nn_sm"]
N["T2_DWPC_SM"] = ["DWPC, same features, filtered softmax", "—", ms(D["means"]["dwpc_sm"]), "—", ms(H["means"]["dwpc_sm"])]
N["T2_DWPC_NN"] = ["Linear DWPC, non-negative coefficients, filtered softmax", "—", ms(D["means"]["dwpc_nn_sm"]), "—",
                   ms(H["means"]["dwpc_nn_sm"])]
N["T2_RANDOM"] = ["Uniformly random ranker (expected MRR)", f3(ch["het_CtD"]["chance_MRR"]), f3(ch["het_DpS"]["chance_MRR"]),
                  "0.0006", f3(ch["hkg_DpS"]["chance_MRR"])]
N["CAP_T2"] = ("**Table 2.** Test MRR (mean ± SD over random splits) of DNFR, its variants and the baselines. Numbers in "
               "parentheses give the number of splits when it differs from that of DNFR (10, 10, 2, 10 splits on the four "
               "tasks). MRR is not comparable across columns, because the number of candidates differs (Table 1); the last "
               "row gives the expected MRR of a uniformly random ranker under the same filtered protocol. Fixed p, selected "
               "on validation: one Schweizer–Sklar p chosen per split from a grid by validation MRR, trained with the "
               "protocol of DNFR. The two DWPC controls of Section 3.4 were run on the two disease–symptom tasks; on the "
               "graded graph, with one metapath, all DWPC models rank identically. DNFR + R1–R4, the validation-selected "
               "fixed p, NBFNet and the DWPC controls were not run on PrimeKG, which is descriptive (two splits). Untuned "
               "TransE and RotatE are in Supplementary Tables S1–S3. Hits@k, intervals and every variant are in "
               "Supplementary Tables S1–S4 and S23.")
N["CAP_T3"] = ("**Table 3.** What each learned component adds. Each cell is the difference in test MRR between the model "
               "with the component and the model without it, averaged over the splits on which both were run, with its 95% "
               "paired bootstrap interval and the number of splits on which the difference was positive (out of the splits "
               "run; a cell based on fewer splits than its column, such as 1/5, says so). Verdicts under the rule of "
               "Section 3.6, with Holm adjustment within the family of component comparisons of each task (all panels): "
               "≈ equivalent within ±0.01; † different; no mark, inconclusive. — not run or not applicable (Hetionet and "
               "PrimeKG edges carry no graded weight).")
EX = json.load(open(f"{R6}/het_DpS_explain_dwpc_nn.json"))["summary"]["dwpc"]
EXH = json.load(open(f"{R6}/hkg_DpS_explain_dwpc_nn.json"))["summary"]["dwpc"]
pc = lambda x: f"{100 * x:.0f}%"
N["T6_NN_SHARE"] = ["Linear non-negative DWPC: share of the leading fact (exact Shapley value)", "—",
                    pc(EX["top_edge_share_median"]), pc(EXH["top_edge_share_median"])]
N["T6_NN_K50"] = ["Linear non-negative DWPC: facts covering 50% of its attribution", "—",
                  f"{EX['k50_edges_median']:.0f}", f"{EXH['k50_edges_median']:.0f}"]
N["T6_NN_CF"] = ["Linear non-negative DWPC: its leading fact / a random fact on its paths", "—",
                 f"{pc(EX['stay_top10_cf_top'])} / {pc(EX['stay_top10_cf_random'])}",
                 f"{pc(EXH['stay_top10_cf_top'])} / {pc(EXH['stay_top10_cf_random'])}"]
N["CAP_T6"] = ("**Table 6.** Quantitative evaluation of path explanations on the seed-0 split. Evidence is −log(1 − P) "
               "above the leak, which the paths share exactly. Summed path shares count a fact once for every path through "
               "it; the Shapley value splits each path's term equally among its facts (Section 3.6) and sums to the total. "
               "DWPC-LR rows use its own correct top-10 predictions and the attribution of Section 3.6; the linear "
               "non-negative DWPC rows use that model's correct top-10 predictions and its exact Shapley value (not run on "
               "compound–treats–disease). Deletion within the pair changes one score with all others fixed; the fact-level "
               "counterfactual removes one fact from the graph and re-scores every candidate of the same head, without "
               "retraining and with degrees held fixed. Metapath classes are given for drug repurposing only, the one task "
               "with a mechanistic route. In \"Same leading fact in DNFR (Shapley) and DWPC-LR\" the number in parentheses "
               "is the number of pairs in the top 10 of both models; in \"Test pairs evaluated\" it is the number of top-10 "
               "hits.")
N["HL4"] = "Path counts matched DNFR in accuracy and, made non-negative, in exact attribution."
N["ABS_M2"] = (f"; trained with DNFR's objective they changed by {f3(sm_lr['delta_MRR'], True)}, and a linear version "
               f"with non-negative coefficients, as exactly attributable as DNFR, was {f3(-dn_nn['delta_MRR'])} more "
               f"accurate on Hetionet disease–symptom")
N["RESP_M2_SUMMARY"] = (f"The objective explained little: trained with the filtered softmax, DWPC changed by "
                        f"{f3(sm_lr['delta_MRR'], True)} on Hetionet disease–symptom (equivalent within ±0.01). The linear "
                        f"non-negative DWPC reached {f3(D['means']['dwpc_nn_sm']['MRR'])} against "
                        f"{f3(D['means']['dnfr_rel']['MRR'])} for DNFR ({f3(dn_nn['delta_MRR'], True)} for DNFR, "
                        f"different under our rule) and has an exact, non-negative attribution to facts. The "
                        f"\"difference in kind\" claim is withdrawn.")
N["P_41_DWPC"] = (
    "Among path methods, DWPC-LR was at least as accurate as DNFR on the three tasks with ten splits. On Hetionet drug "
    "repurposing DNFR minus DWPC-LR was +0.005 [−0.016, +0.025] (5/10 splits, inconclusive); on the two disease–symptom "
    "tasks DWPC-LR was ahead by 0.016 and 0.021, a difference under our rule on both. On PrimeKG DNFR was ahead by 0.070 "
    "[+0.056, +0.084] on two splits, too few for a verdict. Two controls separate the model from its training objective "
    f"(Supplementary Table S23). Trained with DNFR's exact filtered softmax instead of sampled logistic regression, DWPC "
    f"changed by {f3(sm_lr['delta_MRR'], True)} {ci(sm_lr)} on Hetionet disease–symptom ({sm_lr['seed_wins']}/10 splits, "
    f"{word(sm_lr)}), so the objective does not explain DNFR's deficit there. A linear model in the untransformed DWPC with "
    f"non-negative coefficients, trained with the same softmax, reached {f3(D['means']['dwpc_nn_sm']['MRR'])} "
    f"({f3(nn_lr['delta_MRR'], True)} against DWPC-LR, {word(nn_lr)}) and was ahead of DNFR by "
    f"{f3(-dn_nn['delta_MRR'])} {ci({'ci_lo': -dn_nn['ci_hi'], 'ci_hi': -dn_nn['ci_lo']})} ({10 - dn_nn['seed_wins']}/10 "
    f"splits, {word(dn_nn)}); its optimum set {100 * D['coef']['dwpc_nn_sm']['zero_share_mean']:.0f}% of the metapath "
    "coefficients to zero. On the graded graph, with one metapath, every increasing function of the single DWPC feature "
    "ranks the candidates identically, so all DWPC models gave DWPC-LR's MRR (0.331). We kept the damping of DWPC at 0.4, "
    "the value of Himmelstein et al. [1]; tuning it could only add to DWPC's accuracy. Degree and prior features did not "
    "help DWPC-LR. Completion of the target relation alone was far behind on Hetionet drug repurposing (Soft-Impute "
    "0.262), came close on PrimeKG (0.627) and led on the dense graded graph (0.352 against 0.311 for DNFR), where a "
    "low-rank description should work well. AnyBURL was below DNFR and DWPC-LR wherever it ran<<ANYBURL_1000>>. The "
    "compact NBFNet, with about ten thousand parameters, did not differ from DNFR under our rule (graded graph −0.001 "
    "[−0.011, +0.009]; Hetionet disease–symptom +0.021 [+0.014, +0.029], ahead on 9 of 10 splits), and it was behind "
    "DWPC-LR on the graded graph (−0.022, different). Every path method was far above the chance level on every task.")

# ---------------------------------------------------------------- M1
E = S["envelope"]
def resolved_counts(t, seeds):
    return sum(E[t][str(s)]["n_resolved"] for s in seeds), sum(E[t][str(s)]["n_points"] for s in seeds)
rh, nh = resolved_counts("hkg_DpS", (0, 1, 2)); rd, nd = resolved_counts("het_DpS", (0, 2))
maxgap_h = max(E["hkg_DpS"][s]["max_second_minus_best_resolved"] for s in ("0", "1", "2"))
maxgap_d = max(E["het_DpS"][s]["max_second_minus_best_resolved"] for s in ("0", "2"))
gaps1 = [r["second_minus_best"] for r in E["het_DpS"]["1"]["rows"] if not r["resolved"]]
def excess_at(t, s, p):
    return next(r["excess"] for r in E[t][str(s)]["rows"] if abs(r["p"] - p) < 1e-9)
vb = [(s, E["hkg_DpS"][str(s)]["p_best_valid"]) for s in (0, 1, 2)]
vexc = [excess_at("hkg_DpS", s, p) * 1e3 for s, p in vb]
# single-metapath profiles
PM = json.load(open(f"{R6}/profile_permp_het_DpS_seed0.json"))
pm_txt, pm_rows = [], []
shares = PM["metapath_share"]
for mp, chains in PM["profiles"].items():
    best = {}
    for k, cs in chains.items():
        L = sorted(c["loss"] for c in cs)
        cb = min(cs, key=lambda c: c["loss"])
        best[float(k)] = (L[0], L[-1], cb["valid_MRR"], cb["test_MRR"], len(cs))
    ps = sorted(best); m0 = min(v[0] for v in best.values())
    rng = (max(v[0] for v in best.values()) - m0) * 1e3
    gaps = {p: (best[p][1] - best[p][0]) * 1e3 for p in ps}
    res = [p for p in ps if gaps[p] < 0.1]
    gap = max(gaps[p] for p in res)
    inside = [p for p in ps if (best[p][0] - m0) < 7.2e-4]
    vals = [best[p][2] for p in ps]
    pm_rows.append(dict(mp=mp, share=shares[mp], range=rng, gap=gap, n_res=len(res), n=len(ps), inside=inside,
                        vrange=max(vals) - min(vals),
                        rows=[(p, (best[p][0] - m0) * 1e3, gaps[p], best[p][2], best[p][3], best[p][4]) for p in ps]))
N["_PM"] = pm_rows
order_sh = sorted(shares.items(), key=lambda x: -x[1])
N["PERMP_SHARE2"] = f"{100 * (order_sh[0][1] + order_sh[1][1]):.0f}%"
N["PERMP_THIRD"] = f"{100 * order_sh[2][1]:.1f}%"
def mpname(m):
    return {"DrD-DpS": "disease–resembles–disease–presents–symptom",
            "DlA-DlA^-1-DpS": "disease–localizes–anatomy–localizes–disease–presents–symptom"}.get(m, m)
def interval(ins):
    if not ins: return "no point"
    return f"[{ins[0]:g}, {ins[-1]:g}]".replace("-", "−") if len(ins) > 1 else "{" + f"{ins[0]:g}".replace("-", "−") + "}"
pmt = []
for r in pm_rows:
    pmt.append(f"for {mpname(r['mp'])} ({100 * r['share']:.0f}% of the evidence) the loss rose by {r['range']:.1f} × 10⁻³ nat "
               f"from p = −5 to the product, the points within the threshold were {interval(r['inside'])}, and validation MRR "
               f"varied by {r['vrange']:.3f}")
N["P_44_PROFILE"] = (
    "Figure 2 shows the converged profiles with the disagreement between optimisation chains. The largest spreads in "
    "Supplementary Tables S8 and S16 (up to 32.1 × 10⁻³ nat on Hetionet) came mostly from single chains that settled in a "
    "poorer basin at every p, not from imprecision of the profile itself: on the graded graph the two best chains agreed "
    f"within {maxgap_h * 1e3:.2f} × 10⁻³ nat at {rh} of {nh} grid points, and on Hetionet disease–symptom within "
    f"{maxgap_d * 1e3:.2f} × 10⁻³ nat at {rd} of {nd} points on splits 0 and 2 (Supplementary Table S21). On these resolved "
    "profiles the training loss did locate p within a split, but loosely: on the graded graph the points within the "
    "threshold χ²(0.95, 1)/(2N) = 0.65 × 10⁻³ nat were [−0.5, −0.25] on split 1, {0.5} on split 2 and, on split 0, two "
    "disjoint ranges, p ≤ −2.5 and 0.1 to 0.25, separated by a ridge near p = −0.25 that rose 5.1 × 10⁻³ nat; on Hetionet "
    "they were [−20, −10] and [−5, −1]. On Hetionet split 1 the two best chains disagreed by "
    f"{min(gaps1) * 1e3:.0f} to {max(gaps1) * 1e3:.0f} × 10⁻³ nat at 8 of 9 points, more than the profile varied, so that "
    "profile could not be resolved at the precision we attained. Across splits the minimiser moved (graded graph 0.1, "
    "−0.25 and 0.5; Hetionet −20 and −2), which is sampling variability of a weakly curved objective rather than flatness "
    "within a split. Profiles of the two metapaths that carry most of DNFR's evidence on Hetionet disease–symptom, with "
    "the other 15 re-fitted, were resolved (the two sweeps agreed within 0.01 × 10⁻³ nat at all but one cold-started point; "
    "Supplementary Table S22) and only one-sided: " + "; ".join(pmt) + ". The loss thus prefers operators towards the "
    "minimum for these metapaths without fixing a value, and held-out accuracy hardly distinguishes them.")
N["P_44_HELDOUT"] = (
    "Held-out accuracy pointed elsewhere. On the graded graph, validation MRR at the converged fits peaked at p = −1.5, −2 "
    f"and −1.5 (Figure 2C), where the training loss lay {vexc[0]:.1f}, {vexc[1]:.1f} and {vexc[2]:.1f} × 10⁻³ nat above "
    "its minimum, outside the threshold on every split; test MRR at the selected value exceeded that of the product by "
    "+0.014, +0.008 and +0.016. The data thus carry information about p that the training objective barely registers: the "
    "softmax loss averages over all candidates, whereas MRR depends on the order of the few top-ranked ones. This is a "
    "mismatch between the surrogate loss and the metric, not an absence of information, and it is why we profiled "
    "validation MRR, a ranking-aware criterion, beside the loss rather than a second likelihood. The Łukasiewicz t-norm, "
    "which lost 0.080 from the default initialisation because every path starts at strength zero, reached 0.312, 0.333 "
    "and 0.290 when started from the converged product (Supplementary Table S18). Per-metapath values from different "
    "starts were weakly correlated (median Spearman 0.44 on Hetionet disease–symptom), so learned per-metapath values "
    "should not be interpreted either.")

# ---------------------------------------------------------------- M3 (entity-centred splits)
T = C.get("tail10"); TN = C.get("tail_notgt"); HD = C.get("head")
rand_ctd = ch["het_CtD"]["chance_MRR"]
N["T5_A_RANDOM"] = ["Uniformly random ranker (expected)", "0.041", f3(rand_ctd), f"{10 / ch['het_CtD']['mean_filtered_candidates']:.3f}", "—"]
RS = {"dnfr": "0.599", "product": "0.602", "hamacher": "0.575", "dnfr_nodeg": "0.437", "dwpc_lr": "0.594",
      "dwpc_lr_full": "0.577", "softimpute": "0.262", "svd": "0.278", "prior": "0.090", "popularity": "0.205"}
LABEL = {"dnfr": "DNFR (learned p", "product": "Product t-norm (fixed)", "hamacher": "Hamacher product (fixed)",
         "dnfr_nodeg": "DNFR without degree", "dwpc_lr": "DWPC + logistic", "dwpc_lr_full": "DWPC-LR + degree",
         "softimpute": "Soft-thresholded", "svd": "Truncated SVD", "prior": "Edge prior", "popularity": "Popularity"}
def nspl(m, Tm, full):
    return "" if Tm[m]["n"] == full else f" ({Tm[m]['n']})"
if T:
    full = T["n_splits"]; ref = "dnfr" if T["means"]["dnfr"]["n"] == full else "dnfr"
    A = []
    for m in ["dnfr", "product", "hamacher", "dnfr_nodeg", "dwpc_lr", "dwpc_lr_full", "softimpute", "svd", "prior", "popularity"]:
        x = T["means"][m]; v = T["family"].get(f"{m}__product")
        A.append([LABEL[m], [RS[m], f"{ms(x)}{nspl(m, T['means'], full)}", f3(x["H10"]), cell_delta(v) if v else "—"]])
    N["T5_A"] = A
    nd = T["means"]["dnfr"]["n"]
    N["T5_A_TITLE"] = (f"A. Disease-centred split ({full} splits; numbers in parentheses give fewer): every indication of a "
                       "test disease held out; ΔMRR is the method minus the fixed product over the splits both were run on")
fam = T["family"] if T else {}
nodeg = fam.get("dnfr_nodeg__dnfr", {})
N["ABS_TAIL"] = (f"{T['means']['product']['MRR']:.2f}" if T else "0.08") + (
    f" (chance 0.04) but {TN['means']['product_notgt']['MRR']:.2f} for a model trained without them" if TN else " (chance 0.041)")
panels = []
if TN:
    panels.append(dict(title=f"D. Disease-centred split without the 33 metapaths that contain the treats relation "
                             f"({TN['n_splits']} splits); ΔMRR is the model without them minus the model with them",
                       rows=[["Product t-norm (fixed)", f3(TN["means"]["product"]["MRR"]), ms(TN["means"]["product_notgt"]),
                              f3(TN["means"]["product_notgt"]["H10"]), cell_delta(TN["family"]["product_notgt__product"])],
                             ["DWPC + logistic regression", f3(TN["means"]["dwpc_lr"]["MRR"]), ms(TN["means"]["dwpc_lr_notgt"]),
                              f3(TN["means"]["dwpc_lr_notgt"]["H10"]), cell_delta(TN["family"]["dwpc_lr_notgt__dwpc_lr"])]]))
if HD:
    rows = []
    refm = "product"
    for m in ["dnfr", "product", "dwpc_lr", "dwpc_lr_full", "softimpute", "svd", "prior", "popularity"]:
        if m not in HD["means"]: continue
        x = HD["means"][m]
        v = HD["family"].get(f"{m}__product") if m != "product" else None
        rows.append([LABEL[m].replace("DNFR (learned p", "DNFR (learned p per metapath)").replace("DWPC + logistic", "DWPC + logistic regression")
                     .replace("DWPC-LR + degree", "DWPC-LR + degree + prior").replace("Soft-thresholded", "Soft-thresholded SVD (Soft-Impute)")
                     .replace("Truncated SVD", "Truncated SVD (rank chosen on validation)").replace("Edge prior", "Edge prior (degree-preserving permutations)")
                     .replace("Popularity", "Popularity (training positives of the tail)"),
                     RS[m], f"{ms(x)}{nspl(m, HD['means'], HD['n_splits'])}", f3(x["H10"]), cell_delta(v) if v else "—"])
    rows.append(["Uniformly random ranker (expected)", "0.041", f3(rand_ctd), f"{10 / ch['het_CtD']['mean_filtered_candidates']:.3f}", "—"])
    panels.append(dict(title=f"E. Compound-centred split ({HD['n_splits']} splits): every indication of a test compound held "
                             "out; ΔMRR is the method minus the fixed product over the splits both were run on", rows=rows))
N["T5_NEW_PANELS"] = panels
N["CAP_T5"] = ("**Table 5.** Hetionet compound–treats–disease in harder settings. Random-split MRR from Table 2 for "
               "comparison (in panel B, over the same splits; in panel C, the MRR of panel B; in panel D, the disease-centred "
               "MRR of the same model with all metapaths). Each Δ is given with its 95% bootstrap interval and the number of "
               "splits on which it was positive; verdicts as in Table 3 (Holm adjustment within each panel). The expected "
               "MRR and Hits@10 of a uniformly random ranker are given for the entity-centred splits; popularity can fall "
               "below chance there because every held-out disease has no training positive and all such candidates tie.")
json.dump(N, open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "numbers_partial.json"), "w"), indent=1, ensure_ascii=False)
print("ok", [k for k in N if not k.startswith("_")])

# ================================================================ explanations, clinical, discussion
EXD = json.load(open(f"{R0}/het_DpS_explain_dwpc.json"))["summary"]["dwpc"]
N["P_46_FACTS"] = (
    "At the level of facts the picture is clearer. Summed path shares credited the leading fact with 45%, 25% and 28% of "
    "the evidence on the three tasks; its Shapley value was 16%, 12% and 9%, and half of the attribution lay on 7, 6 and 12 "
    "facts. The two criteria picked the same leading fact for 95%, 91% and 100% of pairs, so double counting changes how "
    "much a fact appears to carry more than which fact leads. Removing the leading fact from the graph left 85%, 60% and "
    "76% of correct top-10 predictions in the top 10, against 100%, 99% and 99% for a random fact on the same paths. This "
    "counterfactual keeps degrees, and hence the memberships of neighbouring edges, fixed, as the Shapley decomposition "
    "does, and it measures how much the trained model depends on a fact, not what a model retrained without the fact "
    "would learn. DWPC-LR's counterfactuals were similarly faithful (87%, 55% and 82% against 100%, 99% and 96%), but its "
    "attribution is inexact and of mixed sign: on compound–treats–disease 31% of its positive attribution was offset by "
    "metapaths with negative coefficients (median). Both properties come from the asinh transform and the unconstrained "
    "coefficients, not from path counting. The linear non-negative DWPC model of Section 4.1 is a sum of non-negative "
    "path terms, each a unanimity game over its facts, so its edge-level Shapley value is exact and non-negative for the "
    f"same reason as DNFR's. Its leading fact carried {pc(EX['top_edge_share_median'])} of its attribution on Hetionet "
    f"disease–symptom and half lay on {EX['k50_edges_median']:.0f} facts; removing its leading fact left "
    f"{pc(EX['stay_top10_cf_top'])} of its correct top-10 predictions in the top 10, against "
    f"{pc(EX['stay_top10_cf_random'])} for a random fact, and on the graded graph it coincides with DWPC-LR (Table 6). "
    "Exact, non-negative attribution to facts is therefore a property of additive path models with non-negative terms, "
    "not an advantage of fuzzy composition. On drug repurposing the leading metapath was associative for 100% of "
    "correct top-10 predictions (examples in Supplementary Table S19).")
CL = S["clinical"]
def wil(x): return f"{100 * x[2]:.0f}% [{100 * x[3]:.0f}–{100 * x[4]:.0f}%]"
fr = CL["first_rater_relations"]
N["P_46_CLIN"] = (
    f"In the pilot clinical assessment (Supplementary Sections S6.2, S7 and S8.7), the first rater judged every relation "
    f"on a path to agree with biomedical knowledge for {wil(fr['het_DpS']['yes'])} of paths on Hetionet and "
    f"{wil(fr['hkg_DpS']['yes'])} on the graded graph. The physician's usefulness ratings separated correct from "
    f"incorrect predictions only weakly (AUC {CL['auc_usefulness'][0]:.2f}, 95% CI {CL['auc_usefulness'][1]:.2f}–"
    f"{CL['auc_usefulness'][2]:.2f}), and his agreement with the first rater on the correctness of relations was moderate "
    f"(κ = {CL['kappa_relations_correct'][0]:.2f}, {CL['kappa_relations_correct'][1]:.2f}–{CL['kappa_relations_correct'][2]:.2f}). "
    "The explanations were more often a reason to look further than a reason to believe.")
N["P_51"] = (
    "Noisy-OR aggregation and hub discounting carried DNFR's gains over its ablations, with the same sign on every split "
    "of the three tasks with ten splits; a learned discount helped only with graded clinical evidence. The t-norm, the "
    "most visibly neuro-fuzzy part of the model, gave no consistent gain when learned. The training loss located its "
    "parameter only within wide or disjoint intervals, the location moved between splits, and it disagreed with the "
    "value that held-out accuracy preferred; validation selection helped where learning did not. As a predictor DNFR did "
    "not beat DWPC-LR on any task with ten splits, trailed completion on the graded graph and did not differ from a "
    "compact NBFNet, and the training objective did not explain its deficit against path counts. A linear path-count "
    "model with non-negative coefficients was more accurate than DNFR on Hetionet disease–symptom and has the same exact, "
    "non-negative attribution to facts. For accuracy and for exact attribution alike, then, simpler additive path models "
    "suffice on graphs like these; whether an exact fact-level attribution helps a clinician more than a metapath-level "
    "one is a question for a user study. On drug repurposing, paths through known indications carried almost all of the "
    "evidence on random splits, and when they were available in training they lowered accuracy for diseases without "
    "known treatments.")
N["P_6"] = (
    "In a controlled component analysis on three medical link-prediction tasks with ten splits each, the accuracy of "
    "differentiable neuro-fuzzy path reasoning came from noisy-OR aggregation and hub discounting, and four plausible "
    "refinements did not improve it. The training loss constrained the t-norm parameter only loosely, its optimum moved "
    "between splits and differed from the one held-out accuracy preferred, so the operator should be selected on "
    "validation data and not interpreted when learned. Degree-weighted path counts were at least as accurate, and a "
    "non-negative linear version of them offers the same exact attribution to facts. On drug repurposing most of the "
    "evidence, for every path method, ran through known indications. Held-out compounds kept most of their accuracy "
    "through similar compounds, but held-out diseases did not, and for them a model trained without the "
    "target-relation metapaths was more than three times as accurate. For explainable clinical decision support, effort is better spent on how much each fact counts, how "
    "paths are combined, attribution to facts and tests of what the evidence rests on than on learning the operator that "
    "composes it.")
N["EXCL_NOTE"] = ("only one excluded metapath contains the treats relation "
                  "(compound–causes–side effect–causes–compound–treats–disease), and on the training graph of every split "
                  "it still had 9.9 to 10.2 million paths, so counting per split would have excluded the same metapaths "
                  "(Supplementary Section S8.8). For PrimeKG we did not repeat this check.")
N["NEW_REFS"] = {
    "47": "S. Kim, J.-H. Won, D.S.U. Lee, R. Luo, L. Wu, Y. Xia, T. Qin, H. Lee, Revisiting machine-learning based drug "
          "repurposing: drug indications are not a right prediction target, in: Proceedings of the Conference on Health, "
          "Inference, and Learning (CHIL 2023), PMLR 209, 2023, pp. 100–116.",
    "48": "G. Brière, T. Stosskopf, B. Loire, A. Baudot, Benchmarking the impact of data leakage on the performance of "
          "knowledge graph embedding models for biomedical link prediction, bioRxiv (2025). "
          "https://doi.org/10.1101/2025.01.23.634511",
    "49": "D. Bang, S. Lim, S. Lee, S. Kim, Biomedical knowledge graph learning for drug repurposing by extending "
          "guilt-by-association to multiple layers, Nat. Commun. 14 (2023) 3570. https://doi.org/10.1038/s41467-023-39301-y",
}

# ================================================================ supplement S8 texts and tables
rows21 = []
for t, lab in (("hkg_DpS", "Graded"), ("het_DpS", "Hetionet DpS")):
    for s in ("0", "1", "2"):
        for r in E[t][s]["rows"]:
            rows21.append([lab, s, f"{r['p']:g}".replace("-", "−"), str(r["n_chains"]), f"{r['excess'] * 1e3:.2f}",
                           f"{r['second_minus_best'] * 1e3:.2f}", f"{r['worst_minus_best'] * 1e3:.2f}",
                           "yes" if r["resolved"] else "no", f"{r['valid_MRR']:.3f}", f"{r['test_MRR']:.3f}"])
N["TAB_S21"] = dict(caption="**Table S21.** Chain envelope of the converged profiles over a fixed shared p (Figure 2). For "
                    "each graph, split and p: the number of converged warm-start chains, the lowest loss in excess of the "
                    "split's minimum, the second-best and the worst chain minus the best (×10⁻³ nat), whether the point is "
                    "resolved (two best chains within 10⁻⁴ nat), and validation and test MRR of the best chain.",
                    header=["Graph", "Split", "p", "Chains", "Excess ×10⁻³", "2nd − best ×10⁻³", "Worst − best ×10⁻³",
                            "Resolved", "Valid MRR", "Test MRR"], rows=rows21,
                    widths=[1100, 600, 700, 700, 900, 1000, 1000, 900, 1000, 1000])
rows22 = []
for r in pm_rows:
    for p, ex, g, vm, tm, nc in r["rows"]:
        rows22.append([mpname(r["mp"]), f"{100 * r['share']:.1f}%", f"{p:g}".replace("-", "−"), str(nc), f"{ex:.2f}",
                       f"{g:.2f}", f"{vm:.3f}", f"{tm:.3f}"])
N["TAB_S22"] = dict(caption="**Table S22.** Converged profiles of single metapaths on Hetionet disease–symptom (split 0): the "
                    "t-norm parameter of one metapath fixed on a grid, the parameters of the other 15 metapaths, the "
                    "memberships, reliabilities and leak re-fitted by L-BFGS, all p kept in the non-nilpotent range "
                    "[−20, 0]. Share: mean share of DNFR's evidence over its correct top-10 test predictions. Excess: lowest "
                    "loss minus the minimum of that profile; Gap: the two sweeps' difference (×10⁻³ nat).",
                    header=["Metapath", "Share", "p", "Chains", "Excess ×10⁻³", "Gap ×10⁻³", "Valid MRR", "Test MRR"],
                    rows=rows22, widths=[2600, 700, 600, 700, 1000, 900, 1000, 1000])
def mrow(lab, key, dct):
    x = dct["means"].get(key)
    return [lab, f"{f3(x['MRR'])} ± {f3(x['sd'])}", f3(x["H10"])] if x else [lab, "—", "—"]
rows23 = []
for lab, key in (("DNFR", "dnfr_rel"), ("Product t-norm (fixed)", "product_rel"), ("DWPC-LR (asinh, sampled logistic)", "dwpc_lr"),
                 ("DWPC, asinh, filtered softmax", "dwpc_sm"), ("Linear DWPC ≥ 0, filtered softmax", "dwpc_nn_sm"),
                 ("Linear DWPC ≥ 0, sampled logistic", "dwpc_nn_lr")):
    rows23.append(mrow(lab, key, D) + mrow(lab, key, H)[1:])
comp_lab = {"dwpc_sm__dwpc_lr": "DWPC softmax − DWPC-LR", "dwpc_nn_sm__dwpc_lr": "Linear ≥ 0 softmax − DWPC-LR",
            "dwpc_nn_lr__dwpc_lr": "Linear ≥ 0 logistic − DWPC-LR", "dnfr_rel__dwpc_sm": "DNFR − DWPC softmax",
            "dnfr_rel__dwpc_nn_sm": "DNFR − linear ≥ 0 softmax"}
for k, lab in comp_lab.items():
    v = FD[k]; vh = H["family"][k]
    rows23.append([lab, cell_delta(v), f"p_Holm = {v['p_holm']:.3f}" if v.get("p_holm") is not None else "—",
                   "identical ranks" if abs(vh["delta_MRR"]) < 1e-12 else cell_delta(vh), ""])
N["TAB_S23"] = dict(caption="**Table S23.** DWPC models that separate the model from the objective (ten random splits). Upper "
                    "part: test MRR (mean ± SD) and Hits@10. Lower part: paired differences with 95% bootstrap interval, "
                    "splits positive and verdict (Holm adjustment within the five comparisons of each task). On the graded "
                    "graph, with one metapath, every DWPC model ranks the candidates identically. Coefficients: the "
                    f"softmax-trained asinh model had {100 * D['coef']['dwpc_sm']['neg_share_mean']:.0f}% negative coefficients "
                    f"on Hetionet disease–symptom; the non-negative models set {100 * D['coef']['dwpc_nn_sm']['zero_share_mean']:.0f}% "
                    f"and {100 * D['coef']['dwpc_nn_lr']['zero_share_mean']:.0f}% of them to zero.",
                    header=["Model or comparison", "Hetionet DpS MRR / Δ", "Hetionet DpS H@10 / test", "Graded MRR / Δ", "Graded H@10"],
                    rows=rows23, widths=[2800, 2000, 1500, 1700, 1000])
N["S81"] = (
    "Each point of the converged profiles (Section 4.4) was fitted by two to eight warm-start chains: sweeps up and down "
    "the grid and, at selected points, chains started from the product or from the far end of the grid. Table S21 gives, "
    "for every point, the excess loss of the best chain, the gap to the second-best and to the worst chain, and whether "
    "the two best chains agree within 10⁻⁴ nat. The largest spreads of Tables S8 and S16 come from single chains that "
    "settled in a poorer basin over the whole grid; on the graded graph and on splits 0 and 2 of Hetionet "
    "disease–symptom the two best chains agree closely, whereas on split 1 of Hetionet they do not, and we do not "
    "interpret that profile. The code is dnfr/rev6_stats.py (function envelope); Figure 2 is drawn by "
    "figures/make_fig2_rev6.py.")
N["S82"] = (
    "The two metapaths with the largest mean share of DNFR's evidence over its correct top-10 test predictions on split 0 "
    "of Hetionet disease–symptom were profiled one at a time (dnfr/profile_permp.py). With the parameter of the profiled "
    "metapath fixed, the other 15 parameters were free but kept in [−20, 0]: in a first attempt that allowed p up to 1, "
    "the free parameters of other metapaths drifted to the nilpotent range, where their paths reach strength zero and "
    "stop passing gradient, and the fits stalled at a higher loss. The other parameters ended at the product end of the "
    "range (median 0). The two profiles were fitted from separate cold starts and settle at losses about 6 × 10⁻³ nat "
    "apart, another sign that the full objective has several basins; within each profile the two sweeps agree to 10⁻⁵ "
    "nat except at the cold-started point. Both profiles fall monotonically towards p = −5 and stay within the threshold "
    "over a range of several units, while validation MRR varies by at most 0.005 (Table S22).")
N["S83"] = (
    "Three DWPC models were added to DWPC-LR (dnfr/run_task.py, functions fit_linear_softmax and fit_nonneg_logistic). "
    "(i) The asinh-transformed features of DWPC-LR with a linear scorer trained by DNFR's exact filtered softmax over all "
    "candidate tails; pairs without paths score zero, which is exact because the intercept cancels within a query. "
    "(ii) A linear scorer in the untransformed DWPC, divided by the mean of each metapath as in DWPC-LR, with "
    "coefficients w = softplus(u) ≥ 0, trained by the same softmax. (iii) The same non-negative model trained with "
    "DWPC-LR's objective (20 sampled negatives per positive, balanced class weights). The L2 penalty of each was chosen "
    "from {10⁻⁴, 10⁻³, 10⁻², 10⁻¹} by validation MRR, and every model was fitted by L-BFGS. The score of model (ii) is "
    "a sum over paths of w_m w_π / mean_m with w_π ≥ 0 the degree-damped path weight; removing any fact of a path "
    "removes the path when degrees are held fixed, so each path term is a unanimity game over its facts and the "
    "Shapley value of a fact is exactly the sum of the terms of its paths divided by their lengths, as for DNFR "
    "(dnfr/explain_dwpc_nn.py). DWPC-LR was recomputed for the same splits and reproduced the released ranks exactly. "
    "The controls were run on the two disease–symptom tasks, as requested; on compound–treats–disease they were "
    "stopped for time.")
wex = C["M2_het_DpS"]
N["S86"] = (
    "The rule of Section 3.6 applied to the validation-selected fixed p against the fixed product on the graded graph "
    "(Table 4, last row). The ten per-split differences in MRR were −0.005, +0.018, +0.023, +0.021, +0.023, +0.015, "
    "+0.025, +0.021, +0.029 and +0.025, with mean +0.0195 and 9 of 10 positive. The corrected resampled t-test [44], with "
    "test-to-training ratio 1/8, gave p = 0.0021; its 90% interval was [+0.011, +0.028] and its 95% interval [+0.009, "
    "+0.030]. The stratified paired bootstrap over test queries gave a 90% interval of [+0.015, +0.024] and a 95% "
    "interval of [+0.014, +0.025]. Equivalence requires both 90% intervals inside (−0.01, +0.01); neither is, so the "
    "comparison is not equivalent. The comparison belongs to the family of 14 refinement comparisons of this task; with "
    "the smallest p of the family, its Holm-adjusted p-value is 14 × 0.0021 = 0.030 < 0.05, and the 95% bootstrap "
    "interval excludes zero on the side of the mean difference, so the verdict is different (†).")
k1, k2 = CL["kappa_relations_correct"], CL["kappa_rationale"]
rows25 = [
    ["First rater, Hetionet DpS: all relations correct", wil(fr["het_DpS"]["yes"]), "75 paths"],
    ["First rater, Hetionet DpS: a relation wrong", wil(fr["het_DpS"]["no"]), "75 paths"],
    ["First rater, Hetionet DpS: could not judge", wil(fr["het_DpS"]["unsure"]), "75 paths"],
    ["First rater, graded: all relations correct", wil(fr["hkg_DpS"]["yes"]), "75 paths"],
    ["First rater, graded: a relation wrong", wil(fr["hkg_DpS"]["no"]), "75 paths"],
    ["First rater, graded: could not judge", wil(fr["hkg_DpS"]["unsure"]), "75 paths"],
    ["Physician: correct predictions judged plausible", wil(CL["correct"]["plausible"]), "20 pairs"],
    ["Physician: incorrect predictions judged plausible", wil(CL["incorrect"]["plausible"]), "20 pairs"],
    ["Physician: relations correct, correct predictions", wil(CL["correct"]["relations_correct"]), "60 paths"],
    ["Physician: relations correct, incorrect predictions", wil(CL["incorrect"]["relations_correct"]), "60 paths"],
    ["AUC of usefulness, correct vs incorrect", f"{CL['auc_usefulness'][0]:.2f} [{CL['auc_usefulness'][1]:.2f}–{CL['auc_usefulness'][2]:.2f}]", "20 + 20 pairs"],
    ["AUC of plausibility, correct vs incorrect", f"{CL['auc_plausible'][0]:.2f} [{CL['auc_plausible'][1]:.2f}–{CL['auc_plausible'][2]:.2f}]", "20 + 20 pairs"],
    ["Cohen's κ, relations correct (two raters)", f"{k1[0]:.2f} [{k1[1]:.2f}–{k1[2]:.2f}]", f"{k1[3]} paths"],
    ["Cohen's κ, kind of rationale (two raters)", f"{k2[0]:.2f} [{k2[1]:.2f}–{k2[2]:.2f}]", f"{k2[3]} paths"],
]
N["TAB_S24"] = dict(caption="**Table S24.** The clinical pilot with 95% confidence intervals: Wilson intervals for proportions, "
                    "stratified bootstrap (5000 resamples of pairs within each class) for the AUCs, and bootstrap over "
                    "items for Cohen's κ (dnfr/rev6_stats.py).", header=["Quantity", "Estimate [95% CI]", "Items"],
                    rows=rows25, widths=[4800, 2600, 1600])
N["S87"] = (
    "The clinical assessment is a pilot: it was not pre-registered, the first rater is a co-author trained in clinical "
    "psychology who knew the aims of the study, and the second rater, a physician who is not an author, knows the authors. "
    "Table S24 gives every proportion with its interval. With 20 correct and 20 incorrect predictions, the interval for "
    "the AUC of usefulness includes 0.5, and the agreement between the raters is moderate with wide intervals.")
N["S88"] = (
    "Metapaths with more than five million paths from the query heads were excluded for all path methods. The counts "
    "were computed once on the full graph (dnfr/paths.py). On compound–treats–disease four metapaths were excluded: "
    "compound–causes–side effect–causes–compound–palliates–disease (7.6 million), "
    "compound–causes–side effect–causes–compound–treats–disease (12.7 million), "
    "compound–downregulates–gene–expressed in–anatomy–localizes–disease (5.8 million) and "
    "compound–upregulates–gene–expressed in–anatomy–localizes–disease (5.5 million). Only the second contains the treats "
    "relation; with the validation and test pairs of each split removed it still had 9.9 to 10.2 million paths on every "
    "one of the ten splits, so per-split counting would have excluded the same four metapaths. The other three contain no "
    "held-out edge type, so their counts do not depend on the split. On PrimeKG we did not repeat the check.")
json.dump(N, open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "numbers_partial.json"), "w"), indent=1, ensure_ascii=False)
print("ok2")

# ================================================================ response letter texts (independent of the CtD queues)
N["RESP_M1A"] = (
    "We agree that the earlier wording claimed too much, and we examined where the chain disagreement came from. Each "
    "profile point had been fitted by two to eight warm-start chains, and Table S16 reported the largest difference "
    "between any two of them. Most of that spread came from single chains that settled in a poorer basin over the whole "
    f"grid. The two best chains agreed within {maxgap_h * 1e3:.2f} × 10⁻³ nat at {rh} of {nh} points on the graded graph "
    f"and within {maxgap_d * 1e3:.2f} × 10⁻³ nat at {rd} of {nd} points on splits 0 and 2 of Hetionet disease–symptom, "
    "well below the threshold χ²(0.95, 1)/(2N) of about 0.7 × 10⁻³ nat. On split 1 of Hetionet the two best chains also "
    f"disagreed ({min(gaps1) * 1e3:.0f} to {max(gaps1) * 1e3:.0f} × 10⁻³ nat at 8 of 9 points), and for that split we "
    "now say exactly what you propose: the profile could not be resolved at the precision attained. The new Figure 2 "
    "shows every profile with its chain envelope and marks unresolved points, and Table S21 lists all points.")
N["RESP_M1B"] = (
    "On the resolved profiles the training loss does locate p within a split, but loosely: on the graded graph the "
    "points within the threshold are [−0.5, −0.25] on split 1, {0.5} on split 2 and, on split 0, two disjoint ranges "
    "(p ≤ −2.5 and 0.1–0.25; Table S8 previously showed their hull [−5, 0.25], which we have corrected); on Hetionet "
    "they are [−20, −10] and [−5, −1]. The minimiser moves between splits (0.1, −0.25, 0.5 and −20, −2), which we now "
    "call sampling variability rather than non-identifiability. The Abstract now reads: \"Within a split the converged "
    "training loss located the t-norm parameter only within wide or disjoint intervals, its minimiser moved between "
    "splits, and it did not coincide with the value that maximised validation MRR.\"")
N["RESP_M1C"] = (
    "Thank you; the converged profile supports your reading. On the graded graph the mean excess loss peaks between "
    "p = −0.5 and −0.25 (Table S7) and, on split 0, a ridge of 5.1 × 10⁻³ nat at p = −0.25 separates a flat region at "
    "p ≤ −2.5 from a second basin near 0.1–0.25. Restarts without the penalty that start on either side of the ridge "
    "move away from it (from −2 to −3.51, from 0.5 to 0.91), as a ridge with lower ground on both sides would produce. "
    "Section 4.4 now says this and points to Figure 2A; it also says that the default penalty λ‖θ − θ₀‖² keeps learned "
    "values near their start by construction (minor comment 2).")
pmd = {r["mp"]: r for r in pm_rows}
N["RESP_M1D"] = (
    "We profiled the two metapaths that carry most of DNFR's evidence on Hetionet disease–symptom, "
    f"disease–resembles–disease–presents–symptom ({100 * pmd['DrD-DpS']['share']:.0f}%) and "
    f"disease–localizes–anatomy–localizes–disease–presents–symptom ({100 * pmd['DlA-DlA^-1-DpS']['share']:.0f}%), one at a "
    "time with the other 15 parameters, the memberships, reliabilities and leak re-fitted (Table S22, Section S8.2). "
    "Both profiles were resolved (sweeps within 10⁻⁵ nat except at one cold-started point) and fall monotonically "
    f"towards p = −5: the loss rose by {pmd['DrD-DpS']['range']:.1f} and {pmd['DlA-DlA^-1-DpS']['range']:.1f} × 10⁻³ nat "
    f"from −5 to the product, the points within the threshold were {interval(pmd['DrD-DpS']['inside'])} and "
    f"{interval(pmd['DlA-DlA^-1-DpS']['inside'])}, and validation MRR changed by at most "
    f"{max(pmd['DrD-DpS']['vrange'], pmd['DlA-DlA^-1-DpS']['vrange']):.3f}. Per metapath, then, the loss gives a direction "
    "but not a value. We kept all parameters in the non-nilpotent range p ≤ 0 for these fits, because in a first attempt "
    "the free parameters of other metapaths drifted to p > 0, where paths reach strength zero and stop passing "
    "gradient. The claim in Section 4.4 is now restricted accordingly: we no longer say that no learned value carries "
    "information, only that learned per-metapath values should not be interpreted, given the weak correlation between "
    "restarts and these one-sided profiles.")
N["RESP_M1E"] = (
    "We agree, and Section 4.4 now frames the result as a mismatch between the surrogate loss and the metric. At the "
    "values where validation MRR peaked (−1.5, −2, −1.5), the training loss lay "
    f"{vexc[0]:.1f}, {vexc[1]:.1f} and {vexc[2]:.1f} × 10⁻³ nat above its minimum, outside the threshold on every split, "
    "so the data do carry information about p that the loss does not register. Figure 2C now shows validation MRR along "
    "every profile, which is the ranking-aware criterion we could compute at every converged fit. We did not add a "
    "validation log-likelihood profile: it is the same filtered softmax on another set of queries and would inherit the "
    "insensitivity to the top of the ranking that the comparison with MRR exposes.")
N["RESP_M2A"] = (
    f"Done (Section 4.1, Table 2, Table S23). With the same asinh features and DNFR's exact filtered softmax, DWPC "
    f"changed by {f3(sm_lr['delta_MRR'], True)} {ci(sm_lr)} on Hetionet disease–symptom ({sm_lr['seed_wins']}/10 splits; "
    f"equivalent within ±0.01 under our rule) and remained ahead of DNFR by {f3(-dn_sm['delta_MRR'])} (different). On the "
    "graded graph, which has a single metapath, every increasing function of the single DWPC feature ranks the "
    "candidates identically, so DWPC-LR, its softmax version and the non-negative linear model give the same ranks and "
    "the same MRR (0.331); the objective cannot matter there. DNFR's deficit on the disease–symptom tasks therefore "
    "comes from the model, not from the loss. We ran the controls on the two tasks you named; on compound–treats–disease "
    "we stopped them for time, and on PrimeKG they were not feasible.")
N["RESP_M2B"] = (
    "You are right, and the result removes the claim. The linear DWPC model with non-negative coefficients (no asinh, "
    f"trained with the filtered softmax) reached {f3(D['means']['dwpc_nn_sm']['MRR'])} ± "
    f"{f3(D['means']['dwpc_nn_sm']['sd'])} on Hetionet disease–symptom against {f3(D['means']['dnfr_rel']['MRR'])} for "
    f"DNFR ({f3(-dn_nn['delta_MRR'], True)} {ci({'ci_lo': -dn_nn['ci_hi'], 'ci_hi': -dn_nn['ci_lo']})}, 10/10 splits, "
    "different), and its attribution to facts is exact and non-negative by the same unanimity-game argument. In "
    f"Table 6, its leading fact carried {pc(EX['top_edge_share_median'])} of its attribution, half lay on "
    f"{EX['k50_edges_median']:.0f} facts, and removing the leading fact left {pc(EX['stay_top10_cf_top'])} of its correct "
    f"top-10 predictions in the top 10, against {pc(EX['stay_top10_cf_random'])} for a random fact (DNFR: 12%, 6 facts, "
    "60% against 99%). On the graded graph it coincides with DWPC-LR. Section 4.6, Section 5.1, the Abstract, Highlight 4 "
    "and the Conclusion now say that exact, non-negative attribution to facts is a property of additive path models "
    "with non-negative terms, not an advantage of fuzzy composition, and the phrase \"difference in kind\" is gone.")
N["RESP_M5B"] = (
    "The counts were computed once on the full graph, before splitting. We checked what per-split counting would have "
    "changed (Section 3.5, Section S8.8). On compound–treats–disease only one of the four excluded metapaths contains "
    "the treats relation (compound–causes–side effect–causes–compound–treats–disease, 12.7 million paths); with the "
    "validation and test pairs removed it still had 9.9 to 10.2 million paths on each of the ten splits, so the same "
    "metapaths would have been excluded. The other three contain no held-out edge type. We did not repeat the check "
    "on PrimeKG and say so. We also note in Section 5.3 that the exclusion may lead us to underestimate the effect of "
    "hub discounting.")
N["RESP_M6"] = (
    "We chose option (b). A study with at least three physicians who are not authors was not possible within the "
    "revision period, and we did not want to present the current data as more than they are. The assessment is now "
    "called a pilot in Sections 3.6 and 5.3; Section 4.6 keeps two sentences, with confidence intervals; and the details "
    "are in Supplementary Sections S6.2, S7 and S8.7, with every proportion, the two AUCs and both κ values given with "
    f"95% intervals (Table S24). The AUC of usefulness is {CL['auc_usefulness'][0]:.2f} with a 95% interval of "
    f"{CL['auc_usefulness'][1]:.2f}–{CL['auc_usefulness'][2]:.2f}, which includes 0.5, and κ for the correctness of "
    f"relations is {k1[0]:.2f} ({k1[1]:.2f}–{k1[2]:.2f}). The Acknowledgements no longer call the second rater independent; "
    "they now say that he is not an author but is acquainted with the authors, as Section S7 does. Section 5.3 describes "
    "the design you outline (several non-author physicians, a pre-specified protocol, blinding to the model, DNFR and "
    "DWPC explanations side by side) as the next step.")
N["RESP_MINOR14"] = (
    "14. The GitHub release and the Zenodo record of this revision are public (release <<VER>>; Section 7). We checked "
    "the rating sheets: they contain the raters' qualifications, specialty and years of practice, as the raters agreed, "
    "and the name and workplace fields of the second rater are withheld; the free-text comments contain no names.")
json.dump(N, open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "numbers_partial.json"), "w"), indent=1, ensure_ascii=False)
print("ok3")

# ================================================================ entity-centred splits (CtD queues) and AnyBURL
def g(dct, key):
    return dct["means"].get(key) if dct else None
if T:
    mt = T["means"]; ft = T["family"]
    ndn = mt["dnfr"]["n"]
    nd_v = ft.get("dnfr_nodeg__dnfr", {})
    N["P_45_TAIL"] = (
        f"When every indication of a test disease was held out (Table 5A), the fixed product fell to "
        f"{f3(mt['product']['MRR'])} from 0.602 and DWPC-LR to {f3(mt['dwpc_lr']['MRR'])} from 0.594 over "
        f"{T['n_splits']} splits, and DNFR to {f3(mt['dnfr']['MRR'])} on the {ndn} splits on which it was run, about twice "
        f"the chance level of {f3(rand_ctd)}. Completion, popularity and the edge "
        f"prior ({f3(mt['softimpute']['MRR'])}, {f3(mt['popularity']['MRR'])} and {f3(mt['prior']['MRR'])}) fell to or "
        "below chance: every held-out disease has no training positive, so all of them tie at the bottom of a ranking "
        f"by popularity. Degree and prior features hurt DWPC-LR ({f3(mt['dwpc_lr_full']['MRR'])} against "
        f"{f3(mt['dwpc_lr']['MRR'])}). DNFR without degree terms reached {f3(mt['dnfr_nodeg']['MRR'])} "
        f"({f3(nd_v.get('delta_MRR', 0), True)} {ci(nd_v)}, {nd_v.get('seed_wins')}/{nd_v.get('seeds')} splits, "
        f"{word(nd_v)}), a plausible reversal, since hubs are the diseases with many known treatments. Hits@10 separated "
        "the methods more than MRR did: at similar MRR, on the same four splits, the Hamacher product and DNFR without "
        f"degree terms placed the true compound in the top 10 for {f3(mt['hamacher']['H10'])} and "
        f"{f3(mt['dnfr_nodeg']['H10'])} of test pairs, against {f3(mt['dnfr']['H10'])} for DNFR. Because a clinician inspects a short list, Hits@10 is the more "
        "practical criterion here, but with about ten test diseases per split these differences are not established.")
TN_ok = bool(TN)
HD_ok = bool(HD)
txt = ("On random splits, the 33 metapaths that contain the treats relation carried 99% of DNFR's evidence for its "
       "correct top-10 predictions, against 0.5% for metapaths through a gene that the compound targets. Without them "
       "(Table 5B), DNFR fell from 0.599 to 0.289 (−0.310 [−0.343, −0.277], different), DWPC-LR from 0.594 to 0.235 and "
       "the fixed product from 0.602 to 0.281. Removing the palliates metapaths too (Table 5C) changed DNFR by a further "
       "−0.012, and gene-only metapaths then carried 46.5% of its evidence.")
if TN_ok:
    mn = TN["means"]; fn = TN["family"]
    vp, vd = fn["product_notgt__product"], fn["dwpc_lr_notgt__dwpc_lr"]
    txt += (" On the disease-centred split the picture reversed (Table 5D). Trained without the treats metapaths, the "
            f"fixed product reached {f3(mn['product_notgt']['MRR'])} and DWPC-LR {f3(mn['dwpc_lr_notgt']['MRR'])} for "
            f"held-out diseases, against {f3(mn['product']['MRR'])} and {f3(mn['dwpc_lr']['MRR'])} with them "
            f"({f3(vp['delta_MRR'], True)} {ci(vp)} and {f3(vd['delta_MRR'], True)} {ci(vd)}, {vp['seed_wins']}/10 and "
            f"{vd['seed_wins']}/10 splits, both different), close to their random-split accuracy without those metapaths "
            "(0.281 and 0.235) and about six times the chance level. With the treats metapaths available in training, "
            "both models presumably came to rest on known indications and gave little weight to the rest of the graph, "
            "which is all that a disease without indications has; without them, the same graph supported a useful "
            "zero-shot ranking. Known indications thus act as a shortcut that raises random-split accuracy and lowers "
            "accuracy for new diseases.")
if HD_ok:
    mh = HD["means"]
    dn = mh.get("dnfr")
    txt += (f" When every indication of a test compound was held out instead (Table 5E, {HD['n_splits']} splits), the "
            f"fixed product reached {f3(mh['product']['MRR'])} and DWPC-LR {f3(mh['dwpc_lr']['MRR'])}"
            + (f", DNFR {f3(dn['MRR'])} on {dn['n']} splits" if dn else "") +
            f", against {f3(mh['popularity']['MRR'])} for popularity and {f3(rand_ctd)} for chance, far above the disease-centred "
            "level. A new compound loses its own indications but not those of similar compounds or compounds of its "
            "pharmacologic class, so this accuracy presumably still rests on the same association; we did not ablate the "
            "treats metapaths on this split.")
txt += (" Random-split accuracy on curated drug repurposing thus comes largely from guilt by association, for DWPC-LR as "
        "much as for DNFR: a compound ranks high because similar compounds, or compounds of its pharmacologic class, "
        "treat the disease. The evidence outside the treats relation supports a lower random-split accuracy, but it is "
        "what carries over to a disease without known treatments, provided the model is trained on it alone.")
N["P_45_TREATS"] = txt
N["RESP_M3C"] = (
    "We added the chance level for every task (Section 3.4 gives the formula; Table 2 and Table 5): 0.041 for "
    "compound–treats–disease, 0.016 for Hetionet disease–symptom, 0.021 for the graded graph and about 0.0006 for "
    "PrimeKG. As you note, popularity falls below chance on the disease-centred split because all test diseases tie at "
    "zero, and the text now says so. "
    + (f"The disease-centred split now has {T['n_splits']} splits (splits 4–9 added) for the fixed product, DWPC-LR and "
       f"all baselines, and {T['means']['dnfr']['n']} for DNFR"
       + (" and DNFR without degree terms" if T['means']['dnfr_nodeg']['n'] == T['means']['dnfr']['n'] else "")
       + f"; DNFR's MRR there is {f3(T['means']['dnfr']['MRR'])}. " if T else "")
    + "Leave-one-disease-out on the 97–137 diseases with indications would need one DNFR fit per disease (about ten "
    "minutes each on our CPU), which we could not afford; ten random disease-centred splits hold out each disease about "
    "once on average. "
    + (f"A compound-centred split was added (Table 5E, {HD['n_splits']} splits): the fixed product reached "
       f"{f3(HD['means']['product']['MRR'])}, DWPC-LR {f3(HD['means']['dwpc_lr']['MRR'])} and popularity "
       f"{f3(HD['means']['popularity']['MRR'])}." if HD else ""))
N["RESP_M3D"] = (
    (f"Done (Table 5D, {TN['n_splits']} splits), and the result was the most surprising of the revision. Without the "
     f"treats metapaths the disease-centred MRR rose, from {f3(TN['means']['product']['MRR'])} to "
     f"{f3(TN['means']['product_notgt']['MRR'])} for the fixed product and from {f3(TN['means']['dwpc_lr']['MRR'])} to "
     f"{f3(TN['means']['dwpc_lr_notgt']['MRR'])} for DWPC-LR (10/10 splits each, different under our rule), close to "
     "their random-split accuracy without those metapaths (0.281 and 0.235) and far above chance (0.041). So the "
     "mechanistic and associative evidence outside the treats relation does have zero-shot value, but only when the "
     "model is not trained to lean on known indications. Sections 4.5, 5.2 and 6 and the Abstract now report this. We "
     "ran the fixed product rather than DNFR, which did not differ from it on this task (Table 3), to fit the runs into "
     "the time available.") if TN else "Not completed.")
N["RESP_MINOR5"] = (
    "5. Hits@10 on the disease-centred split: Section 4.5 now comments on it. With ten splits, the Hamacher product "
    + (f"({f3(T['means']['hamacher']['H10'])}) and DNFR without degree terms ({f3(T['means']['dnfr_nodeg']['H10'])}) still "
       f"placed the true compound in the top 10 more often than DNFR ({f3(T['means']['dnfr']['H10'])}) at similar MRR" if T else "")
    + "; since clinicians inspect short lists, we call Hits@10 the more practical criterion there, while noting that "
    "with about ten test diseases per split the differences are not established.")
N["S84"] = (
    "The disease-centred split (dnfr/run_task.py, split_by_tail) holds out 10% of the diseases with at least one "
    "indication, with all their indications; splits 0–3 were run first and splits 4–9 for this revision with the same "
    "code. The compound-centred split (split_by_head) holds out 10% of the compounds with at least one indication, "
    "with all their indications, the setting of a drug without known uses. Without the treats metapaths "
    "(DNFR_DROP_TARGET_MPS=1), the disease-centred split was run for the fixed product and DWPC-LR. Table 5 gives the "
    "results; every comparison uses the per-query ranks released in results_rev6/.")
AB = sorted(glob.glob(f"{R6}/anyburl1000/anyburl_*_seed*.json"))
if AB:
    res = collections.defaultdict(list)
    for f in AB:
        a = json.load(open(f)); res[a["task"]].append((a["seed"], a["metrics"]["MRR"]))
    old = {}
    for t in res:
        for s, _ in res[t]:
            fo = f"{R0}/anyburl_{t}_seed{s}.json"
            if os.path.exists(fo): old.setdefault(t, []).append(json.load(open(fo))["metrics"]["MRR"])
    lab = {"hkg_DpS": "the graded graph", "het_CtD": "compound–treats–disease"}
    parts = []
    for t, v in res.items():
        oldm = [json.load(open(f"{R0}/anyburl_{t}_seed{s}.json"))["metrics"]["MRR"] for s, _ in v]
        newm = [m for _, m in v]
        same = all(abs(a - b) < 1e-9 for a, b in zip(oldm, newm))
        parts.append(f"{lab[t]}, {len(v)} split{'s' if len(v) > 1 else ''}: MRR {f3(np.mean(newm))} with 1000 s and "
                     f"{f3(np.mean(oldm))} with 100 s{', with identical ranks' if same else ''}")
    N["ANYBURL_1000"] = ("; ten times the rule-learning budget left it unchanged (" + "; ".join(parts) +
                         "; Supplementary Section S8.5)")
    N["S85"] = ("AnyBURL 23-1 was re-run with 1000 s of rule learning instead of 100 s, on the same splits and with the same "
                "application settings (top 200 candidates per query): " + "; ".join(parts) + ". On this graph, with a single "
                "relation, the rules found within 100 s already gave the final ranking. The longer run on "
                "compound–treats–disease did not fit into the time available. Result files: results_rev6/anyburl1000/.")
    N["RESP_M4C"] = ("We kept 100 s for the main tables and added a run with 1000 s on the splits we could afford (Section "
                     "S8.5): " + "; ".join(parts) + ". On the graded graph the rules found within 100 s already gave the final ranking. We could "
                     "not afford the longer run on compound–treats–disease, and Section 5.3 notes that the rule sets there "
                     "may not have saturated, so AnyBURL's accuracy on that task is a lower bound.")
else:
    N["ANYBURL_1000"] = ""
    N["S85"] = "The longer AnyBURL run could not be completed in the time available."
    N["RESP_M4C"] = ("We could not complete a longer run in the revision period; Section 5.3 now says that rule sets may not "
                     "have saturated at 100 s on graphs of this size, so AnyBURL's accuracy is a lower bound.")
json.dump(N, open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "numbers.json"), "w"), indent=1, ensure_ascii=False)
print("ok4")
