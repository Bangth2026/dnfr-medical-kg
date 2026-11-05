"""Results tables of the main text (Section 4) and of the supplementary material. Text is in ms_text.py."""
import numpy as np
from ms_common import *
from docx_helpers import para, heading, table, figure

NAMES = {"dnfr": "DNFR (learned p per metapath)", "dnfr2": "DNFR 2.0 (R1–R4)", "fixp_val": "Fixed p, selected on validation",
         "product": "Product t-norm (fixed)", "hamacher": "Hamacher product (fixed)",
         "min": "Minimum t-norm (fixed)", "luk": "Łukasiewicz t-norm (fixed)", "dnfr_shared": "DNFR, one shared p",
         "dnfr_nodeg": "DNFR without degree terms", "dnfr_fixdeg": "DNFR, degree terms fixed (b = c = 0.4)",
         "product_fixdeg": "Product, degree terms fixed (b = c = 0.4)",
         "dnfr_max": "DNFR, strongest path only", "dnfr_binary": "DNFR, edge weights ignored",
         "dwpc_lr": "DWPC + logistic regression", "dwpc_lr_full": "DWPC-LR + degree + prior", "anyburl": "AnyBURL (path rules)",
         "nbfnet": "NBFNet (compact, CPU)", "nbfnet_tuned": "NBFNet, tuned (2 × 2 grid, chosen on validation)",
         "popularity": "Popularity (training positives of the tail)", "prior": "Edge prior (degree-preserving permutations)",
         "svd": "Truncated SVD (rank chosen on validation)", "svd_rank3": "Rank-3 SVD", "softimpute": "Soft-thresholded SVD (Soft-Impute)",
         "kge_TransE": "TransE (untuned)", "kge_RotatE": "RotatE (untuned)",
         "dnfr_overlap": "DNFR + R1 overlap-aware aggregation", "dnfr_gumbel": "DNFR + R2 discrete operator choice",
         "dnfr_ic": "DNFR + R3 information-content discount", "dnfr_anneal": "DNFR + R4 membership annealing",
         "product_overlap": "Product + R1", "dnfr2_no_overlap": "DNFR 2.0 without R1", "dnfr2_no_gumbel": "DNFR 2.0 without R2",
         "dnfr2_no_ic": "DNFR 2.0 without R3", "dnfr2_no_anneal": "DNFR 2.0 without R4",
         "fixp_m4": "Fixed p = −4", "fixp_m2": "Fixed p = −2", "fixp_m1": "Fixed p = −1", "fixp_m05": "Fixed p = −0.5",
         "fixp_05": "Fixed p = 0.5", "dnfr_frozen": "DNFR, memberships frozen", "product_frozen": "Product, memberships frozen"}
R3T = ["het_CtD", "het_DpS", "hkg_DpS"]           # tasks on which the refinements, fixed p and NBFNet were run
ONLY_R3 = {"fixp_val", "nbfnet", "nbfnet_tuned"}


def T(d, *a, **k):
    a = list(a); a[0] = C(a[0])
    if k.get("note"):
        k["note"] = C(k["note"])
    return table(d, *a, **k)


# ---------------------------------------------------------------- third-revision statistics (rev3_stats.json)
def r3sum(t, m):
    return R3.get(t, {}).get("summary", {}).get(m)


def r3p(t, a, b):
    """Paired comparison a minus b from rev3_stats (family 'refine' or 'nbfnet'), oriented as asked."""
    P = R3.get(t, {}).get("paired", {})
    if f"{a}__{b}" in P:
        return P[f"{a}__{b}"]
    if f"{b}__{a}" in P:
        v = dict(P[f"{b}__{a}"]); v["delta"] = -v["delta"]
        for k in ("bs90", "bs95", "nb90", "nb95"):
            if k in v:
                v[k] = [-v[k][1], -v[k][0]]
        v["per_split"] = [-x for x in v["per_split"]]; v["wins"] = sum(x > 0 for x in v["per_split"])
        return v
    return None


def r3cell(p, mark=True):
    if p is None:
        return "—"
    s = f"{sg(p['delta'])} [{sg(p['bs95'][0])}, {sg(p['bs95'][1])}] {p['wins']}/{p['J']}"
    return s + (VMARK.get(p.get("verdict"), "") if mark else "")


def mrr_any(t, m, sd=True):
    """Mean (± SD) test MRR; methods that exist only in the third-revision statistics are read from there."""
    if m in ONLY_R3 or summ(t, m) is None:
        s = r3sum(t, m)
        if s is None:
            return "—"
        out = f3(s["mean"]) + (f" ± {f3(s['sd'])}" if sd and s["n"] > 1 else "")
        return out + (f" ({s['n']})" if s["n"] != nseeds(t, "dnfr") else "")
    return mrr(t, m, sd=sd)


def mean_any(t, m):
    s = r3sum(t, m) if (m in ONLY_R3 or summ(t, m) is None) else None
    if s:
        return s["mean"]
    s = summ(t, m)
    return s["MRR"]["mean"] if s else None


# ---------------------------------------------------------------- Table 2
def table2(d):
    groups = [("§Path methods, fixed or learned weighting of the same paths", ["dnfr", "dnfr2", "product", "hamacher", "fixp_val",
                                                                            "dwpc_lr", "dwpc_lr_full", "anyburl"]),
              ("§Learned path operators", ["nbfnet", "nbfnet_tuned"]),
              ("§Without paths", ["softimpute", "svd", "prior", "popularity"]),
              ("§Knowledge graph embeddings (untuned reference points)", ["kge_RotatE", "kge_TransE"])]
    rows = []
    for g, meths in groups:
        rows.append([g])
        for m in meths:
            rows.append([NAMES[m]] + [mrr_any(t, m) for t in TASKS])
    T(d, "**Table 2.** Test MRR (mean ± SD over random splits) of DNFR, its variants and the baselines. Numbers in parentheses give "
         "the number of splits when it differs from that of DNFR (" + ", ".join(f"{nseeds(t, 'dnfr')}" for t in TASKS) + " splits on "
         "the four tasks). MRR is not comparable across columns, because the number of candidates differs (Table 1). Fixed p, "
         "selected on validation: one Schweizer–Sklar p chosen per split from a grid by validation MRR, trained with the protocol of "
         "DNFR. DNFR 2.0 (all four refinements of Section 3.3), the validation-selected fixed p and NBFNet were not run on PrimeKG. "
         "Hits@k, intervals and every "
         "variant are in Supplementary Tables S1–S4.",
      ["Method"] + [TASK_LABEL[t] for t in TASKS], rows, widths=[2.07, 1.05, 1.05, 1.05, 1.05], size=8)


# ---------------------------------------------------------------- Table 3
def table3(d):
    rows = [["§A. Components, learned memberships: model with the component minus the model without it"]]

    def r(label, a, b, skip=()):
        rows.append([label] + ["—" if t in skip else cell(pr(t, a, b)) for t in TASKS])
    r("Noisy-OR instead of the strongest path", "dnfr", "dnfr_max")
    r("Degree terms instead of none", "dnfr", "dnfr_nodeg")
    r("Learned instead of fixed degree terms", "dnfr", "dnfr_fixdeg")
    r("Graded edge weights instead of binary edges", "dnfr", "dnfr_binary", skip=("het_CtD", "het_DpS", "pkg_ind"))
    rows.append(["§B. t-norm"])
    r("Learned p per metapath instead of the product", "dnfr", "product")
    r("Learned p per metapath instead of the Hamacher product", "dnfr", "hamacher")
    r("Learned p per metapath instead of one shared p", "dnfr", "dnfr_shared")
    r("Hamacher product instead of the product", "hamacher", "product")
    rows.append(["§C. Memberships frozen at their initial values"])
    r("Learned p per metapath instead of the product", "dnfr_frozen", "product_frozen")
    r("Product: learned memberships instead of frozen", "product", "product_frozen")
    T(d, "**Table 3.** What each learned component adds. Each cell is the difference in test MRR between the model with the component "
         "and the model without it, averaged over the splits on which both were run, with its 95% paired bootstrap interval and the "
         "number of splits on which the difference was positive (out of the splits run). Verdicts under the rule of Section 3.6, "
         "with Holm adjustment within the family of component comparisons of each task (all panels; Section 3.6): ≈ equivalent "
         "within ±0.01; † different; no mark, "
         "inconclusive. — not run or not applicable (Hetionet and PrimeKG edges carry no graded weight).",
      ["Component"] + [TASK_LABEL[t] for t in TASKS], rows, widths=[1.87, 1.1, 1.1, 1.1, 1.1], size=7.5)


# ---------------------------------------------------------------- Table 4: refinements
REFINE_ROWS = [("§Added one at a time to DNFR (refined minus DNFR)", None, None),
               ("R1 overlap-aware aggregation", "dnfr_overlap", "dnfr"), ("R2 discrete operator choice", "dnfr_gumbel", "dnfr"),
               ("R3 information-content discount", "dnfr_ic", "dnfr"), ("R4 membership annealing", "dnfr_anneal", "dnfr"),
               ("R1 added to the fixed product", "product_overlap", "product"),
               ("§Removed one at a time from DNFR 2.0 (DNFR 2.0 minus the version without it)", None, None),
               ("R1", "dnfr2", "dnfr2_no_overlap"), ("R2", "dnfr2", "dnfr2_no_gumbel"), ("R3", "dnfr2", "dnfr2_no_ic"),
               ("R4", "dnfr2", "dnfr2_no_anneal"),
               ("§Combined", None, None),
               ("DNFR 2.0 minus DNFR", "dnfr2", "dnfr"), ("DNFR 2.0 minus the fixed product", "dnfr2", "product"),
               ("§Selecting a fixed t-norm on validation data", None, None),
               ("Fixed p (validation) minus DNFR", "fixp_val", "dnfr"), ("Fixed p (validation) minus the fixed product", "fixp_val", "product")]


def table4(d):
    rows = []
    for lab, a, b in REFINE_ROWS:
        rows.append([lab] if a is None else [lab] + [r3cell(r3p(t, a, b)) for t in R3T])
    T(d, "**Table 4.** The four refinements and a validation-selected fixed t-norm. Each cell gives the difference in test MRR, its "
         "95% paired bootstrap interval and the number of splits on which it was positive, out of the splits run. Verdicts as in "
         "Table 3, with Holm adjustment within the family of refinement comparisons of each task (all rows of this table; "
         "Section 3.6). DNFR 2.0 combines R1–R4. — not run. Not run on PrimeKG.",
      ["Comparison"] + [TASK_LABEL[t] for t in R3T], rows, widths=[2.07, 1.4, 1.4, 1.4], size=7.5)


# ---------------------------------------------------------------- Supplementary: identifiability of the operator
def supp_identifiability(d, num):
    V = [("dnfr", "p₀ = −0.1, λ = 10⁻³ (default)"), ("dnfr_l2_0", "p₀ = −0.1, no penalty"), ("dnfr_pm2", "p₀ = −2, no penalty"),
         ("dnfr_pm1", "p₀ = −1, no penalty"), ("dnfr_pp05", "p₀ = 0.5, no penalty")]
    rows = [["§Bounded parameter p = 6σ(θ) − 5, default early stopping: test MRR; learned p, median [IQR]"]]
    for v, lab in V:
        row = [lab]
        for t in TASKS:
            s = summ(t, v); P = S[t].get("p_by_variant", {}).get(v)
            if s is None:
                row.append("—"); continue
            txt = f3(s["MRR"]["mean"]) + (f" ({s['n_seeds']})" if s["n_seeds"] != nseeds(t, "dnfr") else "")
            if P:
                txt += f"; {f3(P['median'], 2)} [{f3(P['q25'], 2)}, {f3(P['q75'], 2)}]"
            row.append(txt)
        rows.append(row)
    rows.append(["§Unbounded parameter (θ clipped to [−20, 0.95]): learned p, median over metapaths and splits"])
    for v, lab in (("dnfr_free_pm2", "from −2"), ("dnfr_free", "from −0.1"), ("dnfr_free_pp05", "from 0.5")):
        row = [lab]
        for t in TASKS:
            F = R3.get(t, {}).get("free_p")
            if not F:
                row.append("—"); continue
            row.append(f"{f3(np.median([r['median_by_init'][v] for r in F]), 2)}; MRR {f3(np.mean([r['mrr'][v] for r in F]))} ({len(F)})")
        rows.append(row)
    rows.append(["Spread of p over the three starts, median over metapaths"] + [
        f3(np.median([r["spread_median"] for r in R3[t]["free_p"]]), 2) if R3.get(t, {}).get("free_p") else "—" for t in TASKS])
    rows.append(["§Discrete operator choice (DNFR 2.0)"])
    rows.append(["Metapaths with the same operator in three restarts"] + [
        f"{pct(R3[t]['restarts']['dnfr2']['all_same_mean'])} ({R3[t]['restarts']['dnfr2']['n_splits']})"
        if R3.get(t, {}).get("restarts", {}).get("dnfr2") else "—" for t in TASKS])

    def ops(t):
        c = R3.get(t, {}).get("ops_single") or {}
        n = sum(c.values())
        return "; ".join(f"{k} {pct(v / n)}" for k, v in sorted(c.items(), key=lambda kv: -kv[1])) if n else "—"
    rows.append(["Operators chosen (share of metapaths × splits)"] + [ops(t) for t in TASKS])
    rows.append(["§Fixed shared p, default early stopping"])

    def choice(t):
        c = R3.get(t, {}).get("fixp_val_choice") or {}
        n = sum(c.values())
        return "; ".join(f"{NAMES.get(k, k).replace('Fixed ', '')} {v}" for k, v in sorted(c.items(), key=lambda kv: -kv[1])) if n else "—"
    rows.append(["Value chosen on validation (splits)"] + [choice(t) for t in TASKS])
    rs = RS["stopping"]
    rows.append(["Epoch of best validation MRR, DNFR (median)"] + [f"{ri(rs[t]['dnfr']['best_epoch']['median'])}"
                                                                if t in rs and "dnfr" in rs[t] else "—" for t in TASKS])
    T(d, f"**Table S{num}.** Does training determine the t-norm? Restarts of the bounded parameterisation, an unbounded parameterisation, "
         "restarts of the discrete operator choice, and the value of a fixed shared p chosen on validation data. Numbers in "
         "parentheses give the number of splits when it differs from that of DNFR.",
      ["Setting"] + [TASK_LABEL[t] for t in TASKS], rows, widths=[1.67, 1.15, 1.15, 1.15, 1.15], size=7.5)


def np_median(x):
    return float(np.median(x))


# ---------------------------------------------------------------- Table 6: held-out diseases
def table_tail(d):
    tg = "_tail"
    meths = ["dnfr", "product", "hamacher", "dnfr_nodeg", "dwpc_lr", "dwpc_lr_full", "softimpute", "svd", "prior", "popularity"]
    rows = [[f"§A. Disease-centred split ({nseeds('het_CtD', 'dnfr', tg)} splits): every indication of a test disease held out; "
             "ΔMRR is the method minus DNFR"]]
    for m in meths:
        s = summ("het_CtD", m, tg)
        if s is None:
            continue
        rows.append([NAMES[m], mrr("het_CtD", m, sd=False, mark_n=False), mrr("het_CtD", m, tg), f3(s["H10"]["mean"]),
                     "—" if m == "dnfr" else cell(_neg(pr("het_CtD", "dnfr", m, tg)), verdict=False)])
    NT = R4.get("notgt")
    if NT:
        rows.append([f"§B. Random splits without the {R4['explain_dwpc']['het_CtD']['dnfr']['n_target_metapaths']} metapaths that "
                     "contain the treats relation; ΔMRR is the model without them minus the model with them"])
        for m in ("dnfr", "product", "dwpc_lr"):
            a_ = NT["means"].get(m + "_notgt"); p_ = NT["paired"].get(f"{m}_notgt__{m}")
            if not a_ or not a_["n"]:
                continue
            rows.append([NAMES[m], f3(NT["means"][m]["mean"]), f"{f3(a_['mean'])} ± {f3(a_['sd'])} ({a_['n']})", "—",
                         r3cell(p_) if p_ else "—"])
    T(d, "**Table 5.** Hetionet compound–treats–disease in two harder settings. Random-split MRR from Table 2 for comparison (in "
         "panel B, over the same splits). Each Δ is given with its 95% bootstrap interval and the number of splits on which it "
         "was positive; in panel B, verdicts as in Table 3 (Holm adjustment within the panel).",
      ["Method", "MRR, random split", "MRR, harder setting", "Hits@10, harder setting", "ΔMRR [95% CI], splits"], rows,
      widths=[2.0, 0.85, 1.05, 0.85, 1.52], size=8)


def _neg(p):
    if p is None:
        return None
    q = dict(p); q["delta_MRR"] = -p["delta_MRR"]; q["ci_lo"], q["ci_hi"] = -p["ci_hi"], -p["ci_lo"]
    q["seed_wins"] = sum(1 for x in p["per_split"] if x < 0); return q


# ---------------------------------------------------------------- Table 7: explanations
def table_expl(d):
    E = XS["explain_eval"]; E2 = RS["explain2"]; E3 = R3.get("explain3", {})
    ts = [t for t in ("het_CtD", "het_DpS", "hkg_DpS") if t in E]
    g = lambda t, k: E[t][k]
    e3 = lambda t, k, f=pct: f(E3[t][k]) if t in E3 and E3[t].get(k) is not None else "—"
    XD = R4.get("explain_dwpc", {})
    dw = lambda t, k, f=pct: f(XD[t]["dwpc"][k]) if t in XD and XD[t]["dwpc"].get(k) is not None else "—"
    dn = lambda t, k, f=pct: f(XD[t]["dnfr"][k]) if t in XD and XD[t]["dnfr"].get(k) is not None else "—"
    pct1 = lambda x: f"{100 * x:.1f}%"
    rows = [["§Concision (all evaluated test pairs; medians)"],
            ["Test pairs evaluated (top-10 hits)"] + [f"{g(t, 'n_eval')} ({g(t, 'n_hits10')})" for t in ts],
            ["Paths per pair"] + [f"{g(t, 'n_paths_median'):.0f}" for t in ts],
            ["Paths covering 50% / 80% of the evidence"] + [f"{g(t, 'k50_median'):.0f} / {g(t, 'k80_median'):.0f}" for t in ts],
            ["Distinct facts (edges) on the paths"] + [e3(t, "n_edges_median", lambda x: f"{x:.0f}") for t in ts],
            ["Facts covering 50% / 80% of the edge attribution"] + [
                f"{E3[t]['k50_edges_median']:.0f} / {E3[t]['k80_edges_median']:.0f}" if t in E3 else "—" for t in ts],
            ["Metapaths per pair; share of the leading metapath"] + [f"{g(t, 'n_mps_median'):.0f}; {pct(g(t, 'top_mp_share_median'))}" for t in ts],
            ["Same leading metapath in DNFR and DWPC-LR"] + [pct(g(t, "same_top_mp")) for t in ts],
            ["§Attribution to facts"],
            ["Pairs whose three leading paths share a fact"] + [pct(E2[t]["top3_share_edge"]) if t in E2 else "—" for t in ts],
            ["Share of the leading fact: summed path shares"] + [e3(t, "top_edge_double_median") for t in ts],
            ["Share of the leading fact: Shapley value"] + [e3(t, "top_edge_shapley_median") for t in ts],
            ["Same leading fact by Shapley value and by summed share"] + [dn(t, "same_shapley_summed") for t in ts],
            ["DWPC-LR: share of the leading fact (proportional allocation)"] + [dw(t, "top_edge_share_median") for t in ts],
            ["DWPC-LR: facts covering 50% of its positive attribution"] + [dw(t, "k50_edges_median", lambda x: f"{x:.0f}") for t in ts],
            ["Same leading fact in DNFR (Shapley) and DWPC-LR"] + [
                f"{pct(XD[t]['dwpc']['same_lead_as_dnfr'])} ({XD[t]['dwpc']['n_both']})" if t in XD else "—" for t in ts],
            ["§Deletion within the pair (top-10 hits that stay in the top 10)"],
            ["3 leading paths / 3 random paths deleted"] + [f"{pct(g(t, 'del_top3_keep10'))} / {pct(g(t, 'del_rand3_keep10'))}" for t in ts],
            ["Random paths with the share of the top 3 deleted"] + [pct(E2[t]["stay_top10"]["matched3"]) if t in E2 else "—" for t in ts],
            ["§Fact-level counterfactual (one fact removed from the graph; competing tails re-scored)"],
            ["DNFR: fact with the largest Shapley share"] + [e3(t, "stay_top10_edge_top_sh") for t in ts],
            ["DNFR: random fact on the pair's paths"] + [pct(E2[t]["stay_top10"]["edge_rand"]) if t in E2 else "—" for t in ts],
            ["DWPC-LR: its leading fact / a random fact on its paths"] + [
                f"{pct(XD[t]['dwpc']['stay_top10_cf_top'])} / {pct(XD[t]['dwpc']['stay_top10_cf_random'])}" if t in XD else "—"
                for t in ts]]
    if "het_CtD" in E3 and E3["het_CtD"].get("lead_class"):
        lc = E3["het_CtD"]["lead_class"]
        rows.append(["§Drug repurposing: metapath classes of correct top-10 predictions"])
        rows.append(["Leading metapath: mechanistic (target gene) / associative / other"] + [
            f"{pct(lc['gene'])} / {pct(lc['associative'])} / {pct(lc['other'])}" if t == "het_CtD" else "—" for t in ts])
        if "het_CtD" in XD:
            sh = XD["het_CtD"]["dnfr"]["share_mean"]
            rows.append(["Share of the evidence: mechanistic / associative / other (mean)"] + [
                f"{pct1(sh['gene'])} / {pct1(sh['associative'])} / {pct1(sh['other'])}" if t == "het_CtD" else "—" for t in ts])
            rows.append(["Share of the evidence on metapaths through the treats relation (mean)"] + [
                pct1(XD["het_CtD"]["dnfr"]["share_target_mean"]) if t == "het_CtD" else "—" for t in ts])
    T(d, "**Table 6.** Quantitative evaluation of path explanations on the seed-0 split. Evidence is −log(1 − P) above the leak, "
         "which the paths share exactly. Summed path shares count a fact once for every path through it; the Shapley value splits "
         "each path's term equally among its facts (Section 3.6) and sums to the total. DWPC-LR rows use its own correct top-10 "
         "predictions and the attribution of Section 3.6. Deletion within the pair changes one score with all others fixed; the "
         "fact-level counterfactual removes one fact from the graph and re-scores every candidate of the same head, without "
         "retraining. Metapath classes are given for drug repurposing only, the one task with a mechanistic route.",
      [""] + [TASK_LABEL[t] for t in ts], rows, widths=[2.67, 1.2, 1.2, 1.2], size=8)


def table_examples(d):
    rows = []
    for e in XS["explanations"]:
        nm = e["paths"][0]["nodes"]
        rows.append([f"§{nm[0]} → {nm[-1]} (rank 1; {e['n_paths']:,} paths)"])
        for p in e["paths"][:3]:
            rows.append([" → ".join(p["nodes"]), p["metapath"].replace("^-1", "⁻¹"), ", ".join(f"{x:.2f}" for x in p["mu"]),
                         f3(p["p"], 2), f"{100 * p['share']:.1f}%"])
    T(d, "**Table 7.** Explanations for three correctly top-ranked test pairs (seed-0 models): the three paths with the largest share "
         "of the evidence, with edge memberships μ, the learned t-norm parameter p of the metapath and the share of −log(1 − P). "
         "Metapaths use Hetionet's relation abbreviations, with ⁻¹ for a relation traversed against its direction (e.g. PCiC⁻¹: "
         "compound to pharmacologic class). The score is a ranking score, not a calibrated probability.",
      ["Path", "Metapath", "μ along path", "p", "Share"], rows, widths=[2.6, 1.35, 0.95, 0.55, 0.62], size=8)


# ================================================================ supplementary material
def supp_task(d, num, t, title):
    order = [("§Proposed models", None), ("dnfr", 0), ("dnfr2", 0), ("§Same model, fixed t-norm", None), ("product", 0),
             ("hamacher", 0), ("min", 0), ("luk", 0), ("fixp_m4", 0), ("fixp_m2", 0), ("fixp_m1", 0), ("fixp_m05", 0),
             ("fixp_05", 0), ("fixp_val", 0), ("§Ablations", None), ("dnfr_shared", 0), ("dnfr_nodeg", 0), ("dnfr_fixdeg", 0),
             ("dnfr_max", 0), ("dnfr_binary", 0), ("dnfr_frozen", 0), ("product_frozen", 0), ("§Refinements", None),
             ("dnfr_overlap", 0), ("dnfr_gumbel", 0), ("dnfr_ic", 0), ("dnfr_anneal", 0), ("product_overlap", 0),
             ("dnfr2_no_overlap", 0), ("dnfr2_no_gumbel", 0), ("dnfr2_no_ic", 0), ("dnfr2_no_anneal", 0),
             ("§Path baselines", None), ("dwpc_lr", 0), ("dwpc_lr_full", 0), ("anyburl", 0), ("nbfnet", 0),
             ("§Without paths", None), ("prior", 0), ("popularity", 0), ("svd", 0), ("svd_rank3", 0), ("softimpute", 0),
             ("§Knowledge graph embeddings (untuned reference)", None), ("kge_TransE", 0), ("kge_RotatE", 0)]
    rows = []
    for m, _ in order:
        if m.startswith("§"):
            rows.append([m]); continue
        if m == "dnfr_binary" and t != "hkg_DpS":
            continue
        s = summ(t, m)
        if s is None and m in ONLY_R3 and r3sum(t, m):
            q = r3sum(t, m); p = r3p(t, m, "dnfr")
            rows.append([NAMES[m] + (f" ({q['n']} splits)" if q["n"] != nseeds(t, "dnfr") else ""),
                         f"{f3(q['mean'])} ± {f3(q['sd'])}", "—", "—",
                         "—" if p is None else f"{sg(p['delta'])} [{sg(p['bs95'][0])}, {sg(p['bs95'][1])}]",
                         "—" if p is None or p.get("p_holm") is None else ("<0.001" if p["p_holm"] < .001 else f"{p['p_holm']:.3f}"),
                         "—" if p is None else (VMARK.get(p["verdict"], "") or "·")])
            continue
        if s is None:
            continue
        p = None if m == "dnfr" else _neg(pr(t, "dnfr", m))
        nb = "" if s["n_seeds"] == nseeds(t, "dnfr") else f" ({s['n_seeds']} split{'s' if s['n_seeds'] > 1 else ''})"
        q = pr(t, "dnfr", m)
        rows.append([NAMES.get(m, m) + nb, mrr(t, m, mark_n=False), f3(s["H1"]["mean"]), f3(s["H10"]["mean"]),
                     "—" if p is None else dci(p), "—" if p is None else ph(q),
                     "—" if p is None else VMARK.get(q.get("verdict"), "") or "·"])
    out = []
    for i, r in enumerate(rows):
        if r[0].startswith("§") and (i + 1 == len(rows) or rows[i + 1][0].startswith("§")):
            continue
        out.append(r)
    T(d, f"**Table S{num}.** {title} Mean ± SD over random splits; filtered ranking against all candidates. ΔMRR is the method minus "
         "DNFR, averaged over shared splits, with a 95% paired bootstrap interval; p from the corrected resampled t-test across splits "
         "[@nadeau2003], Holm-adjusted within the family of the comparison (baselines, components, or other comparisons; the "
         "refinements and NBFNet within their own families); verdict under the rule of Section 3.6 (≈ equivalent, † different, · "
         "inconclusive).",
      ["Method", "MRR", "Hits@1", "Hits@10", "ΔMRR vs DNFR [95% CI]", "p (Holm)", "Verdict"], out,
      widths=[2.05, 0.9, 0.55, 0.6, 1.2, 0.55, 0.45], size=7.5)


def supp_sparsity(d, num):
    rows = []
    for t, lab in (("het_CtD", "Compound–treats–disease"), ("het_DpS", "Disease–presents–symptom")):
        rows.append([f"§{lab}"])
        for m in ("dnfr", "product", "dnfr_nodeg", "dwpc_lr", "softimpute"):
            rows.append([NAMES[m]] + [mrr(t, m, tag, mark_n=False) for tag in ("", "_sp0.50", "_sp0.25", "_spt0.25")])
    T(d, f"**Table S{num}.** MRR with all edges, and with 50% and 25% of the edges of every relation kept after random removal (three "
         "splits on the thinned graphs; same test pairs). Last column: 25% of the edges of every relation except the target relation "
         "kept, so that completion keeps its training edges (Hetionet disease–symptom only). The all-edges column repeats Tables S1 "
         "and S2.",
      ["Method", "All edges", "50% of edges kept", "25% of edges kept", "25% kept, target intact"], rows,
      widths=[2.0, 1.05, 1.05, 1.05, 1.12], size=8)


def supp_long(d, num):
    L = RS.get("long", {})
    rows = []
    for t in [t for t in ("het_DpS", "hkg_DpS") if t in L]:
        rows.append([f"§{TASK_LABEL[t]} ({len(L[t]['seeds'])} splits)"])
        for v in L[t]["inits"]:
            b = L[t]["by_init"][f"{v:g}"]
            rows.append([f"p₀ = {f3(v, 1) if v != -0.1 else '−0.1'}",
                         f"{f3(b['p_best']['median'], 2)} [{f3(b['p_best']['q25'], 2)}, {f3(b['p_best']['q75'], 2)}]",
                         f"{f3(b['p_final']['median'], 2)} [{f3(b['p_final']['q25'], 2)}, {f3(b['p_final']['q75'], 2)}]",
                         f"{ri(np_median(b['best_epoch']))}", f3(b["test_best"]), f3(b["test_final"]), f"{b['loss_final']:.4f}"])
    T(d, f"**Table S{num}.** Learned t-norm (bounded parameterisation) trained for 400 epochs without early stopping and without "
         "penalty. p at the best validation epoch and at epoch 400: median [interquartile range] over metapaths and splits (one "
         "metapath on the graded graph). Test MRR of the parameters with the best validation MRR and of the final parameters.",
      ["Initial value", "p at best validation epoch", "p at epoch 400", "Best epoch (median)", "Test MRR, best validation",
       "Test MRR, epoch 400", "Training loss, epoch 400"], rows, widths=[0.75, 1.15, 1.15, 0.7, 0.85, 0.82, 0.85], size=7.5)


def supp_profile(d, num, key="profile_summary", label="the graded graph"):
    PS = R3.get(key)
    if not PS:
        return 0
    rows = []
    for p, b in sorted(PS["by_p"].items(), key=lambda kv: float(kv[0])):
        rows.append([p.replace("-", "−"), f"{1e3 * b['excess']:.2f}", "—" if b["dLdp"] is None else f"{1e3 * b['dLdp']:+.2f}".replace("-", "−"),
                     f3(b["valid"]), f3(b["test"])])
    T(d, f"**Table S{num}.** Converged profile on {label} (means over {len(PS['seeds'])} splits): for each fixed shared p, "
         "training loss in excess of the split's minimum (×10⁻³ nat), ∂L/∂p at the converged fit (×10⁻³), validation and test MRR. "
         "Values p > 0 were fitted with warm starts from the fit at the product.",
      ["p", "Excess loss ×10⁻³", "∂L/∂p ×10⁻³", "Validation MRR", "Test MRR"], rows, widths=[0.8, 1.3, 1.3, 1.3, 1.3], size=8)
    rows = [[str(r["seed"]), f"{r['threshold']:.1e}", f"[{r['p_lo']:g}, {r['p_hi']:g}]".replace("-", "−"),
             f"{r['p_argmin']:g}".replace("-", "−"), f"{c:.3f}" + ("" if inter else " (end)"),
             f"{r['p_best_valid']:g}".replace("-", "−"), f3(r["test_at_valid_choice"]), f3(r["test_product"])]
            for r, c, inter in zip(PS["intervals"], PS["curvature"], PS["interior"])]
    T(d, f"**Table S{num + 1}.** Profile-likelihood style intervals for p on {label}, per split: values of p whose converged "
         "training loss lies within χ²(0.95, 1)/(2N) of the minimum, treating N times the mean loss as a negative log-likelihood "
         "[@raue2009]; the filtered softmax is not a likelihood of independent observations, so the intervals are descriptive. "
         "Curvature ∂²L/∂p² at the minimum from the three grid points around it; (end): the minimum lies at an end of the grid, "
         "and the value is a one-sided estimate from the three points nearest that end.",
      ["Split", "Threshold (nat)", "Interval for p", "p at minimum loss", "Curvature", "p chosen on validation",
       "Test MRR at that p", "Test MRR, product"], rows, widths=[0.45, 0.8, 0.95, 0.75, 0.75, 0.85, 0.85, 0.85], size=8)
    return 2


def supp_rules(d, num):
    """Verdicts under the current rule and under the rule of the earlier version (reviewer M6)."""
    rows = []
    def add(label, t, p):
        if p is None:
            return
        v2 = p.get("verdict_rule2", "—")
        rows.append([label, TASK_LABEL[t], f"{sg(p['delta'])} ({p['wins']}/{p['J']})", p.get("verdict") or "—", v2 or "—"])
    for t in TASKS:
        for a, b, lab in (("dnfr", "dnfr_max", "Noisy-OR instead of the strongest path"),
                          ("dnfr", "dnfr_nodeg", "Degree terms instead of none"),
                          ("dnfr", "dnfr_fixdeg", "Learned instead of fixed degree terms"),
                          ("dnfr", "product", "Learned p instead of the product"),
                          ("dnfr", "hamacher", "Learned p instead of the Hamacher product"),
                          ("dnfr", "dwpc_lr", "DNFR minus DWPC-LR")):
            q = pr(t, a, b)
            if q:
                add(lab, t, dict(delta=q["delta_MRR"], wins=q["seed_wins"], J=q["seeds"], verdict=q.get("verdict"),
                                 verdict_rule2=q.get("verdict_rule2")))
    for t in R3T:
        for a, b, lab in (("fixp_val", "product", "Fixed p (validation) minus the product"),
                          ("dnfr2", "dnfr", "DNFR 2.0 minus DNFR"), ("dnfr_anneal", "dnfr", "R4 added to DNFR"),
                          ("nbfnet", "dnfr", "NBFNet minus DNFR")):
            add(lab, t, r3p(t, a, b))
    NT = R4.get("notgt", {}).get("paired", {})
    for m in ("dnfr", "dwpc_lr"):
        if f"{m}_notgt__{m}" in NT:
            add(f"{NAMES[m].split(' (')[0]} without treats metapaths minus with", "het_CtD", NT[f"{m}_notgt__{m}"])
    T(d, f"**Table S{num}.** Verdicts for the comparisons on which the Highlights and the Conclusion rest, under the rule of "
         "Section 3.6 and under the rule of the earlier version of this analysis (different when the unadjusted 95% corrected-t "
         "interval excludes zero; equivalent when the 90% corrected-t interval lies inside ±0.01). Δ is the mean difference in "
         "test MRR, with the number of splits on which it was positive.",
      ["Comparison", "Task", "Δ (splits > 0)", "Verdict, current rule", "Verdict, earlier rule"], rows,
      widths=[2.4, 1.05, 1.0, 0.95, 0.95], size=7.5)


def supp_hamacher(d, num):
    H = R4.get("hamacher_ctd") or []
    rows = [[str(h["seed"]), f3(h["hamacher"]), f3(h["product"]), f3(h["dnfr"]),
             "—" if h.get("loss_hamacher") is None else f"{h['loss_hamacher']:.3f}",
             "—" if h.get("loss_product") is None else f"{h['loss_product']:.3f}",
             "—" if h.get("best_epoch") is None else str(h["best_epoch"])] for h in H]
    T(d, f"**Table S{num}.** Fixed Hamacher product on Hetionet compound–treats–disease, per split: test MRR of the Hamacher product, "
         "the product and DNFR, the training loss at the last logged epoch (read from the run logs, available for the runs made "
         "after the second version of this analysis), and the epoch of the best validation MRR of the Hamacher run.",
      ["Split", "Hamacher MRR", "Product MRR", "DNFR MRR", "Hamacher final loss", "Product final loss", "Best epoch"], rows,
      widths=[0.5, 0.9, 0.9, 0.9, 1.0, 1.0, 0.8], size=8)
    E = R4.get("time_per_epoch_ctd") or []
    rows = [[str(e["seed"]), f"{e['dnfr_s']:.0f}", str(e["epochs"]), f"{e['s_per_epoch']:.1f}",
             "—" if e.get("dwpc_s") is None else f"{e['dwpc_s']:.1f}"] for e in E]
    T(d, f"**Table S{num + 1}.** Time of one DNFR run and of DWPC-LR on Hetionet compound–treats–disease, per split. Splits 0–2 "
         "come from an earlier batch whose thread setting was not recorded in the result files (the code default is two threads); "
         "the other splits ran with one thread while a second job shared the two-core machine.",
      ["Split", "DNFR time (s)", "Epochs", "Seconds per epoch", "DWPC-LR time (s)"], rows, widths=[0.6, 1.1, 0.8, 1.2, 1.2], size=8)


def supp_prior(d, num):
    P = R3.get("prior_summary", {})
    lab = {"het_CtD": "Hetionet CtD, random split", "het_CtD_tail": "Hetionet CtD, disease split", "het_DpS": "Hetionet DpS",
           "hkg_DpS": "EHR graph (graded)"}
    rows = [[lab[t], f3(P[t]["popularity_MRR"]), f3(P[t]["prior100_MRR"]), f3(P[t]["prior1000_MRR"]), f3(P[t]["analytic_MRR"]),
             pct(P[t]["tie_any_prior100"]), pct(P[t]["tie_share_prior100"]), pct(P[t]["share_zero_true"])] for t in lab if t in P]
    T(d, f"**Table S{num}.** Edge prior and popularity (reviewer comment M6), mean over splits. Prior with 100 and 1000 "
         "degree-preserving permutations, and the analytical approximation of Zietz et al. [@zietz2024], which for a fixed head ranks "
         "tails in the same order as popularity. Ties: share of test queries whose true tail ties with another candidate, and mean "
         "share of tied candidates, under the 100-permutation prior. Zero prior: share of test queries whose true pair has prior 0.",
      ["Task", "Popularity", "Prior, 100", "Prior, 1000", "Analytic prior", "Queries with a tie", "Tied candidates", "Zero prior"],
      rows, widths=[1.35, 0.7, 0.65, 0.65, 0.7, 0.7, 0.7, 0.62], size=8)


def supp_valid(d, num):
    V = R3.get("ctd_valid_spread") or []
    if not V:
        return
    rows = [[v["job"].split("_")[-1].replace("s", ""), str(v["n_ckpt"]), f3(v["best"]), f3(v["sd_all"]),
             "—" if v["sd_after_10"] is None else f3(v["sd_after_10"]), f3(v["range_top5"])] for v in V]
    T(d, f"**Table S{num}.** Validation MRR of DNFR across checkpoints (every five epochs) on Hetionet compound–treats–disease, which "
         "has about 75 validation pairs (reviewer minor comment 6).",
      ["Split", "Checkpoints", "Best validation MRR", "SD, all checkpoints", "SD, from the third checkpoint", "Range of the five best"],
      rows, widths=[0.6, 0.9, 1.1, 1.1, 1.3, 1.27], size=8)


def supp_nhanes(d, num):
    N = XS["nhanes"]
    lab = [("product", "Product t-norm, strongest path (fixed)"), ("hamacher", "Hamacher product, strongest path (fixed)"),
           ("min", "Minimum, strongest path (fixed)"), ("luk", "Łukasiewicz, strongest path (fixed)"),
           ("ss_shared", "Schweizer–Sklar, one learned p"), ("ss_permp", "Schweizer–Sklar, learned p per metapath"),
           ("calib_product", "Calibrated membership + product, noisy-OR"),
           ("dnfr_max", "DNFR: calibrated membership + p per metapath, strongest path"),
           ("dnfr", "DNFR: calibrated membership + p per metapath, noisy-OR")]
    rows = [[l, f3(N[k]["rho_hold"]), f3(N[k]["rho_dev"]), f3(N[k]["mae_hold"])] for k, l in lab if k in N]
    T(d, f"**Table S{num}.** NHANES population relation (24 concepts, {XS['nhanes_info']['n_pairs']} ordered pairs). Each pair is "
         "predicted from its indirect paths in the 2013–2018 relation and compared with the relation measured in 2007–2012. Fixed rows "
         "reproduce the earlier analysis [@truong2026].",
      ["Composition", "Spearman ρ, held-out", "Spearman ρ, development", "MAE, held-out"], rows, widths=[3.0, 1.1, 1.1, 1.07], size=8)
