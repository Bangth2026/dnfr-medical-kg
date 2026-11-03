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
         "nbfnet": "NBFNet (compact, CPU)",
         "popularity": "Popularity (training positives of the tail)", "prior": "Edge prior (degree-preserving permutations)",
         "svd": "Truncated SVD (rank chosen on validation)", "svd_rank3": "Rank-3 SVD", "softimpute": "Soft-thresholded SVD (Soft-Impute)",
         "kge_TransE": "TransE (untuned)", "kge_RotatE": "RotatE (untuned)",
         "dnfr_overlap": "DNFR + R1 overlap-aware aggregation", "dnfr_gumbel": "DNFR + R2 discrete operator choice",
         "dnfr_ic": "DNFR + R3 information-content discount", "dnfr_anneal": "DNFR + R4 membership annealing",
         "product_overlap": "Product + R1", "dnfr2_no_overlap": "DNFR 2.0 without R1", "dnfr2_no_gumbel": "DNFR 2.0 without R2",
         "dnfr2_no_ic": "DNFR 2.0 without R3", "dnfr2_no_anneal": "DNFR 2.0 without R4",
         "fixp_m4": "Fixed p = −4", "fixp_m2": "Fixed p = −2", "fixp_m1": "Fixed p = −1", "fixp_m05": "Fixed p = −0.5",
         "fixp_05": "Fixed p = 0.5", "dnfr_frozen": "DNFR, memberships frozen", "product_frozen": "Product, memberships frozen"}
R3T = ["het_CtD", "het_DpS", "hkg_DpS"]           # tasks on which the third-revision experiments were run
ONLY_R3 = {"fixp_val", "nbfnet"}


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
              ("§Learned path operators", ["nbfnet"]),
              ("§Without paths", ["softimpute", "svd", "prior", "popularity"]),
              ("§Knowledge graph embeddings (untuned reference points)", ["kge_RotatE", "kge_TransE"])]
    rows = []
    for g, meths in groups:
        rows.append([g])
        for m in meths:
            rows.append([NAMES[m]] + [mrr_any(t, m) for t in TASKS])
    T(d, "**Table 2.** Test MRR (mean ± SD over random splits) of DNFR, DNFR 2.0 and the baselines. Numbers in parentheses give "
         "the number of splits when it differs from that of DNFR (" + ", ".join(f"{nseeds(t, 'dnfr')}" for t in TASKS) + " splits on "
         "the four tasks). MRR is not comparable across columns, because the number of candidates differs (Table 1). Fixed p, "
         "selected on validation: one Schweizer–Sklar p chosen per split from a grid by validation MRR, trained with the protocol of "
         "DNFR. The third-revision experiments (DNFR 2.0, fixed p, NBFNet) were not run on PrimeKG. Hits@k, intervals and every "
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
         "with Holm adjustment over the comparisons of panels A–C of each task: ≈ equivalent within ±0.01; † different; no mark, "
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
         "Table 3, with Holm adjustment over the comparisons of this table within each task. Not run on PrimeKG.",
      ["Comparison"] + [TASK_LABEL[t] for t in R3T], rows, widths=[2.07, 1.4, 1.4, 1.4], size=7.5)


# ---------------------------------------------------------------- Table 5: identifiability of the operator
def table5(d):
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
    T(d, "**Table 5.** Does training determine the t-norm? Restarts of the bounded parameterisation, an unbounded parameterisation, "
         "restarts of the discrete operator choice, and the value of a fixed shared p chosen on validation data. Numbers in "
         "parentheses give the number of splits when it differs from that of DNFR.",
      ["Setting"] + [TASK_LABEL[t] for t in TASKS], rows, widths=[1.67, 1.15, 1.15, 1.15, 1.15], size=7.5)


def np_median(x):
    return float(np.median(x))


# ---------------------------------------------------------------- Table 6: held-out diseases
def table_tail(d):
    tg = "_tail"
    meths = ["dnfr", "product", "hamacher", "dnfr_nodeg", "dwpc_lr", "dwpc_lr_full", "softimpute", "svd", "prior", "popularity"]
    rows = []
    for m in meths:
        s = summ("het_CtD", m, tg)
        if s is None:
            continue
        rows.append([NAMES[m], mrr("het_CtD", m, sd=False, mark_n=False), mrr("het_CtD", m, tg), f3(s["H10"]["mean"]),
                     "—" if m == "dnfr" else cell(_neg(pr("het_CtD", "dnfr", m, tg)), verdict=False)])
    T(d, f"**Table 6.** Hetionet compound–treats–disease with a disease-centred split ({nseeds('het_CtD', 'dnfr', tg)} splits): every "
         "indication of a test disease is held out together. Random-split MRR from Table 2 for comparison. ΔMRR is the method minus "
         "DNFR, with its 95% bootstrap interval and the number of splits on which it was positive.",
      ["Method", "MRR, random split", "MRR, disease split", "Hits@10, disease split", "ΔMRR vs DNFR [95% CI], splits"], rows,
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
            ["§Deletion within the pair (top-10 hits that stay in the top 10)"],
            ["3 leading paths / 3 random paths deleted"] + [f"{pct(g(t, 'del_top3_keep10'))} / {pct(g(t, 'del_rand3_keep10'))}" for t in ts],
            ["Random paths with the share of the top 3 deleted"] + [pct(E2[t]["stay_top10"]["matched3"]) if t in E2 else "—" for t in ts],
            ["§Fact-level counterfactual (one fact removed from the graph; competing tails re-scored)"],
            ["Fact with the largest Shapley share"] + [e3(t, "stay_top10_edge_top_sh") for t in ts],
            ["Fact with the largest summed path share"] + [pct(E2[t]["stay_top10"]["edge_top"]) if t in E2 else "—" for t in ts],
            ["Random fact on the pair's paths"] + [pct(E2[t]["stay_top10"]["edge_rand"]) if t in E2 else "—" for t in ts]]
    if "het_CtD" in E3 and E3["het_CtD"].get("lead_class"):
        lc = E3["het_CtD"]["lead_class"]
        rows.append(["§Leading metapath of correct top-10 predictions (drug repurposing)"])
        rows.append(["Mechanistic (through a target gene) / associative / other"] + [
            f"{pct(lc['gene'])} / {pct(lc['associative'])} / {pct(lc['other'])}" if t == "het_CtD" else "—" for t in ts])
    T(d, "**Table 7.** Quantitative evaluation of path explanations on the seed-0 split. Evidence is −log(1 − P) above the leak, "
         "which the paths share exactly. Summed path shares count a fact once for every path through it; the Shapley value splits "
         "each path's term equally among its facts (Section 3.6) and sums to the total. Deletion within the pair changes one score "
         "with all others fixed; the fact-level counterfactual removes one fact from the graph and re-scores every candidate of the "
         "same head, without retraining.",
      [""] + [TASK_LABEL[t] for t in ts], rows, widths=[2.67, 1.2, 1.2, 1.2], size=8)


def table_examples(d):
    rows = []
    for e in XS["explanations"]:
        nm = e["paths"][0]["nodes"]
        rows.append([f"§{nm[0]} → {nm[-1]} (rank 1; {e['n_paths']:,} paths)"])
        for p in e["paths"][:3]:
            rows.append([" → ".join(p["nodes"]), p["metapath"].replace("^-1", "⁻¹"), ", ".join(f"{x:.2f}" for x in p["mu"]),
                         f3(p["p"], 2), f"{100 * p['share']:.1f}%"])
    T(d, "**Table 8.** Explanations for three correctly top-ranked test pairs (seed-0 models): the three paths with the largest share "
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
            rows.append([NAMES[m]] + [mrr(t, m, tag, mark_n=False) for tag in ("", "_sp0.50", "_sp0.25")])
    T(d, f"**Table S{num}.** MRR with all edges, and with 50% and 25% of the edges of every relation kept after random removal (three "
         "splits on the thinned graphs; same test pairs). The all-edges column repeats Tables S1 and S2.",
      ["Method", "All edges", "50% of edges kept", "25% of edges kept"], rows, widths=[2.4, 1.3, 1.3, 1.27], size=8)


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


def supp_profile(d, num):
    PS = R3.get("profile_summary")
    if not PS:
        return
    rows = []
    for p, b in sorted(PS["by_p"].items(), key=lambda kv: float(kv[0])):
        rows.append([p.replace("-", "−"), f"{1e3 * b['excess']:.2f}", "—" if b["dLdp"] is None else f"{1e3 * b['dLdp']:+.2f}".replace("-", "−"),
                     f3(b["valid"]), f3(b["test"])])
    T(d, f"**Table S{num}.** Converged profile on the graded graph (means over {len(PS['seeds'])} splits): for each fixed shared p, "
         "training loss in excess of the split's minimum (×10⁻³ nat), ∂L/∂p at the converged fit (×10⁻³), validation and test MRR.",
      ["p", "Excess loss ×10⁻³", "∂L/∂p ×10⁻³", "Validation MRR", "Test MRR"], rows, widths=[0.8, 1.3, 1.3, 1.3, 1.3], size=8)
    rows = [[str(r["seed"]), f"{r['threshold']:.1e}", f"[{r['p_lo']:g}, {r['p_hi']:g}]".replace("-", "−"), f"{r['p_argmin']:g}".replace("-", "−"),
             f"{r['p_best_valid']:g}".replace("-", "−"), f3(r["test_at_valid_choice"]), f3(r["test_product"])] for r in PS["intervals"]]
    T(d, f"**Table S{num + 1}.** Profile-likelihood style intervals for p on the graded graph, per split: values of p whose converged "
         "training loss lies within χ²(0.95, 1)/(2N) of the minimum, treating N times the mean loss as a negative log-likelihood "
         "[@raue2009]; the filtered softmax is not a likelihood of independent observations, so the intervals are descriptive.",
      ["Split", "Threshold (nat)", "Interval for p", "p at minimum loss", "p chosen on validation", "Test MRR at that p",
       "Test MRR, product"], rows, widths=[0.5, 0.9, 1.0, 0.9, 1.0, 0.95, 1.0], size=8)


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
