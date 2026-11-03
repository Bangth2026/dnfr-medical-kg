"""Results tables (Section 4 and Appendix A). Text that interprets the results is in ms_text.py."""
from ms_common import *
from docx_helpers import para, heading, table, figure

NAMES = {"dnfr": "DNFR (learned p per metapath)", "product": "Product t-norm (fixed)", "hamacher": "Hamacher product (fixed)",
         "min": "Minimum t-norm (fixed)", "luk": "Łukasiewicz t-norm (fixed)", "dnfr_shared": "DNFR, one shared p",
         "dnfr_nodeg": "DNFR without degree terms", "dnfr_fixdeg": "DNFR, degree terms fixed (b = c = 0.4)",
         "product_fixdeg": "Product, degree terms fixed (b = c = 0.4)",
         "dnfr_max": "DNFR, strongest path only", "dnfr_binary": "DNFR, edge weights ignored",
         "dwpc_lr": "DWPC + logistic regression", "dwpc_lr_full": "DWPC-LR + degree + prior", "anyburl": "AnyBURL (path rules)",
         "popularity": "Popularity (training positives of the tail)", "prior": "Edge prior (degree-preserving permutations)",
         "svd": "Truncated SVD (rank chosen on validation)", "svd_rank3": "Rank-3 SVD", "softimpute": "Soft-thresholded SVD (Soft-Impute)",
         "kge_TransE": "TransE (untuned)", "kge_RotatE": "RotatE (untuned)"}


def T(d, *a, **k):
    a = list(a); a[0] = C(a[0])
    if k.get("note"):
        k["note"] = C(k["note"])
    return table(d, *a, **k)


def table2(d):
    meths = ["dnfr", "product", "hamacher", "dwpc_lr", "dwpc_lr_full", "anyburl", "softimpute", "svd", "prior", "popularity",
             "kge_RotatE", "kge_TransE"]
    rows = [[NAMES[m]] + [mrr(t, m) for t in TASKS] for m in meths]
    T(d, "**Table 2.** Test MRR (mean ± SD over random splits) of DNFR and the baselines. Numbers in parentheses give the number of "
         "splits when it differs from that of DNFR (" + ", ".join(f"{nseeds(t, 'dnfr')}" for t in TASKS) + " splits on the four "
         "tasks). Full results with Hits@k, confidence intervals and all variants are in Appendix Tables A1–A4.",
      ["Method"] + [TASK_LABEL[t] for t in TASKS], rows, widths=[2.07, 1.05, 1.05, 1.05, 1.05], size=8)


def table3(d):
    rows = [["§Full model, learned memberships: model with the component minus the model without it"]]
    def r(label, a, b, skip=()):
        rows.append([label] + ["—" if t in skip else cell(pr(t, a, b)) for t in TASKS])
    r("Noisy-OR instead of the strongest path", "dnfr", "dnfr_max")
    r("Degree terms instead of none", "dnfr", "dnfr_nodeg")
    r("Learned instead of fixed degree terms", "dnfr", "dnfr_fixdeg")
    r("Graded edge weights instead of binary edges", "dnfr", "dnfr_binary", skip=("het_CtD", "het_DpS", "pkg_ind"))
    rows.append(["§t-norm (decision rule of Section 3.5 applies)"])
    r("Learned p per metapath instead of the product", "dnfr", "product")
    r("Learned p per metapath instead of the Hamacher product", "dnfr", "hamacher")
    r("Learned p per metapath instead of one shared p", "dnfr", "dnfr_shared")
    r("Hamacher product instead of the product", "hamacher", "product")
    rows.append(["§Memberships frozen at their initial values"])
    r("Learned p per metapath instead of the product", "dnfr_frozen", "product_frozen")
    rows.append(["§Product t-norm"])
    r("Learned memberships instead of frozen", "product", "product_frozen")
    T(d, "**Table 3.** What each learned component adds. Each cell is the difference in test MRR between the model with the component "
         "and the model without it, averaged over splits, with its 95% paired bootstrap interval and the number of splits on which the "
         "difference was positive. ≈ equivalent within ±0.01 (90% corrected-t interval inside the margin); † different (95% "
         "corrected-t interval excludes zero); no mark, inconclusive. — not run or not applicable (Hetionet and PrimeKG edges carry no "
         "graded weight).",
      ["Component"] + [TASK_LABEL[t] for t in TASKS], rows, widths=[1.87, 1.1, 1.1, 1.1, 1.1], size=7.5)


def table4(d):
    V = [("dnfr", "p₀ = −0.1, λ = 10⁻³ (default)"), ("dnfr_l2_0", "p₀ = −0.1, no penalty"), ("dnfr_l2_1e-2", "p₀ = −0.1, λ = 10⁻²"),
         ("dnfr_l2_1e-1", "p₀ = −0.1, λ = 10⁻¹"), ("dnfr_pm2", "p₀ = −2, no penalty"), ("dnfr_pm1", "p₀ = −1 (Hamacher), no penalty"),
         ("dnfr_pp05", "p₀ = 0.5, no penalty")]
    rows = []
    for v, lab in V:
        row = [lab]
        for t in TASKS:
            s = summ(t, v); P = S[t].get("p_by_variant", {}).get(v)
            if s is None:
                row.append("—"); continue
            txt = f"{f3(s['MRR']['mean'])}"
            if s["n_seeds"] != nseeds(t, "dnfr"):
                txt += f" ({s['n_seeds']})"
            if P:
                txt += f"; p {f3(P['median'], 2)} [{f3(P['q25'], 2)}, {f3(P['q75'], 2)}]"
            row.append(txt)
        rows.append(row)
    rows.append(["§Fixed members of the family"])
    for v, lab in (("product", "Product (p → 0)"), ("hamacher", "Hamacher product (p = −1)"), ("luk", "Łukasiewicz (p = 1)")):
        rows.append([lab] + [mrr(t, v, sd=False) for t in TASKS])
    rows.append(["§Median epoch of the best validation MRR, default model (stopping epoch)"])
    rs = RS["stopping"]
    rows.append(["DNFR, default"] + [f"{ri(rs[t]['dnfr']['best_epoch']['median'])} ({ri(rs[t]['dnfr']['stop_epoch']['median'])})"
                                     if t in rs and "dnfr" in rs[t] else "—" for t in TASKS])
    T(d, "**Table 4.** Learned t-norm with the default early stopping: sensitivity to the initial value p₀ and to the penalty λ that "
         "pulls θ~m~ towards its initial value. Each cell gives the mean test MRR and the median [interquartile range] of the learned p "
         "over metapaths and splits; numbers in parentheses after the MRR give the number of splits when it differs from that of DNFR.",
      ["Initialisation and penalty"] + [TASK_LABEL[t] for t in TASKS], rows, widths=[1.67, 1.15, 1.15, 1.15, 1.15], size=7.5)


def table5(d):
    """Training without early stopping."""
    L = RS.get("long", {})
    tasks = [t for t in ("het_DpS", "hkg_DpS") if t in L]
    rows = []
    for t in tasks:
        rows.append([f"§{TASK_LABEL[t]} ({len(L[t]['seeds'])} splits)"])
        for v in L[t]["inits"]:
            b = L[t]["by_init"][f"{v:g}"]
            rows.append([f"p₀ = {f3(v, 1) if v != -0.1 else '−0.1'}",
                         f"{f3(b['p_best']['median'], 2)} [{f3(b['p_best']['q25'], 2)}, {f3(b['p_best']['q75'], 2)}]",
                         f"{f3(b['p_final']['median'], 2)} [{f3(b['p_final']['q25'], 2)}, {f3(b['p_final']['q75'], 2)}]",
                         f"{ri(np_median(b['best_epoch']))}", f3(b["test_best"]), f3(b["test_final"]),
                         f"{b['loss_final']:.4f}"])
    T(d, "**Table 5.** Learned t-norm trained for 400 epochs without early stopping and without penalty. p at the best validation "
         "epoch and at epoch 400: median [interquartile range] over metapaths and splits (one metapath on the graded graph). Test MRR of "
         "the parameters with the best validation MRR and of the final parameters; training loss at epoch 400.",
      ["Initial value", "p at best validation epoch", "p at epoch 400", "Best epoch (median)", "Test MRR, best validation",
       "Test MRR, epoch 400", "Training loss, epoch 400"], rows,
      widths=[0.75, 1.15, 1.15, 0.7, 0.85, 0.82, 0.85], size=7.5)


def np_median(x):
    import numpy as np
    return float(np.median(x))


def table_tail(d):
    tg = "_tail"
    meths = ["dnfr", "product", "hamacher", "dnfr_nodeg", "dwpc_lr", "dwpc_lr_full", "softimpute", "svd", "prior", "popularity"]
    rows = []
    for m in meths:
        s = summ("het_CtD", m, tg)
        if s is None:
            continue
        rows.append([NAMES[m], mrr("het_CtD", m, sd=False, mark_n=False), mrr("het_CtD", m, tg), f3(s["H10"]["mean"]),
                     "—" if m == "dnfr" else cell(pr("het_CtD", "dnfr", m, tg) and _neg(pr("het_CtD", "dnfr", m, tg)), verdict=False)])
    T(d, f"**Table 6.** Hetionet compound–treats–disease with a disease-centred split ({nseeds('het_CtD', 'dnfr', tg)} splits): every "
         "indication of a test disease is held out together. Random-split MRR from Table A1 for comparison. ΔMRR is the method minus "
         "DNFR, with its 95% bootstrap interval and the number of splits on which it was positive.",
      ["Method", "MRR, random split", "MRR, disease split", "Hits@10, disease split", "ΔMRR vs DNFR [95% CI], splits"], rows,
      widths=[2.0, 0.85, 1.05, 0.85, 1.52], size=8)


def _neg(p):
    if p is None:
        return None
    q = dict(p); q["delta_MRR"] = -p["delta_MRR"]; q["ci_lo"], q["ci_hi"] = -p["ci_hi"], -p["ci_lo"]
    q["seed_wins"] = sum(1 for x in p["per_split"] if x < 0); return q


def table_expl(d):
    E = XS["explain_eval"]; E2 = RS["explain2"]; ts = [t for t in ("het_CtD", "het_DpS", "hkg_DpS") if t in E]
    g = lambda t, k: E[t][k]
    rows = [["§Concision (all evaluated test pairs; medians)"],
            ["Test pairs evaluated (top-10 hits)"] + [f"{g(t, 'n_eval')} ({g(t, 'n_hits10')})" for t in ts],
            ["Paths per pair"] + [f"{g(t, 'n_paths_median'):.0f}" for t in ts],
            ["Paths covering 50% / 80% of the evidence"] + [f"{g(t, 'k50_median'):.0f} / {g(t, 'k80_median'):.0f}" for t in ts],
            ["Share of the three leading paths"] + [pct(g(t, "top3_share_median")) for t in ts],
            ["Metapaths per pair; share of the leading metapath"] + [f"{g(t, 'n_mps_median'):.0f}; {pct(g(t, 'top_mp_share_median'))}" for t in ts],
            ["DWPC-LR: share of the leading metapath"] + [pct(g(t, "lr_top_mp_share_median")) for t in ts],
            ["Same leading metapath in DNFR and DWPC-LR"] + [pct(g(t, "same_top_mp")) for t in ts],
            ["Pairs whose three leading paths share an edge"] + [pct(E2[t]["top3_share_edge"]) if t in E2 else "—" for t in ts],
            ["§Deletion within the pair (top-10 hits that stay in the top 10)"],
            ["Leading path / 3 / 10 leading paths deleted"] + [f"{pct(g(t, 'del_top1_keep10'))} / {pct(g(t, 'del_top3_keep10'))} / {pct(g(t, 'del_top10_keep10'))}" for t in ts],
            ["1 / 3 / 10 random paths deleted"] + [f"{pct(g(t, 'del_rand1_keep10'))} / {pct(g(t, 'del_rand3_keep10'))} / {pct(g(t, 'del_rand10_keep10'))}" for t in ts],
            ["Random paths with the share of the top 1 / 3 / 10 deleted"] + [
                f"{pct(E2[t]['stay_top10']['matched1'])} / {pct(E2[t]['stay_top10']['matched3'])} / {pct(E2[t]['stay_top10']['matched10'])}" if t in E2 else "—" for t in ts],
            ["Random paths needed to match the top-3 share (median)"] + [f"{E2[t]['matched_paths_removed']['3']:.0f}" if t in E2 else "—" for t in ts],
            ["Leading metapath deleted, DNFR"] + [pct(g(t, "del_top_mp_keep10")) for t in ts],
            ["Leading metapath deleted, DWPC-LR (its own hits)"] + [pct(g(t, "lr_del_top_mp_keep10")) for t in ts],
            ["§Edge-level counterfactual (one fact removed from the graph; competing tails re-scored)"],
            ["Edge with the largest share of the pair's evidence"] + [pct(E2[t]["stay_top10"]["edge_top"]) if t in E2 else "—" for t in ts],
            ["Random edge on the pair's paths"] + [pct(E2[t]["stay_top10"]["edge_rand"]) if t in E2 else "—" for t in ts],
            ["Share of the evidence on the leading edge (median)"] + [pct(E2[t]["top_edge_share"]) if t in E2 else "—" for t in ts]]
    T(d, "**Table 7.** Quantitative evaluation of path explanations on the seed-0 split. Evidence is −log(1 − P) above the leak, "
         "which the paths share exactly. Deletion within the pair removes paths from the score of one pair with all other scores fixed; "
         "the edge-level counterfactual removes one fact from the graph and recomputes every pair of the same head, without retraining.",
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


def appendix_task(d, num, t, title):
    order = [("§Proposed", None), ("dnfr", 0), ("§Same model, fixed t-norm", None), ("product", 0), ("hamacher", 0), ("min", 0),
             ("luk", 0), ("§Ablations", None), ("dnfr_shared", 0), ("dnfr_nodeg", 0), ("dnfr_fixdeg", 0), ("dnfr_max", 0),
             ("dnfr_binary", 0), ("§Path baselines", None), ("dwpc_lr", 0), ("dwpc_lr_full", 0), ("anyburl", 0),
             ("§Without paths", None), ("prior", 0), ("popularity", 0), ("svd", 0), ("svd_rank3", 0), ("softimpute", 0),
             ("§Knowledge graph embeddings (untuned reference)", None), ("kge_TransE", 0), ("kge_RotatE", 0)]
    rows = []
    for m, _ in order:
        if m.startswith("§"):
            rows.append([m]); continue
        if m == "dnfr_binary" and t != "hkg_DpS":
            continue
        s = summ(t, m)
        if s is None:
            continue
        p = None if m == "dnfr" else _neg(pr(t, "dnfr", m))
        nb = "" if s["n_seeds"] == nseeds(t, "dnfr") else f" ({s['n_seeds']} split{'s' if s['n_seeds'] > 1 else ''})"
        rows.append([NAMES[m] + nb, mrr(t, m, mark_n=False), f3(s["H1"]["mean"]), f3(s["H10"]["mean"]),
                     "—" if p is None else dci(p), "—" if p is None else ph(pr(t, "dnfr", m)),
                     "—" if p is None else VMARK.get(p.get("verdict", ""), "") or ("·" if p.get("verdict") else "—")])
    # drop empty group labels
    out = []
    for i, r in enumerate(rows):
        if r[0].startswith("§") and (i + 1 == len(rows) or rows[i + 1][0].startswith("§")):
            continue
        out.append(r)
    T(d, f"**Table A{num}.** {title} Mean ± SD over random splits; filtered ranking against all candidates. ΔMRR is the method minus "
         "DNFR, averaged over shared splits, with a 95% paired bootstrap interval; p from the corrected resampled t-test across splits "
         "[@nadeau2003], Holm-adjusted within the table; verdict under the rule of Section 3.5 (≈ equivalent, † different, · "
         "inconclusive), which uses unadjusted corrected intervals and so ignores multiplicity.",
      ["Method", "MRR", "Hits@1", "Hits@10", "ΔMRR vs DNFR [95% CI]", "p (Holm)", "Verdict"], out,
      widths=[2.05, 0.9, 0.55, 0.6, 1.2, 0.55, 0.45], size=7.5)


def table_sparsity(d):
    rows = []
    for t, lab in (("het_CtD", "Compound–treats–disease"), ("het_DpS", "Disease–presents–symptom")):
        rows.append([f"§{lab}"])
        for m in ("dnfr", "product", "dnfr_nodeg", "dwpc_lr", "softimpute"):
            row = [NAMES[m]]
            for tag in ("", "_sp0.50", "_sp0.25"):
                row.append(mrr(t, m, tag, mark_n=False))
            rows.append(row)
    T(d, "**Table A5.** MRR with all edges, and with 50% and 25% of the edges of every relation kept after random removal (three "
         "splits on the thinned graphs; same test pairs). The all-edges column repeats Tables A1 and A2.",
      ["Method", "All edges", "50% of edges kept", "25% of edges kept"], rows, widths=[2.4, 1.3, 1.3, 1.27], size=8)


def table_nhanes(d):
    N = XS["nhanes"]
    lab = [("product", "Product t-norm, strongest path (fixed)"), ("hamacher", "Hamacher product, strongest path (fixed)"),
           ("min", "Minimum, strongest path (fixed)"), ("luk", "Łukasiewicz, strongest path (fixed)"),
           ("ss_shared", "Schweizer–Sklar, one learned p"), ("ss_permp", "Schweizer–Sklar, learned p per metapath"),
           ("calib_product", "Calibrated membership + product, noisy-OR"),
           ("dnfr_max", "DNFR: calibrated membership + p per metapath, strongest path"),
           ("dnfr", "DNFR: calibrated membership + p per metapath, noisy-OR")]
    rows = [[l, f3(N[k]["rho_hold"]), f3(N[k]["rho_dev"]), f3(N[k]["mae_hold"])] for k, l in lab if k in N]
    T(d, f"**Table B1.** NHANES population relation (24 concepts, {XS['nhanes_info']['n_pairs']} ordered pairs). Each pair is predicted "
         "from its indirect paths in the 2013–2018 relation and compared with the relation measured in 2007–2012. Fixed rows reproduce "
         "the earlier analysis [@truong2026].",
      ["Composition", "Spearman ρ, held-out", "Spearman ρ, development", "MAE, held-out"], rows, widths=[3.0, 1.1, 1.1, 1.07], size=8)
