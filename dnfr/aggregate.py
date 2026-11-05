"""Aggregate all result files into one statistics file (single source of truth for the paper)."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import DATA, OUT
import glob, json, os
import numpy as np
from scipy.stats import wilcoxon, t as student_t

R = OUT
EQ_MARGIN = 0.01
TASKS = ["het_CtD", "het_DpS", "pkg_ind", "hkg_DpS"]
KG_OF = {"het_CtD": "het", "het_DpS": "het", "pkg_ind": "pkg", "hkg_DpS": "hkg"}


def load_task(task, tag=""):
    f = f"{R}/{task}{tag}.json"
    if not os.path.exists(f):
        return None
    d = json.load(open(f))
    runs = {r["seed"]: r for r in d["runs"]}
    # attach KGE results for the same seeds
    for kf in glob.glob(f"{R}/kge_{KG_OF[task]}{tag}_seed*.json"):
        k = json.load(open(kf))
        s = k["seed"]
        if s not in runs:
            continue
        for m, per in k["results"].items():
            if task in per:
                runs[s]["methods"]["kge_" + m] = {kk: v for kk, v in per[task].items() if kk != "ranks"}
                runs[s].setdefault("ranks", {})["kge_" + m] = per[task]["ranks"]
    for af in glob.glob(f"{R}/anyburl_{task}{tag}_seed*.json"):
        a = json.load(open(af))
        if a["seed"] in runs:
            runs[a["seed"]]["methods"]["anyburl"] = a["metrics"]
            runs[a["seed"]].setdefault("ranks", {})["anyburl"] = a["ranks"]
    d["runs"] = [runs[s] for s in sorted(runs)]
    return d


def summarise(d):
    out = {}
    methods = sorted({m for r in d["runs"] for m in r["methods"]})
    for m in methods:
        vals = {k: [r["methods"][m][k] for r in d["runs"] if m in r["methods"]] for k in ("MRR", "H1", "H3", "H10")}
        n = len(vals["MRR"])
        out[m] = {k: dict(mean=float(np.mean(v)), sd=float(np.std(v, ddof=1)) if n > 1 else 0.0) for k, v in vals.items()}
        out[m]["n_seeds"] = n
        out[m]["seeds"] = [r["seed"] for r in d["runs"] if m in r["methods"]]
    return out


def paired(d, a, b, n_boot=2000, seed=0):
    """Difference in MRR between methods a and b over the splits where both were run.
    delta_MRR: mean over splits of the per-split MRR difference.
    ci95: stratified paired bootstrap over test queries within each split.
    p: corrected resampled t-test across splits (Nadeau and Bengio, 2003); None with one split.
    p_wilcoxon: Wilcoxon signed-rank on pooled per-query reciprocal ranks (reported for reference only)."""
    per, diffs, ntr, nte = [], [], [], []
    for r in d["runs"]:
        rk = r.get("ranks", {})
        if a in rk and b in rk:
            dq = 1 / np.array(rk[a]) - 1 / np.array(rk[b])
            diffs.append(dq); per.append(dq.mean()); ntr.append(r["n_train"]); nte.append(r["n_test"])
    if not per:
        return None
    J = len(per); per = np.array(per); delta = float(per.mean())
    rng = np.random.default_rng(seed)
    boots = np.zeros(n_boot)
    for dq in diffs:
        idx = rng.integers(0, len(dq), (n_boot, len(dq)))
        boots += dq[idx].mean(1)
    boots /= J
    out = dict(delta_MRR=delta, ci_lo=float(np.quantile(boots, .025)), ci_hi=float(np.quantile(boots, .975)),
               ci90_lo=float(np.quantile(boots, .05)), ci90_hi=float(np.quantile(boots, .95)),
               seeds=J, seed_wins=int((per > 0).sum()), per_split=per.tolist(), n_queries=int(sum(len(x) for x in diffs)))
    if J > 1:
        var = per.var(ddof=1)
        corr = 1 / J + float(np.mean(nte)) / float(np.mean(ntr))
        if var > 0:
            t = delta / np.sqrt(corr * var)
            out["t_nb"] = float(t); out["p"] = float(2 * student_t.sf(abs(t), J - 1))
            se = float(np.sqrt(corr * var))
            for lvl, q in (("90", 0.95), ("95", 0.975)):
                h = float(student_t.ppf(q, J - 1)) * se
                out[f"nb_ci{lvl}_lo"], out[f"nb_ci{lvl}_hi"] = delta - h, delta + h
            out["verdict"] = None   # set by set_verdicts() once the Holm adjustment of the family is known
        else:   # identical per-split differences (e.g. identical rankings): no test
            out["t_nb"] = None; out["p"] = 1.0 if delta == 0 else None
    else:
        out["p"] = None
    pooled = np.concatenate(diffs)
    out["p_wilcoxon"] = float(wilcoxon(pooled).pvalue) if np.any(pooled != 0) else 1.0
    return out


FAM_BASE = {"dwpc_lr", "dwpc_lr_full", "anyburl", "softimpute", "svd", "prior", "popularity"}
FAM_COMP = {"dnfr_max", "dnfr_nodeg", "dnfr_fixdeg", "dnfr_binary", "product", "hamacher", "dnfr_shared",
            "hamacher__vs__product", "dnfr_frozen__vs__product_frozen", "product__vs__product_frozen"}


def set_verdicts(family):
    """Third-revision decision rule (margin 0.01 MRR), applied to every comparison of a family (dict name -> paired()):
    equivalent  both 90% intervals, corrected t and stratified bootstrap, lie inside (-0.01, 0.01);
    different   Holm-adjusted corrected-t p < 0.05 and the 95% bootstrap interval excludes zero on the same side;
    inconclusive otherwise (also when no corrected test is possible)."""
    hp = holm({k: v.get("p") for k, v in family.items() if v})
    for k, v in family.items():
        if not v:
            continue
        v["p_holm"] = hp.get(k)
        if v.get("nb_ci90_lo") is None:
            v["verdict"] = "inconclusive" if v.get("p") is not None or v["seeds"] > 1 else None
            continue
        inside = lambda lo, hi: lo > -EQ_MARGIN and hi < EQ_MARGIN
        # rule of the second revision, kept for the comparison of verdicts under both rules (reviewer M6)
        v["verdict_rule2"] = ("equivalent" if inside(v["nb_ci90_lo"], v["nb_ci90_hi"]) else
                              "different" if (v["nb_ci95_lo"] > 0 or v["nb_ci95_hi"] < 0) else "inconclusive")
        if inside(v["nb_ci90_lo"], v["nb_ci90_hi"]) and inside(v["ci90_lo"], v["ci90_hi"]):
            v["verdict"] = "equivalent"
        elif v["p_holm"] is not None and v["p_holm"] < 0.05 and (
                (v["ci_lo"] > 0 and v["delta_MRR"] > 0) or (v["ci_hi"] < 0 and v["delta_MRR"] < 0)):
            v["verdict"] = "different"
        else:
            v["verdict"] = "inconclusive"


def holm(pvals):
    """Holm-adjusted p-values for a dict name -> p (None entries are skipped)."""
    items = sorted([(p, k) for k, p in pvals.items() if p is not None])
    m, out, run = len(items), {}, 0.0
    for i, (p, k) in enumerate(items):
        run = max(run, min(1.0, (m - i) * p)); out[k] = run
    return out


def p_stats(arr, p0):
    arr = np.asarray(arr)
    return dict(min=float(arr.min()), q25=float(np.quantile(arr, .25)), median=float(np.median(arr)),
                q75=float(np.quantile(arr, .75)), max=float(arr.max()), p_init=p0,
                mean_abs_move=float(np.mean(np.abs(arr - p0))), share_abs_move_lt_01=float((np.abs(arr - p0) < 0.1).mean()),
                share_below_minus05=float((arr < -0.5).mean()), share_abs_lt_02=float((np.abs(arr) < 0.2).mean()))


P_INIT = {"dnfr": -0.1, "dnfr_shared": -0.1, "dnfr_nodeg": -0.1, "dnfr_max": -0.1, "dnfr_binary": -0.1, "dnfr_l2_0": -0.1,
          "dnfr_pm2": -2.0, "dnfr_pm1": -1.0, "dnfr_pp05": 0.5, "dnfr_l2_1e-2": -0.1, "dnfr_l2_1e-1": -0.1,
          "dnfr_fixdeg": -0.1, "dnfr_frozen": -0.1}
EXTRA_PAIRS = [("dnfr_pm1", "hamacher"), ("dnfr_pm1", "dnfr"), ("dnfr_pm2", "dnfr"), ("dnfr_pp05", "dnfr"),
               ("dnfr_l2_0", "dnfr"), ("dnfr", "dnfr_fixdeg"), ("product", "product_fixdeg"), ("dnfr_fixdeg", "dnfr_nodeg"),
               ("dnfr_frozen", "product_frozen"), ("product", "product_frozen"), ("hamacher", "product"), ("dnfr", "popularity"),
               ("dnfr_pm1", "product"), ("dnfr_pm2", "product"), ("dnfr_pp05", "product"), ("dnfr_l2_0", "product"),
               ("dnfr_pm2", "hamacher"), ("dnfr_pp05", "hamacher"), ("dnfr_l2_0", "hamacher"),
               ("product_fixdeg", "product_fixdeg02"), ("product_fixdeg", "product_fixdeg06"),
               ("dwpc_lr_full", "dwpc_lr"), ("prior", "popularity"), ("dnfr_shared", "product")]


if __name__ == "__main__":
    stats = {}
    for task in TASKS:
        for tag in ["", "_sp0.50", "_sp0.25", "_spt0.25", "_tail"]:
            d = load_task(task, tag)
            if d is None:
                continue
            key = task + tag
            s = dict(info={k: d[k] for k in ("n_pos", "n_sources", "n_tails", "n_paths", "n_mps", "n_pairs") if k in d},
                     summary=summarise(d), paired={})
            s["info"]["n_test_mean"] = float(np.mean([r["n_test"] for r in d["runs"]]))
            s["info"]["n_train_mean"] = float(np.mean([r["n_train"] for r in d["runs"]]))
            methods = list(s["summary"])
            for m in methods:
                if m != "dnfr":
                    pr = paired(d, "dnfr", m)
                    if pr:
                        s["paired"][m] = pr
            s["paired_extra"] = {}
            for a_, b_ in EXTRA_PAIRS:
                if a_ in s["summary"] and b_ in s["summary"]:
                    s["paired_extra"][f"{a_}__vs__{b_}"] = paired(d, a_, b_)
            # Holm families by question: RQ1 baselines (Table 2), RQ2 components (Table 3); every other comparison
            # (restarts, penalties, other fixed t-norms, untuned embeddings, refinements in this file) is descriptive
            # and adjusted within its own family. The refinement family is computed in rev3_stats.py.
            allp = dict(s["paired"]); allp.update({k: v for k, v in s["paired_extra"].items() if v})
            fam = {"base": [k for k in allp if k in FAM_BASE],
                   "comp": [k for k in allp if k in FAM_COMP]}
            fam["descr"] = [k for k in allp if k not in fam["base"] and k not in fam["comp"]]
            for name, keys in fam.items():
                set_verdicts({k: allp[k] for k in keys})
                for k in keys:
                    allp[k]["family"] = name
            # learned t-norm parameters of every variant with a learned p
            s["p_by_variant"] = {}
            for v in P_INIT:
                arrs = [np.array(r["p_learned"][v]) for r in d["runs"] if v in r.get("p_learned", {})]
                if not arrs and v == "dnfr":
                    arrs = [np.array(r["dnfr_params"]["p"]) for r in d["runs"] if "dnfr_params" in r]
                if arrs:
                    s["p_by_variant"][v] = p_stats(np.concatenate(arrs), P_INIT[v])
            # agreement of the per-metapath parameters learned from different initial values (same split)
            from scipy.stats import spearmanr
            RV = ["dnfr_pm2", "dnfr_pm1", "dnfr_l2_0", "dnfr_pp05"]
            rhos, signs, rng_small = [], [], []
            for r in d["runs"]:
                pl = r.get("p_learned", {}); have = [v for v in RV if v in pl]
                if len(have) < 2 or len(pl[have[0]]) < 3:
                    continue
                M = np.array([pl[v] for v in have])
                rhos += [float(spearmanr(M[i], M[j]).statistic) for i in range(len(have)) for j in range(i + 1, len(have))]
                signs.append(float(np.mean(np.all(M < 0, 0) | np.all(M > 0, 0))))
                rng_small.append(float(np.mean(M.max(0) - M.min(0) < 0.25)))
            if rhos:
                s["p_agreement"] = dict(spearman_median=float(np.median(rhos)), spearman_min=float(np.min(rhos)),
                                        spearman_max=float(np.max(rhos)), sign_agree_mean=float(np.mean(signs)),
                                        range_lt025_mean=float(np.mean(rng_small)), n_splits=len(signs))
            # timing (seconds per training run)
            s["time_s"] = {m: float(np.mean([r["methods"][m]["time_s"] for r in d["runs"] if m in r["methods"] and "time_s" in r["methods"][m]]))
                           for m in s["summary"] if any("time_s" in r["methods"].get(m, {}) for r in d["runs"])}
            # learned t-norm parameters
            ps = [np.array(r["dnfr_params"]["p"]) for r in d["runs"] if "dnfr_params" in r]
            if ps:
                allp = np.concatenate(ps)
                s["p_learned"] = dict(min=float(allp.min()), q25=float(np.quantile(allp, .25)), median=float(np.median(allp)),
                                      q75=float(np.quantile(allp, .75)), max=float(allp.max()),
                                      share_below_minus05=float((allp < -0.5).mean()), share_above_05=float((allp > 0.5).mean()),
                                      share_abs_lt_02=float((np.abs(allp) < 0.2).mean()))
                # only relations that occur on some path (others keep their initial value 0.3 and get no gradient)
                bs, cs = [], []
                for r in d["runs"]:
                    if "dnfr_params" not in r: continue
                    b = np.array(r["dnfr_params"]["b"]); c = np.array(r["dnfr_params"]["c"])
                    used = (np.abs(b - 0.3) > 1e-6) | (np.abs(c - 0.3) > 1e-6)
                    bs.append(b[used]); cs.append(c[used])
                bs, cs = np.concatenate(bs), np.concatenate(cs)
                s["degree_coef"] = dict(b_median=float(np.median(bs)), c_median=float(np.median(cs)), n=int(len(bs)),
                                        share_b_pos=float((bs > 0).mean()), share_c_pos=float((cs > 0).mean()),
                                        share_any_pos=float(((bs > 0) | (cs > 0)).mean()))
            stats[key] = s
    if os.path.exists(f"{R}/nhanes_dnfr.json"):
        stats["nhanes"] = json.load(open(f"{R}/nhanes_dnfr.json"))
    json.dump(stats, open(f"{R}/paper_stats.json", "w"), indent=1, allow_nan=False)
    for k, s in stats.items():
        if k == "nhanes":
            continue
        print("==", k, s["info"])
        for m, v in sorted(s["summary"].items(), key=lambda x: -x[1]["MRR"]["mean"]):
            pr = s["paired"].get(m, {})
            print(f"  {m:14s} MRR {v['MRR']['mean']:.3f}±{v['MRR']['sd']:.3f} H1 {v['H1']['mean']:.3f} H10 {v['H10']['mean']:.3f} n={v['n_seeds']}"
                  + (f"  dnfr-m {pr['delta_MRR']:+.3f} [{pr['ci_lo']:+.3f},{pr['ci_hi']:+.3f}] pNB={pr['p']} holm={pr.get('p_holm')} wins {pr['seed_wins']}/{pr['seeds']}" if pr else ""))
        for kk, vv in s.get("paired_extra", {}).items():
            print(f"  {kk:34s} {vv['delta_MRR']:+.3f} [{vv['ci_lo']:+.3f},{vv['ci_hi']:+.3f}] pNB={vv['p']}")
        for kk, vv in s.get("p_by_variant", {}).items():
            print(f"  p[{kk}] med {vv['median']:+.2f} IQR [{vv['q25']:+.2f},{vv['q75']:+.2f}] move {vv['mean_abs_move']:.2f}")
        if "p_learned" in s:
            print("  p:", {k: round(v, 3) for k, v in s["p_learned"].items()}, s["degree_coef"])
