"""Statistics for the third revision (DNFR 2.0 components). Merges the second-revision result files (results/)
with the third-revision jobs (results2/jobs/*, completed jobs only) by split, then computes paired comparisons.

Decision rule (margin 0.01 MRR, fixed before the third-revision runs were analysed):
  equivalent   both 90% intervals (Nadeau-Bengio corrected t, stratified query bootstrap) lie inside (-0.01, 0.01);
  different    Holm-adjusted corrected-t p < 0.05 within the task's comparison family AND the 95% bootstrap
               interval excludes zero on the same side;
  inconclusive otherwise.
Output: results2/rev3_stats.json"""
import os, sys, glob, json, collections
import numpy as np
from scipy.stats import t as student_t
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from aggregate import holm

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EQ = 0.01
TASKS = ["het_CtD", "het_DpS", "hkg_DpS"]
GRID = {"fixp_m4": -4.0, "fixp_m2": -2.0, "fixp_m1": -1.0, "fixp_m05": -0.5, "product": 0.0, "fixp_05": 0.5}
FAMILY = [  # (a, b): the comparisons that enter the Holm family of a task
    ("dnfr2", "dnfr"), ("dnfr2", "product"), ("dnfr2", "dwpc_lr"),
    ("dnfr_overlap", "dnfr"), ("dnfr_gumbel", "dnfr"), ("dnfr_ic", "dnfr"), ("dnfr_anneal", "dnfr"),
    ("product_overlap", "product"),
    ("dnfr2", "dnfr2_no_overlap"), ("dnfr2", "dnfr2_no_gumbel"), ("dnfr2", "dnfr2_no_ic"), ("dnfr2", "dnfr2_no_anneal"),
    ("fixp_val", "dnfr"), ("fixp_val", "product"),
]
# separate Holm family: the learned-operator GNN against the main path methods and completion (reviewer M7)
FAMILY_NBF = [("nbfnet", "dnfr"), ("nbfnet", "dwpc_lr"), ("nbfnet", "softimpute"), ("nbfnet", "product"),
              ("nbfnet_tuned", "dnfr"), ("nbfnet_tuned", "dwpc_lr")]
# descriptive checks outside the confirmatory families (no Holm adjustment, no verdict used in the text)
DESCR = [("dnfr_ic_sd", "dnfr"), ("dnfr_ic_sd", "dnfr_ic")]
RESTARTS = {"dnfr2": ["dnfr2", "dnfr2_r1", "dnfr2_r2"], "dnfr_gumbel": ["dnfr_gumbel", "dnfr_gumbel_r1", "dnfr_gumbel_r2"]}
FREE = ["dnfr_free_pm2", "dnfr_free", "dnfr_free_pp05"]
FREE_INIT = {"dnfr_free_pm2": -2.0, "dnfr_free": -0.1, "dnfr_free_pp05": 0.5}


def merge(task):
    runs = {}
    files = [f"{ROOT}/results/{task}.json"] + sorted(glob.glob(f"{ROOT}/results2/jobs/*/{task}.json"))
    meta = None
    for f in files:
        if "/jobs/" in f and not os.path.exists(os.path.join(os.path.dirname(f), "DONE")):
            continue
        if not os.path.exists(f):
            continue
        d = json.load(open(f)); meta = meta or {k: v for k, v in d.items() if k != "runs"}
        for r in d["runs"]:
            R = runs.setdefault(r["seed"], dict(seed=r["seed"], n_train=r["n_train"], n_test=r["n_test"],
                                                methods={}, ranks={}, extra=collections.defaultdict(dict)))
            for m, v in r["methods"].items():
                R["methods"][m] = v
                if m in r.get("ranks", {}):
                    R["ranks"][m] = r["ranks"][m]
            for k in ("ops_chosen", "op_probs", "beta", "tau_best", "p_learned"):
                for m, v in r.get(k, {}).items():
                    R["extra"][k][m] = v
    for f in sorted(glob.glob(f"{ROOT}/results2/nbf/nbfnet_{task}_seed*.json") + glob.glob(f"{ROOT}/results/nbfnet_{task}_seed*.json")):
        a = json.load(open(f))
        if a["seed"] in runs:
            runs[a["seed"]]["methods"]["nbfnet"] = a["metrics"]; runs[a["seed"]]["ranks"]["nbfnet"] = a["ranks"]
    # NBFNet with a 2 x 2 grid of learning rate and dimension, configuration chosen per split by validation MRR (M7)
    for R in runs.values():
        cfg = {"d32_lr5e-3": R["methods"].get("nbfnet")}
        for f in glob.glob(f"{ROOT}/results2/nbf/nbfnet_{task}_d*_lr*_seed{R['seed']}.json"):
            a = json.load(open(f)); c = os.path.basename(f)[len(f"nbfnet_{task}_"):].rsplit("_seed", 1)[0]
            cfg[c] = dict(a["metrics"], ranks=a["ranks"])
        if R["methods"].get("nbfnet"):
            cfg["d32_lr5e-3"] = dict(R["methods"]["nbfnet"], ranks=R["ranks"]["nbfnet"])
        cfg = {k: v for k, v in cfg.items() if v}
        if len(cfg) >= 4:
            best = max(cfg, key=lambda k: cfg[k]["valid_MRR"])
            R["methods"]["nbfnet_tuned"] = {k: v for k, v in cfg[best].items() if k != "ranks"} | {"chosen": best}
            R["ranks"]["nbfnet_tuned"] = cfg[best]["ranks"]
    # validation-selected fixed t-norm (shared p chosen per split on validation MRR among the grid values run)
    for R in runs.values():
        cand = [m for m in GRID if m in R["methods"] and m in R["ranks"]]
        if len([m for m in cand if m != "product"]) >= 2:
            best = max(cand, key=lambda m: R["methods"][m]["valid_MRR"])
            R["methods"]["fixp_val"] = dict(R["methods"][best], chosen=best)
            R["ranks"]["fixp_val"] = R["ranks"][best]
    return meta, [runs[s] for s in sorted(runs)]


def paired(runs, a, b, n_boot=4000, seed=0):
    per, diffs, ntr, nte = [], [], [], []
    for r in runs:
        if a in r["ranks"] and b in r["ranks"]:
            dq = 1 / np.array(r["ranks"][a]) - 1 / np.array(r["ranks"][b])
            diffs.append(dq); per.append(dq.mean()); ntr.append(r["n_train"]); nte.append(r["n_test"])
    if not per:
        return None
    J = len(per); per = np.array(per); delta = float(per.mean())
    rng = np.random.default_rng(seed)
    boots = np.zeros(n_boot)
    for dq in diffs:
        boots += dq[rng.integers(0, len(dq), (n_boot, len(dq)))].mean(1)
    boots /= J
    out = dict(delta=delta, J=J, wins=int((per > 0).sum()), per_split=per.tolist(),
               bs90=[float(np.quantile(boots, .05)), float(np.quantile(boots, .95))],
               bs95=[float(np.quantile(boots, .025)), float(np.quantile(boots, .975))])
    if J > 1 and per.var(ddof=1) > 0:
        se = float(np.sqrt((1 / J + np.mean(nte) / np.mean(ntr)) * per.var(ddof=1)))
        out["p"] = float(2 * student_t.sf(abs(delta / se), J - 1))
        out["nb90"] = [delta - float(student_t.ppf(.95, J - 1)) * se, delta + float(student_t.ppf(.95, J - 1)) * se]
        out["nb95"] = [delta - float(student_t.ppf(.975, J - 1)) * se, delta + float(student_t.ppf(.975, J - 1)) * se]
    else:
        out["p"] = None
    return out


def verdict(v):
    if v is None or v.get("p") is None:
        return "untested"
    if v["nb90"][0] > -EQ and v["nb90"][1] < EQ and v["bs90"][0] > -EQ and v["bs90"][1] < EQ:
        return "equivalent"
    ph = v.get("p_holm")
    if ph is not None and ph < 0.05 and (v["bs95"][0] > 0 or v["bs95"][1] < 0) and np.sign(v["bs95"][0]) == np.sign(v["delta"]):
        return "different"
    return "inconclusive"


def verdict_rule2(v):
    """Rule of the second revision: equivalent if the 90% corrected-t interval lies inside the margin; different if the
    unadjusted 95% corrected-t interval excludes zero (reviewer M6 asks for verdicts under both rules)."""
    if v is None or v.get("p") is None:
        return "untested"
    if v["nb90"][0] > -EQ and v["nb90"][1] < EQ:
        return "equivalent"
    return "different" if (v["nb95"][0] > 0 or v["nb95"][1] < 0) else "inconclusive"


def summary(runs):
    out = {}
    for m in sorted({m for r in runs for m in r["methods"]}):
        x = [r["methods"][m]["MRR"] for r in runs if m in r["methods"]]
        out[m] = dict(mean=float(np.mean(x)), sd=float(np.std(x, ddof=1)) if len(x) > 1 else 0.0, n=len(x),
                      seeds=[r["seed"] for r in runs if m in r["methods"]])
        t = [r["methods"][m].get("time_s") for r in runs if m in r["methods"] and r["methods"][m].get("time_s")]
        if t:
            out[m]["time_s"] = float(np.mean(t))
    return out


def restart_agreement(runs, names):
    """Share of metapaths whose arg-max operator is the same in all restarts of one split; operator counts."""
    agree, counts, n = [], collections.Counter(), 0
    for r in runs:
        oc = r["extra"].get("ops_chosen", {})
        if not all(v in oc for v in names):
            continue
        M = np.array([oc[v] for v in names])
        agree.append(float(np.mean(np.all(M == M[0], 0)))); n += 1
        counts.update(M.flatten().tolist())
    if not n:
        return None
    return dict(n_splits=n, all_same_mean=float(np.mean(agree)), all_same_per_split=agree, op_counts=dict(counts))


def free_p(runs):
    rows = []
    for r in runs:
        pl = r["extra"].get("p_learned", {})
        if not all(v in pl for v in FREE):
            continue
        M = np.array([pl[v] for v in FREE])  # restarts x metapaths
        rows.append(dict(seed=r["seed"], median_by_init={v: float(np.median(pl[v])) for v in FREE},
                         min=float(M.min()), max=float(M.max()), spread_median=float(np.median(M.max(0) - M.min(0))),
                         share_hit_upper=float(np.mean(M >= 0.95 - 1e-6)), share_below_m5=float(np.mean(M < -5)),
                         mrr={v: r["methods"][v]["MRR"] for v in FREE}))
    return rows or None


def prior_summary(pc):
    out = {}
    for t, rows in pc.items():
        if not isinstance(rows, list):
            continue
        g = lambda k, f: float(np.mean([r[k][f] for r in rows if k in r and f in r[k]]))
        out[t] = dict(n=len(rows), **{f"{k}_MRR": g(k, "MRR") for k in ("popularity", "prior100", "prior1000", "analytic")},
                      tie_any_prior100=g("prior100", "tie_any"), tie_share_prior100=g("prior100", "tie_share"),
                      tie_any_popularity=g("popularity", "tie_any"), tie_share_popularity=g("popularity", "tie_share"),
                      share_zero_true=g("prior100", "share_zero_true"),
                      rho_analytic_pop_min=float(np.min([r.get("rho_analytic_pop", np.nan) for r in rows])))
    return out


def curvature_at(ps, L):
    """Second derivative of the profile at its minimum from the three grid points around it (non-uniform central
    difference); at an end of the grid, from the three points nearest that end (a one-sided estimate)."""
    i = int(np.argmin(L)); j = min(max(i, 1), len(ps) - 2)
    h1, h2 = ps[j] - ps[j - 1], ps[j + 1] - ps[j]
    return float(2 * (h1 * L[j + 1] - (h1 + h2) * L[j] + h2 * L[j - 1]) / (h1 * h2 * (h1 + h2))), bool(0 < i < len(ps) - 1)


def merged_profiles(task):
    """Profile fits of the first run and of the extension (grid with p > 0), keeping the lower loss at each p."""
    out = []
    for f in sorted(glob.glob(f"{ROOT}/results/profile_lbfgs_{task}_seed*.json")) + \
            [g for g in sorted(glob.glob(f"{ROOT}/results2/prof_ext/profile_lbfgs_{task}_seed*.json"))
             if not os.path.exists(f"{ROOT}/results/{os.path.basename(g)}")]:
        d = json.load(open(f)); rows = {r["p"]: r for r in d["rows"]}
        g = f"{ROOT}/results2/prof_ext/{os.path.basename(f)}"
        if os.path.exists(g) and g != f:
            for r in json.load(open(g))["rows"]:
                if r["p"] not in rows or r["loss"] < rows[r["p"]]["loss"]:
                    rows[r["p"]] = dict(r, from_extension=True)
        d["rows"] = [rows[p] for p in sorted(rows)]
        ps = np.array([r["p"] for r in d["rows"]]); L = np.array([r["loss"] for r in d["rows"]])
        d["p_min"] = float(ps[np.argmin(L)]); d["loss_range"] = float(L.max() - L.min())
        d["d2L_dp2"], d["interior"] = curvature_at(ps, L)
        d["loss_range_nonpos"] = float(L[ps <= 0].max() - L[ps <= 0].min())
        out.append(d)
    return out


def profile_summary(pl, task="hkg_DpS"):
    rows = merged_profiles(task)
    if not rows:
        return None
    out = dict(seeds=[d["seed"] for d in rows], p_min=[d["p_min"] for d in rows], curvature=[d["d2L_dp2"] for d in rows],
               interior=[d["interior"] for d in rows], loss_range=[d["loss_range"] for d in rows],
               loss_range_nonpos=[d["loss_range_nonpos"] for d in rows], grid=[[r["p"] for r in d["rows"]] for d in rows],
               rows=[d["rows"] for d in rows])
    by_p = {}
    for d in rows:
        L = np.array([r["loss"] for r in d["rows"]])
        for r in d["rows"]:
            b = by_p.setdefault(r["p"], dict(excess=[], dLdp=[], test=[], valid=[]))
            b["excess"].append(r["loss"] - L.min()); b["test"].append(r["test_MRR"]); b["valid"].append(r["valid_MRR"])
            if r["dLdp"] is not None:
                b["dLdp"].append(r["dLdp"])
    out["by_p"] = {f"{p:g}": {k: float(np.mean(v)) if v else None for k, v in b.items()} for p, b in sorted(by_p.items())}
    # profile-likelihood style interval per split: p values whose excess loss is below chi2(0.95, 1) / (2 N),
    # treating N times the mean softmax loss as a negative log-likelihood (Raue et al. 2009); the filtered softmax
    # is not a proper likelihood of independent observations, so these intervals are descriptive only
    out["intervals"] = []
    for d in rows:
        L = np.array([r["loss"] for r in d["rows"]]); ps = np.array([r["p"] for r in d["rows"]])
        thr = 3.841 / (2 * d["n_train"]); inside = ps[L - L.min() < thr]
        te = np.array([r["test_MRR"] for r in d["rows"]]); va = np.array([r["valid_MRR"] for r in d["rows"]])
        out["intervals"].append(dict(seed=d["seed"], threshold=float(thr), p_lo=float(inside.min()), p_hi=float(inside.max()),
                                     p_argmin=float(ps[np.argmin(L)]), p_best_valid=float(ps[np.argmax(va)]),
                                     p_best_test=float(ps[np.argmax(te)]), test_range=float(te.max() - te.min()),
                                     test_at_valid_choice=float(te[np.argmax(va)]), test_product=float(te[ps == 0][0])))
    out["max_abs_dLdp"] = float(max(abs(x) for b in by_p.values() for x in b["dLdp"]))
    out["max_excess_nonpos"] = float(max(x for p, b in by_p.items() if p <= 0 for x in b["excess"]))
    return out


def valid_spread():
    """Spread of validation MRR across the checkpoints of the default DNFR runs on CtD (from the job logs)."""
    import re
    out = []
    for f in sorted(glob.glob(f"{ROOT}/results2/jobs/ctd_v1_s*/log.txt")):
        if not os.path.exists(os.path.join(os.path.dirname(f), "DONE")):
            continue
        cur, vals = None, []
        for line in open(f):
            if line.startswith("[het_CtD"):
                cur = line.strip().split()[-1]
            m = re.search(r"valid MRR ([0-9.]+)", line)
            if m and cur == "dnfr":
                vals.append(float(m.group(1)))
        if len(vals) >= 3:
            v = np.array(vals); top = np.sort(v)[-5:]
            out.append(dict(job=os.path.basename(os.path.dirname(f)), n_ckpt=len(v), best=float(v.max()),
                            sd_all=float(v.std(ddof=1)), sd_after_10=float(v[2:].std(ddof=1)) if len(v) > 3 else None,
                            range_top5=float(top.max() - top.min())))
    return out


def main():
    out = {}
    for task in TASKS:
        meta, runs = merge(task)
        if not runs:
            continue
        S = dict(n_splits=len(runs), summary=summary(runs), paired={})
        for fam_name, fam in (("refine", FAMILY), ("nbfnet", FAMILY_NBF)):
            F = {}
            for a, b in fam:
                v = paired(runs, a, b)
                if v:
                    F[f"{a}__{b}"] = v
            hp = holm({k: v["p"] for k, v in F.items()})
            for k, v in F.items():
                v["p_holm"] = hp.get(k); v["verdict"] = verdict(v); v["family"] = fam_name
                v["verdict_rule2"] = verdict_rule2(v)
            S["paired"].update(F)
        for a_, b_ in DESCR:
            v = paired(runs, a_, b_)
            if v:
                v["family"] = "descr"; S["paired"][f"{a_}__{b_}"] = v
        S["nbf_tuned_choice"] = collections.Counter(r["methods"]["nbfnet_tuned"]["chosen"] for r in runs if "nbfnet_tuned" in r["methods"])
        S["fixp_val_choice"] = collections.Counter(r["methods"]["fixp_val"]["chosen"] for r in runs if "fixp_val" in r["methods"])
        S["restarts"] = {k: restart_agreement(runs, v) for k, v in RESTARTS.items()}
        S["ops_single"] = collections.Counter(o for r in runs for o in r["extra"].get("ops_chosen", {}).get("dnfr2", []))
        S["beta"] = {m: [r["extra"]["beta"][m] for r in runs if m in r["extra"].get("beta", {})]
                     for m in ("dnfr2", "dnfr_overlap", "product_overlap")}
        S["tau_best"] = {m: [r["extra"]["tau_best"][m] for r in runs if m in r["extra"].get("tau_best", {})]
                         for m in ("dnfr2", "dnfr_anneal")}
        S["free_p"] = free_p(runs)
        out[task] = S
    for f, k in (("results/prior_check.json", "prior_check"), ("results/saturation.json", "saturation")):
        if os.path.exists(f"{ROOT}/{f}"):
            out[k] = json.load(open(f"{ROOT}/{f}"))
    out["prior_summary"] = prior_summary(out.get("prior_check", {}))
    out["profile_lbfgs"] = {os.path.basename(f): json.load(open(f)) for f in
                            sorted(set(glob.glob(f"{ROOT}/results2/profile_lbfgs_*.json")) | set(glob.glob(f"{ROOT}/results/profile_lbfgs_*.json")))}
    out["profile_summary"] = profile_summary(out["profile_lbfgs"])
    out["profile_summary_dps"] = profile_summary(out["profile_lbfgs"], "het_DpS")
    out["explain3"] = {os.path.basename(f).split("_explain")[0]: json.load(open(f))["summary"]
                       for f in glob.glob(f"{ROOT}/results/*_explain_eval3.json")}
    out["ctd_valid_spread"] = valid_spread()
    for d in ("results2", "results"):
        json.dump(out, open(f"{ROOT}/{d}/rev3_stats.json", "w"), indent=1, default=lambda o: dict(o))
    return out


if __name__ == "__main__":
    o = main()
    for task in TASKS:
        if task not in o:
            continue
        S = o[task]; print(f"\n== {task} ({S['n_splits']} splits)")
        for m in ("dnfr", "product", "dwpc_lr", "dnfr2", "fixp_val"):
            if m in S["summary"]:
                s = S["summary"][m]; print(f"  {m:16s} {s['mean']:.3f} ± {s['sd']:.3f} (n={s['n']})")
        for k, v in S["paired"].items():
            print(f"  {k:32s} Δ={v['delta']:+.4f} J={v['J']} wins={v['wins']} p_holm={v['p_holm']} -> {v['verdict']}")
        print("  restarts", {k: (v and v['all_same_mean']) for k, v in S["restarts"].items()}, "fixp_val", dict(S["fixp_val_choice"]))
