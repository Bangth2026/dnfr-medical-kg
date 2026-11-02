"""Aggregate all result files into one statistics file (single source of truth for the paper)."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import DATA, OUT
import glob, json, os
import numpy as np
from scipy.stats import wilcoxon

R = OUT
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


def paired(d, a, b):
    """Wilcoxon signed-rank on per-query reciprocal ranks, pooled over the seeds where both exist."""
    xa, xb, wins = [], [], 0
    seeds = 0
    for r in d["runs"]:
        rk = r.get("ranks", {})
        if a in rk and b in rk:
            ra, rb = 1 / np.array(rk[a]), 1 / np.array(rk[b])
            xa.append(ra); xb.append(rb); seeds += 1
            wins += int(ra.mean() > rb.mean())
    if not xa:
        return None
    xa, xb = np.concatenate(xa), np.concatenate(xb)
    diff = xa - xb
    p = float(wilcoxon(xa, xb, zero_method="wilcox").pvalue) if np.any(diff != 0) else 1.0
    return dict(delta_MRR=float(diff.mean()), p=p, n_queries=int(len(diff)), seeds=seeds, seed_wins=wins)


if __name__ == "__main__":
    stats = {}
    for task in TASKS:
        for tag in ["", "_sp0.50", "_sp0.25"]:
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
    json.dump(stats, open(f"{R}/paper_stats.json", "w"), indent=1)
    for k, s in stats.items():
        if k == "nhanes":
            continue
        print("==", k, s["info"])
        for m, v in sorted(s["summary"].items(), key=lambda x: -x[1]["MRR"]["mean"]):
            pr = s["paired"].get(m, {})
            print(f"  {m:14s} MRR {v['MRR']['mean']:.3f}±{v['MRR']['sd']:.3f} H1 {v['H1']['mean']:.3f} H10 {v['H10']['mean']:.3f} n={v['n_seeds']}"
                  + (f"  dnfr-m {pr['delta_MRR']:+.3f} p={pr['p']:.3g} wins {pr['seed_wins']}/{pr['seeds']}" if pr else ""))
        if "p_learned" in s:
            print("  p:", {k: round(v, 3) for k, v in s["p_learned"].items()}, s["degree_coef"])
