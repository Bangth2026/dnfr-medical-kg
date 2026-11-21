"""Revision 6 (JBI reviewer M2, M3, M4): paired comparisons for the new runs, with the decision rule of Section 3.6.
Merges the per-query ranks of the new result files in results_rev6/ with those of the released files in results/ for the
same splits (the split function and seeds are unchanged; test queries are checked to be identical).
Families (Holm adjustment within each):
  M2 controls, per task: DWPC-softmax vs DWPC-LR, linear non-negative DWPC (softmax) vs DWPC-LR, linear non-negative
     DWPC (logistic) vs DWPC-LR, DNFR vs DWPC-softmax, DNFR vs linear non-negative DWPC (softmax);
  disease-centred split (10 splits): every method vs the fixed product (DNFR on four splits), and DNFR without degree terms vs DNFR;
  disease-centred split without the treats metapaths: product and DWPC-LR without minus with;
  compound-centred split: every method vs the fixed product (DNFR was run on fewer splits).
Output: results_rev6/rev6_compare.json"""
import os, sys, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from aggregate import paired, set_verdicts, summarise

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
R0, R6 = os.path.join(ROOT, "results"), os.path.join(ROOT, "results_rev6")


def load(path):
    return json.load(open(path)) if os.path.exists(path) else None


def merged(files_methods):
    """files_methods: list of (path, methods or None for all, suffix). Returns a d with runs keyed by seed."""
    runs = {}
    for path, methods, suf in files_methods:
        d = load(path)
        if d is None:
            continue
        for r in d["runs"]:
            if "ranks" not in r:
                continue
            s = r["seed"]
            R = runs.setdefault(s, dict(seed=s, n_train=r["n_train"], n_test=r["n_test"], methods={}, ranks={},
                                        test_queries=r.get("test_queries")))
            if R["test_queries"] is not None and r.get("test_queries") is not None:
                assert R["test_queries"] == r["test_queries"], (path, s)
            for m, rk in r["ranks"].items():
                if methods is None or m in methods:
                    R["ranks"][m + suf] = rk
                    if m in r["methods"]:
                        R["methods"][m + suf] = r["methods"][m]
    return dict(runs=[runs[s] for s in sorted(runs)])


def fam(d, pairs):
    F = {f"{a}__{b}": paired(d, a, b) for a, b in pairs}
    F = {k: v for k, v in F.items() if v}
    set_verdicts(F)
    return {k: {kk: vv for kk, vv in v.items() if kk != "per_split"} | {"per_split": v["per_split"]} for k, v in F.items()}


def means(d, methods):
    out = {}
    for m in methods:
        mrr = [r["methods"][m]["MRR"] for r in d["runs"] if m in r["methods"]]
        h10 = [r["methods"][m]["H10"] for r in d["runs"] if m in r["methods"]]
        if mrr:
            out[m] = dict(MRR=float(np.mean(mrr)), sd=float(np.std(mrr, ddof=1)) if len(mrr) > 1 else 0.0,
                          H10=float(np.mean(h10)), n=len(mrr))
    return out


OUT = {}
# ---------------- M2: DWPC controls
for task in ("hkg_DpS", "het_DpS", "het_CtD"):
    d = merged([(f"{R0}/{task}.json", {"dnfr", "product", "dwpc_lr"}, "_rel"),
                (f"{R6}/{task}_ctl.json", None, "")])
    if not d["runs"] or not any("dwpc_sm" in r["ranks"] for r in d["runs"]):
        continue
    d["runs"] = [r for r in d["runs"] if "dwpc_sm" in r["ranks"] and "dnfr_rel" in r["ranks"]]
    # reproducibility check of DWPC-LR against the released ranks
    rep = [float(np.max(np.abs(np.array(r["ranks"]["dwpc_lr"]) - np.array(r["ranks"]["dwpc_lr_rel"])))) for r in d["runs"]]
    coef = {}
    c6 = load(f"{R6}/{task}_ctl.json")
    for r in c6["runs"]:
        for m in ("dwpc_sm", "dwpc_nn_sm", "dwpc_nn_lr"):
            if m in r["methods"] and "coef" in r["methods"][m]:
                c = np.array(r["methods"][m]["coef"])
                coef.setdefault(m, []).append(dict(neg=float((c < 0).mean()), zero=float((np.abs(c) < 1e-6).mean())))
    OUT[f"M2_{task}"] = dict(
        n_splits=len(d["runs"]), dwpc_lr_reproduced_max_rank_diff=max(rep),
        means=means(d, ["dnfr_rel", "product_rel", "dwpc_lr", "dwpc_sm", "dwpc_nn_sm", "dwpc_nn_lr"]),
        coef={m: dict(neg_share_mean=float(np.mean([x["neg"] for x in v])), zero_share_mean=float(np.mean([x["zero"] for x in v])))
              for m, v in coef.items()},
        family=fam(d, [("dwpc_sm", "dwpc_lr"), ("dwpc_nn_sm", "dwpc_lr"), ("dwpc_nn_lr", "dwpc_lr"),
                       ("dnfr_rel", "dwpc_sm"), ("dnfr_rel", "dwpc_nn_sm")]))

# ---------------- M3: disease-centred split, 10 splits (released splits 0-3 + new 4-9)
d = merged([(f"{R0}/het_CtD_tail.json", None, ""), (f"{R6}/het_CtD_tail.json", None, "")])
if d["runs"]:
    ms = ["dnfr", "product", "hamacher", "dnfr_nodeg", "dwpc_lr", "dwpc_lr_full", "softimpute", "svd", "prior", "popularity"]
    OUT["tail10"] = dict(n_splits=len(d["runs"]), seeds=[r["seed"] for r in d["runs"]], means=means(d, ms),
                         family=fam(d, [(m, "product") for m in ms if m != "product"] + [("dnfr_nodeg", "dnfr")]))
# disease-centred without treats metapaths
dn = merged([(f"{R0}/het_CtD_tail.json", {"product", "dwpc_lr", "popularity"}, ""), (f"{R6}/het_CtD_tail.json", {"product", "dwpc_lr", "popularity"}, ""),
             (f"{R6}/het_CtD_tail_notgt.json", {"product", "dwpc_lr"}, "_notgt")])
if dn["runs"] and any("product_notgt" in r["ranks"] for r in dn["runs"]):
    dn["runs"] = [r for r in dn["runs"] if "product_notgt" in r["ranks"]]
    OUT["tail_notgt"] = dict(n_splits=len(dn["runs"]), means=means(dn, ["product", "dwpc_lr", "product_notgt", "dwpc_lr_notgt", "popularity"]),
                             family=fam(dn, [("product_notgt", "product"), ("dwpc_lr_notgt", "dwpc_lr"),
                                             ("product_notgt", "popularity"), ("dwpc_lr_notgt", "popularity")]))
# compound-centred split
dh = merged([(f"{R6}/het_CtD_head.json", None, "")])
if dh["runs"]:
    ms = ["dnfr", "product", "dwpc_lr", "dwpc_lr_full", "softimpute", "svd", "prior", "popularity"]
    dh["runs"] = [r for r in dh["runs"] if "product" in r["ranks"]]
    OUT["head"] = dict(n_splits=len(dh["runs"]), seeds=[r["seed"] for r in dh["runs"]], means=means(dh, ms),
                       family=fam(dh, [(m, "product") for m in ms if m != "product"]))
json.dump(OUT, open(f"{R6}/rev6_compare.json", "w"), indent=1)
for k, v in OUT.items():
    print(k, v.get("n_splits"))
    for m, x in v.get("means", {}).items():
        print(f"   {m:16s} {x['MRR']:.3f} ± {x['sd']:.3f}  H10 {x['H10']:.3f} (n={x['n']})")
    for c, x in v.get("family", {}).items():
        print(f"   {c:28s} {x['delta_MRR']:+.3f} [{x['ci_lo']:+.3f}, {x['ci_hi']:+.3f}] {x['seed_wins']}/{x['seeds']} "
              f"p_holm={x.get('p_holm')} {x.get('verdict')}  nb90={x.get('nb_ci90_lo')},{x.get('nb_ci90_hi')}")
    if "coef" in v:
        print("   coef", v["coef"], "repro", v["dwpc_lr_reproduced_max_rank_diff"])
