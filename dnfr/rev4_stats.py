"""Statistics for the analyses added after the third reviewer report: per-split values of the fixed Hamacher product
on compound-treats-disease with the final training loss read from the run logs (M1, Q1), the ablation without
metapaths that contain the target relation (M4, Q3), the DWPC-LR attribution (M2, Q5), the evidence shares by
metapath class (M4) and the time per epoch of each run (minor 7). Output: results/rev4_stats.json"""
import os, re, sys, glob, json
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from rev3_stats import paired, holm, verdict, verdict_rule2, merge

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def final_losses(task, variant):
    """Training loss at the last logged epoch of every run of a variant, by split, from the job logs."""
    out = {}
    pat = re.compile(rf"^\[{re.escape(task)} seed (\d+)\] {re.escape(variant)}$")
    for f in sorted(glob.glob(f"{ROOT}/results2/jobs/*/log.txt")):
        seed, last = None, None
        for line in open(f, errors="ignore"):
            line = line.rstrip("\n")
            mm = pat.match(line)
            if mm:
                seed, last = int(mm.group(1)), None; continue
            if seed is not None:
                lm = re.match(r"\s+ep (\d+) loss ([\d.]+)", line)
                if lm:
                    last = float(lm.group(2))
                elif line.startswith("[") or line.strip().startswith("test"):
                    if last is not None:
                        out[seed] = last
                    seed = None if line.startswith("[") else seed
                    if line.strip().startswith("test"):
                        seed = None
    return out


def main():
    out = {}
    meta, runs = merge("het_CtD")
    lh, lp = final_losses("het_CtD", "hamacher"), final_losses("het_CtD", "product")
    out["hamacher_ctd"] = [dict(seed=r["seed"], hamacher=r["methods"]["hamacher"]["MRR"], product=r["methods"]["product"]["MRR"],
                                dnfr=r["methods"]["dnfr"]["MRR"], best_epoch=r["methods"]["hamacher"].get("best_epoch"),
                                epochs=r["methods"]["hamacher"].get("epochs"), loss_hamacher=lh.get(r["seed"]),
                                loss_product=lp.get(r["seed"]))
                           for r in runs if "hamacher" in r["methods"]]
    out["time_per_epoch_ctd"] = [dict(seed=r["seed"], dnfr_s=r["methods"]["dnfr"]["time_s"], epochs=r["methods"]["dnfr"]["epochs"],
                                      s_per_epoch=r["methods"]["dnfr"]["time_s"] / r["methods"]["dnfr"]["epochs"],
                                      dwpc_s=r["methods"]["dwpc_lr"].get("time_s"))
                                 for r in runs if "dnfr" in r["methods"] and "dwpc_lr" in r["methods"]]
    # ablation without metapaths that contain the target relation (runs of the tagged file, paired with the full graph)
    _, nt = merge("het_CtD_notgt")
    full = {r["seed"]: r for r in runs}
    if nt:
        comb = []
        for r in nt:
            f = full.get(r["seed"])
            if not f:
                continue
            R = dict(seed=r["seed"], n_train=r["n_train"], n_test=r["n_test"], methods={}, ranks={})
            for m in ("dnfr", "product", "dwpc_lr"):
                if m in r["ranks"]:
                    R["ranks"][m + "_notgt"] = r["ranks"][m]; R["methods"][m + "_notgt"] = r["methods"][m]
                if m in f["ranks"]:
                    R["ranks"][m] = f["ranks"][m]; R["methods"][m] = f["methods"][m]
            comb.append(R)
        fam = {}
        for a, b in (("dnfr_notgt", "dnfr"), ("product_notgt", "product"), ("dwpc_lr_notgt", "dwpc_lr"),
                     ("dnfr_notgt", "dwpc_lr_notgt"), ("dnfr_notgt", "product_notgt")):
            v = paired(comb, a, b)
            if v:
                fam[f"{a}__{b}"] = v
        hp = holm({k: v["p"] for k, v in fam.items()})
        for k, v in fam.items():
            v["p_holm"] = hp.get(k); v["verdict"] = verdict(v); v["verdict_rule2"] = verdict_rule2(v); v["family"] = "notgt"
        means = {m: dict(mean=float(np.mean([r["methods"][m]["MRR"] for r in comb if m in r["methods"]])),
                         sd=float(np.std([r["methods"][m]["MRR"] for r in comb if m in r["methods"]], ddof=1)) if len(comb) > 1 else 0.0,
                         n=sum(1 for r in comb if m in r["methods"]))
                 for m in ("dnfr_notgt", "product_notgt", "dwpc_lr_notgt", "dnfr", "product", "dwpc_lr")}
        out["notgt"] = dict(paired=fam, means=means, paths_kept=[r.get("paths_kept") for r in nt],
                            splits=[r["seed"] for r in comb])
    out["explain_dwpc"] = {}
    for t in ("het_CtD", "het_DpS", "hkg_DpS"):
        f = f"{ROOT}/results/{t}_explain_dwpc.json"
        if os.path.exists(f):
            out["explain_dwpc"][t] = json.load(open(f))["summary"]
    json.dump(out, open(f"{ROOT}/results/rev4_stats.json", "w"), indent=1)
    print(json.dumps({k: (v if k != "notgt" else v.get("means")) for k, v in out.items() if k != "explain_dwpc"}, indent=0)[:3000])


if __name__ == "__main__":
    main()
