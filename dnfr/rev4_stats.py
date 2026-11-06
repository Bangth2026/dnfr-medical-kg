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
    # ablations without metapaths that contain the target relation (_notgt), or treats or palliates (_notgtp, round 5),
    # runs of the tagged files paired with the full graph
    full = {r["seed"]: r for r in runs}
    def ablation(tag, suf, extra_pairs=()):
        _, nt = merge(f"het_CtD{tag}")
        if not nt:
            return None
        comb = []
        for r in nt:
            f = full.get(r["seed"])
            if not f:
                continue
            R = dict(seed=r["seed"], n_train=r["n_train"], n_test=r["n_test"], methods={}, ranks={})
            for m in ("dnfr", "product", "dwpc_lr"):
                if m in r["ranks"]:
                    R["ranks"][m + suf] = r["ranks"][m]; R["methods"][m + suf] = r["methods"][m]
                if m in f["ranks"]:
                    R["ranks"][m] = f["ranks"][m]; R["methods"][m] = f["methods"][m]
            comb.append(R)
        return nt, comb
    def family(comb, pairs, name):
        fam = {}
        for a, b in pairs:
            v = paired(comb, a, b)
            if v:
                fam[f"{a}__{b}"] = v
        hp = holm({k: v["p"] for k, v in fam.items()})
        for k, v in fam.items():
            v["p_holm"] = hp.get(k); v["verdict"] = verdict(v); v["verdict_rule2"] = verdict_rule2(v); v["family"] = name
        return fam
    def means_of(comb, ms):
        return {m: dict(mean=float(np.mean([r["methods"][m]["MRR"] for r in comb if m in r["methods"]])),
                        sd=float(np.std([r["methods"][m]["MRR"] for r in comb if m in r["methods"]], ddof=1))
                        if sum(1 for r in comb if m in r["methods"]) > 1 else 0.0,
                        n=sum(1 for r in comb if m in r["methods"]))
                for m in ms if any(m in r["methods"] for r in comb)}
    A1 = ablation("_notgt", "_notgt")
    if A1:
        nt, comb = A1
        fam = family(comb, (("dnfr_notgt", "dnfr"), ("product_notgt", "product"), ("dwpc_lr_notgt", "dwpc_lr"),
                            ("dnfr_notgt", "dwpc_lr_notgt"), ("dnfr_notgt", "product_notgt")), "notgt")
        out["notgt"] = dict(paired=fam, means=means_of(comb, ("dnfr_notgt", "product_notgt", "dwpc_lr_notgt", "dnfr", "product", "dwpc_lr")),
                            paths_kept=[r.get("paths_kept") for r in nt], splits=[r["seed"] for r in comb])
    A2 = ablation("_notgtp", "_notgtp")
    if A2:
        nt2, comb2 = A2
        if A1:   # add the treats-only ablation of the same split, to test what removing palliates adds
            by = {r["seed"]: r for r in A1[1]}
            for R in comb2:
                for k in ("dnfr_notgt", "product_notgt", "dwpc_lr_notgt"):
                    if R["seed"] in by and k in by[R["seed"]]["methods"]:
                        R["methods"][k] = by[R["seed"]]["methods"][k]; R["ranks"][k] = by[R["seed"]]["ranks"][k]
        fam2 = family(comb2, (("dnfr_notgtp", "dnfr"), ("dnfr_notgtp", "dnfr_notgt"), ("product_notgtp", "product_notgt"),
                              ("dwpc_lr_notgtp", "dwpc_lr_notgt"), ("dnfr_notgtp", "dwpc_lr_notgtp")), "notgtp")
        out["notgtp"] = dict(paired=fam2, means=means_of(comb2, ("dnfr_notgtp", "product_notgtp", "dwpc_lr_notgtp", "dnfr_notgt",
                                                                "product_notgt", "dwpc_lr_notgt", "dnfr", "product", "dwpc_lr")),
                             paths_kept=[r.get("paths_kept") for r in nt2], splits=[r["seed"] for r in comb2])
    # evidence shares after each ablation (seed-0 models), from evidence_notgt.py
    out["evidence_ablation"] = {}
    for tag in ("_notgt", "_notgtp"):
        for d_ in (f"{ROOT}/results", f"{ROOT}/results2/jobs/ctd5_notgtp_s0"):
            f = f"{d_}/het_CtD{tag}_evidence.json"
            if os.path.exists(f):
                out["evidence_ablation"][tag] = {k: v for k, v in json.load(open(f)).items() if k != "records"}; break
    # round-5 M3: the fixed Hamacher product on split 3 from other initial memberships
    hr = []
    for v in ("hamacher_b01", "hamacher_b06"):
        lv = final_losses("het_CtD", v)
        for r in runs:
            if v in r["methods"]:
                hr.append(dict(variant=v, b_init=float(v[-2:]) / 10, seed=r["seed"], mrr=r["methods"][v]["MRR"],
                               loss=lv.get(r["seed"]), best_epoch=r["methods"][v].get("best_epoch"), epochs=r["methods"][v].get("epochs")))
    out["hamacher_restarts"] = hr
    out["explain_dwpc"] = {}
    for t in ("het_CtD", "het_DpS", "hkg_DpS"):
        f = f"{ROOT}/results/{t}_explain_dwpc.json"
        if os.path.exists(f):
            out["explain_dwpc"][t] = json.load(open(f))["summary"]
    json.dump(out, open(f"{ROOT}/results/rev4_stats.json", "w"), indent=1)
    print(json.dumps({k: (v if k != "notgt" else v.get("means")) for k, v in out.items() if k != "explain_dwpc"}, indent=0)[:3000])


if __name__ == "__main__":
    main()
