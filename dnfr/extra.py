import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import DATA, OUT
import json, os
R = OUT
n = json.load(open(f"{R}/nhanes_dnfr.json"))
ren = {"product (fixed, max)": "product", "hamacher (fixed, max)": "hamacher", "min (fixed, max)": "min", "luk (fixed, max)": "luk",
       "SS shared p (max)": "ss_shared", "SS per-metapath p (max)": "ss_permp", "DNFR: calib + per-metapath p (max)": "dnfr_max",
       "DNFR: calib + per-metapath p (noisy-OR)": "dnfr", "calib + product (noisy-OR)": "calib_product"}
ex = {"nhanes": {ren[k]: v for k, v in n.items() if k in ren}, "nhanes_info": n["_info"]}
ex["jack"] = {k: v for k, v in json.load(open(f"{R}/nhanes_jack.json")).items() if k != "jack"}
expl = []
for t in ["het_CtD", "het_DpS", "pkg_ind", "hkg_DpS"]:
    f = f"{R}/{t}_explanations.json"
    if os.path.exists(f):
        pick = {"het_CtD": 2, "het_DpS": 1, "hkg_DpS": 0}.get(t)
        if pick is None: continue
        e = json.load(open(f))[pick]
        e["task"] = t; e["top3_share"] = sum(p["share"] for p in e["paths"][:3]); expl.append(e)
ex["explanations"] = expl
# quantitative evaluation of explanations (explain_eval.py)
import numpy as np
ee = {}
for t in ["het_CtD", "het_DpS", "hkg_DpS"]:
    f = f"{R}/{t}_explain_eval.json"
    if not os.path.exists(f):
        continue
    d = json.load(open(f)); rec = d["records"]
    A = lambda k, sel=None: np.array([r[k] for r in rec if k in r and (sel is None or sel(r))], float)
    hit = lambda r: r["rank"] <= 10
    o = dict(n_eval=len(rec), n_test=d["n_test"], n_hits10=int(sum(hit(r) for r in rec)))
    for k in ("n_paths", "k50", "k80", "top3_share", "n_mps", "mp_k50", "top_mp_share", "lr_top_mp_share", "lr_mp_k50"):
        o[k + "_median"] = float(np.median(A(k)))
    o["same_top_mp"] = float(np.mean(A("same_top_mp")))
    lrhit = lambda r: r.get("lr_rank", 1e9) <= 10
    for k in ("del_top1", "del_top3", "del_top10", "del_rand1", "del_rand3", "del_rand10", "del_top_mp", "lr_del_top_mp"):
        sel = lrhit if k.startswith("lr_") else hit   # each method's own top-10 hits
        o[k + "_keep10"] = float(np.mean(A(k, sel) <= 10))          # share of top-10 hits still in the top 10
        o[k + "_median_rank"] = float(np.median(A(k, sel)))
    o["n_lr_hits10"] = int(sum(lrhit(r) for r in rec))
    o["lr_hits10_among"] = float(np.mean(A("lr_rank", hit) <= 10))
    ee[t] = o
ex["explain_eval"] = ee
json.dump(ex, open(f"{R}/extra_stats.json", "w"), indent=1)
print(list(ex), len(expl))
