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
json.dump(ex, open(f"{R}/extra_stats.json", "w"), indent=1)
print(list(ex), len(expl))
