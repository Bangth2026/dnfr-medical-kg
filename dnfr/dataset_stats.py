import os, sys
from config import DATA, OUT
import pickle, sys, json
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from kg import *
out = {}
for n in ["het", "pkg", "hkg"]:
    kg = pickle.load(open(f"{DATA}/{n}.pkl", "rb"))
    out[n] = dict(nodes=int(sum(kg.n(t) for t in kg.names)), edges=int(kg.n_edges()), rel_types=len(kg.rels), node_types=len(kg.names))
raw = pd.read_csv(os.path.join(DATA, "primekg_kg.csv"), usecols=["relation"], dtype={"relation": "category"})
vc = raw["relation"].value_counts()
out["pkg"]["raw_rows"] = int(len(raw))
out["pkg"]["dropped_rows_anatomy_protein"] = int(vc["anatomy_protein_present"] + vc["anatomy_protein_absent"])
out["pkg"]["dropped_edges"] = out["pkg"]["dropped_rows_anatomy_protein"] // 2  # PrimeKG stores each edge in both directions
w = np.array(pickle.load(open(os.path.join(DATA, "hkg.pkl"), "rb")).rels["DpS"]["w"])
out["hkg"].update(w_median=float(np.median(w)), w_q25=float(np.quantile(w, .25)), w_q75=float(np.quantile(w, .75)), w_max=float(w.max()))
for t in ["het_CtD", "het_DpS", "pkg_ind", "hkg_DpS"]:
    P = pickle.load(open(f"{DATA}/paths_{t}.pkl", "rb"))
    c = P["counts"]
    out[t] = dict(metapaths_all=len(c), metapaths_kept=len(P["mps"]), skipped=[k for k, v in c.items() if v > 5e6],
                  paths=int(len(P["src"])), paths_len2=int(sum((P["mp"] == i).sum() for i, m in enumerate(P["mps"]) if len(m) == 2)))
json.dump(out, open(os.path.join(OUT, "data_stats.json"), "w"), indent=1)
print(json.dumps(out, indent=1)[:2500])
