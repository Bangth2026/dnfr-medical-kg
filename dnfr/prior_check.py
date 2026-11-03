"""Third revision, reviewer comment M6: why does the degree-preserving edge prior rank below popularity?
For each split: share of test queries whose true tail ties with at least one other candidate, mean share of tied
candidates, MRR of the Monte-Carlo prior with 100 and 1000 permutations, of the analytical approximation
P = uv / sqrt((uv)^2 + (m - u - v + 1)^2) (Zietz et al., GigaScience 2024), and of popularity.
Output: results/prior_check.json"""
import os, sys, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import OUT
import numpy as np
from run_task import TASKS, load_kg, positives, split, split_by_tail, edge_prior, _dense_ranks
from dnfr import metrics


def analytic(tr_q, n_src, n_t):
    u = np.bincount(tr_q[:, 0], minlength=n_src).astype(float); v = np.bincount(tr_q[:, 1], minlength=n_t).astype(float)
    m = len(tr_q); uv = u[:, None] * v[None, :]
    return uv / np.sqrt(uv ** 2 + (m - u[:, None] - v[None, :] + 1) ** 2)


def ties(Z, q, known):
    tie_any, tie_share = [], []
    for s, t in q:
        row = Z[s].copy(); filt = list(known.get(int(s), set()) - {int(t)}); row[filt] = np.nan
        others = np.delete(row, t); others = others[~np.isnan(others)]
        k = (others == row[t]).sum(); tie_any.append(k > 0); tie_share.append(k / len(others))
    return float(np.mean(tie_any)), float(np.mean(tie_share))


if __name__ == "__main__":
    tasks = sys.argv[1].split(",") if len(sys.argv) > 1 else ["het_CtD", "het_DpS", "hkg_DpS"]
    out = json.load(open(f"{OUT}/prior_check.json")) if os.path.exists(f"{OUT}/prior_check.json") else {}
    for task in tasks:
        for mode in (["random", "tail"] if task == "het_CtD" else ["random"]):
            spec = TASKS[task]; kg = load_kg(spec["kg"]); pos = positives(kg, spec)
            sources = np.unique(pos[:, 0]); loc = {s: i for i, s in enumerate(sources)}; n_t = kg.n(spec["t"])
            posl = np.stack([[loc[s] for s in pos[:, 0]], pos[:, 1]], 1)
            known = {}
            for s, t in posl:
                known.setdefault(int(s), set()).add(int(t))
            rec = []
            for seed in range(5 if mode == "random" else 4):
                tr, va, te = (split if mode == "random" else split_by_tail)(posl, seed)
                pop = np.broadcast_to(np.bincount(tr[:, 1], minlength=n_t).astype(float), (len(sources), n_t))
                r = dict(seed=seed)
                for name, Z in [("popularity", pop), ("prior100", edge_prior(tr, len(sources), n_t, seed, 100)),
                                ("prior1000", edge_prior(tr, len(sources), n_t, seed, 1000)),
                                ("analytic", analytic(tr, len(sources), n_t))]:
                    r[name] = dict(MRR=metrics(_dense_ranks(Z, te, known))["MRR"])
                    r[name]["tie_any"], r[name]["tie_share"] = ties(Z, te, known)
                    if name == "prior100":
                        r[name]["share_zero_true"] = float(np.mean([Z[s, t] == 0 for s, t in te]))
                        r[name]["n_distinct"] = int(len(np.unique(Z)))
                # Spearman correlation, within heads, between the analytical prior and popularity (should be 1)
                A = analytic(tr, len(sources), n_t)
                from scipy.stats import spearmanr
                r["rho_analytic_pop"] = float(np.median([spearmanr(A[s], pop[s]).statistic for s in range(len(sources))
                                                         if np.std(A[s]) > 0]))
                rec.append(r); print(task, mode, r, flush=True)
            out[task + ("_tail" if mode == "tail" else "")] = rec
            json.dump(out, open(f"{OUT}/prior_check.json", "w"), indent=1)
