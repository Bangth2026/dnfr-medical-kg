"""Statistics of the second-revision analyses -> results/revision_stats.json

profile   : training loss and accuracy with p fixed on a grid (identifiability.py profile)
long      : learned p without early stopping from several initial values (identifiability.py long)
stopping  : epoch of the best validation MRR and of stopping for the default runs (run_task.py)
explain2  : share-matched deletion, rank against share removed, overlap and edge-level counterfactual
mu_hkg    : distribution of the learned memberships on the graded graph (seed-0 model) and of the raw weights
n_params  : number of trainable parameters of DNFR, the fixed-product variant and DWPC-LR per task
"""
import os, sys, re, glob, json, pickle
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import DATA, OUT
import numpy as np
from scipy.stats import spearmanr

TASKS = ["het_CtD", "het_DpS", "pkg_ind", "hkg_DpS"]


def q(a):
    a = np.asarray(a, float)
    return dict(median=float(np.median(a)), q25=float(np.quantile(a, .25)), q75=float(np.quantile(a, .75)),
                min=float(a.min()), max=float(a.max()), mean=float(a.mean()), n=int(len(a)))


def profile(task):
    f = f"{OUT}/profile_{task}.json"
    if not os.path.exists(f):
        return None
    d = json.load(open(f))
    seeds = sorted({r["seed"] for r in d.values()}); vals = sorted({r["value"] for r in d.values()})
    out = dict(seeds=seeds, grid=vals, rows=[])
    # per seed, loss relative to the best grid value of that seed (only seeds with the full grid)
    full = [s for s in seeds if all(f"seed{s}_p{v:g}" in d for v in vals)]
    for v in vals:
        rs = [d[f"seed{s}_p{v:g}"] for s in seeds if f"seed{s}_p{v:g}" in d]
        row = dict(p=v, n=len(rs),
                   final_loss=float(np.mean([r["final"]["train_loss"] for r in rs])),
                   best_loss=float(np.mean([r["best_valid"]["train_loss"] for r in rs])),
                   final_epoch=float(np.mean([r["final"]["epoch"] for r in rs])),
                   best_epoch=float(np.mean([r["best_valid"]["epoch"] for r in rs])),
                   test_best=float(np.mean([r["best_valid"]["test_MRR"] for r in rs])),
                   test_best_sd=float(np.std([r["best_valid"]["test_MRR"] for r in rs], ddof=1)) if len(rs) > 1 else 0.0,
                   valid_best=float(np.mean([r["best_valid"]["valid_MRR"] for r in rs])),
                   test_final=float(np.mean([r["final"]["test_MRR"] for r in rs])),
                   loss_drop_last50=float(np.mean([r["history"][-11]["train_loss"] - r["history"][-1]["train_loss"]
                                                   for r in rs if len(r["history"]) > 11])))
        if full:
            rel = []
            for s in full:
                ls = {w: d[f"seed{s}_p{w:g}"]["final"]["train_loss"] for w in vals}
                rel.append(ls[v] - min(ls.values()))
            row["final_loss_minus_seed_min"] = float(np.mean(rel))
        out["rows"].append(row)
    if full:
        argmin = [min(vals, key=lambda w: d[f"seed{s}_p{w:g}"]["final"]["train_loss"]) for s in full]
        out["argmin_final_loss_by_seed"] = argmin
        out["argmax_valid_by_seed"] = [max(vals, key=lambda w: d[f"seed{s}_p{w:g}"]["best_valid"]["valid_MRR"]) for s in full]
    return out


def long(task):
    f = f"{OUT}/long_{task}.json"
    if not os.path.exists(f):
        return None
    d = json.load(open(f))
    seeds = sorted({r["seed"] for r in d.values()}); inits = sorted({r["value"] for r in d.values()})
    out = dict(seeds=seeds, inits=inits, by_init={}, agreement={})
    for v in inits:
        rs = [d[f"seed{s}_p{v:g}"] for s in seeds if f"seed{s}_p{v:g}" in d]
        pb = np.concatenate([r["best_valid"]["p"] for r in rs]); pf = np.concatenate([r["final"]["p"] for r in rs])
        traj = {}
        for r in rs:
            for h in r["history"]:
                traj.setdefault(h["epoch"], []).extend(h["p"])
        out["by_init"][f"{v:g}"] = dict(
            n=len(rs), p_best=q(pb), p_final=q(pf),
            test_best=float(np.mean([r["best_valid"]["test_MRR"] for r in rs])),
            test_final=float(np.mean([r["final"]["test_MRR"] for r in rs])),
            valid_best=float(np.mean([r["best_valid"]["valid_MRR"] for r in rs])),
            best_epoch=[r["best_valid"]["epoch"] for r in rs],
            loss_final=float(np.mean([r["final"]["train_loss"] for r in rs])),
            loss_best=float(np.mean([r["best_valid"]["train_loss"] for r in rs])),
            grad_final=float(np.mean([r["history"][-1]["grad_theta"] for r in rs])),
            grad_first=float(np.mean([r["history"][0]["grad_theta"] for r in rs])),
            p_median_traj={str(e): float(np.median(v_)) for e, v_ in sorted(traj.items())})
    rhos, signs, close = [], [], []
    for s in seeds:
        have = [v for v in inits if f"seed{s}_p{v:g}" in d]
        if len(have) < 2:
            continue
        M = np.array([d[f"seed{s}_p{v:g}"]["final"]["p"] for v in have])
        if M.shape[1] >= 3:
            rhos += [float(spearmanr(M[i], M[j]).statistic) for i in range(len(have)) for j in range(i + 1, len(have))]
        signs.append(float(np.mean(np.all(M < 0, 0) | np.all(M > 0, 0))))
        close.append(float(np.mean(M.max(0) - M.min(0) < 0.25)))
    out["agreement"] = dict(spearman=q(rhos) if rhos else None, sign_agree=float(np.mean(signs)) if signs else None,
                            range_lt025=float(np.mean(close)) if close else None)
    return out


def stopping(task):
    f = f"{OUT}/{task}.json"
    if not os.path.exists(f):
        return None
    d = json.load(open(f)); out = {}
    for v in ["dnfr", "product", "hamacher", "dnfr_pm2", "dnfr_pm1", "dnfr_pp05", "dnfr_l2_0"]:
        st = [r["methods"][v]["epochs"] for r in d["runs"] if v in r["methods"] and "epochs" in r["methods"][v]]
        # best epoch: recorded in second-revision runs; otherwise the stopping epoch minus the patience
        # (10 evaluations x 5 epochs), exact whenever training stopped before the epoch limit
        be = [r["methods"][v].get("best_epoch", r["methods"][v]["epochs"] - 50 if r["methods"][v]["epochs"] < 400 else 400)
              for r in d["runs"] if v in r["methods"] and "epochs" in r["methods"][v]]
        if st:
            out[v] = dict(stop_epoch=q(st), best_epoch=q(be), n_hit_limit=int(sum(e >= 400 for e in st)))
    return out


def explain2(task):
    f = f"{OUT}/{task}_explain_eval2.json"
    if not os.path.exists(f):
        return None
    d = json.load(open(f)); R = d["records"]
    A = {k: np.array([r[k] for r in R]) for k in R[0]}
    out = dict(n_hits=len(R), matched_paths_removed={k: float(np.median(A[f"matched{k}_n"])) for k in (1, 3, 10)},
               stay_top10={k: float(np.mean(A[k] <= 10)) for k in ("matched1", "matched3", "matched10", "edge_top", "edge_rand")},
               curve={f"{f_:g}": dict(top=float(np.mean(A[f"top_f{f_}"] <= 10)), rand=float(np.mean(A[f"rand_f{f_}"] <= 10)))
                      for f_ in d["fracs"]},
               top3_share_edge=float(A["top3_share_edge"].mean()), top3_overlap_share=float(np.median(A["top3_overlap_share"])),
               top_edge_share=float(np.median(A["top_edge_share"])), paths_on_top_edge=float(np.median(A["paths_on_top_edge"])),
               n_edges=float(np.median(A["n_edges"])),
               median_rank_edge_top=float(np.median(A["edge_top"])), median_rank_edge_rand=float(np.median(A["edge_rand"])))
    return out


def mu_hkg():
    import torch
    from explain import setup
    ctx = setup("hkg_DpS", 0)
    m = ctx["model"]; sd = ctx["sd"]
    out = {}
    raw = np.array([x[2] for x in json.load(open(f"{DATA}/hkg_edges.json"))])
    out["raw_weight"] = q(raw)
    # one value per edge of the seed-0 training graph and direction of traversal, with training-graph degrees
    kg_s = ctx["kg_s"]; d_ = kg_s.rels["DpS"]; w = np.clip(d_["w"], 1e-4, 1 - 1e-4); lw = np.log(w / (1 - w))
    sd_ = {}
    for t, tr in enumerate(ctx["trav"]):
        A = kg_s.adj(tr); do = np.log1p(np.asarray(A.sum(1)).ravel()); di = np.log1p(np.asarray(A.sum(0)).ravel())
        u, v = (d_["t"], d_["h"]) if tr[4] else (d_["h"], d_["t"])
        a, b, c, g = (float(m.state_dict()[k][t]) for k in ("a", "b", "c", "g"))
        mu = 1 / (1 + np.exp(-(a - b * do[u] - c * di[v] + g * lw)))
        out[tr[0]] = q(mu)
        out[tr[0] + "_spearman_with_weight"] = float(spearmanr(mu, w).statistic)
    out["params"] = {k: m.state_dict()[k].numpy().tolist() for k in ("a", "b", "c", "g")}
    return out


def n_params():
    out = {}
    for task in TASKS:
        f = f"{OUT}/{task}.json"
        if not os.path.exists(f):
            continue
        d = json.load(open(f))
        names = d["mp_names"]; used = set()
        for nm in names:
            used |= set(re.split(r"(?<!\^)-", nm))
        graded = task == "hkg_DpS"; n_mp = d["n_mps"]
        dn = len(used) * (4 if graded else 3) + 2 * n_mp + 1
        out[task] = dict(n_traversals=len(used), n_metapaths=n_mp, dnfr=dn, product=dn - n_mp, shared_p=dn - n_mp + 1,
                         dwpc_lr=n_mp + 1, dwpc_lr_full=n_mp + 4)
    return out


if __name__ == "__main__":
    S = dict(profile={t: profile(t) for t in TASKS}, long={t: long(t) for t in TASKS},
             stopping={t: stopping(t) for t in TASKS}, explain2={t: explain2(t) for t in TASKS},
             n_params=n_params())
    try:
        S["mu_hkg"] = mu_hkg()
    except Exception as e:  # needs data/hkg.pkl and data/paths_hkg_DpS.pkl
        print("mu_hkg skipped:", e)
    S = {k: ({t: v for t, v in val.items() if v is not None} if isinstance(val, dict) and k != "mu_hkg" else val) for k, val in S.items()}
    json.dump(S, open(f"{OUT}/revision_stats.json", "w"), indent=1)
    print(json.dumps({k: (list(v) if isinstance(v, dict) else v) for k, v in S.items()}, indent=0)[:2000])
