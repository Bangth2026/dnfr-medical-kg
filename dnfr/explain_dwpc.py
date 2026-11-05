"""Path- and fact-level attribution for DWPC-LR next to DNFR's, on the seed-0 split (reviewer M2, Q5), and the share
of DNFR's evidence by metapath class (reviewer M4, Q3).

DWPC-LR scores a pair by z = b + sum_m beta_m asinh(DWPC_m / mean_m), where DWPC_m is a sum over the paths of metapath m
of products of degree-damped edge weights w_pi. The metapath terms beta_m asinh(.) add up exactly; within a metapath we
split its term over the paths in proportion to w_pi, which is exact within the metapath (the paths sum to the term) and
a proportional, not marginal, allocation across the non-linear asinh. A fact receives the equal share of every path
through it, as for DNFR. For every test pair that DWPC-LR ranks in the top 10 we record the share of the leading fact,
the number of facts that carry half of the positive attribution, and the rank after removing the leading fact (or a
random fact of the same paths) from the graph, re-scoring every candidate of the same head without retraining. For
pairs in the top 10 of both models we record whether DWPC-LR and DNFR (Shapley) select the same leading fact, and for
DNFR whether the Shapley and summed-path-share criteria select the same fact (minor 5). For DNFR's correct top-10 pairs
we record the share of the evidence on gene-only, associative and other metapaths, and on metapaths that contain the
target relation. Output: results/<task>_explain_dwpc.json"""
import os, sys, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import OUT
import numpy as np
from scipy.sparse import coo_matrix
from sklearn.linear_model import LogisticRegression
from explain import setup
from dnfr import score_pairs, metrics
from explain_eval2 import path_table
from explain_eval3 import mp_class

CLS = ("gene", "associative", "other")


def main(task, seed=0):
    ctx = setup(task, seed)
    P, kg_s, spec, n_t, ukey = ctx["P"], ctx["kg_s"], ctx["spec"], ctx["n_t"], ctx["ukey"]
    ranker, known, tr, va, te = ctx["ranker"], ctx["known"], ctx["tr"], ctx["va"], ctx["te"]
    n_pairs, n_mp = len(ukey), len(P["mps"])
    z, z0 = score_pairs(ctx["model"], ctx["sd"])
    rank_dnfr = ranker.ranks(z, z0, te, known)
    pairs, contr, edges = path_table(ctx)                       # DNFR path terms, in chunk order
    pidx = np.concatenate([ch["path_index"] for ch in ctx["sd"].chunks])
    mp = P["mp"][pidx].astype(np.int16)
    # degree-damped path weights of DWPC (damping 0.4, degrees in the split's training graph), as in run_task.py
    w = np.ones(len(pidx), np.float32); deg = {}
    for mi, m in enumerate(P["mps"]):
        sel = np.where(mp == mi)[0]
        if len(sel) == 0:
            continue
        ii = pidx[sel]
        s = P["sources"][P["src"][ii]]
        nodes = [s, P["k1"][ii]] + ([P["tgt"][ii]] if len(m) == 2 else [P["k2"][ii], P["tgt"][ii]])
        for h, trv in enumerate(m):
            if trv not in deg:
                A = kg_s.adj(trv)
                deg[trv] = (np.asarray(A.sum(1)).ravel(), np.asarray(A.sum(0)).ravel())
            do, di = deg[trv]
            w[sel] *= ((np.maximum(do[nodes[h]], 1) * np.maximum(di[nodes[h + 1]], 1)) ** (-0.4)).astype(np.float32)
    del pidx
    tgt_mp = np.array([any(trv[3] == spec["rel"] for trv in m) for m in P["mps"]])
    cls_mp = np.array([CLS.index(mp_class(m)) for m in P["mps"]], np.int8)
    X = coo_matrix((w.astype(np.float64), (pairs, mp.astype(np.int64))), shape=(n_pairs, n_mp)).tocsr()
    X.sum_duplicates()
    mean = np.asarray(X.sum(0)).ravel() / np.maximum(np.diff(X.tocsc().indptr), 1)
    mean = np.maximum(mean, 1e-12)
    Xa = X.multiply(1 / mean).tocsr(); Xa.data = np.arcsinh(Xa.data)
    # logistic regression as in run_task.baselines (20 random negatives per positive, C chosen on validation)
    rng = np.random.default_rng(seed)
    key = {int(k): i for i, k in enumerate(ukey)}
    tr_set = set(map(tuple, tr.tolist()))
    pos_rows = [key.get(int(s) * n_t + int(t), -1) for s, t in tr]
    neg_rows = [key.get(int(s) * n_t + int(t), -1) for s in tr[:, 0] for t in rng.integers(0, n_t, 20)
                if (int(s), int(t)) not in tr_set]
    def rows(lst):
        lst = np.array(lst); ok = lst >= 0
        return Xa[np.where(ok, lst, 0)].multiply(ok[:, None]).tocsr()
    from scipy.sparse import vstack
    Xtr = vstack([rows(pos_rows), rows(neg_rows)]).tocsr(); ytr = np.r_[np.ones(len(pos_rows)), np.zeros(len(neg_rows))]
    best = None
    for C in [0.01, 0.1, 1.0, 10.0]:
        lr = LogisticRegression(C=C, max_iter=2000, class_weight="balanced").fit(Xtr, ytr)
        zd = lr.decision_function(Xa); zd0 = float(lr.intercept_[0])
        mv = metrics(ranker.ranks(zd, zd0, va, known))["MRR"]
        if best is None or mv > best[0]:
            best = (mv, C, lr, zd, zd0)
    _, C, lr, zd, zd0 = best
    beta = lr.coef_.ravel()
    rank_dwpc = ranker.ranks(zd, zd0, te, known)
    print(task, "DWPC-LR test MRR", metrics(rank_dwpc)["MRR"], "C", C, "DNFR test MRR", metrics(rank_dnfr)["MRR"], flush=True)
    # sort the path table by head
    psrc = ukey[pairs] // n_t
    o = np.argsort(psrc, kind="stable")
    pairs = pairs[o]; contr = contr[o]; edges = edges[o]; w = w[o]; mp = mp[o]; psrc = psrc[o]; del o
    sb = np.searchsorted(psrc, np.arange(len(ctx["sources"]) + 1))
    s0 = float(ctx["model"].s0())
    kmap = key
    rrng = np.random.default_rng(0)

    def phi_of(c, oe):
        valid = oe >= 0; L = valid.sum(1)
        keys, inv = np.unique(oe[valid], return_inverse=True)
        phi = np.bincount(inv, weights=np.repeat((c / L)[:, None], 3, 1)[valid])
        dbl = np.bincount(inv, weights=np.repeat(c[:, None], 3, 1)[valid])
        return keys, phi, dbl

    def zlogit(S):
        S = np.minimum(S, -1e-7)
        return np.clip(np.log(-np.expm1(S)) - S, -30, 30)

    def dwpc_rescore(sp, sw, smp, keep):
        """New DWPC-LR scores of the pairs in sp after dropping the paths where keep is False."""
        ids, inv = np.unique(sp, return_inverse=True)
        M = coo_matrix((sw * keep, (inv, smp.astype(np.int64))), shape=(len(ids), n_mp)).toarray()
        return ids, zd0 + (np.arcsinh(M / mean) * beta).sum(1)

    rec_dw, rec_dn = [], []
    for qi, (s, t) in enumerate(te):
        pid = kmap.get(int(s) * n_t + int(t))
        if pid is None:
            continue
        a, b = sb[s], sb[s + 1]
        sp, sc, se, sw, smp = pairs[a:b], contr[a:b], edges[a:b], w[a:b], mp[a:b]
        own = np.where(sp == pid)[0]
        if len(own) == 0:
            continue
        # ---- DNFR: Shapley leading fact, summed-share leading fact, evidence by metapath class
        k_dn, phi_dn, dbl_dn = phi_of(sc[own].astype(np.float64), se[own])
        e_dn = keys_top = k_dn[int(np.argmax(phi_dn))]; e_dbl = k_dn[int(np.argmax(dbl_dn))]
        if rank_dnfr[qi] <= 10:
            c = sc[own].astype(np.float64); tot = c.sum(); cl = cls_mp[smp[own]]
            rec_dn.append(dict(same_shapley_summed=bool(e_dn == e_dbl),
                               share={CLS[k]: float(c[cl == k].sum() / tot) for k in range(3)},
                               share_target=float(c[tgt_mp[smp[own]]].sum() / tot)))
        if rank_dwpc[qi] > 10:
            continue
        # ---- DWPC-LR attribution
        cols = smp[own].astype(np.int64)
        xraw = np.asarray(X[pid].toarray()).ravel()
        term = beta * np.arcsinh(xraw / mean)                     # metapath terms
        cpath = term[cols] * sw[own] / np.maximum(xraw[cols], 1e-30)
        k_dw, phi_dw, _ = phi_of(cpath, se[own])
        pos_tot = float(np.clip(term, 0, None).sum())
        j = int(np.argmax(phi_dw)); e_top = k_dw[j]
        cum = np.cumsum(np.sort(np.clip(phi_dw, 0, None))[::-1]) / max(pos_tot, 1e-30)
        def cf(e):
            keep = ~(se == e).any(1)
            ids, znew = dwpc_rescore(sp, sw, smp, keep)
            zz = zd.copy(); zz[ids] = znew
            return float(ranker.ranks(zz, zd0, te[qi:qi + 1], known)[0])
        others = k_dw[k_dw != e_top]
        r_rand = cf(rrng.choice(others)) if len(others) else float(rank_dwpc[qi])
        rec_dw.append(dict(rank=float(rank_dwpc[qi]), n_edges=int(len(k_dw)),
                           top_edge_share=float(phi_dw[j] / max(pos_tot, 1e-30)),
                           k50_edges=int(np.searchsorted(cum, 0.5 - 1e-12) + 1),
                           metapath_top_share=float(term.max() / max(pos_tot, 1e-30)),
                           neg_share=float(-np.clip(term, None, 0).sum() / max(pos_tot, 1e-30)),
                           cf_top=cf(e_top), cf_random=r_rand,
                           both_top10=bool(rank_dnfr[qi] <= 10),
                           same_lead_as_dnfr=bool(e_top == e_dn) if rank_dnfr[qi] <= 10 else None))
    D = {k: np.array([r[k] for r in rec_dw]) for k in ("top_edge_share", "k50_edges", "cf_top", "cf_random", "n_edges",
                                                       "metapath_top_share", "neg_share")}
    both = [r["same_lead_as_dnfr"] for r in rec_dw if r["both_top10"]]
    sh = {k: np.array([r["share"][k] for r in rec_dn]) for k in CLS}
    st = np.array([r["share_target"] for r in rec_dn])
    summary = dict(
        dwpc=dict(test_MRR=float(metrics(rank_dwpc)["MRR"]), C=C, n_hits=len(rec_dw),
                  top_edge_share_median=float(np.median(D["top_edge_share"])),
                  k50_edges_median=float(np.median(D["k50_edges"])), n_edges_median=float(np.median(D["n_edges"])),
                  metapath_top_share_median=float(np.median(D["metapath_top_share"])),
                  neg_share_median=float(np.median(D["neg_share"])),
                  stay_top10_cf_top=float(np.mean(D["cf_top"] <= 10)), stay_top10_cf_random=float(np.mean(D["cf_random"] <= 10)),
                  n_both=len(both), same_lead_as_dnfr=float(np.mean(both)) if both else None),
        dnfr=dict(n_hits=len(rec_dn), same_shapley_summed=float(np.mean([r["same_shapley_summed"] for r in rec_dn])),
                  share_mean={k: float(sh[k].mean()) for k in CLS}, share_median={k: float(np.median(sh[k])) for k in CLS},
                  share_target_mean=float(st.mean()), share_target_median=float(np.median(st)),
                  n_target_metapaths=int(tgt_mp.sum()), n_metapaths=int(n_mp)))
    json.dump(dict(task=task, seed=seed, summary=summary, dwpc_records=rec_dw, dnfr_records=rec_dn),
              open(f"{OUT}/{task}_explain_dwpc.json", "w"), indent=0)
    print(task, json.dumps(summary))


if __name__ == "__main__":
    main(sys.argv[1])
