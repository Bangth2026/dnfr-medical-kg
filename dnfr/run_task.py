"""Run DNFR, fixed-t-norm variants and path/matrix baselines on one task over several splits."""
import os, sys
from config import DATA, OUT
import argparse, json, os, pickle, time, sys
import numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from kg import *
from paths import enumerate_paths, drop_paths_using
from dnfr import DNFR, SplitData, Ranker, metrics, train, score_pairs
from sklearn.linear_model import LogisticRegression

TASKS = {
    "het_CtD": dict(kg="het", rel="CtD", rev=False, s="C", t="D"),
    "het_DpS": dict(kg="het", rel="DpS", rev=False, s="D", t="S"),
    "pkg_ind": dict(kg="pkg", rel="indication", rev=True, s="drug", t="disease"),
    "hkg_DpS": dict(kg="hkg", rel="DpS", rev=False, s="D", t="S"),
}
VARIANTS = {
    "dnfr":        dict(tnorm="ss", p_mode="per_mp", degree=True, agg="noisyor"),
    "dnfr_shared": dict(tnorm="ss", p_mode="shared", degree=True, agg="noisyor"),
    "product":     dict(tnorm="product", p_mode="shared", degree=True, agg="noisyor"),
    "hamacher":    dict(tnorm="hamacher", p_mode="shared", degree=True, agg="noisyor"),
    "min":         dict(tnorm="min", p_mode="shared", degree=True, agg="noisyor"),
    "luk":         dict(tnorm="luk", p_mode="shared", degree=True, agg="noisyor"),
    "dnfr_nodeg":  dict(tnorm="ss", p_mode="per_mp", degree=False, agg="noisyor"),
    "dnfr_max":    dict(tnorm="ss", p_mode="per_mp", degree=True, agg="max"),
    "dnfr_binary": dict(tnorm="ss", p_mode="per_mp", degree=True, agg="noisyor", use_weight=False),
}



def load_kg(name):
    return pickle.load(open(f"{DATA}/{name}.pkl", "rb"))


def positives(kg, spec):
    d = kg.rels[spec["rel"]]
    h, t = (d["t"], d["h"]) if spec["rev"] else (d["h"], d["t"])
    k = np.unique(h.astype(np.int64) * 10**7 + t)
    return np.stack([k // 10**7, k % 10**7], 1)


def split(pos, seed, frac=(0.8, 0.1, 0.1)):
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(pos))
    a, b = int(frac[0] * len(pos)), int((frac[0] + frac[1]) * len(pos))
    return pos[idx[:a]], pos[idx[a:b]], pos[idx[b:]]


def dwpc_features(P, sd_mask, kg_split, n_pairs, pair_id, damp=0.4):
    """Degree-weighted path counts per (pair, metapath), Himmelstein et al. (2017)."""
    from scipy.sparse import coo_matrix
    rows, cols, vals = [], [], []
    deg = {}
    for mi, mp in enumerate(P["mps"]):
        sel = np.where((P["mp"] == mi) & sd_mask)[0]
        if len(sel) == 0:
            continue
        s = P["sources"][P["src"][sel]]
        nodes = [s, P["k1"][sel]] + ([P["tgt"][sel]] if len(mp) == 2 else [P["k2"][sel], P["tgt"][sel]])
        w = np.ones(len(sel))
        for h, tr in enumerate(mp):
            if tr not in deg:
                A = kg_split.adj(tr)
                deg[tr] = (np.asarray(A.sum(1)).ravel(), np.asarray(A.sum(0)).ravel())
            do, di = deg[tr]
            w *= (np.maximum(do[nodes[h]], 1) * np.maximum(di[nodes[h + 1]], 1)) ** (-damp)
        rows.append(pair_id[sel]); cols.append(np.full(len(sel), mi)); vals.append(w)
    X = coo_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))),
                   shape=(n_pairs, len(P["mps"]))).tocsr()
    mean = np.asarray(X.sum(0)).ravel() / np.maximum(np.diff(X.tocsc().indptr), 1)
    X = X.multiply(1 / np.maximum(mean, 1e-12)).tocsr()
    X.data = np.arcsinh(X.data)
    return X


def run(task, seeds, variants, budget, epochs, do_baselines=True, sparsity=1.0, tag=""):
    spec = TASKS[task]
    kg = load_kg(spec["kg"])
    if sparsity < 1.0:
        kg = kg.subsample(sparsity, np.random.default_rng(12345), protect=())
    pos_full = positives(kg if sparsity == 1.0 else load_kg(spec["kg"]), spec)
    # with sparsity < 1 the target relation is also thinned; held-out positives come from the full graph
    sources = np.unique(pos_full[:, 0])
    src_loc = {s: i for i, s in enumerate(sources)}
    n_t = kg.n(spec["t"])
    cache = f"{DATA}/paths_{task}{'_sp%.2f' % sparsity if sparsity < 1 else ''}.pkl"
    if os.path.exists(cache):
        P = pickle.load(open(cache, "rb"))
    else:
        t0 = time.time()
        P = enumerate_paths(kg, spec["s"], spec["t"], sources, budget=budget)
        print(f"enumerated {len(P['src'])} paths over {len(P['mps'])} metapaths in {time.time()-t0:.0f}s", flush=True)
        pickle.dump(P, open(cache, "wb"), protocol=4)
    key = P["src"].astype(np.int64) * n_t + P["tgt"]
    ukey, pair_id = np.unique(key, return_inverse=True)
    pair_id = pair_id.astype(np.int32); del key
    pair_src, pair_tgt = (ukey // n_t).astype(np.int64), (ukey % n_t).astype(np.int64)
    n_pairs = len(ukey)
    trav = kg.traversals(); trav_index = {tr: i for i, tr in enumerate(trav)}
    ranker = Ranker(pair_src, pair_tgt, len(sources), n_t)
    posl = np.stack([[src_loc[s] for s in pos_full[:, 0]], pos_full[:, 1]], 1)
    known = {}
    for s, t in posl:
        known.setdefault(int(s), set()).add(int(t))
    results = dict(task=task, sparsity=sparsity, n_pos=len(posl), n_sources=len(sources), n_tails=n_t,
                   n_paths=int(len(P["src"])), n_mps=len(P["mps"]), mp_names=P["mp_names"],
                   n_pairs=int(n_pairs), runs=[])
    os.makedirs(OUT, exist_ok=True)
    out_file = f"{OUT}/{task}{tag}.json"
    if os.path.exists(out_file):  # merge with runs of other seeds from earlier invocations
        prev = json.load(open(out_file))
        results["runs"] = [r for r in prev.get("runs", []) if r["seed"] not in seeds and "ranks" in r]
        PREV_SEED = {r["seed"]: r for r in prev.get("runs", []) if r["seed"] in seeds and "ranks" in r}
    else:
        PREV_SEED = {}
    for seed in seeds:
        tr_q, va_q, te_q = split(posl, seed)
        if sparsity < 1.0:
            # training positives must exist in the thinned graph
            present = set(map(tuple, np.stack([[src_loc[s] for s in positives(kg, spec)[:, 0]], positives(kg, spec)[:, 1]], 1).tolist()))
            tr_q = np.array([q for q in tr_q.tolist() if tuple(q) in present])
        removed_keys = np.concatenate([sources[va_q[:, 0]].astype(np.int64) * 10**7 + va_q[:, 1],
                                       sources[te_q[:, 0]].astype(np.int64) * 10**7 + te_q[:, 1]])
        removed = {(spec["s"], spec["t"]): removed_keys}
        kg_split = kg.copy_without(removed)
        mask = drop_paths_using(P, kg, removed)
        sd = SplitData(P, kg_split, mask, pair_id, n_pairs, trav_index) if variants else None
        run_res = dict(seed=seed, n_train=len(tr_q), n_valid=len(va_q), n_test=len(te_q),
                       paths_kept=int(mask.sum()), methods={})
        if seed in PREV_SEED:  # keep methods already computed for this split
            for k in ("methods", "ranks", "dnfr_params", "dnfr_test_ranks"):
                if k in PREV_SEED[seed]:
                    run_res[k] = PREV_SEED[seed][k]
        test_tq = te_q
        for v in variants:
            torch.manual_seed(seed)
            cfg = VARIANTS[v]
            model = DNFR(len(trav), len(P["mps"]), **cfg)
            t0 = time.time()
            print(f"[{task}{tag} seed {seed}] {v}", flush=True)
            bestv, hist = train(model, sd, tr_q, None, ranker, va_q, known, len(sources), epochs=epochs)
            z, z0 = score_pairs(model, sd)
            r = ranker.ranks(z, z0, test_tq, known)
            m = metrics(r); m["valid_MRR"] = bestv; m["time_s"] = time.time() - t0; m["epochs"] = hist[-1][0]
            print(f"   test {m}", flush=True)
            run_res["methods"][v] = m
            run_res.setdefault("ranks", {})[v] = r.tolist()
            if v == "dnfr":
                st = {k: v_.detach().numpy().tolist() for k, v_ in model.state_dict().items()}
                st["p"] = model.p().detach().numpy().tolist()
                run_res["dnfr_params"] = st
                run_res["dnfr_test_ranks"] = r.tolist()
                if seed == seeds[0]:
                    torch.save(model.state_dict(), f"{OUT}/{task}{tag}_dnfr_seed{seed}.pt")
                    np.save(f"{OUT}/{task}{tag}_test_q_seed{seed}.npy", te_q)
            json.dump(results | {"runs": results["runs"] + [run_res]}, open(out_file, "w"), indent=1)
        if do_baselines:
            del sd
            import gc; gc.collect()
            RANKS.clear()
            run_res["methods"].update(baselines(P, mask, kg_split, n_pairs, pair_id, pair_src, pair_tgt,
                                                ranker, tr_q, va_q, te_q, known, len(sources), n_t, seed))
            run_res.setdefault("ranks", {}).update(dict(RANKS))
        run_res["test_queries"] = te_q.tolist()
        results["runs"].append(run_res)
        json.dump(results, open(out_file, "w"), indent=1)
    return results


RANKS = {}


def baselines(P, mask, kg_split, n_pairs, pair_id, pair_src, pair_tgt, ranker, tr_q, va_q, te_q, known, n_src, n_t, seed):
    out = {}
    rng = np.random.default_rng(seed)
    # --- DWPC + logistic regression (Rephetio) ---
    t0 = time.time()
    X = dwpc_features(P, mask, kg_split, n_pairs, pair_id)
    key = {(int(s), int(t)): i for i, (s, t) in enumerate(zip(pair_src, pair_tgt))}
    tr_set = set(map(tuple, tr_q.tolist()))
    pos_rows = [key.get((int(s), int(t)), -1) for s, t in tr_q]
    neg_rows = []
    srcs = tr_q[:, 0]
    for s in srcs:
        for t in rng.integers(0, n_t, 20):
            if (int(s), int(t)) not in tr_set:
                neg_rows.append(key.get((int(s), int(t)), -1))
    from scipy.sparse import vstack, csr_matrix
    def rows(lst):
        lst = np.array(lst); Z = csr_matrix((len(lst), X.shape[1]))
        ok = lst >= 0
        M = X[np.where(ok, lst, 0)]
        M = M.multiply(ok[:, None]).tocsr()
        return M
    Xtr = vstack([rows(pos_rows), rows(neg_rows)]).tocsr()
    ytr = np.r_[np.ones(len(pos_rows)), np.zeros(len(neg_rows))]
    best = None
    for C in [0.01, 0.1, 1.0, 10.0]:
        lr = LogisticRegression(C=C, max_iter=2000, class_weight="balanced").fit(Xtr, ytr)
        z = lr.decision_function(X); z0 = float(lr.intercept_[0])
        mv = metrics(ranker.ranks(z, z0, va_q, known))["MRR"]
        if best is None or mv > best[0]:
            best = (mv, C, z, z0)
    m = metrics(ranker.ranks(best[2], best[3], te_q, known)); m["valid_MRR"] = best[0]; m["C"] = best[1]
    m["time_s"] = time.time() - t0
    out["dwpc_lr"] = m; RANKS["dwpc_lr"] = ranker.ranks(best[2], best[3], te_q, known).tolist()
    print("   dwpc_lr", m, flush=True)
    # --- matrix completion on the training relation only ---
    R = np.zeros((n_src, n_t), np.float32)
    R[tr_q[:, 0], tr_q[:, 1]] = 1.0
    U, sv, Vt = np.linalg.svd(R, full_matrices=False) if n_t * n_src < 5e7 else _rsvd(R, 100, seed)
    allq_rank = lambda Z, q: _dense_ranks(Z, q, known)
    best = None
    for k in [1, 2, 3, 5, 10, 20, 50, 100]:
        if k > len(sv): continue
        Z = (U[:, :k] * sv[:k]) @ Vt[:k]
        mv = metrics(allq_rank(Z, va_q))["MRR"]
        if best is None or mv > best[0]: best = (mv, k)
    k = best[1]; Z = (U[:, :k] * sv[:k]) @ Vt[:k]
    rr = allq_rank(Z, te_q); RANKS["svd"] = rr.tolist()
    m = metrics(rr); m["valid_MRR"] = best[0]; m["k"] = k
    out["svd"] = m; print("   svd", m, flush=True)
    Z3 = (U[:, :3] * sv[:3]) @ Vt[:3]
    rr = allq_rank(Z3, te_q); RANKS["svd_rank3"] = rr.tolist()
    m = metrics(rr); out["svd_rank3"] = m; print("   svd_rank3", m, flush=True)
    best = None
    for frac in [0.01, 0.05, 0.1, 0.2, 0.4]:
        lam = frac * sv[0]
        s2 = np.maximum(sv - lam, 0)
        Z = (U * s2) @ Vt
        mv = metrics(allq_rank(Z, va_q))["MRR"]
        if best is None or mv > best[0]: best = (mv, frac)
    s2 = np.maximum(sv - best[1] * sv[0], 0); Z = (U * s2) @ Vt
    rr = allq_rank(Z, te_q); RANKS["softimpute"] = rr.tolist()
    m = metrics(rr); m["valid_MRR"] = best[0]; m["lambda_frac"] = best[1]
    out["softimpute"] = m; print("   softimpute", m, flush=True)
    return out


def _rsvd(R, k, seed):
    from sklearn.utils.extmath import randomized_svd
    U, s, Vt = randomized_svd(R, k, random_state=seed)
    return U, s, Vt


def _dense_ranks(Z, q, known):
    out = np.empty(len(q))
    for i, (s, t) in enumerate(q):
        row = Z[s].copy()
        filt = list(known.get(int(s), set()) - {int(t)})
        row[filt] = -np.inf
        zt = row[t]
        gt = (row > zt).sum(); eq = (row == zt).sum() - 1
        out[i] = 1 + gt + eq / 2.0
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("task"); ap.add_argument("--seeds", default="0,1,2,3,4")
    ap.add_argument("--variants", default=",".join(VARIANTS)); ap.add_argument("--budget", type=float, default=5e6)
    ap.add_argument("--epochs", type=int, default=400); ap.add_argument("--no_baselines", action="store_true")
    ap.add_argument("--sparsity", type=float, default=1.0); ap.add_argument("--tag", default="")
    a = ap.parse_args()
    run(a.task, [int(s) for s in a.seeds.split(",")], [v for v in a.variants.split(",") if v], a.budget, a.epochs,
        not a.no_baselines, a.sparsity, a.tag)
