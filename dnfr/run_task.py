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
    # revision: identifiability of the learned t-norm (initialisation and penalty)
    "dnfr_l2_0":   dict(tnorm="ss", p_mode="per_mp", degree=True, agg="noisyor", l2=0.0),
    "dnfr_pm2":    dict(tnorm="ss", p_mode="per_mp", degree=True, agg="noisyor", l2=0.0, p_init=-2.0),
    "dnfr_pm1":    dict(tnorm="ss", p_mode="per_mp", degree=True, agg="noisyor", l2=0.0, p_init=-1.0),
    "dnfr_pp05":   dict(tnorm="ss", p_mode="per_mp", degree=True, agg="noisyor", l2=0.0, p_init=0.5),
    "dnfr_l2_1e-2": dict(tnorm="ss", p_mode="per_mp", degree=True, agg="noisyor", l2=1e-2),
    "dnfr_l2_1e-1": dict(tnorm="ss", p_mode="per_mp", degree=True, agg="noisyor", l2=1e-1),
    # revision: fixed (DWPC-like) degree discount, and frozen memberships
    "dnfr_fixdeg": dict(tnorm="ss", p_mode="per_mp", degree=True, agg="noisyor", fixed_deg=0.4),
    "product_fixdeg": dict(tnorm="product", p_mode="shared", degree=True, agg="noisyor", fixed_deg=0.4),
    "dnfr_frozen": dict(tnorm="ss", p_mode="per_mp", degree=True, agg="noisyor", freeze_mem=True),
    "product_frozen": dict(tnorm="product", p_mode="shared", degree=True, agg="noisyor", freeze_mem=True),
    # second revision: sensitivity of the fixed degree discount to its value
    "product_fixdeg02": dict(tnorm="product", p_mode="shared", degree=True, agg="noisyor", fixed_deg=0.2),
    "product_fixdeg06": dict(tnorm="product", p_mode="shared", degree=True, agg="noisyor", fixed_deg=0.6),
    # third revision: DNFR 2.0 (overlap-aware aggregation, Gumbel-softmax operator choice, information-content
    # hub discount, membership temperature annealing), leave-one-out and add-one ablations
    "dnfr2":            dict(tnorm="gumbel", p_mode="per_mp", degree=True, agg="overlap", feat="ic", anneal=True),
    "dnfr2_no_overlap": dict(tnorm="gumbel", p_mode="per_mp", degree=True, agg="noisyor", feat="ic", anneal=True),
    "dnfr2_no_gumbel":  dict(tnorm="ss", p_mode="per_mp", degree=True, agg="overlap", feat="ic", anneal=True),
    "dnfr2_no_ic":      dict(tnorm="gumbel", p_mode="per_mp", degree=True, agg="overlap", feat="degree", anneal=True),
    "dnfr2_no_anneal":  dict(tnorm="gumbel", p_mode="per_mp", degree=True, agg="overlap", feat="ic", anneal=False),
    "dnfr_overlap":     dict(tnorm="ss", p_mode="per_mp", degree=True, agg="overlap"),
    "dnfr_gumbel":      dict(tnorm="gumbel", p_mode="per_mp", degree=True, agg="noisyor"),
    "dnfr_ic":          dict(tnorm="ss", p_mode="per_mp", degree=True, agg="noisyor", feat="ic"),
    "dnfr_ic_sd":       dict(tnorm="ss", p_mode="per_mp", degree=True, agg="noisyor", feat="ic_sd"),
    "dnfr_anneal":      dict(tnorm="ss", p_mode="per_mp", degree=True, agg="noisyor", anneal=True),
    "product_overlap":  dict(tnorm="product", p_mode="shared", degree=True, agg="overlap"),
    # restarts of the Gumbel-softmax choice (same split, other random streams for initial noise)
    "dnfr2_r1":         dict(tnorm="gumbel", p_mode="per_mp", degree=True, agg="overlap", feat="ic", anneal=True, restart=1),
    "dnfr2_r2":         dict(tnorm="gumbel", p_mode="per_mp", degree=True, agg="overlap", feat="ic", anneal=True, restart=2),
    # round-5 M3: the fixed Hamacher product from other initial memberships (degree discount b = c = 0.1 or 0.6)
    "hamacher_b01":     dict(tnorm="hamacher", p_mode="shared", degree=True, agg="noisyor", b_init=0.1),
    "hamacher_b06":     dict(tnorm="hamacher", p_mode="shared", degree=True, agg="noisyor", b_init=0.6),
    "dnfr_gumbel_r1":   dict(tnorm="gumbel", p_mode="per_mp", degree=True, agg="noisyor", restart=1),
    "dnfr_gumbel_r2":   dict(tnorm="gumbel", p_mode="per_mp", degree=True, agg="noisyor", restart=2),
    # third revision: t-norm parameter fixed and selected on validation data (grid; p = 0 is the product t-norm)
    "fixp_m4":  dict(tnorm="ss", p_mode="shared", degree=True, agg="noisyor", p_init=-4.0, learn_p=False),
    "fixp_m2":  dict(tnorm="ss", p_mode="shared", degree=True, agg="noisyor", p_init=-2.0, learn_p=False),
    "fixp_m1":  dict(tnorm="ss", p_mode="shared", degree=True, agg="noisyor", p_init=-1.0, learn_p=False),
    "fixp_m05": dict(tnorm="ss", p_mode="shared", degree=True, agg="noisyor", p_init=-0.5, learn_p=False),
    "fixp_05":  dict(tnorm="ss", p_mode="shared", degree=True, agg="noisyor", p_init=0.5, learn_p=False),
    # third revision: unbounded parameterisation of p (p = theta), no penalty, three initial values
    "dnfr_free_pm2":  dict(tnorm="ss", p_mode="per_mp", degree=True, agg="noisyor", l2=0.0, p_init=-2.0, p_param="free"),
    "dnfr_free":      dict(tnorm="ss", p_mode="per_mp", degree=True, agg="noisyor", l2=0.0, p_init=-0.1, p_param="free"),
    "dnfr_free_pp05": dict(tnorm="ss", p_mode="per_mp", degree=True, agg="noisyor", l2=0.0, p_init=0.5, p_param="free"),
}
FIXP = {"fixp_m4": -4.0, "fixp_m2": -2.0, "fixp_m1": -1.0, "fixp_m05": -0.5, "product": 0.0, "fixp_05": 0.5}
L2_DEFAULT = 1e-3



def load_kg(name):
    return pickle.load(open(f"{DATA}/{name}.pkl", "rb"))


def positives(kg, spec):
    d = kg.rels[spec["rel"]]
    h, t = (d["t"], d["h"]) if spec["rev"] else (d["h"], d["t"])
    k = np.unique(h.astype(np.int64) * 10**7 + t)
    return np.stack([k // 10**7, k % 10**7], 1)


def split_by_tail(pos, seed, frac=(0.8, 0.1, 0.1)):
    """Tail-centric split: every positive of a held-out tail (e.g. a disease) is held out together."""
    rng = np.random.default_rng(seed)
    tails = rng.permutation(np.unique(pos[:, 1]))
    a, b = int(frac[0] * len(tails)), int((frac[0] + frac[1]) * len(tails))
    grp = {int(t): (0 if i < a else 1 if i < b else 2) for i, t in enumerate(tails)}
    g = np.array([grp[int(t)] for t in pos[:, 1]])
    return pos[g == 0], pos[g == 1], pos[g == 2]


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


def run(task, seeds, variants, budget, epochs, do_baselines=True, sparsity=1.0, tag="", split_mode="random",
        which=("dwpc_lr", "svd", "softimpute", "popularity", "prior", "dwpc_lr_full")):
    spec = TASKS[task]
    kg = load_kg(spec["kg"])
    protect_target = sparsity < 1.0 and bool(os.environ.get("DNFR_PROTECT_TARGET"))  # thin only non-target relations (minor 9)
    if sparsity < 1.0:
        kg = kg.subsample(sparsity, np.random.default_rng(12345), protect=(spec["rel"],) if protect_target else ())
    pos_full = positives(kg if sparsity == 1.0 else load_kg(spec["kg"]), spec)
    # with sparsity < 1 the target relation is also thinned; held-out positives come from the full graph
    sources = np.unique(pos_full[:, 0])
    src_loc = {s: i for i, s in enumerate(sources)}
    n_t = kg.n(spec["t"])
    cache = f"{DATA}/paths_{task}{'_sp%.2f' % sparsity if sparsity < 1 else ''}{'_pt' if protect_target else ''}.pkl"
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
        tr_q, va_q, te_q = (split_by_tail if split_mode == "tail" else split)(posl, seed)
        if sparsity < 1.0:
            # training positives must exist in the thinned graph
            present = set(map(tuple, np.stack([[src_loc[s] for s in positives(kg, spec)[:, 0]], positives(kg, spec)[:, 1]], 1).tolist()))
            tr_q = np.array([q for q in tr_q.tolist() if tuple(q) in present])
        removed_keys = np.concatenate([sources[va_q[:, 0]].astype(np.int64) * 10**7 + va_q[:, 1],
                                       sources[te_q[:, 0]].astype(np.int64) * 10**7 + te_q[:, 1]])
        removed = {(spec["s"], spec["t"]): removed_keys}
        kg_split = kg.copy_without(removed)
        mask = drop_paths_using(P, kg, removed)
        if os.environ.get("DNFR_DROP_TARGET_MPS"):   # reviewer M4: no metapath that uses the target relation
            bad_mp = [i for i, m in enumerate(P["mps"]) if any(tr[3] == spec["rel"] for tr in m)]
            mask &= ~np.isin(P["mp"], bad_mp)
            print(f"dropped {len(bad_mp)} metapaths that contain {spec['rel']}; {int(mask.sum())} paths kept", flush=True)
        if os.environ.get("DNFR_DROP_RELS"):   # round-5 M4: also drop metapaths through related relations (e.g. CpD)
            rels = set(os.environ["DNFR_DROP_RELS"].split(","))
            bad_mp = [i for i, m in enumerate(P["mps"]) if any(tr[3] in rels for tr in m)]
            mask &= ~np.isin(P["mp"], bad_mp)
            print(f"dropped {len(bad_mp)} metapaths that contain any of {sorted(rels)}; {int(mask.sum())} paths kept", flush=True)
        SD = {}
        def get_sd(overlap, feat):   # one SplitData per (aggregation, feature) setting, built when first needed
            k = (overlap, feat)
            if k not in SD:
                SD.clear(); import gc; gc.collect()
                SD[k] = SplitData(P, kg_split, mask, pair_id, n_pairs, trav_index, overlap=overlap, feat=feat)
            return SD[k]
        run_res = dict(seed=seed, n_train=len(tr_q), n_valid=len(va_q), n_test=len(te_q),
                       paths_kept=int(mask.sum()), methods={})
        if seed in PREV_SEED:  # keep methods already computed for this split
            for k in ("methods", "ranks", "dnfr_params", "dnfr_test_ranks", "p_learned", "w_learned"):
                if k in PREV_SEED[seed]:
                    run_res[k] = PREV_SEED[seed][k]
        test_tq = te_q
        order = sorted(variants, key=lambda v: (VARIANTS[v].get("agg") == "overlap", VARIANTS[v].get("feat", "degree")))
        for v in order:
            cfg = dict(VARIANTS[v]); l2 = cfg.pop("l2", L2_DEFAULT); restart = cfg.pop("restart", 0)
            torch.manual_seed(seed + 1000 * restart)
            sd = get_sd(cfg.get("agg") == "overlap", cfg.get("feat", "degree"))
            model = DNFR(len(trav), len(P["mps"]), **cfg)
            t0 = time.time()
            print(f"[{task}{tag} seed {seed}] {v}", flush=True)
            bestv, hist = train(model, sd, tr_q, None, ranker, va_q, known, len(sources), epochs=epochs, l2=l2)
            z, z0 = score_pairs(model, sd)
            r = ranker.ranks(z, z0, test_tq, known)
            m = metrics(r); m["valid_MRR"] = bestv; m["time_s"] = time.time() - t0; m["epochs"] = hist[-1][0]
            m["best_epoch"] = int(max(hist, key=lambda x: x[2])[0])
            used_tr = sorted({t for ch in sd.chunks for (t, _, _, _) in ch["hops"]})
            m["n_params"] = int(sum(int(p_.requires_grad) * (len(used_tr) if p_.dim() and p_.shape[0] == len(trav) else p_.numel())
                                    for p_ in model.parameters()))
            print(f"   test {m}", flush=True)
            run_res["methods"][v] = m
            run_res.setdefault("ranks", {})[v] = r.tolist()
            if cfg.get("tnorm") == "gumbel":
                run_res.setdefault("ops_chosen", {})[v] = model.chosen_ops()
                run_res.setdefault("op_probs", {})[v] = torch.softmax(model.op_logit, 1).detach().numpy().round(4).tolist()
                run_res.setdefault("ops_trace", {})[v] = [h[3] for h in hist if len(h) > 3]
            if cfg.get("agg") == "overlap":
                run_res.setdefault("beta", {})[v] = float(model.beta())
            if cfg.get("anneal"):
                run_res.setdefault("tau_best", {})[v] = float(model.tau)
            if cfg.get("tnorm") == "ss":
                run_res.setdefault("p_learned", {})[v] = model.p().detach().numpy().tolist()
                run_res.setdefault("w_learned", {})[v] = torch.sigmoid(model.w_raw).detach().numpy().tolist()
            if v in ("dnfr2",) and seed == seeds[0]:
                torch.save(model.state_dict(), f"{OUT}/{task}{tag}_dnfr2_seed{seed}.pt")
                np.save(f"{OUT}/{task}{tag}_test_q_seed{seed}.npy", te_q)
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
            SD.clear()
            import gc; gc.collect()
            RANKS.clear()
            run_res["methods"].update(baselines(P, mask, kg_split, n_pairs, pair_id, pair_src, pair_tgt,
                                                ranker, tr_q, va_q, te_q, known, len(sources), n_t, seed, set(which)))
            run_res.setdefault("ranks", {}).update(dict(RANKS))
        run_res["test_queries"] = te_q.tolist()
        results["runs"].append(run_res)
        json.dump(results, open(out_file, "w"), indent=1)
    return results


RANKS = {}


def baselines(P, mask, kg_split, n_pairs, pair_id, pair_src, pair_tgt, ranker, tr_q, va_q, te_q, known, n_src, n_t, seed,
              which=("dwpc_lr", "svd", "softimpute", "popularity")):
    out = {}
    rng = np.random.default_rng(seed)
    if "popularity" in which:  # rank tails by their number of training positives
        cnt = np.bincount(tr_q[:, 1], minlength=n_t).astype(np.float64)
        Z = np.broadcast_to(cnt, (n_src, n_t))
        rr = _dense_ranks(Z, te_q, known); RANKS["popularity"] = rr.tolist()
        out["popularity"] = metrics(rr); print("   popularity", out["popularity"], flush=True)
    if "prior" in which or "dwpc_lr_full" in which:  # degree-preserving permutation prior (Zietz et al., 2024)
        t0 = time.time()
        prior = edge_prior(tr_q, n_src, n_t, seed)
        rr = _dense_ranks(prior, te_q, known); RANKS["prior"] = rr.tolist()
        out["prior"] = metrics(rr); out["prior"]["time_s"] = time.time() - t0
        print("   prior", out["prior"], flush=True)
    if "dwpc_lr" not in which and "dwpc_lr_full" not in which and "svd" not in which and "softimpute" not in which:
        return out
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
    from scipy.sparse import vstack, hstack, csr_matrix
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
    m["n_params"] = int(X.shape[1] + 1)
    print("   dwpc_lr", m, flush=True)
    if "dwpc_lr_full" in which:
        # Rephetio-style model: DWPC features plus the degrees of source and target in the training target
        # relation and the logit of the permutation prior; scored densely over all (source, tail) pairs.
        t0 = time.time()
        dsrc = np.log1p(np.bincount(tr_q[:, 0], minlength=n_src).astype(np.float64))
        dtgt = np.log1p(np.bincount(tr_q[:, 1], minlength=n_t).astype(np.float64))
        lp = np.log(np.clip(prior, 1e-3, 1 - 1e-3) / (1 - np.clip(prior, 1e-3, 1 - 1e-3)))
        def extra(ss, tt):
            return np.stack([dsrc[ss], dtgt[tt], lp[ss, tt]], 1)
        def full_rows(pairs_st, rowidx):
            M = rows(rowidx)
            return hstack([M, csr_matrix(extra(pairs_st[:, 0], pairs_st[:, 1]))]).tocsr()
        neg_st = []
        lrng = np.random.default_rng(seed)
        for s in srcs:
            for t in lrng.integers(0, n_t, 20):
                if (int(s), int(t)) not in tr_set:
                    neg_st.append((int(s), int(t)))
        neg_st = np.array(neg_st)
        neg_rows2 = [key.get((int(s), int(t)), -1) for s, t in neg_st]
        Xtr2 = vstack([full_rows(tr_q, pos_rows), full_rows(neg_st, neg_rows2)]).tocsr()
        ytr2 = np.r_[np.ones(len(tr_q)), np.zeros(len(neg_st))]
        Xd = X.toarray() if X.shape[0] * X.shape[1] < 2e8 else None
        def dense_scores(lr):
            Z = np.zeros((n_src, n_t))
            Z += lr.intercept_[0] + lr.coef_[0][-3] * dsrc[:, None] + lr.coef_[0][-2] * dtgt[None, :] + lr.coef_[0][-1] * lp
            Z[pair_src, pair_tgt] += X @ lr.coef_[0][:-3]
            return Z
        best2 = None
        for C in [0.01, 0.1, 1.0, 10.0]:
            lr2 = LogisticRegression(C=C, max_iter=3000, class_weight="balanced").fit(Xtr2, ytr2)
            Z = dense_scores(lr2)
            mv = metrics(_dense_ranks(Z, va_q, known))["MRR"]
            if best2 is None or mv > best2[0]:
                best2 = (mv, C, Z)
        rr = _dense_ranks(best2[2], te_q, known); RANKS["dwpc_lr_full"] = rr.tolist()
        m2 = metrics(rr); m2["valid_MRR"] = best2[0]; m2["C"] = best2[1]; m2["time_s"] = time.time() - t0
        m2["n_params"] = int(X.shape[1] + 4)
        out["dwpc_lr_full"] = m2; print("   dwpc_lr_full", m2, flush=True)
    if "svd" not in which and "softimpute" not in which:
        return out
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


def edge_prior(tr_q, n_src, n_t, seed, n_perm=100, mult=10):
    """Degree-preserving permutation prior of the training target relation (XSwap; Hanhijarvi et al. 2009,
    as used by Himmelstein et al. 2017 and Zietz et al. 2024): the share of n_perm permuted networks in which a
    (source, tail) pair is an edge. Each permutation applies mult * n_edges attempted swaps to the original."""
    rng = np.random.default_rng(1000 + seed)
    E0 = tr_q[:, :2].astype(np.int64)
    m = len(E0)
    count = np.zeros((n_src, n_t))
    S0, T0 = E0[:, 0].tolist(), E0[:, 1].tolist()
    for _ in range(n_perm):
        Sx, Tx = list(S0), list(T0)
        present = set(s_ * n_t + t_ for s_, t_ in zip(Sx, Tx))
        ii = rng.integers(0, m, (mult * m, 2)).tolist()
        for a, b in ii:
            s1, t1, s2, t2 = Sx[a], Tx[a], Sx[b], Tx[b]
            if s1 == s2 or t1 == t2:
                continue
            k1, k2 = s1 * n_t + t2, s2 * n_t + t1
            if k1 in present or k2 in present:
                continue
            present.discard(s1 * n_t + t1); present.discard(s2 * n_t + t2)
            present.add(k1); present.add(k2)
            Tx[a], Tx[b] = t2, t1
        count[Sx, Tx] += 1
    return count / n_perm


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
    ap.add_argument("--split", default="random", choices=["random", "tail"])
    ap.add_argument("--baselines", default="dwpc_lr,svd,softimpute,popularity,prior,dwpc_lr_full")
    a = ap.parse_args()
    run(a.task, [int(s) for s in a.seeds.split(",")], [v for v in a.variants.split(",") if v], a.budget, a.epochs,
        not a.no_baselines, a.sparsity, a.tag, a.split, tuple(a.baselines.split(",")))
