"""Quantitative evaluation of path explanations (seed-0 split, all test queries with at least one path).

Concision: number of paths (and metapaths) that account for 50% / 80% of the evidence -log(1-P) above the leak.
Deletion: remove the top-k paths of a pair (k = 1, 3, 10), or k random paths of the same pair, and record the new
filtered rank of the true tail with all other scores fixed. Because the score is a noisy-OR, removing a path changes
-log(1-P) by exactly its attribution; the test measures how much a correct ranking depends on the paths shown.
DWPC-LR comparison: metapath contributions coef_m * x_m of the logistic regression fitted on the same split."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import DATA, OUT
import json
import numpy as np, torch
from explain import setup
from dnfr import score_pairs
from run_task import dwpc_features
from sklearn.linear_model import LogisticRegression
from scipy.sparse import vstack, csr_matrix


def path_contribs(ctx):
    sd, model = ctx["sd"], ctx["model"]
    pairs, contr, mps = [], [], []
    with torch.no_grad():
        for ch in sd.chunks:
            lv = model.path_logvalue(ch)
            contr.append((-torch.log1p(-torch.exp(torch.clamp(lv, max=-1e-6)))).numpy())
            pairs.append(ch["pair32"].numpy()); mps.append(np.full(len(lv), ch["mp"]))
    return np.concatenate(pairs), np.concatenate(contr), np.concatenate(mps)


def k_for(sorted_c, frac):
    cs = np.cumsum(sorted_c); return int(np.searchsorted(cs, frac * cs[-1]) + 1)


def main(task, seed=0, max_q=400):
    ctx = setup(task, seed)
    z, z0 = score_pairs(ctx["model"], ctx["sd"])
    ranker, known, te = ctx["ranker"], ctx["known"], ctx["te"]
    base_rank = ranker.ranks(z, z0, te, known)
    pp, cc, mm = path_contribs(ctx)
    order = np.argsort(pp, kind="stable"); pp, cc, mm = pp[order], cc[order], mm[order]
    bounds = np.searchsorted(pp, np.arange(len(ctx["ukey"]) + 1))
    s0 = float(ctx["model"].s0())
    rng = np.random.default_rng(0)
    # DWPC-LR on the same split
    st = json.load(open(f"{OUT}/{task}.json"))
    run = [r for r in st["runs"] if r["seed"] == seed][0]
    C = run["methods"]["dwpc_lr"]["C"]
    P, sd = ctx["P"], ctx["sd"]
    keep = np.zeros(len(P["src"]), bool)
    for ch in sd.chunks:
        keep[ch["path_index"]] = True
    n_t = ctx["n_t"]; ukey = ctx["ukey"]
    key = P["src"].astype(np.int64) * n_t + P["tgt"]; pair_id = np.searchsorted(ukey, key).astype(np.int32)
    from run_task import split, positives
    X = dwpc_features(P, keep, ctx["kg_s"], len(ukey), pair_id)
    tr = ctx["tr"]
    kmap = {int(k): i for i, k in enumerate(ukey)}
    pos_rows = [kmap.get(int(s) * n_t + int(t), -1) for s, t in tr]
    lrng = np.random.default_rng(seed); tr_set = set(map(tuple, tr.tolist())); neg_rows = []
    for s in tr[:, 0]:
        for t in lrng.integers(0, n_t, 20):
            if (int(s), int(t)) not in tr_set:
                neg_rows.append(kmap.get(int(s) * n_t + int(t), -1))
    def rows(lst):
        lst = np.array(lst); ok = lst >= 0
        return X[np.where(ok, lst, 0)].multiply(ok[:, None]).tocsr()
    Xtr = vstack([rows(pos_rows), rows(neg_rows)]).tocsr(); ytr = np.r_[np.ones(len(pos_rows)), np.zeros(len(neg_rows))]
    lr = LogisticRegression(C=C, max_iter=2000, class_weight="balanced").fit(Xtr, ytr)
    zl = lr.decision_function(X); zl0 = float(lr.intercept_[0]); coef = lr.coef_[0]
    lr_rank = ranker.ranks(zl, zl0, te, known)

    qs = [qi for qi, (s, t) in enumerate(te) if (s * n_t + t) in kmap]
    if len(qs) > max_q:
        qs = sorted(rng.choice(qs, max_q, replace=False).tolist())
    rec = []
    for qi in qs:
        s, t = te[qi]; pid = kmap[int(s) * n_t + int(t)]
        c = cc[bounds[pid]:bounds[pid + 1]]; m = mm[bounds[pid]:bounds[pid + 1]]
        o = np.argsort(-c); cs = c[o]
        mp_tot = np.bincount(m, weights=c); mp_sorted = np.sort(mp_tot[mp_tot > 0])[::-1]
        r = dict(rank=float(base_rank[qi]), n_paths=int(len(c)), k50=k_for(cs, .5), k80=k_for(cs, .8),
                 top3_share=float(cs[:3].sum() / cs.sum()), n_mps=int((mp_tot > 0).sum()),
                 mp_k50=k_for(mp_sorted, .5), top_mp_share=float(mp_sorted[0] / mp_sorted.sum()))
        S = s0 - cs.sum()
        def rank_with(Snew):
            zz = z.copy(); zz[pid] = np.clip(np.log(-np.expm1(min(Snew, -1e-7))) - Snew, -30, 30)
            return float(ranker.ranks(zz, z0, te[qi:qi + 1], known)[0])
        for k in (1, 3, 10):
            r[f"del_top{k}"] = rank_with(S + cs[:k].sum())
            ridx = rng.choice(len(c), min(k, len(c)), replace=False)
            r[f"del_rand{k}"] = rank_with(S + c[ridx].sum())
        top_mp = int(np.argmax(mp_tot)); r["del_top_mp"] = rank_with(S + mp_tot[top_mp])
        # DWPC-LR: metapath contributions
        x = X[pid].toarray().ravel(); contrib = coef * x
        r["lr_rank"] = float(lr_rank[qi])
        pos = contrib[contrib > 0]
        if len(pos):
            r["lr_top_mp_share"] = float(contrib.max() / pos.sum()); r["lr_mp_k50"] = k_for(np.sort(pos)[::-1], .5)
            r["same_top_mp"] = int(np.argmax(contrib) == top_mp)
            zz = zl.copy(); zz[pid] -= contrib.max()
            r["lr_del_top_mp"] = float(ranker.ranks(zz, zl0, te[qi:qi + 1], known)[0])
        rec.append(r)
    json.dump(dict(task=task, seed=seed, n_test=len(te), n_eval=len(rec), records=rec),
              open(f"{OUT}/{task}_explain_eval.json", "w"), indent=0)
    R = {k: np.array([r[k] for r in rec if k in r]) for k in rec[0]}
    hit = R["rank"] <= 10
    print(task, "n", len(rec), "hits@10", hit.sum())
    for k in ("n_paths", "k50", "k80", "top3_share", "n_mps", "mp_k50", "top_mp_share"):
        print(f"  {k}: median {np.median(R[k]):.3g}  (hits {np.median(R[k][hit]):.3g})")
    for k in ("del_top1", "del_top3", "del_top10", "del_rand1", "del_rand3", "del_rand10", "del_top_mp"):
        print(f"  {k}: still top10 among hits {np.mean(R[k][hit] <= 10):.2f}  median rank {np.median(R[k][hit]):.1f}")


if __name__ == "__main__":
    main(sys.argv[1])
