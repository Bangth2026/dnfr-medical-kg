"""Additional controls for the path explanations (second revision; seed-0 split, test pairs ranked in the top 10).

1. Share-matched random deletion: remove random paths of the pair, in random order, until their total share of
   the evidence reaches that of the top-k paths (k = 1, 3, 10).
2. Rank against share removed: remove the leading paths, or random paths, until a share f of the evidence is gone
   (f = 0.1, ..., 0.9), and record whether the pair stays in the top 10.
3. Overlap: share of pairs whose three leading paths share at least one edge (fact), and share of the evidence of
   the three leading paths that lies on paths sharing an edge with another of them.
4. Edge-level counterfactual: remove one fact (edge) from the graph, namely the edge that carries the largest
   total share of the pair's evidence, or a random edge used by the pair's paths, drop every path from the query's
   source that uses it (so competing tails are affected too), and re-rank with all parameters and degrees fixed.

Output: results/<task>_explain_eval2.json
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import DATA, OUT
import json
import numpy as np, torch
from explain import setup
from dnfr import score_pairs

FRACS = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]


def path_table(ctx):
    """Per kept path: pair id, contribution -log(1 - w T), and the edge keys of its hops."""
    sd, model, P, kg = ctx["sd"], ctx["model"], ctx["P"], ctx["kg"]
    bases = sorted(kg.rels); bid = {b: i for i, b in enumerate(bases)}
    N = 1 << 16
    pairs, contr, pidx = [], [], []
    with torch.no_grad():
        for ch in sd.chunks:
            lv = model.path_logvalue(ch)
            contr.append((-torch.log1p(-torch.exp(torch.clamp(lv, max=-1e-6)))).numpy())
            pairs.append(ch["pair32"].numpy()); pidx.append(ch["path_index"])
    pairs, contr, pidx = np.concatenate(pairs), np.concatenate(contr), np.concatenate(pidx)
    mp = P["mp"][pidx]
    s = P["sources"][P["src"][pidx]]
    edges = np.full((len(pidx), 3), -1, np.int64)
    for mi, m in enumerate(P["mps"]):
        sel = np.where(mp == mi)[0]
        if len(sel) == 0:
            continue
        ii = pidx[sel]
        nodes = [s[sel], P["k1"][ii]] + ([P["tgt"][ii]] if len(m) == 2 else [P["k2"][ii], P["tgt"][ii]])
        for h, tr in enumerate(m):
            name, a, b, base, inv = tr
            u, v = (nodes[h + 1], nodes[h]) if inv else (nodes[h], nodes[h + 1])
            if kg.rels[base].get("sym", False) or kg.rels[base]["src"] == kg.rels[base]["dst"]:
                u, v = np.minimum(u, v), np.maximum(u, v)
            edges[sel, h] = (bid[base] * N + u.astype(np.int64)) * N + v.astype(np.int64)
    return pairs, contr, edges


def main(task, seed=0):
    ctx = setup(task, seed)
    z, z0 = score_pairs(ctx["model"], ctx["sd"])
    ranker, known, te, n_t, ukey = ctx["ranker"], ctx["known"], ctx["te"], ctx["n_t"], ctx["ukey"]
    base_rank = ranker.ranks(z, z0, te, known)
    pairs, contr, edges = path_table(ctx)
    psrc = ukey[pairs] // n_t
    o = np.argsort(psrc, kind="stable"); pairs, contr, edges, psrc = pairs[o], contr[o], edges[o], psrc[o]
    sb = np.searchsorted(psrc, np.arange(len(ctx["sources"]) + 1))
    s0 = float(ctx["model"].s0())
    kmap = {int(k): i for i, k in enumerate(ukey)}
    rng = np.random.default_rng(0)

    def zlogit(S):
        S = np.minimum(S, -1e-7)
        return np.clip(np.log(-np.expm1(S)) - S, -30, 30)

    def rank_after(qi, pid, Snew_pair=None, src_paths=None, drop=None):
        zz = z.copy()
        if Snew_pair is not None:
            zz[pid] = zlogit(Snew_pair)
        else:  # recompute every pair of the source without the dropped paths
            pp, cc = src_paths
            keep = ~drop
            ids, inv = np.unique(pp, return_inverse=True)
            S = s0 - np.bincount(inv, weights=cc * keep, minlength=len(ids))
            zz[ids] = zlogit(S)
        return float(ranker.ranks(zz, z0, te[qi:qi + 1], known)[0])

    rec = []
    for qi, (s, t) in enumerate(te):
        if base_rank[qi] > 10 or (int(s) * n_t + int(t)) not in kmap:
            continue
        pid = kmap[int(s) * n_t + int(t)]
        a, b = sb[s], sb[s + 1]
        sp, sc, se = pairs[a:b], contr[a:b], edges[a:b]
        own = np.where(sp == pid)[0]
        if len(own) == 0:
            continue
        c = sc[own]; tot = c.sum(); S = s0 - tot
        order = np.argsort(-c); cs = c[order]; cum = np.cumsum(cs) / tot
        r = dict(rank=float(base_rank[qi]), n_paths=int(len(c)))
        perm = rng.permutation(len(c)); cum_r = np.cumsum(c[perm]) / tot
        for k in (1, 3, 10):
            share = float(cs[:k].sum() / tot)
            n_r = int(np.searchsorted(cum_r, share - 1e-12) + 1)
            r[f"matched{k}_n"] = n_r
            r[f"matched{k}"] = rank_after(qi, pid, S + c[perm[:n_r]].sum())
        for f in FRACS:
            nt_ = int(np.searchsorted(cum, f - 1e-12) + 1); nr_ = int(np.searchsorted(cum_r, f - 1e-12) + 1)
            r[f"top_f{f}"] = rank_after(qi, pid, S + cs[:nt_].sum())
            r[f"rand_f{f}"] = rank_after(qi, pid, S + c[perm[:nr_]].sum())
        # overlap among the three leading paths
        top3 = own[order[:3]]
        E = [set(x for x in se[i] if x >= 0) for i in top3]
        shared = [any(E[i] & E[j] for j in range(len(E)) if j != i) for i in range(len(E))]
        r["top3_share_edge"] = int(any(shared))
        r["top3_overlap_share"] = float(sum(cs[i] for i in range(len(E)) if shared[i]) / max(cs[:3].sum(), 1e-12))
        # edge-level counterfactual
        oe = se[own]; ow = np.repeat(c[:, None], 3, 1)
        valid = oe >= 0
        keys, inv = np.unique(oe[valid], return_inverse=True)
        etot = np.bincount(inv, weights=ow[valid])
        e_top = keys[np.argmax(etot)]; e_rand = keys[rng.integers(len(keys))]
        r["n_edges"] = int(len(keys)); r["top_edge_share"] = float(etot.max() / tot)
        r["paths_on_top_edge"] = int((oe == e_top).any(1).sum())
        for nm, e in (("edge_top", e_top), ("edge_rand", e_rand)):
            drop = (se == e).any(1)
            r[nm] = rank_after(qi, pid, src_paths=(sp, sc), drop=drop)
        rec.append(r)
    json.dump(dict(task=task, seed=seed, n_hits=len(rec), fracs=FRACS, records=rec),
              open(f"{OUT}/{task}_explain_eval2.json", "w"), indent=0)
    R = {k: np.array([x[k] for x in rec]) for k in rec[0]}
    print(task, "hits", len(rec))
    for k in ("matched1", "matched3", "matched10", "edge_top", "edge_rand"):
        print(f"  {k}: stay top10 {np.mean(R[k] <= 10):.2f}")
    print("  matched3 paths removed median", np.median(R["matched3_n"]), "top3 share edge", R["top3_share_edge"].mean(),
          "overlap share", np.median(R["top3_overlap_share"]), "top edge share median", np.median(R["top_edge_share"]))
    for f in FRACS:
        print(f"  f={f}: top {np.mean(R[f'top_f{f}'] <= 10):.2f} rand {np.mean(R[f'rand_f{f}'] <= 10):.2f}")


if __name__ == "__main__":
    main(sys.argv[1])
