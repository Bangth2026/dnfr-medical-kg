"""Explicit metapath enumeration (simple paths, length 2-3) and per-split filtering."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import DATA, OUT
import numpy as np, scipy.sparse as sp
from kg import metapaths, mp_name


def expand(A, u):
    """For CSR A and node vector u: return (rep, nbr) listing every neighbour of every u."""
    ip = A.indptr
    st, cnt = ip[u], ip[u + 1] - ip[u]
    tot = int(cnt.sum())
    rep = np.repeat(np.arange(len(u)), cnt)
    off = np.arange(tot) - np.repeat(np.cumsum(cnt) - cnt, cnt)
    return rep, A.indices[np.repeat(st, cnt) + off]


def enumerate_paths(kg, src_type, dst_type, sources, budget=5e6, exclude_direct=None, verbose=True):
    """Enumerate simple paths of length 2-3 from `sources` to every node of dst_type.
    Metapaths whose path count over `sources` exceeds `budget` are skipped (counted by
    sparse products first). Returns dict with arrays and metapath list."""
    adjc = {}
    def A(tr):
        if tr not in adjc:
            adjc[tr] = kg.adj(tr).tocsr()
        return adjc[tr]
    mps_all = metapaths(kg, src_type, dst_type)
    kept, counts = [], {}
    for mp in mps_all:
        M = A(mp[0])[sources]
        for tr in mp[1:]:
            M = M @ A(tr)
        c = float(M.sum()); counts[mp_name(mp)] = c
        if 0 < c <= budget:
            kept.append(mp)
    out = dict(src=[], tgt=[], k1=[], k2=[], mp=[])
    for mi, mp in enumerate(kept):
        types = [src_type] + [tr[2] for tr in mp]
        s_idx = np.arange(len(sources))
        rep, k1 = expand(A(mp[0]), sources)
        s_loc = s_idx[rep]
        rep, k2 = expand(A(mp[1]), k1)
        s_loc, k1 = s_loc[rep], k1[rep]
        if len(mp) == 2:
            t, kk2 = k2, np.full(len(k2), -1)
            ok = np.ones(len(t), bool)
            if types[1] == types[0]: ok &= k1 != sources[s_loc]
            if types[1] == types[2]: ok &= k1 != t
            if types[2] == types[0]: ok &= t != sources[s_loc]
        else:
            rep, t = expand(A(mp[2]), k2)
            s_loc, k1, kk2 = s_loc[rep], k1[rep], k2[rep]
            s_node = sources[s_loc]
            ok = np.ones(len(t), bool)
            ty = types
            for (a, b, xa, xb) in [(0, 1, s_node, k1), (0, 2, s_node, kk2), (0, 3, s_node, t),
                                   (1, 2, k1, kk2), (1, 3, k1, t), (2, 3, kk2, t)]:
                if ty[a] == ty[b]:
                    ok &= xa != xb
        out["src"].append(s_loc[ok].astype(np.int32)); out["tgt"].append(t[ok].astype(np.int32))
        out["k1"].append(k1[ok].astype(np.int32)); out["k2"].append(kk2[ok].astype(np.int32))
        out["mp"].append(np.full(ok.sum(), mi, np.int16))
        if verbose and mi % 20 == 0:
            print(f"  mp {mi}/{len(kept)} {mp_name(mp)} paths={ok.sum()}", flush=True)
    for k in out:
        out[k] = np.concatenate(out[k])
    out["mps"] = kept
    out["mp_names"] = [mp_name(m) for m in kept]
    out["counts"] = counts
    out["sources"] = np.asarray(sources)
    return out


def hop_nodes(P, i_hop):
    """Endpoints (u, v) of hop i for all paths (vectorised; handles length 2 and 3)."""
    s = P["sources"][P["src"]]
    L = np.array([len(m) for m in P["mps"]])[P["mp"]]
    nodes = [s, P["k1"], np.where(L == 2, P["tgt"], P["k2"]), np.where(L == 3, P["tgt"], -1)]
    return nodes[i_hop], nodes[i_hop + 1], L


def drop_paths_using(P, kg, removed):
    """removed: dict (typeA, typeB) -> int64 keys a*1e7+b of dyads removed in both
    orientations. Returns boolean mask of paths that use none of the removed dyads."""
    keep = np.ones(len(P["src"]), bool)
    Ls = np.array([len(m) for m in P["mps"]])
    for mi, mp in enumerate(P["mps"]):
        sel = np.where(P["mp"] == mi)[0]
        if len(sel) == 0:
            continue
        s = P["sources"][P["src"][sel]]
        nodes = [s, P["k1"][sel]] + ([P["tgt"][sel]] if len(mp) == 2 else [P["k2"][sel], P["tgt"][sel]])
        bad = np.zeros(len(sel), bool)
        for h, tr in enumerate(mp):
            a, b = tr[1], tr[2]
            u, v = nodes[h].astype(np.int64), nodes[h + 1].astype(np.int64)
            for (x, y), keys in removed.items():
                if (a, b) == (x, y):
                    bad |= np.isin(u * 10**7 + v, keys)
                elif (a, b) == (y, x):
                    bad |= np.isin(v * 10**7 + u, keys)
        keep[sel[bad]] = False
    return keep
