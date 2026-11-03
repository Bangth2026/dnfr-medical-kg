"""Edge-level attribution for the path explanations (third revision; seed-0 split, test pairs ranked in the top 10).

With memberships and all other scores fixed, the evidence of a pair, -log(1-P) above the leak, is a sum of path
terms s_pi, and a path term counts only if every edge of the path is present: the evidence is a sum of unanimity
games, one per path. Its Shapley value for edge e is therefore exactly phi_e = sum_{pi containing e} s_pi / L_pi,
and the phi_e sum to the evidence without double counting (Section 3.6). Per pair we record:
  top_edge_shapley   share of the evidence on the edge with the largest phi_e
  k50_edges          number of edges, by decreasing phi_e, that account for 50% of the evidence
  top_edge_double    the second-revision (double-counting) share of the same edge, sum of s_pi over paths through it
  edge_top_sh        filtered rank after removing the Shapley-leading edge from the graph (as in explain_eval2)
  lead_class         class of the leading path's metapath: gene (all interior nodes genes), associative (an interior
                     compound, pharmacologic class or disease), other
  gene_share         share of the evidence on gene-only paths
Output: results/<task>_explain_eval3.json"""
import os, sys, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import OUT
import numpy as np
from explain import setup
from dnfr import score_pairs
from explain_eval2 import path_table


def mp_class(m):
    inner = [tr[2] for tr in m[:-1]]
    if all(x == "G" for x in inner):
        return "gene"
    if any(x in ("C", "PC", "D") for x in inner):
        return "associative"
    return "other"


def main(task, seed=0):
    ctx = setup(task, seed)
    z, z0 = score_pairs(ctx["model"], ctx["sd"])
    ranker, known, te, n_t, ukey, P = ctx["ranker"], ctx["known"], ctx["te"], ctx["n_t"], ctx["ukey"], ctx["P"]
    base_rank = ranker.ranks(z, z0, te, known)
    pairs, contr, edges = path_table(ctx)
    # metapath class of every kept path (same order as path_table)
    pidx = np.concatenate([ch["path_index"] for ch in ctx["sd"].chunks])
    cls_of_mp = np.array([mp_class(m) for m in P["mps"]])
    pcls = cls_of_mp[P["mp"][pidx]]
    psrc = ukey[pairs] // n_t
    o = np.argsort(psrc, kind="stable"); pairs, contr, edges, psrc, pcls = pairs[o], contr[o], edges[o], psrc[o], pcls[o]
    sb = np.searchsorted(psrc, np.arange(len(ctx["sources"]) + 1))
    s0 = float(ctx["model"].s0()); kmap = {int(k): i for i, k in enumerate(ukey)}

    def zlogit(S):
        S = np.minimum(S, -1e-7)
        return np.clip(np.log(-np.expm1(S)) - S, -30, 30)

    rec = []
    for qi, (s, t) in enumerate(te):
        if base_rank[qi] > 10 or (int(s) * n_t + int(t)) not in kmap:
            continue
        pid = kmap[int(s) * n_t + int(t)]
        a, b = sb[s], sb[s + 1]
        sp, sc, se, scl = pairs[a:b], contr[a:b], edges[a:b], pcls[a:b]
        own = np.where(sp == pid)[0]
        if len(own) == 0:
            continue
        c = sc[own]; tot = c.sum(); oe = se[own]; valid = oe >= 0
        L = valid.sum(1)
        keys, inv = np.unique(oe[valid], return_inverse=True)
        w_sh = np.repeat((c / L)[:, None], 3, 1)[valid]; w_dbl = np.repeat(c[:, None], 3, 1)[valid]
        phi = np.bincount(inv, weights=w_sh); dbl = np.bincount(inv, weights=w_dbl)
        j = int(np.argmax(phi)); e_top = keys[j]
        cum = np.cumsum(np.sort(phi)[::-1]) / tot
        drop = (se == e_top).any(1)
        ids, inv2 = np.unique(sp, return_inverse=True)
        zz = z.copy(); zz[ids] = zlogit(s0 - np.bincount(inv2, weights=sc * ~drop, minlength=len(ids)))
        lead = own[np.argmax(c)]
        rec.append(dict(rank=float(base_rank[qi]), n_edges=int(len(keys)), top_edge_shapley=float(phi[j] / tot),
                        top_edge_double=float(dbl[j] / tot), k50_edges=int(np.searchsorted(cum, 0.5 - 1e-12) + 1),
                        k80_edges=int(np.searchsorted(cum, 0.8 - 1e-12) + 1),
                        edge_top_sh=float(ranker.ranks(zz, z0, te[qi:qi + 1], known)[0]),
                        lead_class=str(scl[lead]), gene_share=float(c[scl[own] == "gene"].sum() / tot)))
    R = {k: np.array([x[k] for x in rec]) for k in rec[0]}
    out = dict(task=task, seed=seed, n_hits=len(rec), records=rec,
               summary=dict(top_edge_shapley_median=float(np.median(R["top_edge_shapley"])),
                            top_edge_double_median=float(np.median(R["top_edge_double"])),
                            k50_edges_median=float(np.median(R["k50_edges"])), k80_edges_median=float(np.median(R["k80_edges"])),
                            n_edges_median=float(np.median(R["n_edges"])),
                            stay_top10_edge_top_sh=float(np.mean(R["edge_top_sh"] <= 10)),
                            lead_class={k: float(np.mean(R["lead_class"] == k)) for k in ("gene", "associative", "other")},
                            gene_share_median=float(np.median(R["gene_share"])), gene_share_mean=float(np.mean(R["gene_share"]))))
    json.dump(out, open(f"{OUT}/{task}_explain_eval3.json", "w"), indent=0)
    print(task, json.dumps(out["summary"]))


if __name__ == "__main__":
    main(sys.argv[1])
