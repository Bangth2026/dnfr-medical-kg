"""Share of DNFR's evidence by metapath class after removing metapaths that contain given relations (round-5 M4, Q4).

For the model trained without the metapaths that contain the treats relation (tag _notgt) and the one trained without
the metapaths that contain treats or palliates (tag _notgtp), on the seed-0 split of Hetionet compound-treats-disease,
we take every correctly ranked test pair in the top 10 and split its score over its kept paths (contribution
-log(1 - w T), as in explain_dwpc.py). We report the mean and median share on metapaths that contain CpD (palliates),
on gene-only metapaths (mechanistic), on associative and on other metapaths.
Usage: evidence_notgt.py TAG RELS   e.g.  evidence_notgt.py _notgt CtD   or   evidence_notgt.py _notgtp CtD,CpD
Output: results/het_CtD<TAG>_evidence.json"""
import os, sys, json, pickle
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, torch
from config import OUT, DATA
from run_task import TASKS, VARIANTS, positives, split, drop_paths_using, load_kg
from dnfr import DNFR, SplitData, Ranker, score_pairs, metrics
from explain_eval3 import mp_class

CLS = ("gene", "associative", "other")


def main(tag, rels, task="het_CtD", seed=0):
    rels = set(rels.split(","))
    spec = TASKS[task]; kg = load_kg(spec["kg"])
    pos = positives(kg, spec); sources = np.unique(pos[:, 0]); loc = {s: i for i, s in enumerate(sources)}
    n_t = kg.n(spec["t"])
    P = pickle.load(open(f"{DATA}/paths_{task}.pkl", "rb"))
    ukey, pair_id = np.unique(P["src"].astype(np.int64) * n_t + P["tgt"], return_inverse=True)
    posl = np.stack([[loc[s] for s in pos[:, 0]], pos[:, 1]], 1)
    tr, va, te = split(posl, seed)
    removed = {(spec["s"], spec["t"]): np.concatenate([sources[va[:, 0]].astype(np.int64) * 10**7 + va[:, 1],
                                                       sources[te[:, 0]].astype(np.int64) * 10**7 + te[:, 1]])}
    kg_s = kg.copy_without(removed); mask = drop_paths_using(P, kg, removed)
    bad = [i for i, m in enumerate(P["mps"]) if any(t[3] in rels for t in m)]
    mask &= ~np.isin(P["mp"], bad)
    trav = kg.traversals(); ti = {t: i for i, t in enumerate(trav)}
    sd = SplitData(P, kg_s, mask, pair_id, len(ukey), ti)
    model = DNFR(len(trav), len(P["mps"]), **VARIANTS["dnfr"])
    model.load_state_dict(torch.load(f"{OUT}/{task}{tag}_dnfr_seed{seed}.pt"), strict=False)
    known = {}
    for s, t in posl:
        known.setdefault(int(s), set()).add(int(t))
    ranker = Ranker((ukey // n_t).astype(np.int64), (ukey % n_t).astype(np.int64), len(sources), n_t)
    z, z0 = score_pairs(model, sd)
    rank = ranker.ranks(z, z0, te, known)
    cpd_mp = np.array([any(t[3] == "CpD" for t in m) for m in P["mps"]])
    cls_mp = np.array([CLS.index(mp_class(m)) for m in P["mps"]], np.int8)
    pairs, contr, mps = [], [], []
    with torch.no_grad():
        for ch in sd.chunks:
            lv = model.path_logvalue(ch)
            contr.append((-torch.log1p(-torch.exp(torch.clamp(lv, max=-1e-6)))).numpy())
            pairs.append(ch["pair32"].numpy()); mps.append(P["mp"][ch["path_index"]])
    pairs, contr, mps = np.concatenate(pairs), np.concatenate(contr).astype(np.float64), np.concatenate(mps)
    o = np.argsort(pairs, kind="stable"); pairs, contr, mps = pairs[o], contr[o], mps[o]
    key = {int(k): i for i, k in enumerate(ukey)}
    rec = []
    for qi, (s, t) in enumerate(te):
        pid = key.get(int(s) * n_t + int(t))
        if pid is None or rank[qi] > 10:
            continue
        a, b = np.searchsorted(pairs, pid), np.searchsorted(pairs, pid, side="right")
        if b <= a:
            continue
        c, m = contr[a:b], mps[a:b]; tot = c.sum()
        rec.append(dict(rank=float(rank[qi]), cpd=float(c[cpd_mp[m]].sum() / tot),
                        **{k: float(c[cls_mp[m] == j].sum() / tot) for j, k in enumerate(CLS)}))
    S = {k: np.array([r[k] for r in rec]) for k in ("cpd",) + CLS}
    out = dict(task=task, tag=tag, dropped_relations=sorted(rels), seed=seed, n_dropped_metapaths=len(bad),
               n_metapaths=len(P["mps"]), n_cpd_metapaths_kept=int(sum(cpd_mp[i] for i in range(len(P["mps"])) if i not in set(bad))),
               test_MRR=float(metrics(rank)["MRR"]), n_hits=len(rec),
               share_mean={k: float(v.mean()) for k, v in S.items()} if rec else None,
               share_median={k: float(np.median(v)) for k, v in S.items()} if rec else None, records=rec)
    json.dump(out, open(f"{OUT}/{task}{tag}_evidence.json", "w"), indent=0)
    print(json.dumps({k: v for k, v in out.items() if k != "records"}))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
