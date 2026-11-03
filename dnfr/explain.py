"""Path-level explanations from a trained DNFR model (seed-0 split)."""
import os, sys
from config import DATA, OUT
import json, pickle, sys
import numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from run_task import TASKS, load_kg, positives, split, OUT
from kg import mp_name
from paths import drop_paths_using
from dnfr import DNFR, SplitData, score_pairs, Ranker
from run_task import VARIANTS


_HN = None


def NAME(x):
    global _HN
    if "::" not in x:
        return x
    if _HN is None:
        import pandas as pd
        d = pd.read_csv(os.path.join(DATA, "hetionet-nodes.tsv"), sep="\t", dtype=str)
        _HN = dict(zip(d["id"], d["name"]))
    return _HN.get(x, x)


def setup(task, seed=0):
    spec = TASKS[task]; kg = load_kg(spec["kg"])
    pos = positives(kg, spec); sources = np.unique(pos[:, 0]); loc = {s: i for i, s in enumerate(sources)}
    n_t = kg.n(spec["t"])
    P = pickle.load(open(f"{DATA}/paths_{task}.pkl", "rb"))
    key = P["src"].astype(np.int64) * n_t + P["tgt"]
    ukey, pair_id = np.unique(key, return_inverse=True)
    posl = np.stack([[loc[s] for s in pos[:, 0]], pos[:, 1]], 1)
    tr, va, te = split(posl, seed)
    removed = {(spec["s"], spec["t"]): np.concatenate([sources[va[:, 0]].astype(np.int64) * 10**7 + va[:, 1],
                                                       sources[te[:, 0]].astype(np.int64) * 10**7 + te[:, 1]])}
    kg_s = kg.copy_without(removed); mask = drop_paths_using(P, kg, removed)
    trav = kg.traversals(); ti = {t: i for i, t in enumerate(trav)}
    sd = SplitData(P, kg_s, mask, pair_id, len(ukey), ti)
    model = DNFR(len(trav), len(P["mps"]), **VARIANTS["dnfr"])
    model.load_state_dict(torch.load(f"{OUT}/{task}_dnfr_seed{seed}.pt"), strict=False)
    known = {}
    for s, t in posl:
        known.setdefault(int(s), set()).add(int(t))
    ranker = Ranker((ukey // n_t).astype(np.int64), (ukey % n_t).astype(np.int64), len(sources), n_t)
    return dict(spec=spec, kg=kg, P=P, sd=sd, model=model, sources=sources, te=te, ukey=ukey, n_t=n_t,
                known=known, ranker=ranker, trav=trav, tr=tr, kg_s=kg_s)


def explain(ctx, s_loc, t, top=5):
    P, sd, model, kg, spec = ctx["P"], ctx["sd"], ctx["model"], ctx["kg"], ctx["spec"]
    pid = np.searchsorted(ctx["ukey"], s_loc * ctx["n_t"] + t)
    rows = []
    with torch.no_grad():
        for ch in sd.chunks:
            m = (ch["pair32"] == pid)
            if not m.any():
                continue
            idx = torch.where(m)[0]
            sub = dict(mp=ch["mp"], hops=[(tt, du[idx], dv[idx], None if lw is None else lw[idx]) for (tt, du, dv, lw) in ch["hops"]])
            lv = model.path_logvalue(sub)
            logmu = [torch.nn.functional.logsigmoid(model.a[tt] - model.b[tt] * du - model.c[tt] * dv + (0 if lw is None else model.g[tt] * lw))
                     for (tt, du, dv, lw) in sub["hops"]]
            contrib = -torch.log1p(-torch.exp(torch.clamp(lv, max=-1e-6)))
            for j, pi in enumerate(ch["path_index"][idx.numpy()]):
                rows.append((float(contrib[j]), int(pi), [float(torch.exp(l[j])) for l in logmu], ch["mp"]))
    rows.sort(reverse=True)
    total = sum(r[0] for r in rows)
    out = []
    for c, pi, mus, mi in rows[:top]:
        mp = P["mps"][mi]
        types = [spec["s"]] + [tr[2] for tr in mp]
        nodes = [ctx["sources"][P["src"][pi]], P["k1"][pi]] + ([P["tgt"][pi]] if len(mp) == 2 else [P["k2"][pi], P["tgt"][pi]])
        names = [NAME(str(kg.names[ty][n])) for ty, n in zip(types, nodes)]
        out.append(dict(metapath=mp_name(mp), nodes=names, mu=[round(x, 3) for x in mus],
                        p=round(float(model.p()[mi]), 3), w=round(float(torch.sigmoid(model.w_raw[mi])), 4),
                        share=round(c / total, 3) if total > 0 else 0))
    S = -total + float(model.s0())
    prob = 1 - np.exp(S)
    return dict(n_paths=len(rows), prob=float(prob), paths=out)


if __name__ == "__main__":
    task = sys.argv[1]; k = int(sys.argv[2]) if len(sys.argv) > 2 else 3
    ctx = setup(task)
    z, z0 = score_pairs(ctx["model"], ctx["sd"])
    ranks = ctx["ranker"].ranks(z, z0, ctx["te"], ctx["known"])
    order = np.argsort(ranks)
    res = []
    names_t = ctx["kg"].names[ctx["spec"]["t"]]; names_s = ctx["kg"].names[ctx["spec"]["s"]]
    for qi in order[:k]:
        s, t = ctx["te"][qi]
        e = explain(ctx, int(s), int(t))
        e.update(query=[NAME(str(names_s[ctx["sources"][s]])), NAME(str(names_t[t]))], rank=float(ranks[qi]))
        res.append(e)
        print(json.dumps(e, indent=1))
    json.dump(res, open(f"{OUT}/{task}_explanations.json", "w"), indent=1)
