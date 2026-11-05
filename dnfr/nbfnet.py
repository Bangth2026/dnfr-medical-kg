"""A compact Neural Bellman-Ford network (NBFNet, Zhu et al. 2021) for the reviewer's request M7, CPU only.

Per query head h the representation of every node is computed by L rounds of generalised Bellman-Ford:
  h0[v] = q if v == h else 0;   h_l[v] = LN(ReLU(W_l [ sum_{(u, r, v)} h_{l-1}[u] * R_l[r] ; h0[v] ])) + h_{l-1}[v]
with DistMult messages, sum aggregation, layer-specific relation vectors R_l for every relation and its inverse,
and a final MLP on [h_L[t]; q] giving the logit of tail t. The graph is the leave-dyad-out training graph of the
split restricted to the relations that occur on the task's metapaths (the same evidence DNFR sees, but paths of
any length up to L). During training the target edges of half of each head's training tails, drawn anew every
epoch, are masked for that head and those tails are the positives of the filtered softmax loss; the other half
stay in the graph as context. Validation MRR every 5 epochs, patience 10, best checkpoint kept, as for DNFR.
Usage: nbfnet.py TASK SEEDS [layers] [dim]   ->  results/nbfnet_<task>_seed<k>.json"""
import os, sys, json, time, pickle
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import DATA, OUT
import numpy as np, torch
import torch.nn as nn
from run_task import TASKS, load_kg, positives, split
torch.set_num_threads(int(os.environ.get("NT", "1")))


class NBF(nn.Module):
    def __init__(self, n_rel, d=32, L=3):
        super().__init__()
        self.q = nn.Parameter(torch.randn(d) * 0.1)
        self.R = nn.Parameter(torch.randn(L, n_rel, d) * 0.1 + 1.0)
        self.W = nn.ModuleList([nn.Linear(2 * d, d) for _ in range(L)])
        self.N = nn.ModuleList([nn.LayerNorm(d) for _ in range(L)])
        self.mlp = nn.Sequential(nn.Linear(2 * d, 2 * d), nn.ReLU(), nn.Linear(2 * d, 1))
        self.L, self.d = L, d

    def forward(self, heads, n_nodes, src, dst, rel, emask=None):
        """heads: (B,) global node ids; emask: (B, E) 0/1 edge mask per head or None. Returns (B, n_nodes, d)."""
        B = len(heads)
        h0 = torch.zeros(B, n_nodes, self.d)
        h0[torch.arange(B), heads] = self.q
        h = h0
        for l in range(self.L):
            msg = h[:, src] * self.R[l, rel]
            if emask is not None:
                msg = msg * emask[:, :, None]
            agg = torch.zeros(B, n_nodes, self.d).index_add_(1, dst, msg)
            h = self.N[l](torch.relu(self.W[l](torch.cat([agg, h0], -1)))) + h
        return h

    def score(self, h, tails):
        q = self.q.expand(h.shape[0], len(tails), self.d)
        return self.mlp(torch.cat([h[:, tails], q], -1)).squeeze(-1)


def build(task, seed):
    spec = TASKS[task]; kg = load_kg(spec["kg"])
    pos = positives(kg, spec); sources = np.unique(pos[:, 0]); loc = {s: i for i, s in enumerate(sources)}
    posl = np.stack([[loc[s] for s in pos[:, 0]], pos[:, 1]], 1)
    tr, va, te = split(posl, seed)
    removed = {(spec["s"], spec["t"]): np.concatenate([sources[va[:, 0]].astype(np.int64) * 10**7 + va[:, 1],
                                                       sources[te[:, 0]].astype(np.int64) * 10**7 + te[:, 1]])}
    kg_s = kg.copy_without(removed)
    P = pickle.load(open(f"{DATA}/paths_{task}.pkl", "rb"))
    bases = sorted({tr_[3] for m in P["mps"] for tr_ in m} | {spec["rel"]})
    types = sorted({kg.rels[b]["src"] for b in bases} | {kg.rels[b]["dst"] for b in bases})
    off, n = {}, 0
    for t in types:
        off[t] = n; n += kg.n(t)
    src, dst, rel, is_target = [], [], [], []
    for i, b in enumerate(bases):
        d = kg_s.rels[b]
        u = d["h"].astype(np.int64) + off[d["src"]]; v = d["t"].astype(np.int64) + off[d["dst"]]
        src += [u, v]; dst += [v, u]; rel += [np.full(len(u), 2 * i), np.full(len(u), 2 * i + 1)]
        is_target += [np.full(len(u), b == spec["rel"])] * 2
    src, dst, rel, is_target = map(np.concatenate, (src, dst, rel, is_target))
    # head/tail of the target relation in canonical query orientation (head = spec["s"])
    hs, ts = (dst, src) if spec["rev"] else (src, dst)
    head_g = sources.astype(np.int64) + off[spec["s"]]
    tail_g = np.arange(kg.n(spec["t"])) + off[spec["t"]]
    known = {}
    for s, t in posl:
        known.setdefault(int(s), set()).add(int(t))
    return dict(n=n, src=torch.from_numpy(src), dst=torch.from_numpy(dst), rel=torch.from_numpy(rel), n_rel=2 * len(bases),
                tgt_edge=is_target, e_head=src, e_tail=dst, head_g=head_g, tail_g=torch.from_numpy(tail_g),
                tail_off=off[spec["t"]], head_off=off[spec["s"]], tr=tr, va=va, te=te, known=known, n_t=kg.n(spec["t"]),
                n_edges=len(src) // 2, bases=bases)


def edge_mask(G, heads_loc, masked_tails):
    """(B, E) mask that removes, for each head, the target-relation edges to its masked tails (both directions)."""
    E = len(G["src"]); M = torch.ones(len(heads_loc), E)
    tg = G["tgt_edge"]
    for b, (hl, tails) in enumerate(zip(heads_loc, masked_tails)):
        if not tails:
            continue
        hg = G["head_g"][hl]; tgl = np.array(sorted(tails)) + G["tail_off"]
        a, c = G["e_head"], G["e_tail"]
        hit = tg & (((a == hg) & np.isin(c, tgl)) | ((c == hg) & np.isin(a, tgl)))
        M[b, torch.from_numpy(np.where(hit)[0])] = 0
    return M


def ranks(model, G, Q, batch=16):
    """Filtered ranks (mean rank for ties) of the tails of queries Q (local head, tail)."""
    out = np.zeros(len(Q))
    heads = np.unique(Q[:, 0])
    with torch.no_grad():
        for i in range(0, len(heads), batch):
            hb = heads[i:i + batch]
            h = model(torch.from_numpy(G["head_g"][hb]), G["n"], G["src"], G["dst"], G["rel"])
            S = model.score(h, G["tail_g"]).numpy()
            for b, hl in enumerate(hb):
                for qi in np.where(Q[:, 0] == hl)[0]:
                    t = int(Q[qi, 1]); row = S[b]
                    other = np.ones(G["n_t"], bool); other[list(G["known"][int(hl)])] = False; other[t] = False
                    out[qi] = 1 + (row[other] > row[t]).sum() + 0.5 * (row[other] == row[t]).sum()
    return out


def mrr_of(r):
    return dict(MRR=float(np.mean(1 / r)), H1=float(np.mean(r <= 1)), H3=float(np.mean(r <= 3)), H10=float(np.mean(r <= 10)))


def run(task, seed, L=3, d=32, epochs=int(os.environ.get("NBF_EPOCHS", "300")), lr=float(os.environ.get("NBF_LR", "5e-3")), batch=int(os.environ.get("NBF_BATCH", "16"))):
    torch.manual_seed(seed); rng = np.random.default_rng(seed)
    G = build(task, seed)
    model = NBF(G["n_rel"], d, L)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    tr = G["tr"]; by_head = {}
    for s, t in tr:
        by_head.setdefault(int(s), []).append(int(t))
    heads = np.array(sorted(by_head))
    best, best_state, bad, t0, hist = -1, None, 0, time.time(), []
    for ep in range(1, epochs + 1):
        model.train(); tot = 0.0
        perm = rng.permutation(heads)
        for i in range(0, len(perm), batch):
            hb = perm[i:i + batch]
            qt = []
            for hl in hb:
                ts = by_head[int(hl)]
                k = max(1, len(ts) // 2)
                qt.append(set(rng.choice(ts, k, replace=False).tolist()))
            M = edge_mask(G, hb, qt)
            h = model(torch.from_numpy(G["head_g"][hb]), G["n"], G["src"], G["dst"], G["rel"], M)
            S = model.score(h, G["tail_g"])                       # (B, n_t)
            loss = 0.0; cnt = 0
            for b, hl in enumerate(hb):
                ctx = set(by_head[int(hl)]) - qt[b]               # context tails stay in the graph and are filtered out
                mask = torch.zeros(G["n_t"], dtype=torch.bool)
                if ctx:
                    mask[list(ctx)] = True
                row = S[b].masked_fill(mask, -1e9)
                for t in qt[b]:
                    others = row.clone(); others[list(qt[b] - {t})] = -1e9
                    loss = loss + torch.logsumexp(others, 0) - row[t]; cnt += 1
            loss = loss / cnt
            opt.zero_grad(); loss.backward(); opt.step(); tot += float(loss)
        if ep % 5 == 0:
            model.eval()
            v = mrr_of(ranks(model, G, G["va"]))["MRR"]; hist.append((ep, tot, v))
            print(f"    ep {ep} loss {tot:.3f} valid MRR {v:.4f} ({time.time()-t0:.0f}s)", flush=True)
            if v > best + 1e-4:
                best, bad = v, 0; best_state = {k: x.clone() for k, x in model.state_dict().items()}
            else:
                bad += 1
                if bad >= 10:
                    break
    model.load_state_dict(best_state); model.eval()
    r = ranks(model, G, G["te"])
    m = mrr_of(r); m.update(valid_MRR=best, time_s=time.time() - t0, epochs=ep, best_epoch=max(hist, key=lambda x: x[2])[0],
                           n_params=int(sum(p.numel() for p in model.parameters())), layers=L, dim=d)
    print(task, seed, m, flush=True)
    return m, r.tolist(), G


if __name__ == "__main__":
    task = sys.argv[1]; seeds = [int(x) for x in sys.argv[2].split(",")]
    L = int(sys.argv[3]) if len(sys.argv) > 3 else 3; d = int(sys.argv[4]) if len(sys.argv) > 4 else 32
    for s in seeds:
        m, r, G = run(task, s, L, d)
        json.dump(dict(task=task, seed=s, metrics=m, ranks=r, relations=G["bases"], n_edges=G["n_edges"]),
                  open(f"{OUT}/nbfnet_{task}{'' if L == 3 else f'_L{L}'}{'' if (d == 32 and os.environ.get('NBF_LR', '5e-3') == '5e-3') else f'_d{d}_lr' + os.environ.get('NBF_LR', '5e-3')}_seed{s}.json", "w"))
