"""Differentiable Neuro-Fuzzy Reasoning (DNFR) and fixed-t-norm variants.

Score of a query pair (h, t):
    P(h,t) = 1 - (1 - beta0) * prod_{paths pi} (1 - w_m(pi) * T_{p_m(pi)}(mu(e_1), ..., mu(e_L)))
with learned edge membership mu(e) = sigmoid(a_tau - b_tau log(1+d_out(u)) - c_tau log(1+d_in(v))),
a Schweizer-Sklar t-norm whose parameter p_m is learned per metapath, and metapath reliability w_m.
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import DATA, OUT
import math, time, os
import numpy as np, torch
import torch.nn.functional as F

torch.set_num_threads(int(os.environ.get("NT", "2")))


class SplitData:
    """Per-split tensors: paths grouped by metapath, with degree features per hop."""

    def __init__(self, P, kg_split, keep_mask, pair_id, n_pairs, trav_index):
        self.n_pairs = n_pairs
        self.chunks = []
        deg = {}
        for mi, mp in enumerate(P["mps"]):
            sel = np.where((P["mp"] == mi) & keep_mask)[0]
            if len(sel) == 0:
                continue
            s = P["sources"][P["src"][sel]]
            nodes = [s, P["k1"][sel]] + ([P["tgt"][sel]] if len(mp) == 2 else [P["k2"][sel], P["tgt"][sel]])
            hops = []
            for h, tr in enumerate(mp):
                if tr not in deg:
                    A = kg_split.adj(tr)
                    deg[tr] = (np.log1p(np.asarray(A.sum(1)).ravel()).astype(np.float32),
                               np.log1p(np.asarray(A.sum(0)).ravel()).astype(np.float32))
                do, di = deg[tr]
                lw = None
                base = kg_split.rels[tr[3]]
                if "w" in base:  # graded edge evidence: logit of the stored weight
                    W = edge_weight_matrix(kg_split, tr)
                    wv = np.asarray(W[nodes[h], nodes[h + 1]]).ravel().astype(np.float64)
                    wv = np.clip(wv, 1e-4, 1 - 1e-4)
                    lw = torch.from_numpy(np.log(wv / (1 - wv)).astype(np.float32))
                hops.append((trav_index[tr], torch.from_numpy(do[nodes[h]]), torch.from_numpy(di[nodes[h + 1]]), lw))
            self.chunks.append(dict(mp=mi, pair32=torch.from_numpy(pair_id[sel].astype(np.int32)), hops=hops,
                                    path_index=sel.astype(np.int32)))


_WCACHE = {}


def edge_weight_matrix(kg, tr):
    import scipy.sparse as sp
    k = (id(kg), tr)
    if k not in _WCACHE:
        name, a, b, base, inv = tr
        d = kg.rels[base]
        h, t = (d["t"], d["h"]) if inv else (d["h"], d["t"])
        _WCACHE[k] = sp.csr_matrix((d["w"], (h, t)), shape=(kg.n(a), kg.n(b)))
    return _WCACHE[k]


def ss_log_tnorm(logmu, p):
    """log of the Schweizer-Sklar t-norm of L memberships (list of log tensors), parameter p."""
    p = torch.where(p.abs() < 1e-4, p + 2e-4, p)  # keeps the gradient
    s = sum(torch.expm1(p * lm) for lm in logmu)
    s = torch.clamp(s, min=-1 + 1e-7)
    return torch.log1p(s) / p


def log_tnorm(kind, logmu, p=None):
    if kind == "ss":
        return ss_log_tnorm(logmu, p)
    if kind == "product":
        return sum(logmu)
    if kind == "min":
        out = logmu[0]
        for lm in logmu[1:]:
            out = torch.minimum(out, lm)
        return out
    if kind == "luk":
        x = sum(torch.exp(lm) for lm in logmu) - (len(logmu) - 1)
        return torch.log(torch.clamp(x, min=1e-7))
    if kind == "hamacher":  # Hamacher product, lambda = 0, applied left to right
        x = torch.exp(logmu[0])
        for lm in logmu[1:]:
            y = torch.exp(lm)
            x = x * y / torch.clamp(x + y - x * y, min=1e-12)
        return torch.log(torch.clamp(x, min=1e-30))
    raise ValueError(kind)


P_RAW0 = math.log(4.9 / 1.1)  # p0 = 6*4.9/6 - 5 = -0.1


class DNFR(torch.nn.Module):
    def __init__(self, n_trav, n_mp, tnorm="ss", p_mode="per_mp", degree=True, agg="noisyor", use_weight=True,
                 p_init=-0.1, fixed_deg=None, freeze_mem=False):
        """p_init: initial t-norm parameter. fixed_deg: if a number, b_r = c_r = fixed_deg for every relation
        (not learned; DWPC-style damping). freeze_mem: keep a, b, c, g at their initial values."""
        super().__init__()
        learn_deg = degree and fixed_deg is None and not freeze_mem
        b0 = 0.0 if not degree else (0.3 if fixed_deg is None else float(fixed_deg))
        self.g = torch.nn.Parameter(torch.full((n_trav,), 1.0 if use_weight else 0.0), requires_grad=use_weight and not freeze_mem)
        self.a = torch.nn.Parameter(torch.full((n_trav,), 1.0), requires_grad=not freeze_mem)
        self.b = torch.nn.Parameter(torch.full((n_trav,), b0), requires_grad=learn_deg)
        self.c = torch.nn.Parameter(torch.full((n_trav,), b0), requires_grad=learn_deg)
        self.w_raw = torch.nn.Parameter(torch.full((n_mp,), -3.0))
        n_p = n_mp if p_mode == "per_mp" else 1
        p_raw0 = math.log((p_init + 5) / (1 - p_init))  # inverse of p = 6 sigmoid(theta) - 5
        self.register_buffer("p_raw0", torch.tensor(p_raw0), persistent=False)
        self.p_raw = torch.nn.Parameter(torch.full((n_p,), p_raw0), requires_grad=(tnorm == "ss"))
        self.leak_raw = torch.nn.Parameter(torch.tensor(-6.0))
        self.tnorm, self.p_mode, self.agg = tnorm, p_mode, agg

    def p(self):
        return 6 * torch.sigmoid(self.p_raw) - 5

    def p_of(self, mi):
        pr = self.p()
        return pr[mi] if self.p_mode == "per_mp" else pr[0]

    def path_logvalue(self, ch):
        """log(w_m * T(mu)) for every path in a chunk."""
        logmu = [F.logsigmoid(self.a[t] - self.b[t] * du - self.c[t] * dv + (0 if lw is None else self.g[t] * lw))
                 for (t, du, dv, lw) in ch["hops"]]
        lt = log_tnorm(self.tnorm, logmu, self.p_of(ch["mp"]) if self.tnorm == "ss" else None)
        return lt + F.logsigmoid(self.w_raw[ch["mp"]])

    def s0(self):
        return F.logsigmoid(-self.leak_raw)  # log(1 - beta0)


def to_logit(S):
    """S = log(1-P) -> logit(P)."""
    S = torch.clamp(S, max=-1e-7)
    return torch.log(-torch.expm1(S)) - S


def forward_all(model, sd, grad_S=None):
    """Return S = log(1-P) for all pairs. If grad_S given, backprop it chunk by chunk."""
    S = torch.full((sd.n_pairs,), 0.0)
    if model.agg == "noisyor":
        for ch in sd.chunks:
            ch["pair"] = ch["pair32"].long()
            if grad_S is None:
                with torch.no_grad():
                    lv = model.path_logvalue(ch)
                    S.index_add_(0, ch["pair"], torch.log1p(-torch.exp(torch.clamp(lv, max=-1e-6))))
            else:
                lv = model.path_logvalue(ch)
                contrib = torch.log1p(-torch.exp(torch.clamp(lv, max=-1e-6)))
                contrib.backward(grad_S[ch["pair"]])
            del ch["pair"]
    else:  # max aggregation: P = 1-(1-beta0)(1-max w T); gradient through the arg-max path
        best = torch.full((sd.n_pairs,), -1e9)
        for ch in sd.chunks:
            ch["pair"] = ch["pair32"].long()
        with torch.no_grad():
            for ch in sd.chunks:
                lv = model.path_logvalue(ch)
                ch["_lv"] = lv
                best.scatter_reduce_(0, ch["pair"], lv, reduce="amax")
        if grad_S is None:
            S = torch.log1p(-torch.exp(torch.clamp(best, max=-1e-6)))
            S[best < -1e8] = 0.0
        else:
            for ch in sd.chunks:
                is_best = ch["_lv"] >= best[ch["pair"]] - 1e-9
                idx = torch.where(is_best)[0]
                sub = dict(mp=ch["mp"], hops=[(t, du[idx], dv[idx], None if lw is None else lw[idx]) for (t, du, dv, lw) in ch["hops"]])
                pr = ch["pair"][idx]
                # avoid double counting ties: keep first occurrence per pair
                pr_np = pr.numpy(); _, first = np.unique(pr_np, return_index=True)
                first = torch.from_numpy(first)
                sub = dict(mp=ch["mp"], hops=[(t, du[first], dv[first], None if lw is None else lw[first]) for (t, du, dv, lw) in sub["hops"]])
                lv = model.path_logvalue(sub)
                contrib = torch.log1p(-torch.exp(torch.clamp(lv, max=-1e-6)))
                g = grad_S[pr[first]].clone()
                # a pair may have its best path in several chunks: only the global best gets gradient
                contrib.backward(g)
    return S if grad_S is None else None


class Ranker:
    """Filtered ranking of tails for each query, given pair scores."""

    def __init__(self, pair_src, pair_tgt, n_src, n_tgt):
        self.pair_src, self.pair_tgt = pair_src, pair_tgt
        self.n_tgt = n_tgt
        order = np.argsort(pair_src, kind="stable")
        self.order = order
        self.bounds = np.searchsorted(pair_src[order], np.arange(n_src + 1))

    def ranks(self, z_pairs, z0, queries, known):
        """queries: array (src_local, tgt). known: dict src_local -> set of true tails (all splits).
        Returns realistic (average) filtered rank of each query tail."""
        out = np.empty(len(queries))
        for qi, (s, t) in enumerate(queries):
            idx = self.order[self.bounds[s]:self.bounds[s + 1]]
            tg, zz = self.pair_tgt[idx], z_pairs[idx]
            filt = known.get(s, set()) - {t}
            if filt:
                m = ~np.isin(tg, list(filt))
                tg_f, zz_f = tg[m], zz[m]
            else:
                tg_f, zz_f = tg, zz
            hit = np.where(tg == t)[0]
            zt = zz[hit[0]] if len(hit) else z0
            n_with = len(tg_f) - (1 if len(hit) else 0)
            others = zz_f[tg_f != t]
            tgs = set(tg.tolist())
            n_nopath = self.n_tgt - len(tg) - sum(1 for x in filt if x not in tgs)
            if not len(hit):
                n_nopath -= 1  # t itself is a no-path tail
            gt = (others > zt).sum() + (n_nopath if z0 > zt else 0)
            eq = (others == zt).sum() + (n_nopath if z0 == zt else 0)
            out[qi] = 1 + gt + eq / 2.0
        return out


def metrics(r):
    r = np.asarray(r, float)
    return dict(MRR=float(np.mean(1 / r)), H1=float(np.mean(r <= 1)), H3=float(np.mean(r <= 3)),
                H10=float(np.mean(r <= 10)), MR=float(np.mean(r)))


def train(model, sd, train_q, train_pos_by_src, ranker, valid_q, known, n_src, epochs=150, lr=0.1,
          l2=1e-3, eval_every=5, patience=10, verbose=True, time_limit=None):
    """Full-batch filtered softmax over all tails of each training query source."""
    opt = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=lr)
    pair_src = torch.from_numpy(ranker.pair_src.astype(np.int64))
    pair_tgt = ranker.pair_tgt
    # pair index of each training positive (or -1 if it has no paths)
    key = {(int(s), int(t)): i for i, (s, t) in enumerate(zip(ranker.pair_src, ranker.pair_tgt))}
    pos_pair = torch.tensor([key.get((int(s), int(t)), -1) for s, t in train_q])
    pos_src = torch.from_numpy(train_q[:, 0].astype(np.int64))
    is_pos_pair = torch.zeros(sd.n_pairs, dtype=torch.bool)
    is_pos_pair[pos_pair[pos_pair >= 0]] = True
    n_pos_src = torch.zeros(n_src)  # positives per source (incl. no-path ones)
    n_pos_src.index_add_(0, pos_src, torch.ones(len(pos_src)))
    n_pairs_src = torch.zeros(n_src); n_pairs_src.index_add_(0, pair_src, torch.ones(sd.n_pairs))
    n_pos_with_path = torch.zeros(n_src)
    n_pos_with_path.index_add_(0, pair_src[is_pos_pair], torch.ones(int(is_pos_pair.sum())))
    # number of non-positive tails with no path, per source
    n0 = ranker.n_tgt - n_pairs_src - (n_pos_src - n_pos_with_path)
    best, best_state, bad, hist = -1, None, 0, []
    t_start = time.time()
    for ep in range(1, epochs + 1):
        Sp = forward_all(model, sd).detach().requires_grad_(True)   # path part, no grad yet
        S0 = model.s0()
        S = Sp + S0
        z = torch.clamp(to_logit(S), -30, 30)
        z0 = torch.clamp(to_logit(S0), -30, 30)
        ez = torch.exp(z) * (~is_pos_pair)
        A = torch.zeros(n_src).index_add(0, pair_src, ez) + n0 * torch.exp(z0)
        zp = torch.where(pos_pair >= 0, z[pos_pair.clamp(min=0)], z0)
        loss = (-zp + torch.log(A[pos_src] + torch.exp(zp))).mean()
        reg = l2 * ((model.p_raw - model.p_raw0) ** 2).sum() + l2 * (model.w_raw ** 2).sum() * 0.01
        opt.zero_grad()
        (loss + reg).backward()            # grads for S (leaf) and for leak, p_raw (reg)
        gS = Sp.grad.detach()
        forward_all(model, sd, grad_S=gS)  # accumulate path-parameter gradients
        opt.step()
        if ep % eval_every == 0 or ep == epochs:
            with torch.no_grad():
                Sv = forward_all(model, sd) + model.s0()
                zv = torch.clamp(to_logit(Sv), -30, 30).numpy()
                z0v = float(torch.clamp(to_logit(model.s0()), -30, 30))
            m = metrics(ranker.ranks(zv, z0v, valid_q, known))
            hist.append((ep, float(loss.detach()), m["MRR"]))
            if verbose:
                print(f"    ep {ep} loss {float(loss):.4f} valid MRR {m['MRR']:.4f} ({time.time()-t_start:.0f}s)", flush=True)
            if m["MRR"] > best + 1e-4:
                best, bad = m["MRR"], 0
                best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            else:
                bad += 1
                if bad >= patience:
                    break
        if time_limit and time.time() - t_start > time_limit:
            break
    model.load_state_dict(best_state)
    return best, hist


def score_pairs(model, sd):
    with torch.no_grad():
        S = forward_all(model, sd) + model.s0()
        return (torch.clamp(to_logit(S), -30, 30).numpy(), float(torch.clamp(to_logit(model.s0()), -30, 30)))
