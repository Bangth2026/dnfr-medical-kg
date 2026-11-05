"""Differentiable Neuro-Fuzzy Reasoning (DNFR, DNFR 2.0) and fixed-t-norm variants.

Score of a query pair (h, t):
    P(h,t) = 1 - (1 - beta0) * prod_{paths pi} (1 - w_m(pi) * T_{p_m(pi)}(mu(e_1), ..., mu(e_L)))
with learned edge membership mu(e) = sigmoid(a_tau - b_tau log(1+d_out(u)) - c_tau log(1+d_in(v))),
a Schweizer-Sklar t-norm whose parameter p_m is learned per metapath, and metapath reliability w_m.

DNFR 2.0 (third revision) adds four optional components, each of which can be switched off on its own:
  agg="overlap"   overlap-aware noisy-OR: log(1-P) = log(1-beta0) + sum_pi omega_pi log(1 - v_pi), with
                  omega_pi = prod_{x interior node of pi} (v_pi / max_{pi' of the same pair through x} v_pi')^beta.
                  Two paths of a pair have node-set Jaccard > 0.5 exactly when they share an interior node, so the
                  rule discounts the weaker of every pair of paths with Jaccard > 0.5; beta = 0 recovers noisy-OR.
  tnorm="gumbel"  per-metapath choice among product, minimum and Hamacher t-norms, relaxed by Gumbel-softmax
                  during training (temperature annealed) and hard (arg-max) at evaluation.
  feat="ic"       hub discount from information content: IC(v) = -log P(v), with P(v) the share of the enumerated
                  (training-graph) paths of the task that visit v; enters the membership in place of log(1+degree).
  anneal=True     membership temperature: mu = sigmoid(z / tau), tau annealed geometrically from 2 to 0.5.
p_param="free"   (identifiability check) p = theta, clipped to [-20, 0.95], instead of p = 6 sigmoid(theta) - 5.
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

    def __init__(self, P, kg_split, keep_mask, pair_id, n_pairs, trav_index, overlap=False, feat="degree"):
        self.n_pairs = n_pairs
        self.chunks = []
        deg = {}
        types = sorted(kg_split.names)
        tix = {t: i for i, t in enumerate(types)}
        maxn = max(kg_split.n(t) for t in types) + 1
        ic = None
        if feat in ("ic", "ic_sd"):   # path-occurrence counts per (type, node) on the kept (training-graph) paths
            cnt = {t: np.zeros(kg_split.n(t), np.float64) for t in types}
            src_t = P["mps"][0][0][1]
            for mi, mp in enumerate(P["mps"]):
                sel = np.where((P["mp"] == mi) & keep_mask)[0]
                if len(sel) == 0:
                    continue
                s = P["sources"][P["src"][sel]]
                nodes = [s, P["k1"][sel]] + ([P["tgt"][sel]] if len(mp) == 2 else [P["k2"][sel], P["tgt"][sel]])
                ntypes = [mp[0][1]] + [tr[2] for tr in mp]
                for x, t in zip(nodes, ntypes):
                    cnt[t] += np.bincount(x, minlength=kg_split.n(t))
            ic = {t: np.log1p(c).astype(np.float32) for t, c in cnt.items()}   # = -IC(v) + const(type)
            self.ic_counts = cnt
        groups = []
        for mi, mp in enumerate(P["mps"]):
            sel = np.where((P["mp"] == mi) & keep_mask)[0]
            if len(sel) == 0:
                continue
            s = P["sources"][P["src"][sel]]
            nodes = [s, P["k1"][sel]] + ([P["tgt"][sel]] if len(mp) == 2 else [P["k2"][sel], P["tgt"][sel]])
            hops = []
            ntypes = [mp[0][1]] + [tr[2] for tr in mp]
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
                fu, fv = do[nodes[h]], di[nodes[h + 1]]
                if ic is not None and feat == "ic":   # rescaled so that its mean matches the degree feature it replaces
                    gu, gv = ic[ntypes[h]][nodes[h]], ic[ntypes[h + 1]][nodes[h + 1]]
                    fu = (gu * (fu.mean() / max(gu.mean(), 1e-6))).astype(np.float32)
                    fv = (gv * (fv.mean() / max(gv.mean(), 1e-6))).astype(np.float32)
                elif ic is not None:   # "ic_sd" (reviewer M5 check): matched in mean and standard deviation
                    gu, gv = ic[ntypes[h]][nodes[h]], ic[ntypes[h + 1]][nodes[h + 1]]
                    fu = ((gu - gu.mean()) * (fu.std() / max(gu.std(), 1e-6)) + fu.mean()).astype(np.float32)
                    fv = ((gv - gv.mean()) * (fv.std() / max(gv.std(), 1e-6)) + fv.mean()).astype(np.float32)
                hops.append((trav_index[tr], torch.from_numpy(np.ascontiguousarray(fu)), torch.from_numpy(np.ascontiguousarray(fv)), lw))
            ch = dict(mp=mi, pair32=torch.from_numpy(pair_id[sel].astype(np.int32)), hops=hops,
                      path_index=sel.astype(np.int32))
            if overlap:   # keys of the interior nodes: (pair, node type, node)
                pid = pair_id[sel].astype(np.int64)
                ks = [(pid * len(types) + tix[ntypes[1]]) * maxn + nodes[1]]
                if len(mp) == 3:
                    ks.append((pid * len(types) + tix[ntypes[2]]) * maxn + nodes[2])
                groups.append(ks)
            self.chunks.append(ch)
        self.n_groups = 0
        if overlap:
            allk = np.concatenate([k for ks in groups for k in ks])
            uk = np.unique(allk); del allk
            self.n_groups = len(uk)
            for ch, ks in zip(self.chunks, groups):
                ch["g"] = [torch.from_numpy(np.searchsorted(uk, k).astype(np.int32)) for k in ks]
            del uk


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
    xs = [p * lm for lm in logmu]                  # p log mu_i; large positive when p < 0 and mu_i is small
    over = sum((x > 85.0).float() for x in xs) > 0  # exp would overflow float32 in the direct formula
    s = sum(torch.expm1(torch.clamp(x, max=85.0)) for x in xs)
    s = torch.clamp(s, min=-1 + 1e-7)
    direct = torch.log1p(s) / p
    # overflow branch (p < 0): log(sum mu^p - (L-1)) = A + log1p(-(L-1) e^-A), A = logsumexp(p log mu)
    A = torch.logsumexp(torch.stack(xs), 0)
    A = torch.where(over, A, torch.full_like(A, 100.0))
    stable = (A + torch.log1p(-(len(xs) - 1) * torch.exp(-A))) / p
    return torch.where(over, stable, direct)


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
                 p_init=-0.1, fixed_deg=None, freeze_mem=False, learn_p=True, p_param="sigmoid", anneal=False,
                 beta_init=1.0, ops=("product", "min", "hamacher"), feat="degree"):
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
        self.p_param = p_param
        p_raw0 = math.log((p_init + 5) / (1 - p_init)) if p_param == "sigmoid" else float(p_init)  # p = 6 sigmoid(theta) - 5
        self.register_buffer("p_raw0", torch.tensor(p_raw0), persistent=False)
        self.p_raw = torch.nn.Parameter(torch.full((n_p,), p_raw0), requires_grad=(tnorm == "ss" and learn_p))
        self.leak_raw = torch.nn.Parameter(torch.tensor(-6.0))
        self.tnorm, self.p_mode, self.agg, self.feat = tnorm, p_mode, agg, feat
        # DNFR 2.0 components
        self.beta_raw = torch.nn.Parameter(torch.tensor(math.log(math.expm1(beta_init))), requires_grad=(agg == "overlap"))
        self.ops = tuple(ops)
        self.op_logit = torch.nn.Parameter(torch.zeros(n_mp, len(ops)), requires_grad=(tnorm == "gumbel"))
        self.anneal = anneal
        self.register_buffer("tau", torch.tensor(2.0 if anneal else 1.0), persistent=anneal)
        self.gumbel_tau = 1.0
        self.stochastic = False
        self._y = None
        self._gmax = None

    # ---- schedule hooks called by the training loops ----
    def begin_epoch(self, ep):
        """Set temperatures for epoch ep and draw the Gumbel noise used by both passes of this epoch."""
        if self.anneal:
            self.tau.fill_(max(0.5, 2.0 * (0.25 ** (ep / 100.0))))
        if self.tnorm == "gumbel":
            self.gumbel_tau = max(0.2, math.exp(-0.02 * ep))
            g = -torch.log(-torch.log(torch.rand_like(self.op_logit).clamp(1e-9, 1 - 1e-9)))
            self._g = g
        self.stochastic = True

    def end_epoch(self):
        self.stochastic = False

    def op_weights(self):
        """Weights of the candidate t-norms per metapath: relaxed sample in training, one-hot arg-max otherwise."""
        if self.stochastic:
            return torch.softmax((self.op_logit + self._g) / self.gumbel_tau, dim=1)
        return F.one_hot(self.op_logit.argmax(1), len(self.ops)).float()

    def beta(self):
        return F.softplus(self.beta_raw)

    def p(self):
        if self.p_param == "free":
            return torch.clamp(self.p_raw, -20.0, 0.95)
        return 6 * torch.sigmoid(self.p_raw) - 5

    def p_of(self, mi):
        pr = self.p()
        return pr[mi] if self.p_mode == "per_mp" else pr[0]

    def path_logvalue(self, ch):
        """log(w_m * T(mu)) for every path in a chunk."""
        logmu = [F.logsigmoid((self.a[t] - self.b[t] * du - self.c[t] * dv + (0 if lw is None else self.g[t] * lw)) / self.tau)
                 for (t, du, dv, lw) in ch["hops"]]
        if self.tnorm == "gumbel":
            y = self.op_weights()[ch["mp"]]
            if self.stochastic:
                lt = torch.logsumexp(torch.stack([torch.log(y[k].clamp(min=1e-30)) + log_tnorm(op, logmu)
                                                  for k, op in enumerate(self.ops)]), 0)
            else:
                lt = log_tnorm(self.ops[int(y.argmax())], logmu)
        else:
            lt = log_tnorm(self.tnorm, logmu, self.p_of(ch["mp"]) if self.tnorm == "ss" else None)
        return lt + F.logsigmoid(self.w_raw[ch["mp"]])

    def chosen_ops(self):
        return [self.ops[i] for i in self.op_logit.argmax(1).tolist()]

    def s0(self):
        return F.logsigmoid(-self.leak_raw)  # log(1 - beta0)


def to_logit(S):
    """S = log(1-P) -> logit(P)."""
    S = torch.clamp(S, max=-1e-7)
    return torch.log(-torch.expm1(S)) - S


def forward_all(model, sd, grad_S=None):
    """Return S = log(1-P) for all pairs. If grad_S given, backprop it chunk by chunk."""
    S = torch.full((sd.n_pairs,), 0.0)
    if model.agg == "overlap":
        if grad_S is None or model._gmax is None:   # strongest path through each (pair, interior node)
            gmax = torch.full((sd.n_groups,), -1e9)
            with torch.no_grad():
                for ch in sd.chunks:
                    lv = model.path_logvalue(ch)
                    for g in ch["g"]:
                        gmax.scatter_reduce_(0, g.long(), lv, reduce="amax")
            model._gmax = gmax
        gmax = model._gmax
        for ch in sd.chunks:
            pair = ch["pair32"].long()
            def contrib_of(lv):
                beta = model.beta()                                    # rebuilt per chunk: each chunk backpropagates separately
                lw = sum(lv - gmax[g.long()] for g in ch["g"])        # log of prod (v / v_max)
                return torch.exp(beta * lw) * torch.log1p(-torch.exp(torch.clamp(lv, max=-1e-6)))
            if grad_S is None:
                with torch.no_grad():
                    S.index_add_(0, pair, contrib_of(model.path_logvalue(ch)))
            else:
                contrib_of(model.path_logvalue(ch)).backward(grad_S[pair])
        if grad_S is not None:
            model._gmax = None
        return S if grad_S is None else None
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
        if not np.all(np.isfinite(z_pairs)) or not np.isfinite(z0):
            raise FloatingPointError("non-finite scores")
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
        if hasattr(model, "begin_epoch"):
            model.begin_epoch(ep)
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
        if hasattr(model, "end_epoch"):
            model.end_epoch()
        if ep % eval_every == 0 or ep == epochs:
            with torch.no_grad():
                Sv = forward_all(model, sd) + model.s0()
                zv = torch.clamp(to_logit(Sv), -30, 30).numpy()
                z0v = float(torch.clamp(to_logit(model.s0()), -30, 30))
            m = metrics(ranker.ranks(zv, z0v, valid_q, known))
            hist.append((ep, float(loss.detach()), m["MRR"]))
            if getattr(model, "tnorm", "") == "gumbel":
                hist[-1] = hist[-1] + (model.chosen_ops(),)
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
