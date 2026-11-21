"""Revision 6 (JBI reviewer M1b): converged profile of the training loss in the t-norm parameter of ONE metapath,
with the parameters of all other metapaths, the memberships, reliabilities and leak re-fitted by L-BFGS.
The metapaths profiled are those with the largest mean share of DNFR's evidence over its correct top-10 test
predictions on the seed-0 split. Two warm-start sweeps (up and down the grid) give two chains per point.
All parameters are kept in [-20, 0]: above 0 paths can reach strength exactly zero and pass no gradient (Section 4.4),
which in a first attempt trapped the free parameters of other metapaths at the clip. Usage: profile_permp.py TASK SEED N_MP [GRID]   Output: $DNFR_RESULTS/profile_permp_<task>_seed<seed>.json"""
import os, sys, json, time
import numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import explain
from dnfr import DNFR, forward_all, to_logit, metrics, score_pairs
from run_task import split, positives

torch.set_num_threads(1)
task, seed, n_mp_prof = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
grid = [float(x) for x in (sys.argv[4].split(",") if len(sys.argv) > 4 else "-5,-2,-1,-0.5,-0.25,0".split(","))]
ctx = explain.setup(task, 0)          # trained seed-0 model, used only to choose the metapaths
P, sd0, ranker0 = ctx["P"], ctx["sd"], ctx["ranker"]
z, z0 = score_pairs(ctx["model"], sd0)
r = ranker0.ranks(z, z0, ctx["te"], ctx["known"])
hits = ctx["te"][r <= 10]
hit_pid = set(np.searchsorted(ctx["ukey"], hits[:, 0].astype(np.int64) * ctx["n_t"] + hits[:, 1]).tolist())
share = np.zeros(len(P["mps"])); n_h = 0
per_pair = {}
with torch.no_grad():
    for ch in sd0.chunks:
        lv = ctx["model"].path_logvalue(ch)
        c = (-torch.log1p(-torch.exp(torch.clamp(lv, max=-1e-6)))).numpy()
        pr = ch["pair32"].numpy()
        sel = np.isin(pr, list(hit_pid))
        for pid, cc in zip(pr[sel], c[sel]):
            per_pair.setdefault(int(pid), np.zeros(len(P["mps"])))[ch["mp"]] += cc
for v in per_pair.values():
    share += v / v.sum()
share /= max(len(per_pair), 1)
order = np.argsort(-share)
chosen = [int(i) for i in order[:n_mp_prof]]
print("metapath shares:", [(P["mp_names"][i], round(float(share[i]), 3)) for i in order[:6]], flush=True)
del ctx, sd0
# split of the requested seed
_DN = explain.DNFR
class _Stub(torch.nn.Module):
    def __init__(self, *a, **k): super().__init__()
    def load_state_dict(self, *a, **k): return None
explain.DNFR = _Stub; _tl = explain.torch.load; explain.torch.load = lambda *a, **k: {}
ctx = explain.setup(task, seed); explain.DNFR, explain.torch.load = _DN, _tl
sd, ranker, known, tr, te = ctx["sd"], ctx["ranker"], ctx["known"], ctx["tr"], ctx["te"]
pos = positives(ctx["kg"], ctx["spec"]); loc = {s: i for i, s in enumerate(ctx["sources"])}
_, va, _ = split(np.stack([[loc[s] for s in pos[:, 0]], pos[:, 1]], 1), seed)
n_src = len(ctx["sources"])
pair_src = torch.from_numpy(ranker.pair_src.astype(np.int64))
key = {(int(s), int(t)): i for i, (s, t) in enumerate(zip(ranker.pair_src, ranker.pair_tgt))}
pos_pair = torch.tensor([key.get((int(s), int(t)), -1) for s, t in tr]); pos_src = torch.from_numpy(tr[:, 0].astype(np.int64))
is_pos = torch.zeros(sd.n_pairs, dtype=torch.bool); is_pos[pos_pair[pos_pair >= 0]] = True
n_pos_src = torch.zeros(n_src).index_add_(0, pos_src, torch.ones(len(pos_src)))
n_pairs_src = torch.zeros(n_src).index_add_(0, pair_src, torch.ones(sd.n_pairs))
n_pwp = torch.zeros(n_src).index_add_(0, pair_src[is_pos], torch.ones(int(is_pos.sum())))
n0 = ranker.n_tgt - n_pairs_src - (n_pos_src - n_pwp)


def loss_and_grad(model):
    Sp = forward_all(model, sd).detach().requires_grad_(True)
    S0 = model.s0(); S = Sp + S0
    zz = torch.clamp(to_logit(S), -30, 30); zz0 = torch.clamp(to_logit(S0), -30, 30)
    A = torch.zeros(n_src).index_add(0, pair_src, torch.exp(zz) * (~is_pos)) + n0 * torch.exp(zz0)
    zp = torch.where(pos_pair >= 0, zz[pos_pair.clamp(min=0)], zz0)
    loss = (-zp + torch.log(A[pos_src] + torch.exp(zp))).mean()
    loss.backward(); forward_all(model, sd, grad_S=Sp.grad.detach())
    return float(loss)


def evaluate(model, q):
    with torch.no_grad():
        Sv = forward_all(model, sd) + model.s0()
        zv = torch.clamp(to_logit(Sv), -30, 30).numpy(); z0v = float(torch.clamp(to_logit(model.s0()), -30, 30))
    return metrics(ranker.ranks(zv, z0v, q, known))["MRR"]


def fit(mi, p, warm=None, max_outer=60):
    m = DNFR(len(ctx["trav"]), len(P["mps"]), tnorm="ss", p_mode="per_mp", degree=True, agg="noisyor",
             p_init=-0.1, p_param="free")
    m.p_hi = 0.0    # every metapath in the non-nilpotent range [-20, 0] (from the minimum to the product): for p > 0 a
                    # metapath's paths can reach strength zero and stop passing gradient, a trap of the optimiser
    if warm is not None:
        m.load_state_dict(warm, strict=False)
    with torch.no_grad():
        m.p_raw[mi] = p if p != 0.0 else -1e-4     # p = 0 is the product limit; -1e-4 is numerically the product
    mask = torch.ones_like(m.p_raw); mask[mi] = 0.0
    m.p_raw.register_hook(lambda g: g * mask)
    params = [q for q in m.parameters() if q.requires_grad]
    opt = torch.optim.LBFGS(params, lr=1, max_iter=25, history_size=20, line_search_fn="strong_wolfe",
                            tolerance_grad=1e-6, tolerance_change=1e-10)
    last, t0 = None, time.time()
    def closure():
        opt.zero_grad(); return torch.tensor(loss_and_grad(m))
    for it in range(max_outer):
        opt.step(closure); opt.zero_grad(); L = loss_and_grad(m)
        g = torch.cat([(q.grad if q.grad is not None else torch.zeros_like(q)).flatten() for q in params]).norm().item()
        done = g < 1e-5 or (last is not None and abs(last - L) < 1e-9); last = L
        if done:
            break
    return m, dict(mp=P["mp_names"][mi], p=p, loss=L, grad_norm=g, outer=it + 1, time_s=time.time() - t0,
                   valid_MRR=evaluate(m, va), test_MRR=evaluate(m, te), converged=bool(done),
                   p_others_median=float(np.median(np.delete(m.p().detach().numpy(), mi))))


out = dict(task=task, seed=seed, metapath_share={P["mp_names"][i]: float(share[i]) for i in order}, profiles={})
d = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results_rev6")
for mi in chosen:
    chains, warm = {}, None
    for sweep, ordr in (("up", grid), ("down", grid[::-1])):
        for p in ordr:
            m, rr = fit(mi, p, warm); rr["sweep"] = sweep
            warm = {k: v.detach().clone() for k, v in m.state_dict().items() if k != "p_raw"} | \
                   {"p_raw": m.p_raw.detach().clone()}
            chains.setdefault(p, []).append(rr); print(rr, flush=True)
            out["profiles"][P["mp_names"][mi]] = {str(k): v for k, v in chains.items()}
            json.dump(out, open(f"{d}/profile_permp_{task}_seed{seed}.json", "w"), indent=1)
print("done")
