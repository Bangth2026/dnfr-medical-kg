"""Converged profile of the training loss in a shared Schweizer-Sklar parameter p (reviewer M1.1, M1.3, Q1).
For each fixed p, all other parameters (memberships, metapath weights, leak) are fitted by L-BFGS on the
unpenalised full-batch filtered softmax loss until the gradient norm falls below 1e-5 or the loss changes by
less than 1e-9. At the converged point we report the loss, dL/dp (autograd, with p as a free scalar), valid and
test MRR. Curvature d2L/dp2 is taken from central differences of dL/dp. Usage: profile_lbfgs.py TASK SEED"""
import os, sys, json, time, math
import numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import explain
from dnfr import DNFR, forward_all, to_logit, metrics

torch.set_num_threads(1)
task, seed = sys.argv[1], int(sys.argv[2])
grid = [float(x) for x in (sys.argv[3].split(",") if len(sys.argv) > 3 else
        "-5,-4,-3,-2.5,-2,-1.5,-1,-0.75,-0.5,-0.25,0".split(","))]
# reuse the split and path data of explain.setup without loading a trained model
DNFR_load = explain.DNFR
class _Stub(torch.nn.Module):
    def __init__(self, *a, **k): super().__init__()
    def load_state_dict(self, *a, **k): return None
explain.DNFR = _Stub
_tl = explain.torch.load; explain.torch.load = lambda *a, **k: {}
ctx = explain.setup(task, seed)
explain.DNFR, explain.torch.load = DNFR_load, _tl
sd, ranker, known = ctx["sd"], ctx["ranker"], ctx["known"]
tr, va, te = ctx["tr"], None, ctx["te"]
# validation split as in run_task.split
from run_task import split
posl_tr = tr
spec = ctx["spec"]
n_src = len(ctx["sources"])
# rebuild valid from the same split function
import pickle
from run_task import positives
pos = positives(ctx["kg"], spec); loc = {s: i for i, s in enumerate(ctx["sources"])}
posl = np.stack([[loc[s] for s in pos[:, 0]], pos[:, 1]], 1)
_, va, _ = split(posl, seed)

pair_src = torch.from_numpy(ranker.pair_src.astype(np.int64))
key = {(int(s), int(t)): i for i, (s, t) in enumerate(zip(ranker.pair_src, ranker.pair_tgt))}
pos_pair = torch.tensor([key.get((int(s), int(t)), -1) for s, t in tr])
pos_src = torch.from_numpy(tr[:, 0].astype(np.int64))
is_pos = torch.zeros(sd.n_pairs, dtype=torch.bool); is_pos[pos_pair[pos_pair >= 0]] = True
n_pos_src = torch.zeros(n_src).index_add_(0, pos_src, torch.ones(len(pos_src)))
n_pairs_src = torch.zeros(n_src).index_add_(0, pair_src, torch.ones(sd.n_pairs))
n_pwp = torch.zeros(n_src).index_add_(0, pair_src[is_pos], torch.ones(int(is_pos.sum())))
n0 = ranker.n_tgt - n_pairs_src - (n_pos_src - n_pwp)


def loss_and_grad(model):
    """Unpenalised training loss; accumulates gradients of all trainable parameters (two-pass as in train())."""
    Sp = forward_all(model, sd).detach().requires_grad_(True)
    S0 = model.s0(); S = Sp + S0
    z = torch.clamp(to_logit(S), -30, 30); z0 = torch.clamp(to_logit(S0), -30, 30)
    A = torch.zeros(n_src).index_add(0, pair_src, torch.exp(z) * (~is_pos)) + n0 * torch.exp(z0)
    zp = torch.where(pos_pair >= 0, z[pos_pair.clamp(min=0)], z0)
    loss = (-zp + torch.log(A[pos_src] + torch.exp(zp))).mean()
    loss.backward()
    forward_all(model, sd, grad_S=Sp.grad.detach())
    return float(loss)


def evaluate(model, q):
    with torch.no_grad():
        Sv = forward_all(model, sd) + model.s0()
        zv = torch.clamp(to_logit(Sv), -30, 30).numpy(); z0v = float(torch.clamp(to_logit(model.s0()), -30, 30))
    return metrics(ranker.ranks(zv, z0v, q, known))["MRR"]


def fit(p, warm=None, max_outer=60):
    if p == 0.0:
        m = DNFR(len(ctx["trav"]), len(ctx["P"]["mps"]), tnorm="product", p_mode="shared", degree=True, agg="noisyor")
    else:
        m = DNFR(len(ctx["trav"]), len(ctx["P"]["mps"]), tnorm="ss", p_mode="shared", degree=True, agg="noisyor",
                 p_init=p, learn_p=False, p_param="free")
    m.p_hi = 1.0     # allow p = 1 (Lukasiewicz) on the profile grid
    if warm is not None:
        st = {k: v for k, v in warm.items() if "p_raw" not in k}
        m.load_state_dict(st, strict=False)
    params = [q for q in m.parameters() if q.requires_grad]
    opt = torch.optim.LBFGS(params, lr=1, max_iter=25, history_size=20, line_search_fn="strong_wolfe",
                            tolerance_grad=1e-6, tolerance_change=1e-10)
    last, t0, n_eval = None, time.time(), [0]
    def closure():
        opt.zero_grad(); n_eval[0] += 1
        return torch.tensor(loss_and_grad(m))
    for it in range(max_outer):
        L = float(opt.step(closure))
        opt.zero_grad(); L = loss_and_grad(m)
        # a parameter that no training path reaches has no gradient; count it as zero (as torch's L-BFGS does)
        g = torch.cat([(q.grad if q.grad is not None else torch.zeros_like(q)).flatten() for q in params]).norm().item()
        done = g < 1e-5 or (last is not None and abs(last - L) < 1e-9)
        last = L
        if done:
            break
    # dL/dp at the converged point
    dLdp = None
    if p != 0.0:
        m.p_raw.requires_grad_(True); m.zero_grad(); loss_and_grad(m)
        dLdp = float(m.p_raw.grad.sum()); m.p_raw.requires_grad_(False)
    return m, dict(p=p, loss=L, grad_norm=g, dLdp=dLdp, outer=it + 1, n_eval=n_eval[0], time_s=time.time() - t0,
                   valid_MRR=evaluate(m, va), test_MRR=evaluate(m, te), converged=bool(done))


# two sweeps with warm starts, upward and then downward through the grid; the profile keeps, for each p, the fit
# with the lower loss (a profile is a minimum over the other parameters, and the loss is not convex in them)
best, warm = {}, None
for sweep, order in (("up", grid), ("down", grid[::-1])):
    for p in order:
        m, r = fit(p, warm)
        r["sweep"] = sweep
        warm = {k: v.detach().clone() for k, v in m.state_dict().items()}
        print(r, flush=True)
        if p not in best or r["loss"] < best[p]["loss"]:
            if p in best:
                r["loss_other_sweep"] = best[p]["loss"]
            elif sweep == "down":
                pass
            best[p] = r
        else:
            best[p]["loss_other_sweep"] = r["loss"]
rows = [best[p] for p in grid]
# curvature at the profile minimum from central differences of the loss and of dL/dp
L = np.array([r["loss"] for r in rows]); i = int(np.argmin(L)); ps = np.array(grid)
curv = None
if 0 < i < len(rows) - 1:
    h1, h2 = ps[i] - ps[i - 1], ps[i + 1] - ps[i]
    curv = 2 * (h1 * L[i + 1] - (h1 + h2) * L[i] + h2 * L[i - 1]) / (h1 * h2 * (h1 + h2))
out = dict(task=task, seed=seed, rows=rows, p_min=float(ps[i]), d2L_dp2=curv,
           loss_range=float(L.max() - L.min()), n_train=int(len(tr)))
d = os.environ.get("DNFR_RESULTS", "results2")
json.dump(out, open(f"{d}/profile_lbfgs_{task}_seed{seed}.json", "w"), indent=1)
print("p_min", out["p_min"], "curv", curv, "range", out["loss_range"])
