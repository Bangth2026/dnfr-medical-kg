"""Identifiability of the learned t-norm parameter (second revision).

profile : p fixed (shared by all metapaths) on a grid; every other parameter is trained for up to `epochs`
          epochs without early stopping on validation MRR, until the training loss stops decreasing
          (decrease of less than `tol` over the last 50 epochs). Records the training loss, and the
          validation and test MRR of the parameters with the best validation MRR and of the final parameters.
long    : p learned per metapath from several initial values without penalty, trained for the full `epochs`
          epochs with no early stopping. Records the trajectory of every p_m, the training loss, the norm of
          the gradient with respect to theta, and test MRR at the best-validation and final parameters.

Usage: python3 dnfr/identifiability.py profile het_DpS --seeds 0,1,2
       python3 dnfr/identifiability.py long hkg_DpS --seeds 0,1,2,3,4
Output: results/profile_<task>.json, results/long_<task>.json (merged across invocations).
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import DATA, OUT
import argparse, json, pickle, time
import numpy as np, torch
from run_task import TASKS, load_kg, positives, split
from paths import drop_paths_using
from dnfr import DNFR, SplitData, Ranker, metrics, forward_all, to_logit, score_pairs

GRID = [-4.0, -3.0, -2.0, -1.0, -0.5, -0.1, 0.0, 0.25, 0.5, 0.75]


def context(task):
    spec = TASKS[task]; kg = load_kg(spec["kg"])
    pos = positives(kg, spec); sources = np.unique(pos[:, 0]); loc = {s: i for i, s in enumerate(sources)}
    n_t = kg.n(spec["t"])
    P = pickle.load(open(f"{DATA}/paths_{task}.pkl", "rb"))
    key = P["src"].astype(np.int64) * n_t + P["tgt"]
    ukey, pair_id = np.unique(key, return_inverse=True); pair_id = pair_id.astype(np.int32)
    posl = np.stack([[loc[s] for s in pos[:, 0]], pos[:, 1]], 1)
    known = {}
    for s, t in posl:
        known.setdefault(int(s), set()).add(int(t))
    ranker = Ranker((ukey // n_t).astype(np.int64), (ukey % n_t).astype(np.int64), len(sources), n_t)
    trav = kg.traversals()
    return dict(spec=spec, kg=kg, P=P, ukey=ukey, pair_id=pair_id, posl=posl, known=known, ranker=ranker,
                sources=sources, n_t=n_t, trav=trav, ti={t: i for i, t in enumerate(trav)})


def split_data(ctx, seed):
    spec, sources = ctx["spec"], ctx["sources"]
    tr, va, te = split(ctx["posl"], seed)
    removed = {(spec["s"], spec["t"]): np.concatenate([sources[va[:, 0]].astype(np.int64) * 10**7 + va[:, 1],
                                                       sources[te[:, 0]].astype(np.int64) * 10**7 + te[:, 1]])}
    kg_s = ctx["kg"].copy_without(removed); mask = drop_paths_using(ctx["P"], ctx["kg"], removed)
    sd = SplitData(ctx["P"], kg_s, mask, ctx["pair_id"], len(ctx["ukey"]), ctx["ti"])
    return sd, tr, va, te


def train_long(model, sd, train_q, ranker, valid_q, test_q, known, n_src, epochs=400, lr=0.1, l2=0.0,
               plateau_tol=None, record_p=False):
    """Same loss and optimiser as dnfr.train, without early stopping on validation MRR."""
    opt = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=lr)
    pair_src = torch.from_numpy(ranker.pair_src.astype(np.int64))
    key = {(int(s), int(t)): i for i, (s, t) in enumerate(zip(ranker.pair_src, ranker.pair_tgt))}
    pos_pair = torch.tensor([key.get((int(s), int(t)), -1) for s, t in train_q])
    pos_src = torch.from_numpy(train_q[:, 0].astype(np.int64))
    is_pos = torch.zeros(sd.n_pairs, dtype=torch.bool); is_pos[pos_pair[pos_pair >= 0]] = True
    n_pos_src = torch.zeros(n_src).index_add_(0, pos_src, torch.ones(len(pos_src)))
    n_pairs_src = torch.zeros(n_src).index_add_(0, pair_src, torch.ones(sd.n_pairs))
    n_pos_wp = torch.zeros(n_src).index_add_(0, pair_src[is_pos], torch.ones(int(is_pos.sum())))
    n0 = ranker.n_tgt - n_pairs_src - (n_pos_src - n_pos_wp)
    hist, losses = [], []
    best, best_state, t0 = -1, None, time.time()
    for ep in range(1, epochs + 1):
        Sp = forward_all(model, sd).detach().requires_grad_(True)
        S0 = model.s0(); z = torch.clamp(to_logit(Sp + S0), -30, 30); z0 = torch.clamp(to_logit(S0), -30, 30)
        A = torch.zeros(n_src).index_add(0, pair_src, torch.exp(z) * (~is_pos)) + n0 * torch.exp(z0)
        zp = torch.where(pos_pair >= 0, z[pos_pair.clamp(min=0)], z0)
        loss = (-zp + torch.log(A[pos_src] + torch.exp(zp))).mean()
        reg = l2 * ((model.p_raw - model.p_raw0) ** 2).sum() + 1e-3 * (model.w_raw ** 2).sum() * 0.01
        opt.zero_grad(); (loss + reg).backward()
        forward_all(model, sd, grad_S=Sp.grad.detach())
        gnorm = float(model.p_raw.grad.norm()) if model.p_raw.grad is not None else 0.0
        opt.step(); losses.append(float(loss.detach()))
        if ep % 5 == 0 or ep == epochs:
            with torch.no_grad():
                Sv = forward_all(model, sd) + model.s0()
                zv = torch.clamp(to_logit(Sv), -30, 30).numpy(); z0v = float(torch.clamp(to_logit(model.s0()), -30, 30))
            mv = metrics(ranker.ranks(zv, z0v, valid_q, known))["MRR"]
            mt = metrics(ranker.ranks(zv, z0v, test_q, known))["MRR"]
            h = dict(epoch=ep, train_loss=losses[-1], valid_MRR=mv, test_MRR=mt, grad_theta=gnorm)
            if record_p:
                h["p"] = [round(x, 4) for x in model.p().detach().numpy().tolist()]
            hist.append(h)
            if mv > best + 1e-4:
                best, best_ep = mv, ep
                best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            if plateau_tol is not None and ep >= 100 and losses[-51] - losses[-1] < plateau_tol:
                break
    final = dict(epoch=hist[-1]["epoch"], train_loss=hist[-1]["train_loss"], valid_MRR=hist[-1]["valid_MRR"],
                 test_MRR=hist[-1]["test_MRR"], p=model.p().detach().numpy().tolist())
    model.load_state_dict(best_state)
    bh = [h for h in hist if h["epoch"] == best_ep][0]
    bestd = dict(epoch=best_ep, train_loss=bh["train_loss"], valid_MRR=bh["valid_MRR"], test_MRR=bh["test_MRR"],
                 p=model.p().detach().numpy().tolist())
    return dict(final=final, best_valid=bestd, history=hist, time_s=time.time() - t0)


def merge_save(path, key, rec):
    d = json.load(open(path)) if os.path.exists(path) else {}
    d[key] = rec
    json.dump(d, open(path, "w"), indent=0)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["profile", "long"]); ap.add_argument("task")
    ap.add_argument("--seeds", default="0,1,2"); ap.add_argument("--grid", default=",".join(map(str, GRID)))
    ap.add_argument("--inits", default="-2,-0.1,0.5"); ap.add_argument("--epochs", type=int, default=400)
    a = ap.parse_args()
    ctx = context(a.task); nmp = len(ctx["P"]["mps"])
    out = f"{OUT}/{a.mode}_{a.task}.json"
    done = json.load(open(out)) if os.path.exists(out) else {}
    for seed in [int(s) for s in a.seeds.split(",")]:
        sd, tr, va, te = split_data(ctx, seed)
        values = [float(x) for x in (a.grid if a.mode == "profile" else a.inits).split(",")]
        for v in values:
            key = f"seed{seed}_p{v:g}"
            if key in done:
                continue
            torch.manual_seed(seed)
            if a.mode == "profile":
                model = DNFR(len(ctx["trav"]), nmp, tnorm="ss", p_mode="shared", p_init=v)
                model.p_raw.requires_grad_(False)
                rec = train_long(model, sd, tr, ctx["ranker"], va, te, ctx["known"], len(ctx["sources"]),
                                 epochs=a.epochs, plateau_tol=1e-4)
            else:
                model = DNFR(len(ctx["trav"]), nmp, tnorm="ss", p_mode="per_mp", p_init=v)
                rec = train_long(model, sd, tr, ctx["ranker"], va, te, ctx["known"], len(ctx["sources"]),
                                 epochs=a.epochs, record_p=True)
            rec.update(seed=seed, value=v, mp_names=ctx["P"]["mp_names"])
            print(f"[{a.mode} {a.task}] seed {seed} p {v:g}: best ep {rec['best_valid']['epoch']} "
                  f"test {rec['best_valid']['test_MRR']:.4f} | final ep {rec['final']['epoch']} loss {rec['final']['train_loss']:.5f} "
                  f"test {rec['final']['test_MRR']:.4f} ({rec['time_s']:.0f}s)", flush=True)
            merge_save(out, key, rec); done[key] = rec
