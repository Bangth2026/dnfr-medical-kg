"""Membership saturation check (reviewer M1/annealing premise): share of hop memberships mu on kept training-graph
paths that are > 0.99 or < 0.01 for trained DNFR models, and share of path values with T(mu) < 1e-6."""
import os, sys, json, torch, numpy as np
import torch.nn.functional as F
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from explain import setup
from dnfr import log_tnorm
out = {}
for task in sys.argv[1:]:
    rows = []
    for seed in range(5):
        try:
            ctx = setup(task, seed)
        except Exception as e:
            print(task, seed, "skip", e); continue
        m, sd = ctx["model"], ctx["sd"]; m.eval()
        hi = lo = n = 0; small = npth = 0
        with torch.no_grad():
            for ch in sd.chunks:
                lm = [F.logsigmoid((m.a[t] - m.b[t] * du - m.c[t] * dv + (0 if lw is None else m.g[t] * lw)) / m.tau)
                      for (t, du, dv, lw) in ch["hops"]]
                for x in lm:
                    mu = x.exp(); hi += int((mu > 0.99).sum()); lo += int((mu < 0.01).sum()); n += mu.numel()
                lt = log_tnorm(m.tnorm, lm, m.p_of(ch["mp"]) if m.tnorm == "ss" else None)
                small += int((lt < np.log(1e-6)).sum()); npth += lt.numel()
        r = dict(seed=seed, mu_gt_099=hi / n, mu_lt_001=lo / n, path_lt_1e6=small / npth, n_hops=n,
                 mu_median=None)
        print(task, r, flush=True); rows.append(r)
    out[task] = rows
json.dump(out, open(os.path.join(os.environ.get("DNFR_RESULTS", "results"), "saturation.json"), "w"), indent=1)
