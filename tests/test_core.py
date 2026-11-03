"""Checks of the components the conclusions rest on: leave-dyad-out removal, the Schweizer-Sklar t-norm and
its limits, and filtered ranking with mean-rank ties. Run: python3 tests/test_core.py (needs data/hkg.pkl)."""
import os, sys, pickle
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "dnfr"))
import numpy as np, torch
from dnfr import log_tnorm, Ranker
from config import DATA


def test_tnorm_limits():
    rng = np.random.default_rng(0)
    mu = [torch.tensor(rng.uniform(0.05, 1, 200)) for _ in range(3)]
    lm = [torch.log(m) for m in mu]
    prod = torch.exp(log_tnorm("product", lm))
    assert torch.allclose(torch.exp(log_tnorm("ss", lm, torch.tensor(1e-3))), prod, rtol=2e-2)
    # p = -1 is the Hamacher product (lambda = 0)
    assert torch.allclose(torch.exp(log_tnorm("ss", lm, torch.tensor(-1.0))), torch.exp(log_tnorm("hamacher", lm)), rtol=1e-4)
    # p = 1 is the Lukasiewicz t-norm where it is positive
    luk = mu[0] + mu[1] + mu[2] - 2
    ss1 = torch.exp(log_tnorm("ss", lm, torch.tensor(1.0)))
    pos = luk > 1e-3
    assert torch.allclose(ss1[pos], luk[pos], rtol=1e-3)
    # p -> -inf approaches the minimum
    mn = torch.exp(log_tnorm("min", lm))
    assert torch.all(torch.abs(torch.exp(log_tnorm("ss", lm, torch.tensor(-60.0))) - mn) < 0.03)
    # numerically stable for tiny memberships and p < 0 (log-sum-exp branch)
    tiny = [torch.log(torch.tensor([1e-30, 1e-3])), torch.log(torch.tensor([0.5, 0.5]))]
    v = log_tnorm("ss", tiny, torch.tensor(-4.0))
    assert torch.all(torch.isfinite(v))


def test_leave_dyad_out():
    from run_task import TASKS, positives, split
    from paths import drop_paths_using
    kg = pickle.load(open(f"{DATA}/hkg.pkl", "rb")); spec = TASKS["hkg_DpS"]
    pos = positives(kg, spec); src = np.unique(pos[:, 0]); loc = {s: i for i, s in enumerate(src)}
    posl = np.stack([[loc[s] for s in pos[:, 0]], pos[:, 1]], 1)
    tr, va, te = split(posl, 0)
    held = np.concatenate([src[va[:, 0]].astype(np.int64) * 10**7 + va[:, 1], src[te[:, 0]].astype(np.int64) * 10**7 + te[:, 1]])
    kg_s = kg.copy_without({(spec["s"], spec["t"]): held})
    d = kg_s.rels["DpS"]
    assert not np.isin(d["h"].astype(np.int64) * 10**7 + d["t"], held).any()
    assert len(d["h"]) == len(kg.rels["DpS"]["h"]) - len(held)
    P = pickle.load(open(f"{DATA}/paths_hkg_DpS.pkl", "rb"))
    keep = drop_paths_using(P, kg, {(spec["s"], spec["t"]): held})
    s = P["sources"][P["src"][keep]]; k1, k2, t = P["k1"][keep], P["k2"][keep], P["tgt"][keep]
    # path D(s) -> S(k1) <- D(k2) -> S(t): none of its three edges may be a held-out dyad
    for a, b in ((s, k1), (k2, k1), (k2, t)):
        assert not np.isin(a.astype(np.int64) * 10**7 + b, held).any()


def test_filtered_ranking():
    rng = np.random.default_rng(1)
    n_src, n_t = 4, 30
    Z = rng.normal(size=(n_src, n_t)).round(1)  # rounding creates ties
    keep = rng.random((n_src, n_t)) < 0.6       # pairs with a path; others share z0
    z0 = -5.0
    ps, pt = np.where(keep)
    r = Ranker(ps, pt, n_src, n_t)
    known = {s: set(rng.choice(n_t, 4, replace=False).tolist()) for s in range(n_src)}
    queries = np.array([(s, t) for s in range(n_src) for t in known[s]])
    got = r.ranks(Z[ps, pt], z0, queries, known)
    for (s, t), g in zip(queries, got):
        row = np.where(keep[s], Z[s], z0).astype(float)
        others = [u for u in range(n_t) if u != t and u not in known[s]]
        gt = sum(row[u] > row[t] for u in others); eq = sum(row[u] == row[t] for u in others)
        assert abs(g - (1 + gt + eq / 2)) < 1e-9


if __name__ == "__main__":
    for f in (test_tnorm_limits, test_leave_dyad_out, test_filtered_ranking):
        f(); print(f.__name__, "ok")
