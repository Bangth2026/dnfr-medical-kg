import os, sys
from config import DATA, OUT
import json, sys, numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import nhanes_dnfr as N
from scipy.stats import spearmanr
node_sets = [set(r[1]) for r in N.rows]
def run(cfg, drop):
    pm = torch.tensor([drop not in s for s in node_sets])
    qm = np.array([drop not in (i, j) for i, j in N.pairs])
    # restrict globals
    saved = (N.rows, N.pair_of, N.mp_of, N.hop_mu, N.hop_rt, N.is3)
    N.pair_of = N.pair_of[pm]; N.mp_of = N.mp_of[pm]; N.hop_mu = [h[pm] for h in N.hop_mu]; N.hop_rt = [h[pm] for h in N.hop_rt]; N.is3 = N.is3[pm]
    torch.manual_seed(0)
    m = N.NhanesDNFR(**cfg)
    params = [p for p in m.parameters() if p.requires_grad]
    qmt = torch.tensor(qm)
    if params:
        opt = torch.optim.Adam(params, lr=0.05)
        for _ in range(600):
            pred = m(); loss = ((pred - N.y_dev)[qmt] ** 2).mean() + 1e-2 * ((m.p_raw - N.P_RAW0) ** 2).mean()
            opt.zero_grad(); loss.backward(); opt.step()
    with torch.no_grad():
        pred = m().numpy()
    N.pair_of, N.mp_of, N.hop_mu, N.hop_rt, N.is3 = saved[1:]
    return float(spearmanr(pred[qm], N.y_hold[qm])[0])
cfgs = {"product": dict(tnorm="product", p_mode="shared", calib=False, agg="max"),
        "dnfr": dict(tnorm="ss", p_mode="per_mp", calib=True, agg="noisyor"),
        "calib_product": dict(tnorm="product", p_mode="shared", calib=True, agg="noisyor")}
out = {k: [run(c, d) for d in range(N.n)] for k, c in cfgs.items()}
diff = np.array(out["dnfr"]) - np.array(out["product"]); diff2 = np.array(out["dnfr"]) - np.array(out["calib_product"])
n = N.n
se = lambda x: float(np.sqrt((n - 1) / n * ((x - x.mean()) ** 2).sum()))
res = dict(jack=out, dnfr_minus_product=dict(min=float(diff.min()), max=float(diff.max()), jack_se=se(diff), all_positive=bool((diff > 0).all())),
           dnfr_minus_calibproduct=dict(min=float(diff2.min()), max=float(diff2.max()), jack_se=se(diff2), n_positive=int((diff2 > 0).sum())))
json.dump(res, open(os.path.join(OUT, "nhanes_jack.json"), "w"), indent=1)
print({k: v for k, v in res.items() if k != "jack"})
