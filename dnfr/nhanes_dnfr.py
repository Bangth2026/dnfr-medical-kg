"""Secondary test on graded population evidence (NHANES fuzzy relation, 24 concepts).
Trains on the development relation (leave-edge-out) and evaluates rank agreement
(Spearman) with the relation measured in independent held-out survey cycles."""
import os, sys
from config import DATA, OUT
import itertools, json, sys
import numpy as np, torch, torch.nn.functional as F
from scipy.stats import spearmanr
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dnfr import log_tnorm, P_RAW0

torch.set_num_threads(1)
D = os.path.join(DATA, "nhanes")
Edev, Ehold = np.load(f"{D}/E_dev.npy"), np.load(f"{D}/E_hold.npy")
n = Edev.shape[0]
TYPES = [0] * 9 + [1] * 4 + [2] * 4 + [3] * 7    # symptom, socioeconomic, lifestyle, health (Table 1 order)

# enumerate simple paths of length 2 and 3 for every ordered pair
pairs = [(i, j) for i in range(n) for j in range(n) if i != j]
rows = []  # (pair index, nodes tuple)
for pi, (i, j) in enumerate(pairs):
    others = [k for k in range(n) if k not in (i, j)]
    for k in others:
        rows.append((pi, (i, k, j)))
    for k, l in itertools.permutations(others, 2):
        rows.append((pi, (i, k, l, j)))
pair_of = torch.tensor([r[0] for r in rows])
L = np.array([len(r[1]) - 1 for r in rows])
mp_key = [tuple(TYPES[v] for v in r[1]) for r in rows]
mps = sorted(set(mp_key)); mp_index = {m: i for i, m in enumerate(mps)}
mp_of = torch.tensor([mp_index[m] for m in mp_key])
reltype = lambda a, b: TYPES[a] * 4 + TYPES[b]
hop_mu, hop_rt = [], []
for h in range(3):
    mu = np.array([Edev[r[1][h], r[1][h + 1]] if len(r[1]) - 1 > h else 1.0 for r in rows])
    rt = np.array([reltype(r[1][h], r[1][h + 1]) if len(r[1]) - 1 > h else 0 for r in rows])
    hop_mu.append(torch.tensor(np.clip(mu, 1e-4, 1 - 1e-4), dtype=torch.float32)); hop_rt.append(torch.tensor(rt))
is3 = torch.tensor(L == 3)
y_dev = torch.tensor([Edev[i, j] for i, j in pairs], dtype=torch.float32)
y_hold = np.array([Ehold[i, j] for i, j in pairs])


class NhanesDNFR(torch.nn.Module):
    def __init__(self, tnorm="ss", p_mode="per_mp", calib=True, agg="max"):
        super().__init__()
        self.a = torch.nn.Parameter(torch.zeros(16)); self.g = torch.nn.Parameter(torch.ones(16))
        for prm in (self.a, self.g):
            prm.requires_grad = calib
        self.p_raw = torch.nn.Parameter(torch.full((len(mps) if p_mode == "per_mp" else 1,), P_RAW0))
        self.w_raw = torch.nn.Parameter(torch.zeros(len(mps)), requires_grad=(agg == "noisyor"))
        self.tnorm, self.p_mode, self.calib, self.agg = tnorm, p_mode, calib, agg

    def forward(self):
        logmu = []
        for h in range(3):
            lm = torch.log(hop_mu[h])
            if self.calib:
                x = torch.logit(hop_mu[h])
                lm = F.logsigmoid(self.a[hop_rt[h]] + self.g[hop_rt[h]] * x)
            if h == 2:
                lm = torch.where(is3, lm, torch.zeros_like(lm))  # length-2 paths: neutral third hop
            logmu.append(lm)
        p = 6 * torch.sigmoid(self.p_raw) - 5
        pp = p[mp_of] if self.p_mode == "per_mp" else p[0]
        if self.tnorm == "ss":
            lt3 = log_tnorm("ss", logmu, pp)
            lt2 = log_tnorm("ss", logmu[:2], pp)
        else:
            lt3 = log_tnorm(self.tnorm, logmu); lt2 = log_tnorm(self.tnorm, logmu[:2])
        lt = torch.where(is3, lt3, lt2)
        if self.agg == "max":
            out = torch.full((len(pairs),), -1e9).scatter_reduce(0, pair_of, lt, reduce="amax")
            return torch.exp(out)
        lv = lt + F.logsigmoid(self.w_raw[mp_of])
        S = torch.zeros(len(pairs)).index_add(0, pair_of, torch.log1p(-torch.exp(torch.clamp(lv, max=-1e-6))))
        return -torch.expm1(S)


def fit(model, steps=600, lr=0.05, l2=1e-2):
    params = [p for p in model.parameters() if p.requires_grad]
    if params:
        opt = torch.optim.Adam(params, lr=lr)
        for _ in range(steps):
            pred = model()
            loss = ((pred - y_dev) ** 2).mean() + l2 * ((model.p_raw - P_RAW0) ** 2).mean()
            opt.zero_grad(); loss.backward(); opt.step()
    with torch.no_grad():
        pred = model().numpy()
    return dict(rho_hold=float(spearmanr(pred, y_hold)[0]), rho_dev=float(spearmanr(pred, y_dev.numpy())[0]),
                mae_hold=float(np.abs(pred - y_hold).mean()),
                p=(6 / (1 + np.exp(-model.p_raw.detach().numpy())) - 5).round(3).tolist() if model.tnorm == "ss" else None)


if __name__ == "__main__":
    torch.manual_seed(0)
    res = {}
    configs = {
        "product (fixed, max)": dict(tnorm="product", p_mode="shared", calib=False, agg="max"),
        "hamacher (fixed, max)": dict(tnorm="hamacher", p_mode="shared", calib=False, agg="max"),
        "min (fixed, max)": dict(tnorm="min", p_mode="shared", calib=False, agg="max"),
        "luk (fixed, max)": dict(tnorm="luk", p_mode="shared", calib=False, agg="max"),
        "SS shared p (max)": dict(tnorm="ss", p_mode="shared", calib=False, agg="max"),
        "SS per-metapath p (max)": dict(tnorm="ss", p_mode="per_mp", calib=False, agg="max"),
        "DNFR: calib + per-metapath p (max)": dict(tnorm="ss", p_mode="per_mp", calib=True, agg="max"),
        "DNFR: calib + per-metapath p (noisy-OR)": dict(tnorm="ss", p_mode="per_mp", calib=True, agg="noisyor"),
        "calib + product (noisy-OR)": dict(tnorm="product", p_mode="shared", calib=True, agg="noisyor"),
    }
    for name, cfg in configs.items():
        r = fit(NhanesDNFR(**cfg))
        if r["p"] is not None and len(r["p"]) > 1:
            r["p_summary"] = [float(np.min(r["p"])), float(np.median(r["p"])), float(np.max(r["p"]))]; del r["p"]
        res[name] = r
        print(name, {k: v for k, v in r.items()}, flush=True)
    res["_info"] = dict(n_pairs=len(pairs), n_paths=len(rows), n_metapaths=len(mps))
    json.dump(res, open(os.path.join(OUT, "nhanes_dnfr.json"), "w"), indent=1)
