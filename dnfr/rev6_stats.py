"""Revision 6 (JBI reviewer): statistics computed from released result files, without new training.
- chance MRR: expected MRR of a uniformly random ranker under the same filtered protocol, per task and split type
  (for each test query with n filtered candidates, E[1/R] = H_n / n; averaged over test queries and splits);
- chain envelope of the converged profiles (M1a): for each p, the losses of all warm-start chains, the best and second
  best converged fit, and whether the two best agree within a tolerance of 1e-4 nat;
- confidence intervals for the clinical assessment (M6): Wilson intervals for proportions, stratified bootstrap
  intervals (5000 resamples of pairs within each class) for the AUC of usefulness and of plausibility, and bootstrap
  intervals for Cohen's kappa (resampling items).
Output: results_rev6/rev6_stats.json"""
import os, sys, json, glob, collections, math
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
R = os.path.join(ROOT, "results"); OUT6 = os.path.join(ROOT, "results_rev6")


def harmonic(n):
    return float(np.sum(1.0 / np.arange(1, n + 1)))


def chance_mrr(task_file):
    d = json.load(open(os.path.join(R, task_file)))
    n_t = d["n_tails"]
    # known tails per (local) source: all positives over train, valid and test of the first run
    vals = []
    for r in d["runs"]:
        tq = np.array(r["test_queries"])
        known = collections.Counter()
        # reconstruct positives: training + valid + test sizes are not stored per pair, so use the test queries and
        # the overall positives per source when available
        vals.append((tq, r))
    return n_t, vals


def chance_from_positives(task):
    """Exact expected MRR of a random ranker: requires the positives per source; rebuilt from the graph pickle."""
    import pickle
    from run_task import TASKS, positives, load_kg
    spec = TASKS[task]; kg = load_kg(spec["kg"])
    pos = positives(kg, spec); n_t = kg.n(spec["t"])
    per_src = collections.Counter(pos[:, 0].tolist())
    # every positive (s, t) is ranked among n_t - (known(s) - 1) candidates
    e = [harmonic(n_t - per_src[s] + 1) / (n_t - per_src[s] + 1) for s, _ in pos]
    return dict(n_candidates=int(n_t), mean_filtered_candidates=float(np.mean([n_t - per_src[s] + 1 for s, _ in pos])),
                chance_MRR=float(np.mean(e)), chance_MRR_unfiltered=harmonic(n_t) / n_t)


def envelope(task, tol=1e-4):
    out = {}
    for seed in (0, 1, 2):
        L = collections.defaultdict(list); V = {}
        for f in glob.glob(f"{R}/**/profile_lbfgs_{task}_seed{seed}.json", recursive=True):
            for r in json.load(open(f))["rows"]:
                L[r["p"]].append(r["loss"])
                if r.get("loss_other_sweep") is not None:
                    L[r["p"]].append(r["loss_other_sweep"])
                if r["p"] not in V or r["loss"] <= min(L[r["p"]]):
                    V[r["p"]] = (r["valid_MRR"], r["test_MRR"])
        ps = sorted(L); best = {p: min(L[p]) for p in ps}; m0 = min(best.values())
        rows = []
        for p in ps:
            s = sorted(L[p])
            rows.append(dict(p=p, n_chains=len(s), excess=(s[0] - m0), worst_minus_best=(s[-1] - s[0]),
                             second_minus_best=(s[1] - s[0]) if len(s) > 1 else None,
                             resolved=bool(len(s) > 1 and s[1] - s[0] < tol), valid_MRR=V[p][0], test_MRR=V[p][1]))
        res = [r for r in rows if r["resolved"]]
        ex = [r["excess"] for r in res]
        out[seed] = dict(rows=rows, n_points=len(rows), n_resolved=len(res),
                         profile_range_resolved=(max(ex) - min(ex)) if ex else None,
                         max_second_minus_best_resolved=max(r["second_minus_best"] for r in res) if res else None,
                         p_min=min(rows, key=lambda r: r["excess"])["p"],
                         p_best_valid=max(rows, key=lambda r: r["valid_MRR"])["p"])
    return out


def wilson(k, n, z=1.96):
    if n == 0:
        return None
    ph = k / n; den = 1 + z * z / n
    c = (ph + z * z / (2 * n)) / den; h = z * math.sqrt(ph * (1 - ph) / n + z * z / (4 * n * n)) / den
    return [k, n, ph, max(0.0, c - h), min(1.0, c + h)]


def auc(pos, neg):
    pos, neg = np.asarray(pos, float), np.asarray(neg, float)
    return float(((pos[:, None] > neg[None]) + 0.5 * (pos[:, None] == neg[None])).mean())


def boot_auc(pos, neg, B=5000, seed=0):
    rng = np.random.default_rng(seed); pos, neg = np.asarray(pos, float), np.asarray(neg, float)
    a = [auc(rng.choice(pos, len(pos)), rng.choice(neg, len(neg))) for _ in range(B)]
    return [auc(pos, neg), float(np.percentile(a, 2.5)), float(np.percentile(a, 97.5))]


def kappa(a, b):
    cats = sorted(set(a) | set(b)); n = len(a)
    po = np.mean([x == y for x, y in zip(a, b)])
    pe = sum(np.mean([x == c for x in a]) * np.mean([y == c for y in b]) for c in cats)
    return (po - pe) / (1 - pe) if pe < 1 else float("nan")


def boot_kappa(a, b, B=5000, seed=0):
    rng = np.random.default_rng(seed); a, b = list(a), list(b); n = len(a)
    ks = []
    for _ in range(B):
        i = rng.integers(0, n, n); ks.append(kappa([a[j] for j in i], [b[j] for j in i]))
    ks = np.array(ks); ks = ks[np.isfinite(ks)]
    return [float(kappa(a, b)), float(np.percentile(ks, 2.5)), float(np.percentile(ks, 97.5)), n]


def clinical():
    from openpyxl import load_workbook
    norm = lambda x: (str(x).strip().lower() if x is not None and str(x).strip() else None)
    def rows_of(f):
        ws = load_workbook(f, data_only=True)["Ratings"]
        return {r[0]: r for r in ws.iter_rows(min_row=2, values_only=True) if r[0]}
    key = json.load(open(f"{R}/expert_review2_key.json"))
    R2 = rows_of(f"{R}/expert_review2_filled.xlsx"); R1 = rows_of(f"{R}/expert_review_filled.xlsx")
    pairs = [k for k in key if "-p" not in k]; paths = [k for k in key if "-p" in k]
    ov_paths = [k for k in paths if key[k]["first_rater_id"]]
    out = {}
    for col, name in ((5, "relations_correct"), (6, "rationale")):
        a = [norm(R1[key[k]["first_rater_id"]][col]) for k in ov_paths]; b = [norm(R2[k][col]) for k in ov_paths]
        ab = [(x, y) for x, y in zip(a, b) if x and y]
        out["kappa_" + name] = boot_kappa([x for x, _ in ab], [y for _, y in ab])
    def grp(correct):
        P = [k for k in pairs if key[k]["correct"] == correct]
        Q = [k for k in paths if key[k.rsplit("-p", 1)[0]]["correct"] == correct]
        pl = [norm(R2[k][7]) for k in P]; rc = [norm(R2[k][5]) for k in Q if norm(R2[k][5])]
        us = [int(R2[k][8]) for k in P if R2[k][8] not in (None, "")]
        pl_ = [x for x in pl if x]
        return dict(plausible=wilson(sum(x == "yes" for x in pl_), len(pl_)),
                    relations_correct=wilson(sum(x == "yes" for x in rc), len(rc)), usefulness=us,
                    plaus_score=[{"yes": 1, "unsure": 0.5, "no": 0}[x] for x in pl_])
    G1, G0 = grp(True), grp(False)
    out["correct"] = {k: v for k, v in G1.items() if k not in ("usefulness", "plaus_score")}
    out["incorrect"] = {k: v for k, v in G0.items() if k not in ("usefulness", "plaus_score")}
    out["auc_usefulness"] = boot_auc(G1["usefulness"], G0["usefulness"])
    out["auc_plausible"] = boot_auc(G1["plaus_score"], G0["plaus_score"])
    # first rater: proportions with Wilson intervals, from her sheet
    first = {}
    for task in ("het_DpS", "hkg_DpS"):
        ks = [k for k in R1 if str(k).startswith(task) and "-p" in str(k)]
        rc = [norm(R1[k][5]) for k in ks]
        first[task] = {lab: wilson(sum(x == lab for x in rc), len(rc)) for lab in ("yes", "no", "unsure")}
        first[task]["n_missing"] = sum(x is None for x in rc)
    out["first_rater_relations"] = first
    return out


if __name__ == "__main__":
    res = dict(chance={t: chance_from_positives(t) for t in ("het_CtD", "het_DpS", "hkg_DpS")},
               envelope={t: envelope(t) for t in ("hkg_DpS", "het_DpS")})
    try:
        res["clinical"] = clinical()
    except Exception as e:
        res["clinical_error"] = repr(e)
    os.makedirs(OUT6, exist_ok=True)
    json.dump(res, open(f"{OUT6}/rev6_stats.json", "w"), indent=1)
    print(json.dumps({k: v for k, v in res.items() if k != "envelope"}, indent=1)[:6000])
    for t, e in res["envelope"].items():
        for s, v in e.items():
            print(t, s, {k: v[k] for k in v if k != "rows"})
