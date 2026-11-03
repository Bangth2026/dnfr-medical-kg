"""Merge the third-revision jobs (results2/jobs/<id>/, completed only) into results/<task>.json by split, so that the
second-revision aggregation (aggregate.py, revision_stats.py) sees every split and variant. Per-run dictionaries
(methods, ranks, p_learned, ops_chosen, ...) are merged key by key; a method present in both keeps the job's value
only if it was absent before (rev2 runs are never overwritten). Models and query files are copied alongside."""
import os, glob, json, shutil
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DICT_FIELDS = ("methods", "ranks", "ops_chosen", "op_probs", "ops_trace", "beta", "tau_best", "p_learned", "w_learned")

for task in ("het_CtD", "het_DpS", "hkg_DpS"):
    f0 = f"{ROOT}/results/{task}.json"
    base = json.load(open(f0))
    runs = {r["seed"]: r for r in base["runs"]}
    added = 0
    for f in sorted(glob.glob(f"{ROOT}/results2/jobs/*/{task}.json")):
        jd = os.path.dirname(f)
        if not os.path.exists(f"{jd}/DONE"):
            continue
        for r in json.load(open(f))["runs"]:
            R = runs.get(r["seed"])
            if R is None:
                runs[r["seed"]] = r; added += len(r["methods"]); continue
            for k in DICT_FIELDS:
                for m, v in r.get(k, {}).items():
                    if m not in R.setdefault(k, {}):
                        R[k][m] = v; added += k == "methods"
            for k, v in r.items():
                if k not in R:
                    R[k] = v
        for g in glob.glob(f"{jd}/*.pt") + glob.glob(f"{jd}/*.npy"):
            dst = f"{ROOT}/results/{os.path.basename(g)}"
            if not os.path.exists(dst):
                shutil.copy(g, dst)
    base["runs"] = [runs[s] for s in sorted(runs)]
    json.dump(base, open(f0, "w"))
    print(task, "splits", sorted(runs), "methods added", added)
