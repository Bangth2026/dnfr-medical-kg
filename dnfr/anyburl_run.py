"""AnyBURL (Meilicke et al., 2019) on the same leave-dyad-out splits as DNFR.

Usage: python anyburl_run.py <task> --seeds 0,1,2 --jar AnyBURL-23-1.jar [--learn_s 100]
Writes the training graph G' of each split (every relation, val/test dyads removed), learns rules for the
target relation only (SINGLE_RELATIONS), applies them to the test pairs and ranks the true tail with the
filtered, mean-rank-tie protocol of run_task.py. Tails not in AnyBURL's top-k share the remaining positions."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import DATA, OUT
import argparse, json, subprocess, time
import numpy as np
from run_task import TASKS, load_kg, positives, split
from dnfr import metrics


def ent(ty, i):
    return f"{ty}_{int(i)}"


def write_split(task, seed, work):
    spec = TASKS[task]; kg = load_kg(spec["kg"])
    pos = positives(kg, spec); sources = np.unique(pos[:, 0]); loc = {s: i for i, s in enumerate(sources)}
    posl = np.stack([[loc[s] for s in pos[:, 0]], pos[:, 1]], 1)
    tr, va, te = split(posl, seed)
    removed = {(spec["s"], spec["t"]): np.concatenate([sources[va[:, 0]].astype(np.int64) * 10**7 + va[:, 1],
                                                       sources[te[:, 0]].astype(np.int64) * 10**7 + te[:, 1]])}
    kg_s = kg.copy_without(removed)
    os.makedirs(work, exist_ok=True)
    with open(f"{work}/train.txt", "w") as f:
        for r, d in kg_s.rels.items():
            rn = r.replace(" ", "_")
            for h, t in zip(d["h"], d["t"]):
                f.write(f"{ent(d['src'], h)}\t{rn}\t{ent(d['dst'], t)}\n")
    rel = spec["rel"].replace(" ", "_"); d = kg.rels[spec["rel"]]
    def trip(q):  # query (source, tail) -> stored orientation
        s, t = sources[q[0]], q[1]
        return (ent(spec["t"], t), rel, ent(spec["s"], s)) if spec["rev"] else (ent(spec["s"], s), rel, ent(spec["t"], t))
    for name, qs in (("valid", va), ("test", te)):
        with open(f"{work}/{name}.txt", "w") as f:
            for q in qs:
                f.write("\t".join(trip(q)) + "\n")
    known = {}
    for s, t in posl:
        known.setdefault(int(s), set()).add(int(t))
    return dict(spec=spec, kg=kg, sources=sources, te=te, known=known, rel=rel, n_t=kg.n(spec["t"]))


def run(task, seed, jar, learn_s=100, topk=200):
    work = f"{DATA}/anyburl/{task}_s{seed}"
    ctx = write_split(task, seed, work)
    open(f"{work}/config-learn.properties", "w").write(
        f"PATH_TRAINING = {work}/train.txt\nPATH_OUTPUT = {work}/rules\nSNAPSHOTS_AT = {learn_s}\n"
        f"WORKER_THREADS = 1\nSINGLE_RELATIONS = {ctx['rel']}\n")
    open(f"{work}/config-apply.properties", "w").write(
        f"PATH_TRAINING = {work}/train.txt\nPATH_VALID = {work}/valid.txt\nPATH_TEST = {work}/test.txt\n"
        f"PATH_RULES = {work}/rules-{learn_s}\nPATH_OUTPUT = {work}/preds-{learn_s}\nTOP_K_OUTPUT = {topk}\nWORKER_THREADS = 1\n")
    t0 = time.time()
    jv = ["java", os.environ.get("ANYBURL_XMX", "-Xmx2G"), "-cp", jar]
    # AnyBURL's Learn ends with System.exit(1) after storing the rules, so success is checked by the output files
    subprocess.run(jv + ["de.unima.ki.anyburl.Learn", f"{work}/config-learn.properties"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    assert os.path.exists(f"{work}/rules-{learn_s}"), "AnyBURL learning failed"
    subprocess.run(jv + ["de.unima.ki.anyburl.Apply", f"{work}/config-apply.properties"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    assert os.path.exists(f"{work}/preds-{learn_s}"), "AnyBURL application failed"
    elapsed = time.time() - t0
    lines = open(f"{work}/preds-{learn_s}").read().split("\n")
    pred = {}
    for i in range(0, len(lines) - 2, 3):
        h, r, t = lines[i].split()
        side = lines[i + 1] if ctx["spec"]["rev"] else lines[i + 2]   # "Heads: ..." or "Tails: ..."
        side = side.split(":", 1)[1].strip().split("\t") if ":" in side else []
        cands = [(side[j], float(side[j + 1])) for j in range(0, len(side) - 1, 2)]
        pred[(h, t)] = cands
    ty = ctx["spec"]["t"]; ranks = []
    for s, t in ctx["te"]:
        src = ctx["sources"][s]
        key = (ent(ty, t), ent(ctx["spec"]["s"], src)) if ctx["spec"]["rev"] else (ent(ctx["spec"]["s"], src), ent(ty, t))
        filt = ctx["known"].get(int(s), set()) - {int(t)}
        cands = [(int(c.split("_")[-1]), sc) for c, sc in pred.get(key, []) if c.startswith(ty + "_")]
        cands = [(c, sc) for c, sc in cands if c not in filt]
        n_valid = ctx["n_t"] - len(filt)
        sc_t = dict(cands).get(int(t))
        if sc_t is not None:
            others = [sc for c, sc in cands if c != int(t)]
            rank = 1 + sum(x > sc_t for x in others) + sum(x == sc_t for x in others) / 2
        else:
            U = n_valid - len(cands)
            rank = len(cands) + (U + 1) / 2
        ranks.append(rank)
    m = metrics(ranks); m["time_s"] = elapsed; m["learn_s"] = learn_s
    return m, ranks


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("task"); ap.add_argument("--seeds", default="0"); ap.add_argument("--jar", required=True)
    ap.add_argument("--learn_s", type=int, default=100)
    a = ap.parse_args()
    for seed in [int(x) for x in a.seeds.split(",")]:
        m, ranks = run(a.task, seed, a.jar, a.learn_s)
        print(a.task, seed, m, flush=True)
        json.dump(dict(task=a.task, seed=seed, metrics=m, ranks=list(map(float, ranks))),
                  open(f"{OUT}/anyburl_{a.task}_seed{seed}.json", "w"))
