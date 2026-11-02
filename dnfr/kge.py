"""KGE baselines (PyKEEN) with type-restricted, filtered ranking on the target relation."""
import os, sys
from config import DATA, OUT
import argparse, json, os, pickle, sys, time
import numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from run_task import TASKS, positives, split, load_kg
from dnfr import metrics
from pykeen.triples import TriplesFactory

torch.set_num_threads(int(os.environ.get("NT", "2")))

NEGS = 4


def build(kgname, tasks, seed, sparsity=1.0):
    kg = load_kg(kgname)
    full = kg
    if sparsity < 1.0:
        kg = kg.subsample(sparsity, np.random.default_rng(12345))
    offsets, off = {}, 0
    for ty in sorted(kg.names):
        offsets[ty] = off; off += kg.n(ty)
    removed_by_task, evals = {}, {}
    for task in tasks:
        spec = TASKS[task]
        pos = positives(full, spec)
        srcs = np.unique(pos[:, 0]); loc = {s: i for i, s in enumerate(srcs)}
        posl = np.stack([[loc[s] for s in pos[:, 0]], pos[:, 1]], 1)
        tr, va, te = split(posl, seed)
        keys = np.concatenate([srcs[va[:, 0]].astype(np.int64) * 10**7 + va[:, 1], srcs[te[:, 0]].astype(np.int64) * 10**7 + te[:, 1]])
        removed_by_task[(spec["s"], spec["t"])] = keys
        known = {}
        for s, t in pos:
            known.setdefault(int(s), set()).add(int(t))
        evals[task] = dict(spec=spec, valid=np.stack([srcs[va[:, 0]], va[:, 1]], 1), test=np.stack([srcs[te[:, 0]], te[:, 1]], 1), known=known)
    kg_s = kg.copy_without(removed_by_task)
    trip = []
    for r, d in kg_s.rels.items():
        h = d["h"] + offsets[d["src"]]; t = d["t"] + offsets[d["dst"]]
        trip.append(np.stack([h, np.full(len(h), -1), t], 1).astype(object))
        trip[-1][:, 1] = r
    trip = np.concatenate(trip)
    trip = np.stack([trip[:, 0].astype(str), trip[:, 1].astype(str), trip[:, 2].astype(str)], 1)
    return kg_s, offsets, trip, evals


def evaluate(model, tf, offsets, kg, ev, which):
    spec = ev["spec"]
    q = ev[which]
    rel = spec["rel"]
    ent2id, rel2id = tf.entity_to_id, tf.relation_to_id
    cand = np.arange(kg.n(spec["t"])) + offsets[spec["t"]]
    ids = np.array([ent2id.get(str(c), -1) for c in cand])
    present = ids >= 0
    cand_ids = torch.tensor(np.where(present, ids, 0))
    ranks = []
    model.eval()
    with torch.no_grad():
        for s, t in q:
            sg = s + offsets[spec["s"]]
            if str(sg) not in ent2id:  # head isolated in the training graph: all candidates tie
                n_f = len(cand) - len([x for x in ev["known"].get(int(s), set()) if x != t])
                ranks.append((n_f + 1) / 2); continue
            if spec["rev"]:   # stored as (disease, indication, drug): score heads
                sc = model.score_h(torch.tensor([[rel2id[rel], ent2id[str(sg)]]]))[0]
            else:
                sc = model.score_t(torch.tensor([[ent2id[str(sg)], rel2id[rel]]]))[0]
            sc = sc[cand_ids].numpy().astype(np.float64).copy()
            sc[~present] = -1e30   # entities isolated in the training graph
            filt = [x for x in ev["known"].get(int(s), set()) if x != t]
            sc[filt] = -np.inf
            zt = sc[t]
            ranks.append(1 + (sc > zt).sum() + ((sc == zt).sum() - 1) / 2)
    return np.array(ranks)


def run(kgname, tasks, seed, models, epochs, dim, sparsity=1.0, tag=""):
    kg_s, offsets, trip, evals = build(kgname, tasks, seed, sparsity)
    tf = TriplesFactory.from_labeled_triples(trip, create_inverse_triples=False)
    print(f"{kgname} seed {seed}: {tf.num_triples} triples, {tf.num_entities} entities", flush=True)
    res = {}
    for mname in models:
        t0 = time.time()
        kw = dict(embedding_dim=dim)
        if mname in ("ComplEx", "RotatE"):
            kw = dict(embedding_dim=dim // 2)
        from pykeen.models import model_resolver
        from pykeen.training import SLCWATrainingLoop
        torch.manual_seed(seed)
        loss = "softplus" if mname in ("DistMult", "ComplEx") else "nssa"
        model = model_resolver.make(mname, kw, triples_factory=tf, loss=loss, random_seed=seed)
        opt = torch.optim.Adam(model.parameters(), lr=0.001 if mname == "TransE" else 0.005)
        loop = SLCWATrainingLoop(model=model, triples_factory=tf, optimizer=opt,
                                 negative_sampler_kwargs=dict(num_negs_per_pos=NEGS))
        print(f"  training {mname}...", flush=True)
        losses = loop.train(triples_factory=tf, num_epochs=epochs, batch_size=8192, use_tqdm=False)
        print(f"  trained {mname} in {time.time()-t0:.0f}s, last loss {losses[-1]:.4f}", flush=True)
        res[mname] = {}
        for task, ev in evals.items():
            rv = evaluate(model, tf, offsets, kg_s, ev, "valid"); rt = evaluate(model, tf, offsets, kg_s, ev, "test")
            m = metrics(rt); m["valid_MRR"] = metrics(rv)["MRR"]; m["time_s"] = time.time() - t0
            m["ranks"] = rt.tolist()
            res[mname][task] = m
            print(f"  {mname} {task}: {m}", flush=True)
        out = f"{OUT}/kge_{kgname}{tag}_seed{seed}.json"
        json.dump(dict(seed=seed, epochs=epochs, dim=dim, sparsity=sparsity, results=res), open(out, "w"), indent=1)
    return res


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("kg"); ap.add_argument("--tasks"); ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--models", default="RotatE,TransE,ComplEx"); ap.add_argument("--epochs", type=int, default=25)
    ap.add_argument("--dim", type=int, default=128); ap.add_argument("--sparsity", type=float, default=1.0)
    ap.add_argument("--tag", default="")
    a = ap.parse_args()
    run(a.kg, a.tasks.split(","), a.seed, a.models.split(","), a.epochs, a.dim, a.sparsity, a.tag)
