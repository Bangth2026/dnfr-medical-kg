"""Knowledge-graph loading, splitting and metapath path enumeration for DNFR experiments."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import DATA, OUT
import gzip, itertools, json, os
import numpy as np, pandas as pd, scipy.sparse as sp



HET_TYPES = {"Compound": "C", "Disease": "D", "Gene": "G", "Anatomy": "A", "Symptom": "S",
             "Side Effect": "SE", "Pharmacologic Class": "PC", "Biological Process": "BP",
             "Cellular Component": "CC", "Molecular Function": "MF", "Pathway": "PW"}


class KG:
    """Typed multi-relational graph. Each relation r has a source type, target type and
    an edge list in type-local indices. Inverse relations are added for traversal."""

    def __init__(self, names, rels):
        self.names = names            # type -> np.array of node names
        self.rels = rels              # rel -> dict(src, dst, h, t)

    def n(self, typ):
        return len(self.names[typ])

    def copy_without(self, rel_drop_pairs):
        """rel_drop_pairs: dict type_pair -> set of (h,t) dyads to remove in BOTH directions,
        from every relation linking those two types."""
        rels = {}
        for r, d in self.rels.items():
            h, t = d["h"], d["t"]
            keep = np.ones(len(h), bool)
            for (a, b), dy in rel_drop_pairs.items():
                if (d["src"], d["dst"]) == (a, b):
                    keep &= ~np.isin(h.astype(np.int64) * 10**7 + t, dy)
                if (d["src"], d["dst"]) == (b, a):
                    keep &= ~np.isin(t.astype(np.int64) * 10**7 + h, dy)
            rels[r] = _masked(d, keep)
        return KG(self.names, rels)

    def subsample(self, frac_keep, rng, protect=()):
        rels = {}
        for r, d in self.rels.items():
            if r in protect:
                rels[r] = d; continue
            m = rng.random(len(d["h"])) < frac_keep
            rels[r] = _masked(d, m)
        return KG(self.names, rels)

    def n_edges(self):
        return sum(len(d["h"]) for d in self.rels.values())

    def traversals(self):
        """Directed traversal relations: (name, src_type, dst_type, base_rel, inverse)."""
        out = []
        for r, d in self.rels.items():
            out.append((r, d["src"], d["dst"], r, False))
            if not (d["src"] == d["dst"] and d.get("sym", False)):
                out.append((r + "^-1", d["dst"], d["src"], r, True))
        return out

    def adj(self, trav):
        name, s, t, base, inv = trav
        d = self.rels[base]
        h, tt = (d["t"], d["h"]) if inv else (d["h"], d["t"])
        A = sp.csr_matrix((np.ones(len(h), np.float32), (h, tt)), shape=(self.n(s), self.n(t)))
        A.data[:] = 1.0  # collapse duplicates
        A.sum_duplicates(); A.data[:] = 1.0
        if d.get("sym", False) and s == t:
            A = ((A + A.T) > 0).astype(np.float32).tocsr()
        return A


def _masked(d, keep):
    out = dict(d)
    for k in ("h", "t", "w"):
        if k in d:
            out[k] = d[k][keep]
    return out


def load_hetionet():
    nodes = pd.read_csv(f"{DATA}/hetionet-nodes.tsv", sep="\t", dtype=str)
    names, idx = {}, {}
    for kind, g in nodes.groupby("kind"):
        ty = HET_TYPES[kind]
        names[ty] = g["id"].values
        idx[ty] = {v: i for i, v in enumerate(g["id"].values)}
    kind_of = {k: HET_TYPES[k] for k in HET_TYPES}
    e = pd.read_csv(f"{DATA}/h.sif.gz", sep="\t")
    rels = {}
    for me, g in e.groupby("metaedge"):
        sk = g["source"].str.split("::").str[0].iloc[0]
        tk = g["target"].str.split("::").str[0].iloc[0]
        s, t = kind_of[sk], kind_of[tk]
        h = np.array([idx[s][v] for v in g["source"]])
        tt = np.array([idx[t][v] for v in g["target"]])
        sym = s == t and ">" not in me
        rels[me] = dict(src=s, dst=t, h=h, t=tt, sym=sym)
    return KG(names, rels)


def _id(v):
    return v.split("::", 1)[1]


PK_TYPE = {"drug": "drug", "disease": "disease", "gene/protein": "gene", "effect/phenotype": "phen",
           "anatomy": "anat", "biological_process": "bp", "molecular_function": "mf",
           "cellular_component": "cc", "pathway": "pw", "exposure": "exp"}


def load_primekg(drop=("anatomy_protein_present", "anatomy_protein_absent")):
    cache = f"{DATA}/primekg_cache.npz"
    df = pd.read_csv(f"{DATA}/primekg_kg.csv", usecols=["relation", "x_index", "x_type", "x_name", "y_index", "y_type", "y_name"],
                     dtype={"relation": "category", "x_type": "category", "y_type": "category"}, low_memory=False)
    # node tables by global index
    nodes = pd.concat([df[["x_index", "x_type", "x_name"]].set_axis(["i", "ty", "nm"], axis=1),
                       df[["y_index", "y_type", "y_name"]].set_axis(["i", "ty", "nm"], axis=1)]).drop_duplicates("i")
    nodes["ty"] = nodes["ty"].map(PK_TYPE)
    names, loc = {}, np.full(nodes["i"].max() + 1, -1, np.int64)
    for ty, g in nodes.groupby("ty"):
        g = g.sort_values("i")
        names[ty] = g["nm"].values
        loc[g["i"].values] = np.arange(len(g))
    rels = {}
    df = df[~df["relation"].isin(drop)]
    for (r, xt, yt), g in df.groupby(["relation", "x_type", "y_type"], observed=True):
        s, t = PK_TYPE[xt], PK_TYPE[yt]
        if s > t or (s == t and False):
            continue  # PrimeKG stores both directions; keep one canonical orientation
        h, tt = loc[g["x_index"].values], loc[g["y_index"].values]
        if s == t:
            m = h < tt
            h, tt = h[m], tt[m]
        key = r.replace(" ", "_").replace("-", "_")
        if key in rels:  # same relation, same canonical orientation
            rels[key]["h"] = np.concatenate([rels[key]["h"], h]); rels[key]["t"] = np.concatenate([rels[key]["t"], tt])
        else:
            rels[key] = dict(src=s, dst=t, h=h, t=tt, sym=(s == t))
    for d in rels.values():
        k = np.unique(d["h"].astype(np.int64) * 10**7 + d["t"])
        d["h"], d["t"] = (k // 10**7).astype(np.int64), (k % 10**7).astype(np.int64)
    return KG(names, rels)


def metapaths(kg, s, t, lengths=(2, 3)):
    trav = kg.traversals()
    by_src = {}
    for tr in trav:
        by_src.setdefault(tr[1], []).append(tr)
    out = []
    def rec(path, cur):
        if len(path) in lengths and cur == t:
            out.append(tuple(path))
        if len(path) == max(lengths):
            return
        for tr in by_src.get(cur, []):
            rec(path + [tr], tr[2])
    rec([], s)
    return out


def mp_name(mp):
    return "-".join(tr[0] for tr in mp)


def load_hkg(path=None):
    """Disease -> symptom graph of Rotmensch et al. (2017), weights = noisy-OR importance.
    Built from data/hkg_edges.json (see build_hkg.py). Three disease-symptom pairs occur
    twice in the parsed table; the first occurrence is kept."""
    rows = json.load(open(path or f"{DATA}/hkg_edges.json"))
    seen, keep = set(), []
    for d, s, w in rows:
        if (d, s) not in seen:
            seen.add((d, s)); keep.append((d, s, w))
    D = np.array(sorted({r[0] for r in keep})); S = np.array(sorted({r[1] for r in keep}))
    di = {v: i for i, v in enumerate(D)}; si = {v: i for i, v in enumerate(S)}
    e = sorted((di[d], si[s], w) for d, s, w in keep)
    h = np.array([x[0] for x in e], np.int64); t = np.array([x[1] for x in e], np.int64)
    w = np.array([x[2] for x in e], np.float64)
    return KG({"D": D, "S": S}, {"DpS": dict(src="D", dst="S", h=h, t=t, w=w, sym=False)})
