"""Build the pickled graphs used by run_task.py: data/het.pkl, data/pkg.pkl, data/hkg.pkl."""
import os, sys, pickle
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import DATA
from kg import load_hetionet, load_primekg, load_hkg

which = sys.argv[1:] or ["het", "pkg", "hkg"]
for name in which:
    kg = {"het": load_hetionet, "pkg": load_primekg, "hkg": load_hkg}[name]()
    pickle.dump(kg, open(os.path.join(DATA, f"{name}.pkl"), "wb"), protocol=4)
    print(name, {t: len(v) for t, v in kg.names.items()}, kg.n_edges(), "edges")
