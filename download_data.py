"""Download the public inputs into data/ and check them against the checksums of the files
the published results were computed from."""
import hashlib, os, subprocess, sys, urllib.request

ROOT = os.path.dirname(os.path.abspath(__file__)); D = os.path.join(ROOT, "data")
FILES = {
    # Hetionet v1.0 (Himmelstein et al., eLife 2017), CC0
    "hetionet-nodes.tsv": ("https://github.com/hetio/hetionet/raw/main/hetnet/tsv/hetionet-v1.0-nodes.tsv",
                           "d2789861ab5b977dcee3c23ac9c9f7ce"),
    "h.sif.gz": ("https://media.githubusercontent.com/media/hetio/hetionet/main/hetnet/tsv/hetionet-v1.0-edges.sif.gz",
                 "b4b2ab8890d403c2077a65e0f34fc4d9"),
    # PrimeKG (Chandak et al., Sci Data 2023), Harvard Dataverse doi:10.7910/DVN/IXA7BM, kg.csv
    "primekg_kg.csv": ("https://dataverse.harvard.edu/api/access/datafile/6180620",
                       "aac8191d4fbc5bf09cdf8c3c78b4e75f"),
}


def md5(p):
    h = hashlib.md5()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


os.makedirs(D, exist_ok=True)
for name, (url, ref) in FILES.items():
    p = os.path.join(D, name)
    if not os.path.exists(p):
        print("downloading", name); urllib.request.urlretrieve(url, p)
    got = md5(p)
    print(f"{name}: {'OK' if got == ref else 'CHECKSUM DIFFERS (' + got + ')'}")
