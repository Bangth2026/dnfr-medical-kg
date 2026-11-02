# Differentiable neuro-fuzzy reasoning on medical knowledge graphs (DNFR)

Code, derived data and result files for the manuscript

> Truong Hai Bang, Van An Phan Thi. *Learning edge memberships matters more than learning t-norms: differentiable neuro-fuzzy reasoning on sparse medical knowledge graphs.* Submitted to *Artificial Intelligence in Medicine*.

DNFR scores a pair of entities by a noisy-OR over metapath-instantiated paths of length two and three. Each edge receives a learned membership (degree-aware, and reading graded evidence where the graph has it), each path is composed with a Schweizer–Sklar t-norm whose parameter is learned per metapath, and each metapath has a learned reliability. Every score decomposes exactly into the contributions of its paths.

Every number in the manuscript is read from `results/paper_stats.json`, `results/data_stats.json` and `results/extra_stats.json`, which `dnfr/aggregate.py`, `dnfr/dataset_stats.py` and `dnfr/extra.py` compute from the per-task result files in `results/`.

## Tasks

| Task | Graph | Relation predicted | Splits |
| --- | --- | --- | --- |
| `het_CtD` | Hetionet v1.0 | Compound–treats–Disease | 3 |
| `het_DpS` | Hetionet v1.0 | Disease–presents–Symptom | 5 |
| `pkg_ind` | PrimeKG | drug–indication–disease | 2 |
| `hkg_DpS` | Health knowledge graph of Rotmensch et al. (2017), graded edges | disease–symptom | 5 |
| NHANES | 24-concept fuzzy relation (earlier study) | all pairs, held-out survey cycles | – |

All tasks use leave-dyad-out splits (80/10/10): every edge between a validation or test pair is removed from the graph, and paths using such edges are dropped. Ranking is filtered against all candidate tails, with mean-rank ties.

## Layout

```
download_data.py        fetch Hetionet and PrimeKG into data/ and check MD5 sums
build_hkg.py            parse the health knowledge graph from the supplementary information of Rotmensch et al.
dnfr/config.py          data and result locations (override with DNFR_DATA, DNFR_RESULTS)
dnfr/kg.py              graph loading, leave-dyad-out removal, sparsification, metapaths
dnfr/paths.py           enumeration of simple paths per metapath (budget 5e6 paths)
dnfr/dnfr.py            DNFR model, t-norms in log space, training, filtered ranking
dnfr/run_task.py        DNFR, fixed t-norms, ablations, DWPC + logistic regression, SVD, Soft-Impute
dnfr/kge.py             RotatE, TransE, ComplEx (PyKEEN) on the same splits
dnfr/explain.py         exact path attributions for top-ranked test pairs
dnfr/nhanes_dnfr.py     secondary test on the NHANES fuzzy relation; nhanes_jack.py: concept jackknife
dnfr/aggregate.py       summary statistics and paired Wilcoxon tests -> results/paper_stats.json
dnfr/extra.py           NHANES, jackknife and explanation tables -> results/extra_stats.json
dnfr/dataset_stats.py   graph and task statistics -> results/data_stats.json
figures/                Figures 1-3 and make_figures.py (Figures 2-3)
data/hkg_edges.json     parsed health knowledge graph (3,707 rows; 3,704 distinct disease-symptom edges)
data/nhanes/            E_dev.npy, E_hold.npy: NHANES relations from doi:10.5281/zenodo.23790456
results/                per-task results (with per-query ranks), trained seed models (.pt), test-query ids
```

## Requirements

Python 3.13 with the versions in `requirements.txt` (`pip install -r requirements.txt`). All experiments ran on a two-core CPU with 8 GB of memory; on PrimeKG, run one split at a time.

## Data

- **Hetionet v1.0** (Himmelstein et al., eLife 2017; CC0): `hetionet-v1.0-nodes.tsv`, `hetionet-v1.0-edges.sif.gz` from github.com/hetio/hetionet.
- **PrimeKG** (Chandak et al., Sci Data 2023): `kg.csv` from Harvard Dataverse, doi:10.7910/DVN/IXA7BM. Anatomy–protein expression edges are dropped.
- **Health knowledge graph** (Rotmensch et al., Sci Rep 2017;7:5994, doi:10.1038/s41598-017-05778-z; CC BY 4.0): the noisy-OR graph learned from emergency-department records, as published in the article's supplementary information. `build_hkg.py` reproduces `data/hkg_edges.json` from `pdftotext -layout` output of that file. Three disease–symptom pairs appear twice in the published table; the first occurrence is kept.
- **NHANES relations**: derived from public NHANES data in our earlier study (Zenodo doi:10.5281/zenodo.23790456, MIT licence).

Hetionet and PrimeKG are not redistributed here; `python3 download_data.py` fetches them and checks them against the MD5 sums of the files the results were computed from.

## Reproducing the results

`bash run_all.sh` runs the full pipeline in the order used for the paper (many hours on two CPU cores). Individual tasks, e.g.

```
python3 dnfr/prepare_data.py hkg
python3 dnfr/run_task.py hkg_DpS --seeds 0,1,2,3,4
```

To recompute only the statistics and figures from the result files shipped here:

```
python3 dnfr/aggregate.py && python3 dnfr/extra.py && python3 figures/make_figures.py
```

## Licence

Code: MIT (`LICENSE`). Derived data follow the licences of their sources listed above.

## Citation

See `CITATION.cff`.
