# Differentiable neuro-fuzzy reasoning on medical knowledge graphs (DNFR)

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.23879352.svg)](https://doi.org/10.5281/zenodo.23879352)

Concept DOI (all versions, resolves to the latest): [10.5281/zenodo.23879352](https://doi.org/10.5281/zenodo.23879352). Version 1.0.0: [10.5281/zenodo.23879353](https://doi.org/10.5281/zenodo.23879353). Version 1.1.0: [10.5281/zenodo.23948874](https://doi.org/10.5281/zenodo.23948874). Version v3.0-rev3: [10.5281/zenodo.23949261](https://doi.org/10.5281/zenodo.23949261).

Code, derived data and result files for the manuscript

> Truong Hai Bang, Van An Phan Thi. *Which parts of a neuro-fuzzy path reasoner should be learned? A controlled component analysis on medical knowledge graphs.* Submitted to *Artificial Intelligence in Medicine* (fourth revision, release v4.0 of this repository).

DNFR scores a pair of entities by a noisy-OR over metapath-instantiated paths of length two and three. Each edge receives a learned membership (degree-aware, and reading graded evidence where the graph has it), each path is composed with a Schweizer–Sklar t-norm whose parameter is learned per metapath, and each metapath has a learned reliability. Every score decomposes exactly into the contributions of its paths.

Every number in the manuscript is read from `results/paper_stats.json`, `results/data_stats.json`, `results/extra_stats.json` and `results/revision_stats.json`, which `dnfr/aggregate.py`, `dnfr/dataset_stats.py`, `dnfr/extra.py` and `dnfr/revision_stats.py` compute from the per-task result files in `results/`. `manuscript/build_manuscript.py` builds the manuscript (.docx) from these files.

## Tasks

| Task | Graph | Relation predicted | Splits |
| --- | --- | --- | --- |
| `het_CtD` | Hetionet v1.0 | Compound–treats–Disease | 3 random; 4 disease-centred (`_tail`) |
| `het_DpS` | Hetionet v1.0 | Disease–presents–Symptom | 10 (t-norm comparisons), 5 (other variants) |
| `pkg_ind` | PrimeKG | drug–indication–disease | 2 |
| `hkg_DpS` | Health knowledge graph of Rotmensch et al. (2017), graded edges | disease–symptom | 10 (t-norm comparisons), 5 (other variants) |
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
dnfr/run_task.py        DNFR, fixed t-norms, ablations, DWPC + logistic regression (also with degree and prior features),
                        degree-preserving edge prior, SVD, Soft-Impute, popularity
dnfr/identifiability.py profile over a fixed t-norm parameter; training without early stopping (p trajectories)
dnfr/kge.py             RotatE, TransE, ComplEx (PyKEEN) on the same splits
dnfr/anyburl_run.py     AnyBURL rule learning on the same leave-dyad-out splits
dnfr/explain.py         exact path attributions for top-ranked test pairs
dnfr/explain_eval.py    concision and deletion tests of the explanations
dnfr/explain_eval2.py   share-matched deletion, overlap of leading paths, edge-level counterfactual
dnfr/nhanes_dnfr.py     secondary test on the NHANES fuzzy relation; nhanes_jack.py: concept jackknife
dnfr/aggregate.py       summary statistics, bootstrap and corrected resampled t-test, equivalence verdicts -> results/paper_stats.json
dnfr/extra.py           NHANES, jackknife and explanation tables -> results/extra_stats.json
dnfr/revision_stats.py  profile, trajectories, stopping epochs, explanation controls, learned memberships -> results/revision_stats.json
dnfr/dataset_stats.py   graph and task statistics -> results/data_stats.json
tests/test_core.py      checks of leave-dyad-out removal, the Schweizer-Sklar t-norm limits and filtered ranking
manuscript/             build_manuscript.py and its modules: the manuscript is generated from the statistics files
figures/                Figures 1-4 and A1; make_fig1.py (Figure 1), make_figures.py (Figures 2-4, A1)
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
python3 dnfr/aggregate.py && python3 dnfr/extra.py && python3 dnfr/revision_stats.py && python3 figures/make_figures.py && python3 manuscript/build_manuscript.py
```

## Versions

- **v4.0** (fourth revision): 10 splits on Hetionet compound–treats–disease for the fixed Hamacher product, fixed degree terms, annealing, frozen memberships, DNFR 2.0, the shared and fixed t-norm parameters and AnyBURL (`scripts/queue_rev4.py`); an ablation without metapaths that contain the target relation (`DNFR_DROP_TARGET_MPS=1`); fact-level attribution of DWPC-LR and evidence share by metapath class (`dnfr/explain_dwpc.py`); an information-content check with matched spread (`dnfr_ic_sd`); a profile of the t-norm parameter extended to p > 0 and run on Hetionet disease–symptom; an NBFNet tuning grid; statistics in `dnfr/rev4_stats.py`; a clean-checkout re-run of a key result (`scripts/verify_clean.sh`); the clinical assessment of explanations (`dnfr/expert_sheet.py` builds the rating sheet, `dnfr/expert_summary.py` summarises the returned ratings). The secondary NHANES test is no longer part of the article.
- **v3.0-rev3** (third revision): DNFR 2.0 and its four refinements, each added alone and removed from the combination (overlap-aware aggregation, Gumbel-softmax operator choice, information-content hub discount, membership annealing); a fixed t-norm selected on validation data; a converged L-BFGS profile of the t-norm parameter and an unbounded parameterisation; a compact NBFNet (`dnfr/nbfnet.py`); an exact edge-level Shapley attribution and a mechanistic classification of explanations (`dnfr/explain_eval3.py`); a sparsity variant that keeps the target relation; 10 splits for the main comparisons on Hetionet CtD, Hetionet DpS and the graded graph; manuscript build in `manuscript3/`.
- **1.1.0** (second revision): profile analysis of the t-norm parameter and training without early stopping (`identifiability.py`); splits 5-9 of both disease-symptom tasks for the t-norm comparisons; split 3 of the disease-centred split; degree-preserving edge prior and DWPC-LR with degree and prior features; sensitivity of the fixed degree discount; share-matched and edge-level explanation tests; corrected-t intervals and equivalence verdicts (margin 0.01 MRR); tests; manuscript build script; redrawn Figure 1. The first-revision experiments (restarts of p, fixed-degree and frozen-membership variants, popularity, AnyBURL, disease-centred split, deletion tests) are also included.
- **1.0.0**: code and results of the first submission.

## Licence

Code: MIT (`LICENSE`). Derived data follow the licences of their sources listed above.

## Citation

See `CITATION.cff`.
