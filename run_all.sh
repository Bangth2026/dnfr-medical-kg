#!/usr/bin/env bash
# Reproduces every result file in results/ and the statistics read by the manuscript.
# Results were computed on a two-core CPU with 8 GB of memory; PrimeKG splits are run one at a time.
set -e
cd "$(dirname "$0")"
python3 download_data.py
python3 dnfr/prepare_data.py            # data/het.pkl, data/pkg.pkl, data/hkg.pkl

# Main tasks (path enumeration is cached in data/paths_<task>.pkl on first run)
python3 dnfr/run_task.py het_DpS --seeds 0,1,2,3,4
python3 dnfr/run_task.py hkg_DpS --seeds 0,1,2,3,4
python3 dnfr/run_task.py het_CtD --seeds 0,1,2 --variants dnfr,dnfr_shared,dnfr_nodeg,product,hamacher,min,luk
python3 dnfr/run_task.py pkg_ind --seeds 0 --variants dnfr,dnfr_nodeg,product,hamacher,min
python3 dnfr/run_task.py pkg_ind --seeds 1 --variants dnfr,dnfr_nodeg,product

# Sparsity (50% and 75% of edges removed)
for t in het_CtD het_DpS; do for s in 0.50 0.25; do
  python3 dnfr/run_task.py $t --seeds 0,1,2 --variants dnfr,dnfr_nodeg,product --sparsity $s --tag _sp$s
done; done

# Knowledge graph embeddings
python3 dnfr/kge.py het --tasks het_CtD,het_DpS --seed 0
for s in 1 2; do python3 dnfr/kge.py het --tasks het_CtD,het_DpS --seed $s --models RotatE,TransE; done
python3 dnfr/kge.py pkg --tasks pkg_ind --seed 0 --models RotatE,TransE

# Explanations, NHANES secondary test, statistics
for t in het_CtD het_DpS hkg_DpS; do python3 dnfr/explain.py $t; done
python3 dnfr/nhanes_dnfr.py
python3 dnfr/nhanes_jack.py
python3 dnfr/dataset_stats.py
python3 dnfr/aggregate.py
python3 dnfr/extra.py
python3 figures/make_figures.py
