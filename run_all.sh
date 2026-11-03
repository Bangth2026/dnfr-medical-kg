#!/usr/bin/env bash
# Reproduces every result file in results/ and the statistics and figures read by the manuscript.
# Results were computed on a shared two-core CPU with 8 GB of memory (NT=1 or 2 torch threads per run);
# PrimeKG splits are run one at a time. The full pipeline takes several days on such a machine.
set -e
cd "$(dirname "$0")"
python3 download_data.py
python3 dnfr/prepare_data.py            # data/het.pkl, data/pkg.pkl, data/hkg.pkl

# ---------- version 1.0.0: main tasks (path enumeration is cached in data/paths_<task>.pkl) ----------
python3 dnfr/run_task.py het_DpS --seeds 0,1,2,3,4 --variants dnfr,dnfr_shared,product,hamacher,min,luk,dnfr_nodeg,dnfr_max,dnfr_binary --baselines dwpc_lr,svd,softimpute
python3 dnfr/run_task.py hkg_DpS --seeds 0,1,2,3,4 --variants dnfr,dnfr_binary,product,hamacher,min,luk,dnfr_nodeg,dnfr_max --baselines dwpc_lr,svd,softimpute
python3 dnfr/run_task.py het_CtD --seeds 0,1,2 --variants dnfr,dnfr_shared,dnfr_nodeg,product,hamacher,min,luk,dnfr_max --baselines dwpc_lr,svd,softimpute
python3 dnfr/run_task.py pkg_ind --seeds 0 --variants dnfr,dnfr_nodeg,product,hamacher,min --baselines dwpc_lr,svd,softimpute
python3 dnfr/run_task.py pkg_ind --seeds 1 --variants dnfr,dnfr_nodeg,product --baselines dwpc_lr,svd,softimpute

# Sparsity (50% and 75% of edges removed)
for t in het_CtD het_DpS; do for s in 0.50 0.25; do
  python3 dnfr/run_task.py $t --seeds 0,1,2 --variants dnfr,dnfr_nodeg,product --sparsity $s --tag _sp$s --baselines dwpc_lr,svd,softimpute
done; done

# Knowledge graph embeddings
python3 dnfr/kge.py het --tasks het_CtD,het_DpS --seed 0
for s in 1 2; do python3 dnfr/kge.py het --tasks het_CtD,het_DpS --seed $s --models RotatE,TransE; done
python3 dnfr/kge.py pkg --tasks pkg_ind --seed 0 --models RotatE,TransE

# ---------- version 1.0.0 -> first revision: identifiability, fixed degree, frozen memberships, popularity ----------
R1=dnfr_l2_0,dnfr_pm2,dnfr_pm1,dnfr_pp05,dnfr_fixdeg,product_fixdeg,dnfr_frozen,product_frozen
python3 dnfr/run_task.py het_DpS --seeds 0,1,2,3,4 --variants $R1,dnfr_l2_1e-2,dnfr_l2_1e-1 --baselines popularity
python3 dnfr/run_task.py hkg_DpS --seeds 0,1,2,3,4 --variants $R1,dnfr_l2_1e-2,dnfr_l2_1e-1 --baselines popularity
python3 dnfr/run_task.py het_CtD --seeds 0,1,2 --variants $R1 --baselines popularity
python3 dnfr/run_task.py pkg_ind --seeds 0 --variants luk,dnfr_shared,dnfr_pm1,dnfr_fixdeg --baselines popularity
python3 dnfr/run_task.py pkg_ind --seeds 1 --variants hamacher,min,luk,dnfr_shared,dnfr_pm1,dnfr_fixdeg --baselines popularity
# disease-centred split (every indication of a test disease held out together)
python3 dnfr/run_task.py het_CtD --split tail --tag _tail --seeds 0,1,2 --variants dnfr,product,hamacher,dnfr_nodeg --baselines dwpc_lr,svd,softimpute,popularity
# AnyBURL (needs Java and AnyBURL-23-1.jar)
for t in het_CtD; do python3 dnfr/anyburl_run.py $t --seeds 0,1,2 --jar AnyBURL-23-1.jar; done
for t in het_DpS hkg_DpS; do python3 dnfr/anyburl_run.py $t --seeds 0,1,2,3,4 --jar AnyBURL-23-1.jar; done
for t in het_CtD het_DpS hkg_DpS; do python3 dnfr/explain_eval.py $t; done

# ---------- second revision (v1.1.0) ----------
# splits 5-9 of the disease-symptom tasks for the t-norm comparisons (with all baselines)
python3 dnfr/run_task.py het_DpS --seeds 5,6,7,8,9 --variants dnfr,product,hamacher
python3 dnfr/run_task.py hkg_DpS --seeds 5,6,7,8,9 --variants dnfr,product,hamacher,dnfr_l2_0,dnfr_pm2,dnfr_pm1,dnfr_pp05
# edge prior (Zietz et al. 2024) and DWPC-LR with degree and prior features on the earlier splits
python3 dnfr/run_task.py het_DpS --seeds 0,1,2,3,4 --variants "" --baselines dwpc_lr,prior,dwpc_lr_full
python3 dnfr/run_task.py hkg_DpS --seeds 0,1,2,3,4 --variants "" --baselines dwpc_lr,prior,dwpc_lr_full
python3 dnfr/run_task.py het_CtD --seeds 0,1,2 --variants "" --baselines dwpc_lr,prior,dwpc_lr_full
# sensitivity of the fixed degree discount
python3 dnfr/run_task.py hkg_DpS --seeds 0,1,2,3,4 --variants product_fixdeg02,product_fixdeg06 --no_baselines
# disease-centred split: split 3 (split 4 was stopped for time and is not reported), and the new baselines on splits 0-2
python3 dnfr/run_task.py het_CtD --split tail --tag _tail --seeds 3 --variants dnfr,product,hamacher,dnfr_nodeg
python3 dnfr/run_task.py het_CtD --split tail --tag _tail --seeds 0,1,2 --variants "" --baselines dwpc_lr,prior,dwpc_lr_full
# identifiability: training without early stopping, and profile over a fixed p
python3 dnfr/identifiability.py long hkg_DpS --seeds 0,1,2 --inits=-2,-0.1,0.5 --epochs 400
python3 dnfr/identifiability.py long het_DpS --seeds 0,1 --inits=-2,-0.1,0.5 --epochs 400
python3 dnfr/identifiability.py profile hkg_DpS --seeds 0,1,2 --grid=-4,-2,-1,-0.5,0,0.25,0.5,0.75 --epochs 300
python3 dnfr/identifiability.py profile het_DpS --seeds 0,1 --grid=-4,-1,0,0.5,0.75 --epochs 250
# explanations: share-matched deletion, overlap, edge-level counterfactual
for t in het_CtD het_DpS hkg_DpS; do python3 dnfr/explain.py $t; python3 dnfr/explain_eval2.py $t; done

# ---------- NHANES secondary test, statistics, figures, manuscript ----------
python3 dnfr/nhanes_dnfr.py
python3 dnfr/nhanes_jack.py
python3 tests/test_core.py
python3 dnfr/dataset_stats.py
python3 dnfr/aggregate.py
python3 dnfr/extra.py
python3 dnfr/revision_stats.py
python3 figures/make_fig1.py
python3 figures/make_figures.py
python3 manuscript/build_manuscript.py
