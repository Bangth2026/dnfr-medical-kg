# JBI revision (release v7.0-jbi)

Builds the revised manuscript, supplement and response to the reviewer from the submitted JBI version
(`Manuscript_JBI_DNFR_v2.docx`, `Supplementary_JBI_DNFR.docx`, built by `../build_jbi.py` and `../build_supp.py`) and the
statistics of the revision.

```
# new experiments (two-core CPU; run one at a time, not in parallel)
cd dnfr
DNFR_RESULTS=../results_rev6 python3 run_task.py hkg_DpS --seeds 0,...,9 --variants "" --baselines dwpc_lr,dwpc_sm,dwpc_nn_sm,dwpc_nn_lr --tag _ctl
DNFR_RESULTS=../results_rev6 python3 run_task.py het_DpS --seeds 0,...,9 --variants "" --baselines dwpc_lr,dwpc_sm,dwpc_nn_sm,dwpc_nn_lr --tag _ctl
python3 explain_dwpc_nn.py hkg_DpS; python3 explain_dwpc_nn.py het_DpS
python3 profile_permp.py het_DpS 0 2
DNFR_RESULTS=../results_rev6 python3 run_task.py het_CtD --split tail --tag _tail --seeds 4,...,9 --variants product --baselines dwpc_lr,svd,softimpute,popularity,prior,dwpc_lr_full
DNFR_RESULTS=../results_rev6 python3 run_task.py het_CtD --split head --tag _head --seeds 0,...,9 --variants product --baselines dwpc_lr,svd,softimpute,popularity,prior,dwpc_lr_full
DNFR_DROP_TARGET_MPS=1 DNFR_RESULTS=../results_rev6 python3 run_task.py het_CtD --split tail --tag _tail_notgt --seeds 0,...,9 --variants product --baselines dwpc_lr,popularity
DNFR_RESULTS=../results_rev6/anyburl1000 python3 anyburl_run.py hkg_DpS --seeds 0,1 --jar AnyBURL-23-1.jar --learn_s 1000
python3 rev6_stats.py; python3 rev6_compare.py; python3 ../figures/make_fig2_rev6.py
# documents
cd ../manuscript_jbi/rev6
python3 compute_numbers.py
python3 build_ms.py Manuscript_JBI_DNFR_v2.docx Manuscript_JBI_DNFR_v3.docx numbers.json ../../figures/fig2_profile_envelope.png
python3 build_supp.py Supplementary_JBI_DNFR.docx Supplementary_JBI_DNFR_v3.docx numbers.json Manuscript_JBI_DNFR_v3.docx.refmap.json
python3 build_response.py numbers.json Response_to_Reviewer_JBI_DNFR.docx
python3 check_numbers_rev6.py Manuscript_JBI_DNFR_v3.docx Supplementary_JBI_DNFR_v3.docx -- Manuscript_JBI_DNFR_v2.docx Supplementary_JBI_DNFR.docx numbers.json
```

`numbers.json` holds every new number of the revision, computed by `compute_numbers.py` from `results/` and
`results_rev6/`; `check_numbers_rev6.py` lists the numbers of the revised documents that occur neither in the submitted
version nor in `numbers.json` (only citation numbers and bibliographic details remain).
