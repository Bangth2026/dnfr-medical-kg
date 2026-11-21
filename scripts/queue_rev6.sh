export NT=2 DNFR_RESULTS=/home/claude/dnfr-medical-kg/results_rev6
cd /home/claude/dnfr-medical-kg/dnfr
BL=dwpc_lr,svd,softimpute,popularity,prior,dwpc_lr_full
python3 run_task.py het_CtD --split tail --tag _tail --seeds 5 --variants "" --baselines $BL
python3 run_task.py het_CtD --split tail --tag _tail --seeds 6,7,8,9 --variants product --baselines $BL
echo Q5A DONE
python3 run_task.py het_CtD --split head --tag _head --seeds 1,2,3,4,5,6,7,8,9 --variants product --baselines $BL
echo Q5B DONE
DNFR_DROP_TARGET_MPS=1 python3 run_task.py het_CtD --split tail --tag _tail_notgt --seeds 0,1,2,3,4,5,6,7,8,9 --variants product --baselines dwpc_lr,popularity
echo Q5C DONE
python3 run_task.py het_CtD --split tail --tag _tail --seeds 4,5,6,7,8,9 --variants dnfr,dnfr_nodeg --no_baselines
echo Q5D DONE
