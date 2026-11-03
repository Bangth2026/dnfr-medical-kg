#!/bin/bash
# wait for NBFNet DpS split 0, stop that process (pid in $1), then run the CtD queue
cd "$(dirname "$0")/.."
until [ -f results2/nbf/nbfnet_het_DpS_seed0.json ]; do sleep 30; done
kill "$1"; echo "[$(date +%H:%M:%S)] nbfnet DpS split 0 done; stopped it for memory" >> logs2/queue_A.log
python3 scripts/queue_rev3.py A >> logs2/queue_A.log 2>&1
