#!/usr/bin/env bash
# Independent re-run of one key result from a clean checkout (reviewer minor 12).
# Re-trains DNFR, DNFR with the strongest path only, and DNFR without degree terms on split 0 of the graded
# disease-symptom graph (the noisy-OR and degree-term rows of Table 3), then compares the test MRR with the released
# result file. Needs Python 3 with the packages in requirements.txt; takes about 10 minutes on a laptop.
# Usage: bash verify_clean.sh [release tag]
set -euo pipefail
TAG="${1:-v4.0}"
WORK="$(mktemp -d)"
git clone --quiet --depth 1 --branch "$TAG" https://github.com/Bangth2026/dnfr-medical-kg.git "$WORK/repo"
cd "$WORK/repo"
mkdir -p rerun
python3 -W ignore dnfr/prepare_data.py hkg > rerun/log.txt 2>&1      # builds data/hkg.pkl from the released data/hkg_edges.json
DNFR_RESULTS="$PWD/rerun" python3 -W ignore dnfr/run_task.py hkg_DpS --seeds 0 --variants dnfr,dnfr_max,dnfr_nodeg --no_baselines \
    >> rerun/log.txt 2>&1
python3 - <<'EOF'
import json
new = {r["seed"]: r for r in json.load(open("rerun/hkg_DpS.json"))["runs"]}[0]["methods"]
old = {r["seed"]: r for r in json.load(open("results/hkg_DpS.json"))["runs"]}[0]["methods"]
print(f"{'variant':12s} {'released':>9s} {'re-run':>9s} {'diff':>9s}")
for v in ("dnfr", "dnfr_max", "dnfr_nodeg"):
    print(f"{v:12s} {old[v]['MRR']:9.4f} {new[v]['MRR']:9.4f} {new[v]['MRR'] - old[v]['MRR']:+9.4f}")
print("noisy-OR minus strongest path:", f"{new['dnfr']['MRR'] - new['dnfr_max']['MRR']:+.4f}",
      "| degree terms minus none:", f"{new['dnfr']['MRR'] - new['dnfr_nodeg']['MRR']:+.4f}")
EOF
echo "work directory: $WORK"
