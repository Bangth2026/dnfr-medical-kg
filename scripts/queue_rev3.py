"""Job queue for the third-revision experiments (DNFR 2.0). Each job writes to results2/jobs/<id>/ and is skipped
once its DONE marker exists, so a worker can be stopped and restarted. Usage: python3 scripts/queue_rev3.py A|B"""
import os, subprocess, sys, time
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
V2 = "dnfr2,dnfr2_no_overlap,dnfr2_no_gumbel,dnfr2_no_ic,dnfr2_no_anneal"
ADD = "dnfr_overlap,dnfr_gumbel,dnfr_ic,dnfr_anneal,product_overlap"
GRID = "fixp_m4,fixp_m2,fixp_m1,fixp_m05,fixp_05"
RST = "dnfr2_r1,dnfr2_r2,dnfr_gumbel_r1,dnfr_gumbel_r2"
FREE = "dnfr_free_pm2,dnfr_free,dnfr_free_pp05"
BASE = "dwpc_lr,svd,softimpute,popularity,prior,dwpc_lr_full"
J = {"A": [], "B": []}
# worker A: Hetionet CtD (memory-heavy, one process)
J["A"].append(("ctd_v1_s3", "het_CtD", 3, "dnfr,dnfr_nodeg,dnfr_max,product,hamacher", BASE))
# splits 4-9: headline comparison only (the operator/degree ablations stay at splits 0-3 for CPU reasons)
for s in range(4, 10):
    J["A"].append((f"ctd_v1_s{s}", "het_CtD", s, "dnfr,product", BASE))
# reviewer M3: component ablations on all ten splits
for s in range(4, 10):
    J["A"].append((f"ctd_abl_s{s}", "het_CtD", s, "dnfr_nodeg,dnfr_max", ""))
for s in range(0, 5):
    J["A"].append((f"ctd_v2_s{s}", "het_CtD", s, "dnfr2", ""))
for s in range(0, 3):
    J["A"].append((f"ctd_add_s{s}", "het_CtD", s, "dnfr_overlap", ""))
for s in range(0, 5):
    J["A"].append((f"ctd_grid_s{s}", "het_CtD", s, "fixp_m2,fixp_m1,fixp_m05", ""))
for s in range(0, 3):
    J["A"].append((f"ctd_add2_s{s}", "het_CtD", s, "dnfr_gumbel,dnfr_ic", ""))
# worker B: graded graph and Hetionet DpS
for s in range(0, 10):
    J["B"].append((f"hkg_main_s{s}", "hkg_DpS", s, V2 + "," + ADD + "," + GRID, ""))
for s in range(0, 5):
    J["B"].append((f"hkg_rst_s{s}", "hkg_DpS", s, RST + "," + FREE, ""))
for s in range(0, 10):
    J["B"].append((f"dps_main_s{s}", "het_DpS", s, "dnfr2,dnfr_overlap,dnfr_gumbel,fixp_m2,fixp_m1,fixp_m05,fixp_05", ""))
ABL = "dnfr_max,dnfr_nodeg,dnfr_fixdeg,dnfr_frozen,product_frozen"
for s in range(5, 10):
    J["B"].append((f"hkg_abl_s{s}", "hkg_DpS", s, ABL + ",dnfr_binary", ""))
    J["B"].append((f"dps_abl_s{s}", "het_DpS", s, ABL, ""))
for s in range(0, 3):
    J["B"].append((f"dps_loo_s{s}", "het_DpS", s, "dnfr2_no_overlap,dnfr2_no_gumbel,dnfr2_no_ic,dnfr2_no_anneal,dnfr_ic,dnfr_anneal,product_overlap", ""))
for s in range(0, 2):
    J["B"].append((f"dps_rst_s{s}", "het_DpS", s, RST + "," + FREE, ""))

if __name__ == "__main__":
    w = sys.argv[1]
    for jid, task, seed, variants, base in J[w]:
        d = f"{ROOT}/results2/jobs/{jid}"
        if os.path.exists(f"{d}/DONE"):
            continue
        os.makedirs(d, exist_ok=True)
        cmd = ["python3", f"{ROOT}/dnfr/run_task.py", task, "--seeds", str(seed), "--variants", variants]
        cmd += ["--baselines", base] if base else ["--no_baselines"]
        env = dict(os.environ, DNFR_RESULTS=d, NT="1")
        t0 = time.time()
        print(f"[{time.strftime('%H:%M:%S')}] start {jid}", flush=True)
        with open(f"{d}/log.txt", "w") as f:
            rc = subprocess.call(cmd, stdout=f, stderr=subprocess.STDOUT, env=env, cwd=ROOT)
        if rc == 0:
            open(f"{d}/DONE", "w").write(f"{time.time()-t0:.0f}\n")
        print(f"[{time.strftime('%H:%M:%S')}] end {jid} rc={rc} {time.time()-t0:.0f}s", flush=True)
