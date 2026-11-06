"""Job queue for the fifth-round experiments (reviewer report on revision 4). Same mechanics as queue_rev4.py: a worker
takes the first job that has neither a DONE marker nor a live lock, so two workers can run side by side and either can be
stopped and restarted. A job may list jobs it depends on; it is skipped until they are DONE.
Usage: python3 scripts/queue_rev5.py all|light [job_id]"""
import os, subprocess, sys, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from queue_rev4 import alive, claim, rt, PY, ROOT

J = []   # (job id, heavy, command, extra environment, dependencies)
CTD = "het_CtD"
PROF = lambda task, s, grid: (PY + [f"{ROOT}/dnfr/profile_lbfgs.py", task, str(s), grid], {"DNFR_RESULTS": f"{ROOT}/results2/prof_ext2"})
# M3: the fixed Hamacher product on split 3 from two other initial memberships
J.append(("ctd5_ham3", True, rt(CTD, 3, "hamacher_b01,hamacher_b06"), {}, []))
# M1: profile on the graded graph up to p = 1 (Lukasiewicz), warm starts from the product
for s in range(3):
    c, e = PROF("hkg_DpS", s, "0,0.25,0.5,0.75,1"); J.append((f"prof5_hkg_s{s}", False, c, e, []))
# M4: without the metapaths that contain treats or palliates, interleaved with M2: profile on Hetionet DpS down to -20
for s in range(10):
    J.append((f"ctd5_notgtp_s{s}", True, rt(CTD, s, "dnfr,product", "dwpc_lr", ["--tag", "_notgtp"]),
              {"DNFR_DROP_RELS": "CtD,CpD"}, []))
    if s < 3:
        c, e = PROF("het_DpS", s, "-20,-10,-5"); J.append((f"prof5_dps_s{s}", False, c, e, []))
# M4, Q4: evidence shares after each ablation (seed-0 models)
J.append(("ev5_notgt", True, PY + [f"{ROOT}/dnfr/evidence_notgt.py", "_notgt", "CtD"], {"DNFR_RESULTS": f"{ROOT}/results"}, []))
J.append(("ev5_notgtp", True, PY + [f"{ROOT}/dnfr/evidence_notgt.py", "_notgtp", "CtD,CpD"],
          {"DNFR_RESULTS": f"{ROOT}/results2/jobs/ctd5_notgtp_s0"}, ["ctd5_notgtp_s0"]))


if __name__ == "__main__":
    mode = sys.argv[1]; only = sys.argv[2] if len(sys.argv) > 2 else None
    os.makedirs(f"{ROOT}/results2/prof_ext2", exist_ok=True)
    done = lambda x: os.path.exists(f"{ROOT}/results2/jobs/{x}/DONE")
    failed = set()   # jobs that failed in this worker are not retried by it

    def run_one(jid, cmd, env_x):
        d = f"{ROOT}/results2/jobs/{jid}"
        os.makedirs(d, exist_ok=True)
        if os.path.exists(f"{d}/DONE") or not claim(d):
            return False
        env = dict(os.environ, NT=os.environ.get("QNT", "1"), **env_x)
        if "DNFR_RESULTS" not in env_x:
            env["DNFR_RESULTS"] = d
        t0 = time.time()
        print(f"[{time.strftime('%H:%M:%S')}] start {jid}", flush=True)
        with open(f"{d}/log.txt", "w") as f:
            rc = subprocess.call(cmd, stdout=f, stderr=subprocess.STDOUT, env=env, cwd=ROOT)
        if rc == 0:
            open(f"{d}/DONE", "w").write(f"{time.time()-t0:.0f}\n")
        else:
            failed.add(jid)
        os.remove(f"{d}/LOCK")
        print(f"[{time.strftime('%H:%M:%S')}] end {jid} rc={rc} {time.time()-t0:.0f}s", flush=True)
        return True

    while True:
        ran = False
        for jid, heavy, cmd, env_x, deps in J:
            if jid in failed or (only and jid != only) or (mode == "light" and heavy) or any(not done(x) for x in deps):
                continue
            ran = run_one(jid, cmd, env_x) or ran
        if ran:
            continue
        # nothing could be started: wait while a job still waits for a dependency, otherwise stop
        if not any(not done(j) and j not in failed and any(not done(x) and x not in failed for x in dp)
                   and (not only or j == only) for j, _, _, _, dp in J):
            break
        time.sleep(60)
