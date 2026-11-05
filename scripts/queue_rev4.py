"""Job queue for the fourth-round experiments (reviewer report on revision 3). One ordered list shared by any
number of workers: a worker takes the first job that has neither a DONE marker nor a live lock, so two workers can
run side by side and either can be stopped and restarted. Jobs marked heavy (Hetionet CtD) are skipped by a worker
started with 'light'. Usage: python3 scripts/queue_rev4.py all|light [job_id]"""
import os, subprocess, sys, time
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PY = ["python3", "-W", "ignore"]
RUN = PY + [f"{ROOT}/dnfr/run_task.py"]


def rt(task, seed, variants, base="", extra=()):
    cmd = RUN + [task, "--seeds", str(seed), "--variants", variants]
    return cmd + (["--baselines", base] if base else ["--no_baselines"]) + list(extra)


J = []   # (job id, heavy, command, extra environment)
CTD = "het_CtD"
LOO = "dnfr2_no_overlap,dnfr2_no_gumbel,dnfr2_no_ic,dnfr2_no_anneal"
# priority 1: Hamacher and fixed degree terms on CtD (M1, Q1); annealing on CtD and DpS (M5, Q4); the ablation without
# metapaths through the target relation (M4, Q3); the remaining DpS refinement splits
for s in range(3, 10):
    J.append((f"ctd4_ham_s{s}", True, rt(CTD, s, ("hamacher," if s >= 4 else "") + "dnfr_fixdeg"), {}))
    J.append((f"dps4_s{s}", False, rt("het_DpS", s, "dnfr_anneal,dnfr_ic,product_overlap,dnfr_shared," + LOO), {}))
J.append(("hkg4_icsd", False, RUN + ["hkg_DpS", "--seeds", "0,1,2,3,4,5,6,7,8,9", "--variants", "dnfr_ic_sd", "--no_baselines"], {}))
for s in range(0, 10):
    J.append((f"ctd4_ann_s{s}", True, rt(CTD, s, "dnfr_anneal"), {}))
    J.append((f"ctd4_notgt_s{s}", True, rt(CTD, s, "dnfr,product", "dwpc_lr", ["--tag", "_notgt"]), {"DNFR_DROP_TARGET_MPS": "1"}))
# priority 2: profile with p > 0 on the graded graph (M3); DNFR 2.0, the validation-selected fixed p and the
# frozen-membership pair on all CtD splits (M1); NBFNet tuning grid on the graded graph (M7)
for s in range(0, 3):
    J.append((f"prof_hkg_s{s}", False, PY + [f"{ROOT}/dnfr/profile_lbfgs.py", "hkg_DpS", str(s), "-1,-0.5,-0.25,0,0.1,0.25,0.5"],
              {"DNFR_RESULTS": f"{ROOT}/results2/prof_ext"}))
for s in range(5, 10):
    J.append((f"ctd4_v2_s{s}", True, rt(CTD, s, "dnfr2"), {}))
    J.append((f"ctd4_fix_s{s}", True, rt(CTD, s, "fixp_m2,fixp_m1,fixp_m05"), {}))
for s in range(3, 10):
    J.append((f"ctd4_frz_s{s}", True, rt(CTD, s, "dnfr_frozen,product_frozen"), {}))
for lr in ("5e-3", "1e-2"):
    for d in ("32", "64"):
        if (lr, d) == ("5e-3", "32"):
            continue
        J.append((f"nbf_hkg_d{d}_lr{lr}", False, PY + [f"{ROOT}/dnfr/nbfnet.py", "hkg_DpS", "0,1,2,3,4,5,6,7,8,9", "3", d],
                  {"NBF_LR": lr, "DNFR_RESULTS": f"{ROOT}/results2/nbf"}))
# priority 3: the remaining refinements on CtD; NBFNet on all DpS splits; converged profile on Hetionet DpS (M3)
for s in range(3, 10):
    J.append((f"ctd4_ref_s{s}", True, rt(CTD, s, "dnfr_gumbel,dnfr_ic,dnfr_shared"), {}))
for s in range(3, 10):
    J.append((f"nbf_dps_s{s}", False, PY + [f"{ROOT}/dnfr/nbfnet.py", "het_DpS", str(s)],
              {"NBF_BATCH": "8", "NBF_EPOCHS": "150", "DNFR_RESULTS": f"{ROOT}/results2/nbf"}))
for s in range(0, 3):
    J.append((f"prof_dps_s{s}", False, PY + [f"{ROOT}/dnfr/profile_lbfgs.py", "het_DpS", str(s),
                                             "-5,-3,-2,-1.5,-1,-0.5,-0.25,0,0.1,0.25"],
              {"DNFR_RESULTS": f"{ROOT}/results2/prof_ext"}))
for s in range(3, 10):
    J.append((f"ctd4_ovl_s{s}", True, rt(CTD, s, "dnfr_overlap"), {}))
# priority 4: AnyBURL on all CtD splits
J.append(("anyburl_ctd", False, PY + [f"{ROOT}/dnfr/anyburl_run.py", CTD, "--seeds", "3,4,5,6,7,8,9",
                                      "--jar", f"{ROOT}/tools/AnyBURL-23-1.jar"], {"DNFR_RESULTS": f"{ROOT}/results2/anyburl"}))


def alive(pid):
    try:
        os.kill(pid, 0); return True
    except OSError:
        return False


def claim(d):
    lock = f"{d}/LOCK"
    if os.path.exists(lock):
        try:
            if alive(int(open(lock).read().strip() or 0)):
                return False
        except ValueError:
            pass
        os.remove(lock)
    try:
        fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError:
        return False
    os.write(fd, str(os.getpid()).encode()); os.close(fd)
    return True


if __name__ == "__main__":
    mode = sys.argv[1]; only = sys.argv[2] if len(sys.argv) > 2 else None
    for d_ in ("results2/prof_ext", "results2/nbf", "results2/anyburl"):
        os.makedirs(f"{ROOT}/{d_}", exist_ok=True)
    for jid, heavy, cmd, env_x in J:
        if (only and jid != only) or (mode == "light" and heavy):
            continue
        d = f"{ROOT}/results2/jobs/{jid}"
        os.makedirs(d, exist_ok=True)
        if os.path.exists(f"{d}/DONE") or not claim(d):
            continue
        env = dict(os.environ, NT="1", **env_x)
        env.setdefault("DNFR_RESULTS", d)
        if "DNFR_RESULTS" not in env_x:
            env["DNFR_RESULTS"] = d
        t0 = time.time()
        print(f"[{time.strftime('%H:%M:%S')}] start {jid}", flush=True)
        with open(f"{d}/log.txt", "w") as f:
            rc = subprocess.call(cmd, stdout=f, stderr=subprocess.STDOUT, env=env, cwd=ROOT)
        if rc == 0:
            open(f"{d}/DONE", "w").write(f"{time.time()-t0:.0f}\n")
        os.remove(f"{d}/LOCK")
        print(f"[{time.strftime('%H:%M:%S')}] end {jid} rc={rc} {time.time()-t0:.0f}s", flush=True)
