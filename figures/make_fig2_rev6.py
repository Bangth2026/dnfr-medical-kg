"""Revision 6 Figure 2 (JBI reviewer M1a): converged profiles with their chain envelope.
For each split, the line is the lowest converged training loss at each fixed shared p (excess over the split's minimum);
the band spans all warm-start chains at that p; open markers flag points where the two best chains differ by more than
1e-4 nat (not resolved). The grey band is the profile-likelihood threshold chi2(0.95,1)/(2N). Right: validation MRR of
the lowest-loss fit. Reads results_rev6/rev6_stats.json."""
import json, os
import numpy as np, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
S = json.load(open(f"{ROOT}/results_rev6/rev6_stats.json"))["envelope"]
plt.rcParams.update({"font.size": 8.5, "axes.spines.top": False, "axes.spines.right": False})
TH = {"hkg_DpS": 6.5e-4, "het_DpS": 7.2e-4}
TT = {"hkg_DpS": "A. Graded EHR graph", "het_DpS": "B. Hetionet disease–symptom"}
COL = ["#1f77b4", "#ff7f0e", "#2ca02c"]
fig, ax = plt.subplots(1, 3, figsize=(10.2, 3.0), gridspec_kw=dict(width_ratios=[1, 1, 1]))
for j, t in enumerate(("hkg_DpS", "het_DpS")):
    a = ax[j]
    for s, v in S[t].items():
        R = v["rows"]; p = np.array([r["p"] for r in R]); ex = np.array([r["excess"] for r in R]) * 1e3
        hi = ex + np.array([r["worst_minus_best"] for r in R]) * 1e3
        res = np.array([r["resolved"] for r in R]); c = COL[int(s)]
        x = np.sign(p) * np.log1p(np.abs(p)) if t == "het_DpS" else p
        a.fill_between(x, ex, np.minimum(hi, 12), color=c, alpha=0.09, lw=0)
        a.plot(x, ex, "-", color=c, lw=1, label=f"split {s}")
        a.plot(x[res], ex[res], "o", color=c, ms=3.2)
        a.plot(x[~res], ex[~res], "o", mfc="white", mec=c, ms=3.6)
    a.axhspan(0, TH[t] * 1e3, color="#d9d9d9", zorder=0)
    a.set_ylim(-0.2, 12); a.set_title(TT[t], fontsize=9, loc="left")
    a.set_ylabel("training loss − minimum (×10⁻³ nat)")
    if t == "het_DpS":
        tk = [-20, -10, -5, -2, -1, -0.5, 0]
        a.set_xticks([np.sign(q) * np.log1p(abs(q)) for q in tk]); a.set_xticklabels([str(q).replace("-", "−") for q in tk])
        a.set_xlabel("fixed shared p (log-scaled axis)")
    else:
        a.set_xlabel("fixed shared p")
        for q in (-1, 0, 1): a.axvline(q, color="#aaaaaa", lw=0.7, ls="--", zorder=0)
ax[0].legend(frameon=False, fontsize=7, loc="upper left")
a = ax[2]
for t, mk in (("hkg_DpS", "-o"), ("het_DpS", "--s")):
    for s, v in S[t].items():
        R = v["rows"]; p = np.array([r["p"] for r in R]); vm = np.array([r["valid_MRR"] for r in R])
        sel = p >= -5
        a.plot(p[sel], vm[sel] - vm[sel].max(), mk, color=COL[int(s)], ms=2.8, lw=0.9,
               label=("graded" if t == "hkg_DpS" else "Hetionet") + f", split {s}")
a.set_title("C. Validation MRR of the same fits", fontsize=9, loc="left")
a.set_xlabel("fixed shared p"); a.set_ylabel("validation MRR − its maximum")
for q in (-1, 0, 1): a.axvline(q, color="#aaaaaa", lw=0.7, ls="--", zorder=0)
a.legend(frameon=False, fontsize=6, ncol=2, loc="lower left")
plt.tight_layout(); plt.savefig(f"{ROOT}/figures/fig2_profile_envelope.png", dpi=300); plt.close()
print("ok")
