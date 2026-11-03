"""Third-revision figures: Figure 2 (converged L-BFGS profile on the graded graph) and Figure 3 (refinements:
add-one and leave-one-out differences in MRR with 95% bootstrap intervals). Reads results/rev3_stats.json."""
import json, os, glob
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
R = os.environ.get("DNFR_RESULTS", os.path.join(ROOT, "results"))
F = os.path.join(ROOT, "figures")
plt.rcParams.update({"font.size": 8.5, "axes.spines.top": False, "axes.spines.right": False})
ST = json.load(open(f"{R}/rev3_stats.json"))

# ---- Figure 2: converged profile ----
prof = [v for k, v in sorted(ST.get("profile_lbfgs", {}).items()) if "hkg_DpS" in k]
if prof:
    fig, ax = plt.subplots(1, 3, figsize=(10.2, 2.9))
    for d in prof:
        p = np.array([r["p"] for r in d["rows"]]); L = np.array([r["loss"] for r in d["rows"]])
        g = np.array([np.nan if r["dLdp"] is None else r["dLdp"] for r in d["rows"]])
        te = np.array([r["test_MRR"] for r in d["rows"]])
        lab = f"split {d['seed']}"
        ax[0].plot(p, (L - L.min()) * 1e3, "-o", ms=3, lw=1, label=lab)
        ax[1].plot(p, g * 1e3, "-o", ms=3, lw=1, label=lab)
        ax[2].plot(p, te, "-o", ms=3, lw=1, label=lab)
    for a in ax:
        for x in (-1, 0):
            a.axvline(x, color="#aaaaaa", lw=0.7, ls="--", zorder=0)
        a.set_xlabel("fixed t-norm parameter p")
    ax[1].axhline(0, color="#555555", lw=0.6)
    ax[0].set_ylabel("training loss − minimum (×10⁻³ nat)"); ax[1].set_ylabel("∂L/∂p at the converged fit (×10⁻³)")
    ax[2].set_ylabel("test MRR of the converged fit"); ax[0].legend(frameon=False, fontsize=7)
    plt.tight_layout(); plt.savefig(f"{F}/fig2_profile_lbfgs.png", dpi=300); plt.close()

# ---- Figure 3: refinements ----
TL = {"het_CtD": "Hetionet compound–disease", "het_DpS": "Hetionet disease–symptom", "hkg_DpS": "EHR graph (graded)"}
ROWS = [("dnfr_overlap__dnfr", "R1 overlap-aware aggregation, added"), ("dnfr_gumbel__dnfr", "R2 discrete operator choice, added"),
        ("dnfr_ic__dnfr", "R3 information-content discount, added"), ("dnfr_anneal__dnfr", "R4 membership annealing, added"),
        ("product_overlap__product", "R1 added to the fixed product"),
        ("dnfr2__dnfr2_no_overlap", "R1 in DNFR 2.0 (vs. without)"), ("dnfr2__dnfr2_no_gumbel", "R2 in DNFR 2.0 (vs. without)"),
        ("dnfr2__dnfr2_no_ic", "R3 in DNFR 2.0 (vs. without)"), ("dnfr2__dnfr2_no_anneal", "R4 in DNFR 2.0 (vs. without)"),
        ("dnfr2__dnfr", "DNFR 2.0 vs. DNFR"), ("fixp_val__dnfr", "fixed p (validation) vs. DNFR")]
tasks = [t for t in TL if t in ST]
if tasks:
    fig, axes = plt.subplots(1, len(tasks), figsize=(3.4 * len(tasks), 4.2), sharey=True, squeeze=False)
    for j, t in enumerate(tasks):
        a = axes[0, j]; PRD = ST[t]["paired"]
        for i, (k, lab) in enumerate(ROWS):
            v = PRD.get(k)
            if not v:
                continue
            col = {"different": "#b5523b", "equivalent": "#2c6e8f"}.get(v["verdict"], "#555555")
            a.errorbar(v["delta"], i, xerr=[[v["delta"] - v["bs95"][0]], [v["bs95"][1] - v["delta"]]], fmt="o", ms=3.5,
                       color=col, lw=1, capsize=2)
            a.text(0.99, i, f"{v['J']}", transform=a.get_yaxis_transform(), ha="right", va="center", fontsize=6.5, color="#777777")
        a.axvline(0, color="#333333", lw=0.7); a.axvspan(-0.01, 0.01, color="#e8e8e8", zorder=0)
        a.set_title(TL[t], fontsize=9); a.set_xlabel("difference in test MRR")
        a.set_yticks(range(len(ROWS))); a.set_yticklabels([r[1] for r in ROWS]); a.invert_yaxis()
    plt.tight_layout(); plt.savefig(f"{F}/fig3_refinements.png", dpi=300); plt.close()
print("figures written")
