"""Figures 2-4 and A1 from the result files (Figure 1: make_fig1.py)."""
import json, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
R = os.environ.get("DNFR_RESULTS", os.path.join(ROOT, "results"))
F = os.path.join(ROOT, "figures")
TASKS = [("het_CtD", "Hetionet compound–disease"), ("het_DpS", "Hetionet disease–symptom"),
         ("pkg_ind", "PrimeKG drug–disease"), ("hkg_DpS", "EHR graph disease–symptom (graded)")]
LAB = dict(TASKS)
COL = {-2.0: "#b5523b", -1.0: "#d99a3e", -0.1: "#2c6e8f", 0.5: "#5a8f4e"}
plt.rcParams.update({"font.size": 8.5, "axes.spines.top": False, "axes.spines.right": False})


def ref_lines(ax, vertical=True):
    for x, nm in ((-1, "Hamacher"), (0, "product"), (1, "Łukasiewicz")):
        (ax.axvline if vertical else ax.axhline)(x, color="#aaaaaa", lw=0.7, ls="--", zorder=0)


# ---- Figure 2: profile over fixed p ----
prof = [(t, json.load(open(f"{R}/profile_{t}.json"))) for t, _ in TASKS if os.path.exists(f"{R}/profile_{t}.json")]
if prof:
    fig, axes = plt.subplots(2, len(prof), figsize=(3.7 * len(prof), 5.0), squeeze=False)
    for j, (t, d) in enumerate(prof):
        seeds = sorted({r["seed"] for r in d.values()}); vals = sorted({r["value"] for r in d.values()})
        for s in seeds:
            v = [x for x in vals if f"seed{s}_p{x:g}" in d]
            if len(v) < 2:
                continue
            loss = np.array([d[f"seed{s}_p{x:g}"]["final"]["train_loss"] for x in v])
            te = np.array([d[f"seed{s}_p{x:g}"]["best_valid"]["test_MRR"] for x in v])
            keep = [k for k, x in enumerate(v) if loss[k] - loss.min() < 0.2]   # untrainable p > 0 runs drawn off-scale
            axes[0, j].plot([v[k] for k in keep], (loss - loss.min())[keep], "-o", ms=3, lw=1, label=f"split {s}")
            axes[1, j].plot(v, te, "-o", ms=3, lw=1, label=f"split {s}")
        for i in range(2):
            ref_lines(axes[i, j]); axes[i, j].set_xlabel("fixed t-norm parameter p")
        axes[0, j].set_title(LAB[t], fontsize=9)
        axes[0, j].set_ylabel("final training loss − minimum over p"); axes[1, j].set_ylabel("test MRR (best validation epoch)")
        axes[0, j].legend(frameon=False, fontsize=7)
    plt.tight_layout(); plt.savefig(f"{F}/fig2_profile.png", dpi=300); plt.close()

# ---- Figure 3: trajectories of p without early stopping ----
lng = [(t, json.load(open(f"{R}/long_{t}.json"))) for t, _ in TASKS if os.path.exists(f"{R}/long_{t}.json")]
if lng:
    fig, axes = plt.subplots(1, len(lng), figsize=(3.7 * len(lng), 2.9), squeeze=False)
    for j, (t, d) in enumerate(lng):
        ax = axes[0, j]
        for k, r in d.items():
            ep = [h["epoch"] for h in r["history"]]; P = np.array([h["p"] for h in r["history"]])
            col = COL.get(r["value"], "k")
            if P.shape[1] == 1:
                ax.plot(ep, P[:, 0], color=col, lw=1.1, alpha=0.9)
            else:
                if r["seed"] == min(x["seed"] for x in d.values()):
                    for m in range(P.shape[1]):
                        ax.plot(ep, P[:, m], color=col, lw=0.5, alpha=0.35)
                ax.plot(ep, np.median(P, 1), color=col, lw=1.6)
            ax.axvline(r["best_valid"]["epoch"], color=col, lw=0.6, ls=":", alpha=0.8)
        for x in (-1, 0, 1):
            ax.axhline(x, color="#aaaaaa", lw=0.7, ls="--", zorder=0)
        ax.set_title(LAB[t], fontsize=9); ax.set_xlabel("epoch"); ax.set_ylabel("learned p")
        ax.set_ylim(-5.05, 1.05)
        for v, c in COL.items():
            if any(r["value"] == v for r in d.values()):
                ax.plot([], [], color=c, label=f"p₀ = {v:g}".replace("-", "−"))
        ax.legend(frameon=False, fontsize=7, loc="lower right" if P.shape[1] == 1 else "lower left")
    plt.tight_layout(); plt.savefig(f"{F}/fig3_trajectories.png", dpi=300); plt.close()

# ---- Figure A1: learned p at early stopping, per task, lines join the same metapath across restarts ----
VAR = [("dnfr_pm2", -2.0, "p₀ = −2"), ("dnfr_pm1", -1.0, "p₀ = −1"), ("dnfr_l2_0", -0.1, "p₀ = −0.1"),
       ("dnfr_pp05", 0.5, "p₀ = 0.5"), ("dnfr", -0.1, "p₀ = −0.1, λ = 10⁻³")]
avail = [(t, l) for t, l in TASKS if os.path.exists(f"{R}/{t}.json")]
fig, axes = plt.subplots(1, len(avail), figsize=(3.0 * len(avail), 3.2), squeeze=False, sharey=True)
for j, (t, lab) in enumerate(avail):
    ax = axes[0, j]; d = json.load(open(f"{R}/{t}.json"))
    run = [r for r in d["runs"] if all(v in r.get("p_learned", {}) for v, _, _ in VAR[:4])]
    run = run[0] if run else None
    for i, (v, p0, vl) in enumerate(VAR):
        ps = []
        for r in d["runs"]:
            if v in r.get("p_learned", {}):
                ps += r["p_learned"][v]
            elif v == "dnfr" and "dnfr_params" in r:
                ps += r["dnfr_params"]["p"]
        if ps:
            ax.scatter(ps, np.full(len(ps), i) + np.random.default_rng(i).uniform(-0.12, 0.12, len(ps)), s=5, alpha=0.4,
                       color=COL.get(p0, "#555555") if v != "dnfr" else "#222222", edgecolor="none")
            ax.plot([p0], [i], marker="|", color="black", ms=10, mew=1.3)
    if run is not None and len(run["p_learned"][VAR[0][0]]) > 1:
        M = np.array([run["p_learned"][v] for v, _, _ in VAR[:4]])
        for m in range(M.shape[1]):
            ax.plot(M[:, m], range(4), color="#555555", lw=0.4, alpha=0.5)
    ref_lines(ax); ax.set_title(lab, fontsize=8.5); ax.set_xlabel("learned p"); ax.set_xlim(-5.1, 1.1)
    ax.set_yticks(range(len(VAR))); ax.set_yticklabels([x[2] for x in VAR], fontsize=7.5)
plt.tight_layout(); plt.savefig(f"{F}/figA1_restarts.png", dpi=300); plt.close()

# ---- Figure 4: sparsity ----
st = json.load(open(f"{R}/paper_stats.json"))
meths = [("dnfr", "DNFR", "-o"), ("product", "product (fixed)", "--s"), ("dnfr_nodeg", "DNFR, no degree", ":^"),
         ("dwpc_lr", "DWPC-LR", "-.d"), ("softimpute", "Soft-Impute", "--x")]
fig, axes = plt.subplots(1, 2, figsize=(7.4, 3.0), sharey=False)
for ax, (t, lab) in zip(axes, TASKS[:2]):
    for m, ml, sty in meths:
        xs, ys, es = [], [], []
        for x, tag in [(1.0, ""), (0.5, "_sp0.50"), (0.25, "_sp0.25")]:
            S = st.get(t + tag, {}).get("summary", {})
            if m in S:
                sub = S[m]
                # all-edges point over the same three splits as the thinned graphs
                xs.append(x); ys.append(sub["MRR"]["mean"]); es.append(sub["MRR"]["sd"])
        if len(xs) >= 2:
            ax.errorbar(xs, ys, yerr=es, fmt=sty, ms=4, lw=1.1, capsize=2, label=ml)
    ax.set_title(lab, fontsize=9); ax.set_xlabel("share of edges kept")
    ax.set_xticks([0.25, 0.5, 1.0]); ax.invert_xaxis()
axes[0].set_ylabel("test MRR (mean ± SD over splits)")
axes[1].legend(fontsize=7, frameon=False, loc="upper right")
plt.tight_layout(); plt.savefig(f"{F}/fig4_sparsity.png", dpi=300); plt.close()
print("figs ok")
