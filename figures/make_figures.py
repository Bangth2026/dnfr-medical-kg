import json, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import os
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
R = os.environ.get("DNFR_RESULTS", os.path.join(ROOT, "results"))
TASKS = [("het_CtD", "Hetionet\ncompound–disease"), ("het_DpS", "Hetionet\ndisease–symptom"),
         ("pkg_ind", "PrimeKG\ndrug–disease"), ("hkg_DpS", "EHR health KG\ndisease–symptom")]
ink, acc = "#1f2a36", "#2c6e8f"

# Figure 2: learned p per metapath and split, for each initial value p0 (no penalty except the default)
VAR = [("dnfr", -0.1, "p₀ = −0.1 (default, λ = 10⁻³)", "#2c6e8f"), ("dnfr_l2_0", -0.1, "p₀ = −0.1, no penalty", "#7fb3cc"),
       ("dnfr_pm2", -2.0, "p₀ = −2, no penalty", "#b5523b"), ("dnfr_pm1", -1.0, "p₀ = −1, no penalty", "#d99a3e"),
       ("dnfr_pp05", 0.5, "p₀ = 0.5, no penalty", "#5a8f4e")]
fig, ax = plt.subplots(figsize=(7.4, 3.9))
rng = np.random.default_rng(0)
yt = []
for i, (t, lab) in enumerate(TASKS):
    f = f"{R}/{t}.json"
    if not os.path.exists(f):
        continue
    d = json.load(open(f))
    for j, (v, p0, vl, col) in enumerate(VAR):
        ps = []
        for r in d["runs"]:
            if v in r.get("p_learned", {}):
                ps += r["p_learned"][v]
            elif v == "dnfr" and "dnfr_params" in r:
                ps += r["dnfr_params"]["p"]
        if not ps:
            continue
        y = i + (j - 2) * 0.15 + rng.uniform(-0.05, 0.05, len(ps))
        ax.scatter(ps, y, s=7, alpha=0.5, color=col, edgecolor="none", label=vl if i == 0 or t == "het_DpS" else None)
        ax.plot([p0], [i + (j - 2) * 0.15], marker="|", color="black", ms=8, mew=1.2)
    yt.append((i, lab))
h, l = ax.get_legend_handles_labels(); seen = {}
for hh, ll in zip(h, l):
    seen.setdefault(ll, hh)
ax.legend(seen.values(), seen.keys(), fontsize=7, frameon=False, loc="upper left", bbox_to_anchor=(0.0, -0.2), ncol=3)
for x in (-1, 0, 1):
    ax.axvline(x, color="#999999", lw=0.7, ls="--")
ax.text(-1, len(TASKS) - 0.45, "Hamacher", ha="center", fontsize=7.5, color="#555555")
ax.text(0, len(TASKS) - 0.45, "product", ha="center", fontsize=7.5, color="#555555")
ax.text(1, len(TASKS) - 0.45, "Łukasiewicz", ha="center", fontsize=7.5, color="#555555")
ax.set_yticks([i for i, _ in yt]); ax.set_yticklabels([l for _, l in yt], fontsize=8.5)
ax.set_xlabel("learned Schweizer–Sklar parameter p (one point per metapath and split; | marks p₀)", fontsize=8.5)
ax.set_xlim(-5.1, 1.1); ax.set_ylim(-0.6, len(TASKS) - 0.2)
for s_ in ("top", "right"): ax.spines[s_].set_visible(False)
plt.tight_layout(); plt.savefig(os.path.join(ROOT, "figures", "fig2_p.png"), dpi=300)

# Figure 3: sparsity
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
                xs.append(x); ys.append(S[m]["MRR"]["mean"]); es.append(S[m]["MRR"]["sd"])
        if len(xs) >= 2:
            ax.errorbar(xs, ys, yerr=es, fmt=sty, ms=4, lw=1.1, capsize=2, label=ml)
    ax.set_title(lab.replace("\n", " "), fontsize=9); ax.set_xlabel("share of edges kept", fontsize=8.5)
    ax.set_xticks([0.25, 0.5, 1.0]); ax.invert_xaxis(); ax.tick_params(labelsize=8)
    for s in ("top", "right"): ax.spines[s].set_visible(False)
axes[0].set_ylabel("test MRR", fontsize=8.5)
axes[1].legend(fontsize=7, frameon=False, loc="upper right")
plt.tight_layout(); plt.savefig(os.path.join(ROOT, "figures", "fig3_sparsity.png"), dpi=300)
print("figs ok")
