import json, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import os
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
R = os.path.join(ROOT, "results")
TASKS = [("het_CtD", "Hetionet\ncompound–disease"), ("het_DpS", "Hetionet\ndisease–symptom"),
         ("pkg_ind", "PrimeKG\ndrug–disease"), ("hkg_DpS", "EHR health KG\ndisease–symptom")]
ink, acc = "#1f2a36", "#2c6e8f"

# Figure 2: learned p per metapath
fig, ax = plt.subplots(figsize=(7.2, 3.4))
rng = np.random.default_rng(0)
yt = []
for i, (t, lab) in enumerate(TASKS):
    f = f"{R}/{t}.json"
    if not os.path.exists(f):
        continue
    d = json.load(open(f))
    for r in d["runs"]:
        if "dnfr_params" not in r:
            continue
        p = np.array(r["dnfr_params"]["p"]); w = 1 / (1 + np.exp(-np.array(r["dnfr_params"]["w_raw"])))
        ax.scatter(p, i + rng.uniform(-0.25, 0.25, len(p)), s=8 + 120 * w / max(w.max(), 1e-9), alpha=0.45, color=acc, edgecolor="none")
    yt.append((i, lab))
for x, lab in [(-1, "min-like"), (0, "product"), (1, "Łukasiewicz")]:
    ax.axvline(x, color="#999999", lw=0.8, ls="--"); ax.text(x, len(TASKS) - 0.45, lab, ha="center", fontsize=8, color="#555555")
ax.set_yticks([i for i, _ in yt]); ax.set_yticklabels([l for _, l in yt], fontsize=8.5)
ax.set_xlabel("learned Schweizer–Sklar parameter p (per metapath and split)", fontsize=9)
ax.set_xlim(-5.1, 1.1); ax.set_ylim(-0.6, len(TASKS) - 0.2)
for s in ("top", "right"): ax.spines[s].set_visible(False)
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
