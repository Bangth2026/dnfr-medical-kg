import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
INK, INK2, MUTED, GRID, SURF = "#0b0b0b", "#52514e", "#8a8984", "#e4e3df", "#fcfcfb"
C = ["#2a78d6", "#eb6834", "#1baf7a"]
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11, "text.color": INK, "axes.labelcolor": INK2,
                     "xtick.color": INK2, "ytick.color": INK2})
fig = plt.figure(figsize=(13.28, 5.31), dpi=200, facecolor=SURF)
fig.text(0.015, 0.94, "Which parts of a neuro-fuzzy path reasoner should be learned?", fontsize=17, weight="bold")
fig.text(0.015, 0.885, "DNFR on Hetionet, PrimeKG and a graded graph learned from 273,174 emergency-department records",
         fontsize=11.5, color=INK2)
# panel A: model
ax = fig.add_axes([0.01, 0.10, 0.27, 0.73]); ax.axis("off"); ax.set_xlim(0, 1); ax.set_ylim(0, 1)
steps = [("1  Paths", "length 2–3, grouped by metapath;\nleave-dyad-out"),
         ("2  Edge membership μ", "hub discount (degree)\n+ graded evidence"),
         ("3  t-norm per metapath", "Schweizer–Sklar p:\nmin … Hamacher … product"),
         ("4  Noisy-OR over paths", "score = sum of path terms\n→ exact fact attribution")]
for i, (h, s) in enumerate(steps):
    y = 0.84 - i * 0.245
    ax.add_patch(FancyBboxPatch((0.02, y - 0.085), 0.94, 0.19, boxstyle="round,pad=0.01,rounding_size=0.03",
                                fc="white", ec=GRID, lw=1.2))
    ax.text(0.06, y + 0.045, h, fontsize=11.5, weight="bold", va="center")
    ax.text(0.06, y - 0.035, s, fontsize=9.5, color=INK2, va="center")
    if i < 3:
        ax.annotate("", xy=(0.49, y - 0.12), xytext=(0.49, y - 0.09), arrowprops=dict(arrowstyle="-|>", color=MUTED))
ax.text(0.02, 1.0, "A. The model", fontsize=12, weight="bold")
# panel B: component deltas
bx = fig.add_axes([0.37, 0.17, 0.33, 0.57], facecolor=SURF)
comps = ["Noisy-OR vs\nstrongest path", "Degree terms\nvs none", "Learned p vs\nfixed product"]
vals = {"Hetionet drug–disease": [0.083, 0.162, -0.003], "Hetionet disease–symptom": [0.096, 0.020, 0.003],
        "EHR graph (graded)": [0.083, 0.072, 0.006]}
marks = ["o", "s", "D"]
for k, (lab, v) in enumerate(vals.items()):
    ys = [2 - j - (k - 1) * 0.22 for j in range(3)]
    bx.hlines(ys, 0, v, color=C[k], lw=2)
    bx.scatter(v, ys, s=46, marker=marks[k], color=C[k], edgecolor=SURF, linewidth=1.5, zorder=3, label=lab)
bx.axvline(0, color=INK2, lw=1)
bx.set_yticks([2, 1, 0]); bx.set_yticklabels(comps, fontsize=10.5)
bx.set_xlim(-0.03, 0.18); bx.set_xticks([0, 0.05, 0.10, 0.15]); bx.set_xlabel("Change in test MRR (mean over 10 splits)", fontsize=10)
bx.grid(axis="x", color=GRID, lw=0.8); bx.set_axisbelow(True)
for s in ("top", "right", "left"): bx.spines[s].set_visible(False)
bx.spines["bottom"].set_color(GRID); bx.tick_params(axis="y", length=0)
bx.legend(loc="lower left", bbox_to_anchor=(0, 1.0), ncol=3, fontsize=8.6, frameon=False, handlelength=0.8, columnspacing=0.8)
bx.text(0.02, 0.0, "no consistent gain;\nloss flat in p", fontsize=9, color=INK2, va="center", ha="left")
fig.text(0.29, 0.83, "B. What carries the accuracy", fontsize=12, weight="bold")
# panel C: known indications
cx = fig.add_axes([0.79, 0.17, 0.19, 0.57], facecolor=SURF)
labs = ["Random\nsplits", "No treats\nmetapaths", "Disease-\ncentred"]
v = [0.599, 0.289, 0.079]
cx.bar(range(3), v, width=0.62, color=C[0], edgecolor=SURF, linewidth=2)
for i, x in enumerate(v): cx.text(i, x + 0.015, f"{x:.3f}", ha="center", fontsize=10.5, color=INK)
cx.set_xticks(range(3)); cx.set_xticklabels(labs, fontsize=9.5); cx.set_ylim(0, 0.7)
cx.set_ylabel("DNFR test MRR", fontsize=10); cx.grid(axis="y", color=GRID, lw=0.8); cx.set_axisbelow(True)
for s in ("top", "right"): cx.spines[s].set_visible(False)
cx.spines["left"].set_color(GRID); cx.spines["bottom"].set_color(GRID)
fig.text(0.735, 0.83, "C. Drug repurposing vs indications", fontsize=12, weight="bold")
fig.text(0.015, 0.035, "Aggregate all paths and discount hubs; select the operator on validation data, do not interpret "
         "a learned one; test what the evidence rests on.", fontsize=11, color=INK, style="italic")
fig.savefig("../figures/Graphical_abstract_JBI_DNFR.png", facecolor=SURF)
