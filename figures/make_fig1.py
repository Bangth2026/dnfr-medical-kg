"""Figure 1: schematic of DNFR (four steps)."""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
plt.rcParams["mathtext.fontset"] = "dejavusans"
ink, acc, fill = "#1f2a36", "#2c6e8f", "#e8f0f4"
boxes = [
    ("1  Path extraction",
     ["query pair $(h, t)$", "simple paths of length 2–3,", "grouped by metapath $m$;",
      "every $h$–$t$ edge removed", "(leave-dyad-out)"]),
    ("2  Edge membership",
     [r"$\mu(e)=\sigma(a_r-b_r\,\ell^{out}_u-c_r\,\ell^{in}_v+g_r\,\mathrm{logit}\,w_e)$",
      r"$\ell=\log(1+\mathrm{degree})$ in relation $r$", r"$w_e$: graded evidence, if any",
      r"$a_r, b_r, c_r, g_r$ learned per $r$"]),
    ("3  Schweizer–Sklar t-norm",
     [r"$T_{p}=\left[\max\left(0,\ \sum_i \mu_i^{\,p}-(L-1)\right)\right]^{1/p}$",
      r"one $p_m$ per metapath", r"$p\to-\infty$: minimum;  $p=-1$: Hamacher",
      r"$p\to 0$: product;  $p=1$: Łukasiewicz"]),
    ("4  Noisy-OR evidence",
     [r"$P(h,t)=1-(1-\beta_0)\prod_{\pi}\,(1-\omega_m T_{p_m}(\pi))$",
      r"$\omega_m$: metapath reliability", "explanation: each path's share",
      r"of $-\log(1-P)$ above the leak"]),
]
fig, ax = plt.subplots(figsize=(13.8, 4.0)); ax.set_xlim(0, 13.8); ax.set_ylim(0, 4); ax.axis("off")
W = [2.55, 3.35, 3.3, 3.35]; gap = 0.38; x = 0.05
for (title, lines), w in zip(boxes, W):
    ax.add_patch(FancyBboxPatch((x, 0.95), w, 2.65, boxstyle="round,pad=0.02,rounding_size=0.08",
                                fc=fill, ec=acc, lw=1.6))
    ax.text(x + w / 2, 3.3, title, ha="center", va="center", fontsize=12.5, weight="bold", color=ink)
    for i, ln in enumerate(lines):
        ax.text(x + w / 2, 2.8 - i * 0.37, ln, ha="center", va="center", fontsize=9.6, color=ink)
    if x + w + gap < 13.4:
        ax.annotate("", xy=(x + w + gap - 0.02, 2.27), xytext=(x + w + 0.02, 2.27),
                    arrowprops=dict(arrowstyle="-|>", color=acc, lw=1.8))
    x += w + gap
ax.text(6.9, 0.55, "Trained end to end with a filtered softmax over all candidate tails; "
        "early stopping on validation MRR (default) or a fixed number of epochs (Section 4.3)",
        ha="center", fontsize=10, style="italic", color=ink)
plt.savefig(os.path.join(ROOT, "figures", "fig1_dnfr.png"), dpi=220, bbox_inches="tight")
print("fig1 ok")
