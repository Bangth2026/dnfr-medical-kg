"""Shared state for the third-revision manuscript build: statistics files, number formatting, citation numbering."""
import os, re, json
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.environ.get("DNFR_RESULTS", os.path.join(ROOT, "results"))
from references import REFS

S = json.load(open(f"{RES}/paper_stats.json"))
RS = json.load(open(f"{RES}/revision_stats.json"))
DS = json.load(open(f"{RES}/data_stats.json"))
XS = json.load(open(f"{RES}/extra_stats.json"))
R3 = json.load(open(f"{RES}/rev3_stats.json")) if os.path.exists(f"{RES}/rev3_stats.json") else {}
R4 = json.load(open(f"{RES}/rev4_stats.json")) if os.path.exists(f"{RES}/rev4_stats.json") else {}
XREV = json.load(open(f"{RES}/expert_review.json")) if os.path.exists(f"{RES}/expert_review.json") else {}
PC = json.load(open(f"{RES}/prior_check.json")) if os.path.exists(f"{RES}/prior_check.json") else {}

TASKS = ["het_CtD", "het_DpS", "pkg_ind", "hkg_DpS"]
TASK_LABEL = {"het_CtD": "Hetionet CtD", "het_DpS": "Hetionet DpS", "pkg_ind": "PrimeKG drug–disease", "hkg_DpS": "EHR graph (graded)"}
MINUS = "−"


def f3(x, d=3):
    s = f"{x:.{d}f}"
    if s.startswith("-"):
        s = MINUS + s[1:]
    if s in (MINUS + "0." + "0" * d,):
        s = "0." + "0" * d
    return s


def sg(x, d=3):
    s = f"{x:+.{d}f}"
    if s == "+0." + "0" * d and x < 0:
        s = "-0." + "0" * d          # keep the sign of a small negative value (interval bounds)
    return s.replace("-", MINUS)


def ri(x):
    """round half up (epochs)"""
    import math
    return int(math.floor(x + 0.5))


def pct(x):
    return f"{100 * x:.0f}%"


STUB = bool(os.environ.get("MS_STUB"))  # test builds on incomplete results only
_STUB_S = {k: {"mean": float("nan"), "sd": 0.0} for k in ("MRR", "H1", "H3", "H10")}
_STUB_P = dict(delta_MRR=float("nan"), ci_lo=float("nan"), ci_hi=float("nan"), nb_ci95_lo=float("nan"), nb_ci95_hi=float("nan"),
               nb_ci90_lo=float("nan"), nb_ci90_hi=float("nan"), seed_wins=0, seeds=0, per_split=[], verdict="inconclusive")


def summ(task, m, tag=""):
    v = S.get(task + tag, {}).get("summary", {}).get(m)
    if v is None and STUB:
        return dict(_STUB_S, n_seeds=0)
    return v


def mrr(task, m, tag="", sd=True, mark_n=True):
    s = summ(task, m, tag)
    if s is None:
        return "—"
    out = f3(s["MRR"]["mean"]) + (f" ± {f3(s['MRR']['sd'])}" if sd else "")
    nref = summ(task, "dnfr", tag)["n_seeds"] if summ(task, "dnfr", tag) else s["n_seeds"]
    if mark_n and s["n_seeds"] != nref:
        out += f" ({s['n_seeds']})"
    return out


def nseeds(task, m, tag=""):
    s = summ(task, m, tag)
    return s["n_seeds"] if s else 0


def pr(task, a, b, tag=""):
    """Paired comparison a minus b (dict with delta_MRR, ci, nb intervals, verdict), or None."""
    st = S.get(task + tag)
    if st is None:
        return dict(_STUB_P) if STUB else None
    if a == "dnfr" and b in st["paired"]:
        return st["paired"][b]
    k = f"{a}__vs__{b}"
    if k in st.get("paired_extra", {}) and st["paired_extra"][k]:
        return st["paired_extra"][k]
    k2 = f"{b}__vs__{a}"
    src = None
    if b == "dnfr" and a in st["paired"]:
        src = st["paired"][a]
    elif k2 in st.get("paired_extra", {}) and st["paired_extra"][k2]:
        src = st["paired_extra"][k2]
    if src is None:
        return dict(_STUB_P) if STUB else None
    neg = dict(src)
    neg["delta_MRR"] = -src["delta_MRR"]; neg["ci_lo"], neg["ci_hi"] = -src["ci_hi"], -src["ci_lo"]
    for lvl in ("90", "95"):
        if f"nb_ci{lvl}_lo" in src:
            neg[f"nb_ci{lvl}_lo"], neg[f"nb_ci{lvl}_hi"] = -src[f"nb_ci{lvl}_hi"], -src[f"nb_ci{lvl}_lo"]
    neg["seed_wins"] = sum(1 for x in src["per_split"] if x < 0)
    neg["per_split"] = [-x for x in src["per_split"]]
    return neg


VMARK = {"equivalent": "≈", "different": "†", "inconclusive": ""}


def cell(p, verdict=True, wins=True, ci="boot"):
    if p is None:
        return "—"
    s = sg(p["delta_MRR"])
    if ci == "boot":
        s += f" [{sg(p['ci_lo'])}, {sg(p['ci_hi'])}]"
    elif ci == "nb" and "nb_ci95_lo" in p:
        s += f" [{sg(p['nb_ci95_lo'])}, {sg(p['nb_ci95_hi'])}]"
    if wins:
        s += f" {p['seed_wins']}/{p['seeds']}"
    if verdict:
        s += VMARK.get(p.get("verdict", "inconclusive"), "")
    return s


def dci(p, nb=False):
    if p is None:
        return "—"
    if nb:
        return f"{sg(p['delta_MRR'])} [{sg(p['nb_ci95_lo'])}, {sg(p['nb_ci95_hi'])}]"
    return f"{sg(p['delta_MRR'])} [{sg(p['ci_lo'])}, {sg(p['ci_hi'])}]"


def ph(p):
    if p is None or p.get("p_holm") is None:
        return "—"
    v = p["p_holm"]
    return "<0.001" if v < 0.001 else f"{v:.3f}"


# ---------------- citations ----------------
ORDER = []


def C(text):
    """Replace [@a;b] by numbered citations in order of first appearance."""
    def rep(m):
        keys = [k.strip() for k in m.group(1).split(";")]
        nums = []
        for k in keys:
            if k not in REFS:
                raise KeyError(k)
            if k not in ORDER:
                ORDER.append(k)
            nums.append(ORDER.index(k) + 1)
        nums = sorted(set(nums)); parts = []; i = 0
        while i < len(nums):
            j = i
            while j + 1 < len(nums) and nums[j + 1] == nums[j] + 1:
                j += 1
            parts.append(f"{nums[i]}–{nums[j]}" if j - i >= 2 else ",".join(str(x) for x in nums[i:j + 1]))
            i = j + 1
        return "[" + ",".join(parts) + "]"
    return re.sub(r"\[@([^\]]+)\]", rep, text)

VC = json.load(open(f"{RES}/verify_clean.json")) if os.path.exists(f"{RES}/verify_clean.json") else {}
