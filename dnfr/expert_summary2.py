"""Agreement and discrimination from the second rater's sheet (expert_sheet2.py), round-5 M6.
Input: results/expert_review2_filled.xlsx (as returned by the second rater), results/expert_review2_key.json, and the first
rater's sheet results/expert_review_filled.xlsx. Output: results/expert_review2.json with
- agreement on the overlapping correct pairs: Cohen's kappa and raw agreement for relations_correct and rationale (paths)
  and for pair_plausible (pairs), with the number of items rated by both;
- discrimination: for correct and for incorrect predictions, the share of pairs judged plausible, the share of paths whose
  relations were judged correct, and the mean usefulness, and the AUC of usefulness for separating correct from incorrect;
- the rater's self-reported qualification (sheet 'Rater').
Blank cells are missing, not 'unsure'."""
import os, sys, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import OUT
import numpy as np

norm = lambda x: (str(x).strip().lower() if x is not None and str(x).strip() else None)


def kappa(a, b):
    pairs = [(x, y) for x, y in zip(a, b) if x is not None and y is not None]
    if not pairs:
        return dict(n=0, kappa=None, agreement=None)
    cats = sorted({c for p in pairs for c in p}); n = len(pairs)
    po = sum(x == y for x, y in pairs) / n
    pe = sum((sum(x == c for x, _ in pairs) / n) * (sum(y == c for _, y in pairs) / n) for c in cats)
    return dict(n=n, agreement=po, kappa=(po - pe) / (1 - pe) if pe < 1 else None)


def auc(pos, neg):
    pos, neg = [x for x in pos if x is not None], [x for x in neg if x is not None]
    if not pos or not neg:
        return None
    return float(np.mean([(p > q) + 0.5 * (p == q) for p in pos for q in neg]))


def rows_of(f):
    from openpyxl import load_workbook
    ws = load_workbook(f, data_only=True)["Ratings"]
    return {r[0]: r for r in ws.iter_rows(min_row=2, values_only=True) if r[0]}


def main():
    f2 = f"{OUT}/expert_review2_filled.xlsx"
    if not os.path.exists(f2):
        print("no filled sheet at", f2); return
    from openpyxl import load_workbook
    key = json.load(open(f"{OUT}/expert_review2_key.json"))
    R2 = rows_of(f2); R1 = rows_of(f"{OUT}/expert_review_filled.xlsx")
    rater = {str(r[0]): r[1] for r in load_workbook(f2, data_only=True)["Rater"].iter_rows(values_only=True) if r[0]}
    pairs = [k for k in key if "-p" not in k]
    paths = [k for k in key if "-p" in k]
    ov_pairs = [k for k in pairs if key[k]["correct"] and key[k]["first_rater_id"]]
    ov_paths = [k for k in paths if key[k]["first_rater_id"]]
    agree = dict(
        relations_correct=kappa([norm(R1[key[k]["first_rater_id"]][5]) for k in ov_paths], [norm(R2[k][5]) for k in ov_paths]),
        rationale=kappa([norm(R1[key[k]["first_rater_id"]][6]) for k in ov_paths], [norm(R2[k][6]) for k in ov_paths]),
        pair_plausible=kappa([norm(R1[key[k]["first_rater_id"]][7]) for k in ov_pairs], [norm(R2[k][7]) for k in ov_pairs]))
    def grp(correct):
        P = [k for k in pairs if key[k]["correct"] == correct]
        Q = [k for k in paths if key[k.rsplit("-p", 1)[0]]["correct"] == correct]
        pl = [norm(R2[k][7]) for k in P]; rc = [norm(R2[k][5]) for k in Q]
        us = [int(R2[k][8]) for k in P if R2[k][8] not in (None, "")]
        sh = lambda v, lab: (sum(x == lab for x in v if x) / len([x for x in v if x])) if any(v) else None
        return dict(n_pairs=len(P), plausible=sh(pl, "yes"), implausible=sh(pl, "no"), relations_correct=sh(rc, "yes"),
                    usefulness_mean=float(np.mean(us)) if us else None, usefulness=us)
    G1, G0 = grp(True), grp(False)
    out = dict(rater=rater, agreement=agree, correct=G1, incorrect=G0,
               auc_usefulness=auc(G1["usefulness"], G0["usefulness"]),
               auc_plausible=auc([{"yes": 1, "unsure": 0.5, "no": 0}.get(norm(R2[k][7])) for k in pairs if key[k]["correct"]],
                                 [{"yes": 1, "unsure": 0.5, "no": 0}.get(norm(R2[k][7])) for k in pairs if not key[k]["correct"]]))
    json.dump(out, open(f"{OUT}/expert_review2.json", "w"), indent=1)
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
