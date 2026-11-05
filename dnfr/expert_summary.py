"""Summary of the second author's clinical assessment of DNFR's explanations (see expert_sheet.py).
Input: results/expert_review_filled.xlsx (the rating sheet as returned by the rater) and results/expert_review_key.json.
Output: results/expert_review.json. Shares are of rated rows; blanks are counted as missing, not as 'unsure'. For the
rationale we also report the share by the model's evidence share of the path (first, second, third path) and the
association between the rationale and the metapath class (paths through a gene or anatomy node vs disease-only paths)."""
import os, sys, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import OUT
import numpy as np


def main():
    from openpyxl import load_workbook
    f = f"{OUT}/expert_review_filled.xlsx"
    if not os.path.exists(f):
        print("no filled sheet at", f); return
    key = json.load(open(f"{OUT}/expert_review_key.json"))
    ws = load_workbook(f, data_only=True)["Ratings"]
    rows = list(ws.iter_rows(min_row=2, values_only=True))
    norm = lambda x: (str(x).strip().lower() if x is not None and str(x).strip() else None)
    out = {}
    for task in sorted({v["task"] for v in key.values() if "task" in v}):
        pairs = [r for r in rows if r[0] and r[0] in key and "task" in key[r[0]] and key[r[0]]["task"] == task]
        paths = [r for r in rows if r[0] and "-p" in str(r[0]) and str(r[0]).rsplit("-p", 1)[0] in {p[0] for p in pairs}]
        def share(vals, lab):
            v = [x for x in vals if x is not None]
            return dict(n=len(v), share=(sum(1 for x in v if x == lab) / len(v)) if v else None)
        rc = [norm(r[5]) for r in paths]; ra = [norm(r[6]) for r in paths]
        pl = [norm(r[7]) for r in pairs]
        us = [int(r[8]) for r in pairs if r[8] not in (None, "")]
        gene = lambda r: any(x in key[r[0]]["metapath"] for x in ("G", "A"))
        by_pos = {j: share([norm(r[6]) for r in paths if str(r[0]).endswith(f"-p{j}")], "m") for j in (1, 2, 3)}
        out[task] = dict(
            n_pairs=len(pairs), n_paths=len(paths),
            relations_correct=share(rc, "yes"), relations_wrong=share(rc, "no"),
            rationale={lab: share(ra, lab.lower()) for lab in ("M", "A", "I")},
            mechanistic_by_position=by_pos,
            mechanistic_gene_or_anatomy=share([norm(r[6]) for r in paths if gene(r)], "m"),
            mechanistic_disease_only=share([norm(r[6]) for r in paths if not gene(r)], "m"),
            pair_plausible=share(pl, "yes"),
            wrong_paths=dict(n_no=sum(1 for r in paths if norm(r[5]) == "no"),
                             n_dla=sum(1 for r in paths if norm(r[5]) == "no" and key[r[0]]["metapath"].startswith("DlA-DlA")),
                             mu_mid=float(np.median([key[r[0]]["mu"][1] for r in paths if norm(r[5]) == "no"
                                                     and key[r[0]]["metapath"].startswith("DlA-DlA")] or [np.nan])),
                             mu_yes=float(np.median([min(key[r[0]]["mu"]) for r in paths if norm(r[5]) == "yes"] or [np.nan]))),
            usefulness=dict(n=len(us), mean=(sum(us) / len(us)) if us else None,
                            median=(sorted(us)[len(us) // 2]) if us else None))
    json.dump(out, open(f"{OUT}/expert_review.json", "w"), indent=1)
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
