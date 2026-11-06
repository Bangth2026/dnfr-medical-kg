"""Blinded rating sheet for a second, medically trained rater (round-5 M6).

Per disease-symptom task (Hetionet DpS, graded health knowledge graph) the sheet holds 20 pairs, 40 in all:
- 10 correctly predicted test pairs drawn with a fixed seed from the 25 that the first rater assessed (expert_sheet.py), with
  the same three leading paths, so that agreement between the two raters can be estimated (Cohen's kappa);
- 10 incorrect top-10 predictions: for test diseases drawn with a fixed seed, a symptom that DNFR ranks in its top 10 among
  the candidates that are not known positives (filtered ranking), with its three leading paths.
The 40 pairs are shuffled with a fixed seed and renumbered, so neither the order nor the identifier reveals whether a pair
is a correct prediction or which pairs the first rater saw. The rater records the same judgements as the first rater.
Output: results/expert_review2_sheet.xlsx (to fill in) and results/expert_review2_key.json (correctness and model values)."""
import os, sys, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from config import OUT
from explain import setup, explain, NAME
from dnfr import score_pairs
from expert_sheet import TASKS, rel_text

N_OVERLAP, N_WRONG, N_PATHS = 10, 10, 3


def wrong_pairs(task, n, seed=2027):
    """n incorrect top-10 predictions: (disease, symptom, three leading paths) for distinct test diseases."""
    ctx = setup(task, 0)
    z, z0 = score_pairs(ctx["model"], ctx["sd"])
    ukey, n_t = ctx["ukey"], ctx["n_t"]
    src, tgt = ukey // n_t, ukey % n_t
    ns, nt = ctx["kg"].names[ctx["spec"]["s"]], ctx["kg"].names[ctx["spec"]["t"]]
    test_pos = {}
    for s, t in ctx["te"]:
        test_pos.setdefault(int(s), set()).add(int(t))
    heads = sorted(test_pos)
    rng = np.random.default_rng(seed)
    out = []
    for s in rng.permutation(heads):
        sel = np.where(src == s)[0]
        known = ctx["known"].get(int(s), set()) | test_pos[int(s)]
        cand = [(float(z[i]), int(tgt[i])) for i in sel if int(tgt[i]) not in known]
        # candidates without any path score z0 and rank below every candidate with paths
        cand.sort(reverse=True)
        top = [t for _, t in cand[:10]]
        if not top:
            continue
        t = int(rng.choice(top))
        e = explain(ctx, int(s), t, top=N_PATHS)
        if not e["paths"]:
            continue
        out.append(dict(query=[NAME(str(ns[ctx["sources"][s]])), NAME(str(nt[t]))], paths=e["paths"],
                        rank_among_non_positive=1 + [x for _, x in cand].index(t)))
        if len(out) == n:
            break
    return out


def main():
    from openpyxl import Workbook
    from openpyxl.styles import Alignment, Font, PatternFill
    from openpyxl.worksheet.datavalidation import DataValidation
    key1 = json.load(open(f"{OUT}/expert_review_key.json"))
    blocks = []
    for task, label in TASKS.items():
        rated = sorted(k for k, v in key1.items() if v.get("task") == task)
        rng = np.random.default_rng(5)
        for rid in sorted(rng.choice(rated, size=N_OVERLAP, replace=False)):
            paths = [dict(nodes=key1[f"{rid}-p{j}"]["nodes"], metapath=key1[f"{rid}-p{j}"]["metapath"],
                          mu=key1[f"{rid}-p{j}"]["mu"], share=key1[f"{rid}-p{j}"]["share"])
                     for j in range(1, N_PATHS + 1) if f"{rid}-p{j}" in key1]
            blocks.append(dict(task=task, label=label, correct=True, first_rater_id=rid, query=key1[rid]["query"], paths=paths))
        for w in wrong_pairs(task, N_WRONG):
            blocks.append(dict(task=task, label=label, correct=False, first_rater_id=None, query=w["query"],
                               paths=[dict(nodes=p["nodes"], metapath=p["metapath"], mu=p["mu"], share=p["share"]) for p in w["paths"]],
                               rank_among_non_positive=w["rank_among_non_positive"]))
        print(task, "blocks so far", len(blocks), flush=True)
    order = np.random.default_rng(31).permutation(len(blocks))
    wb = Workbook()
    ins = wb.active; ins.title = "Instructions"
    lines = [
        "Clinical assessment of path explanations, second rater / Đánh giá lâm sàng các đường giải thích, người chấm thứ hai",
        "",
        "Each block is one disease–symptom pair that a model ranked among its ten most likely symptoms for that disease, followed "
        "by the three paths the model relies on most. Some of the pairs are true disease–symptom links and some are not; the sheet "
        "does not say which, and the order is random. Model scores are hidden on purpose. Please do not use AI tools.",
        "Mỗi khối là một cặp bệnh–triệu chứng mà mô hình xếp trong mười triệu chứng khả dĩ nhất của bệnh đó, kèm ba đường mô hình "
        "dựa vào nhiều nhất. Một số cặp là liên kết bệnh–triệu chứng có thật, một số thì không; phiếu không cho biết cặp nào, và thứ "
        "tự là ngẫu nhiên. Điểm số của mô hình được ẩn có chủ ý. Xin không dùng công cụ AI.",
        "",
        "Per path / Cho từng đường:",
        "  relations_correct: yes / no / unsure — every relation on the path agrees with biomedical knowledge "
        "(mọi quan hệ trên đường đúng với tri thức y sinh).",
        "  rationale: M = mechanistic (shared cause, pathway or organ system that explains the symptom / cơ chế); "
        "A = associative only (comorbidity or similar disease, no 'why' / chỉ liên kết); I = irrelevant or misleading "
        "(không liên quan hoặc gây hiểu sai).",
        "Per pair (on the pair row) / Cho từng cặp (ở dòng của cặp):",
        "  pair_plausible: yes / no / unsure — the disease plausibly presents with the symptom (bệnh có thể biểu hiện triệu chứng này).",
        "  usefulness: 1–5 — how useful the three paths are to a clinician judging the prediction (mức hữu ích cho người làm lâm sàng).",
        "  comment: optional / tùy chọn.",
        "",
        "Metapath abbreviations (Hetionet): D = disease, S = symptom, G = gene, A = anatomy; DpS = presents symptom, "
        "DrD = resembles disease, DaG = associates gene, DlA = localizes to anatomy; ⁻¹ = relation read backwards.",
        "Graded graph: disease–symptom edges were learned from emergency-department records by Rotmensch et al. (2017).",
        "",
        "Rater / Người chấm: name, medical qualification, specialty, years of clinical practice — please fill in on the 'Rater' sheet.",
    ]
    for i, l in enumerate(lines, 1):
        ins.cell(i, 1, l).alignment = Alignment(wrap_text=True)
    ins.column_dimensions["A"].width = 140
    ins["A1"].font = Font(bold=True, size=13)
    rs = wb.create_sheet("Rater")
    for i, l in enumerate(["Name / Họ tên", "Qualification / Học vị, chức danh", "Specialty / Chuyên khoa",
                           "Years of clinical practice / Số năm hành nghề", "Date / Ngày chấm",
                           "Relationship to the authors / Quan hệ với các tác giả"], 1):
        rs.cell(i, 1, l).font = Font(bold=True); rs.cell(i, 2, "")
    rs.column_dimensions["A"].width = 52; rs.column_dimensions["B"].width = 60
    ws = wb.create_sheet("Ratings")
    hdr = ["row_id", "task", "pair", "path", "relations", "relations_correct", "rationale", "pair_plausible", "usefulness", "comment"]
    ws.append(hdr)
    for c in ws[1]:
        c.font = Font(bold=True)
    yn = DataValidation(type="list", formula1='"yes,no,unsure"', allow_blank=True)
    ra = DataValidation(type="list", formula1='"M,A,I"', allow_blank=True)
    us = DataValidation(type="list", formula1='"1,2,3,4,5"', allow_blank=True)
    for v in (yn, ra, us):
        ws.add_data_validation(v)
    pair_fill = PatternFill("solid", fgColor="EAF1FB")
    key = {}
    for k, bi in enumerate(order, 1):
        b = blocks[bi]; rid = f"R2-{k:02d}"
        ws.append([rid, b["label"], f"{b['query'][0]} → {b['query'][1]}", "", "", "", "", "", "", ""])
        r = ws.max_row
        for c in ws[r]:
            c.fill = pair_fill; c.font = Font(bold=True)
        yn.add(f"H{r}"); us.add(f"I{r}")
        key[rid] = {kk: vv for kk, vv in b.items() if kk not in ("paths", "label")}
        for j, p in enumerate(b["paths"], 1):
            ws.append([f"{rid}-p{j}", "", "", " → ".join(p["nodes"]), rel_text(p["metapath"]), "", "", "", "", ""])
            r = ws.max_row; yn.add(f"F{r}"); ra.add(f"G{r}")
            key[f"{rid}-p{j}"] = dict(metapath=p["metapath"], nodes=p["nodes"], mu=p["mu"], share=p["share"],
                                      first_rater_id=f"{b['first_rater_id']}-p{j}" if b["first_rater_id"] else None)
    for col, wdt in zip("ABCDEFGHIJ", (10, 26, 40, 70, 22, 16, 10, 14, 11, 40)):
        ws.column_dimensions[col].width = wdt
    ws.freeze_panes = "A2"
    wb.save(f"{OUT}/expert_review2_sheet.xlsx")
    json.dump(key, open(f"{OUT}/expert_review2_key.json", "w"), indent=1)
    print("saved", len(blocks), "pairs")


if __name__ == "__main__":
    main()
