"""Rating sheet for the clinical assessment of DNFR's path explanations by the second author (clinical psychology).

For each disease-symptom task (Hetionet DpS, graded health knowledge graph) we draw, with a fixed seed, 25 held-out test
pairs of the seed-0 split that DNFR ranks in its top 10, and list the three paths with the largest share of the evidence.
The rater sees the disease, the symptom and each path with its nodes and relations, but not the rank, the score, the
memberships or the share, so that the judgement is not anchored on the model's confidence. For each path the rater
records (a) whether every relation on the path agrees with biomedical knowledge (yes / no / unsure) and (b) the kind of
rationale the path offers for the disease-symptom link: M = mechanistic (a shared cause, pathway or organ system that
would explain the symptom), A = associative only (comorbidity or similar diseases, no explanation of why), I =
irrelevant or misleading. For each pair the rater gives (c) whether the pair itself is clinically plausible and (d) how
useful the three paths would be to a clinician judging the prediction (1 = not useful, 5 = very useful).
Output: results/expert_review_sheet.xlsx (to fill in) and results/expert_review_key.json (model values per row)."""
import os, sys, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from config import OUT
from explain import setup, explain, NAME
from dnfr import score_pairs

TASKS = {"het_DpS": "Hetionet disease–symptom", "hkg_DpS": "Graded health knowledge graph"}
N_PAIRS, N_PATHS = 25, 3


def rel_text(mp):
    return mp.replace("^-1", "⁻¹")


def main():
    from openpyxl import Workbook
    from openpyxl.styles import Alignment, Font, PatternFill
    from openpyxl.worksheet.datavalidation import DataValidation
    wb = Workbook()
    ins = wb.active; ins.title = "Instructions"
    lines = [
        "Clinical assessment of DNFR path explanations / Đánh giá lâm sàng các đường giải thích của DNFR",
        "",
        "Each block is one disease–symptom pair that the model predicted correctly (held out from training), followed by the three "
        "paths the model relies on most. Model scores are hidden on purpose.",
        "Mỗi khối là một cặp bệnh–triệu chứng mà mô hình dự đoán đúng (cặp này bị giữ lại khi huấn luyện), kèm ba đường mô hình dựa "
        "vào nhiều nhất. Điểm số của mô hình được ẩn có chủ ý.",
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
        "Graded graph: DpS edges were learned from emergency-department records by Rotmensch et al. (2017).",
    ]
    for i, l in enumerate(lines, 1):
        ins.cell(i, 1, l).alignment = Alignment(wrap_text=True)
    ins.column_dimensions["A"].width = 140
    ins["A1"].font = Font(bold=True, size=13)
    key = {}
    hdr = ["row_id", "task", "pair", "path", "relations", "relations_correct", "rationale", "pair_plausible", "usefulness", "comment"]
    yn = DataValidation(type="list", formula1='"yes,no,unsure"', allow_blank=True)
    ra = DataValidation(type="list", formula1='"M,A,I"', allow_blank=True)
    us = DataValidation(type="list", formula1='"1,2,3,4,5"', allow_blank=True)
    ws = wb.create_sheet("Ratings")
    ws.append(hdr)
    for c in ws[1]:
        c.font = Font(bold=True)
    for v in (yn, ra, us):
        ws.add_data_validation(v)
    pair_fill = PatternFill("solid", fgColor="EAF1FB")
    for task, label in TASKS.items():
        ctx = setup(task, 0)
        z, z0 = score_pairs(ctx["model"], ctx["sd"])
        ranks = ctx["ranker"].ranks(z, z0, ctx["te"], ctx["known"])
        hit = np.where(ranks <= 10)[0]
        rng = np.random.default_rng(2026)
        pick = np.sort(rng.choice(hit, size=min(N_PAIRS, len(hit)), replace=False))
        ns, nt = ctx["kg"].names[ctx["spec"]["s"]], ctx["kg"].names[ctx["spec"]["t"]]
        for k, qi in enumerate(pick, 1):
            s, t = ctx["te"][qi]
            e = explain(ctx, int(s), int(t), top=N_PATHS)
            d_, s_ = NAME(str(ns[ctx["sources"][s]])), NAME(str(nt[t]))
            rid = f"{task}-{k:02d}"
            ws.append([rid, label, f"{d_} → {s_}", "", "", "", "", "", "", ""])
            r = ws.max_row
            for c in ws[r]:
                c.fill = pair_fill; c.font = Font(bold=True)
            yn.add(f"H{r}"); us.add(f"I{r}")
            key[rid] = dict(task=task, query=[d_, s_], rank=float(ranks[qi]), n_paths=e["n_paths"])
            for j, p in enumerate(e["paths"], 1):
                ws.append([f"{rid}-p{j}", "", "", " → ".join(p["nodes"]), rel_text(p["metapath"]), "", "", "", "", ""])
                r = ws.max_row; yn.add(f"F{r}"); ra.add(f"G{r}")
                key[f"{rid}-p{j}"] = dict(metapath=p["metapath"], nodes=p["nodes"], mu=p["mu"], p=p["p"], share=p["share"])
        print(task, len(pick), "pairs from", len(hit), "top-10 hits", flush=True)
        del ctx
    for col, wdt in zip("ABCDEFGHIJ", (14, 26, 40, 70, 22, 16, 10, 14, 11, 40)):
        ws.column_dimensions[col].width = wdt
    ws.freeze_panes = "A2"
    wb.save(f"{OUT}/expert_review_sheet.xlsx")
    json.dump(key, open(f"{OUT}/expert_review_key.json", "w"), indent=1)


if __name__ == "__main__":
    main()
