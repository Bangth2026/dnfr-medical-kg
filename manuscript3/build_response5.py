"""Response to the reviewer and the editor (fifth round, report on the fourth version). Every number is read from the same
statistics files as the manuscript, so the letter and the article cannot disagree. Parts that depend on the second rater's
sheet are written only when results/expert_review2.json exists; otherwise a PENDING marker is left and reported.
Usage: python3 manuscript3/build_response5.py [out_dir]"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from ms_common import *
from docx_helpers import new_document, para, heading, bullets
from ms_results import r3p, r3sum, R3T
import ms_text as MT
from build_response4 import c, r, ch, dci3, vt, ptxt, itv

SH, joinand, num, Num = MT.SH, MT.joinand, MT.num, MT.Num
CTD = "het_CtD"
PENDING = []


def pending(what):
    PENDING.append(what)
    return f"[PENDING: {what}]"


def rows_of(PS, seed):
    return {x["p"]: x for x in PS["rows"][PS["seeds"].index(seed)]}


def build(outdir):
    d = new_document()
    PS = R3.get("profile_summary") or {}; PSd = R3.get("profile_summary_dps") or {}
    XR, X2 = XREV, XREV2
    H, G = "het_DpS", "hkg_DpS"
    para(d, "**Response to the reviewer and the editor**", align="left", size=14)
    para(d, f"Manuscript: *{MT.TITLE}* (fifth version)", align="left")
    para(d, "We thank the reviewer for a fourth report and for the clear list of what remained, and the editor for the question on "
            "authorship. We ran every analysis the reviewer asked for and changed the text where the report found a tension or an "
            "error. Reviewer comments are summarised in italics. Section, table and figure numbers refer to the revised manuscript; "
            "S-numbers refer to the supplementary file. All numbers in this letter are generated from the released result files by "
            "the script that builds the manuscript.")

    # ------------------------------------------------------------------ editor
    heading(d, "Note to the editor: the second author's CRediT statement", 1)
    para(d, "The reviewer pointed out that the second author's CRediT entry has grown between versions. We confirm that the "
            "statement in the earlier versions was incomplete. Van An Phan Thi took part in defining the medical prediction tasks "
            "from the start of the study: the choice of disease–symptom prediction as a second clinical task, the disease-centred "
            "split for drug repurposing, and the question of what an explanation should let a clinician check. The earlier "
            "statement listed only her check of the explanation paths, and that omission was ours. The Conceptualization entry "
            "therefore describes work that predates the earlier versions. The clinical assessment of explanations (Investigation) "
            "and the drafting of the clinical interpretation (Sections 5.3–5.5) are new work for the fourth version, as the "
            "earlier letter said.")
    para(d, "The editor may also note that in the fourth version the only clinical assessment was performed by an author. "
            + (f"In this version a second rater, {X2['rater_desc']}, who is not an author and did not know the hypothesis, rated a "
               "blinded sheet that mixes correct and incorrect predictions (M6). The physician is thanked in the Acknowledgements "
               "and is not an author." if X2 else pending("second rater: description and role")))

    # ------------------------------------------------------------------ major
    heading(d, "Major comments", 1)

    # M1
    heading(d, "M1. The extended profile and the claim that the data exclude the nilpotent end", 2)
    c(d, "Report test and validation MRR of the warm-started fits at p = 0.1–0.5; extend the grid to p = 0.75 and 1; rewrite the "
         "sentence on the nilpotent end according to whether the warm-started fits are as accurate as the product.")
    iv = PS["intervals"]
    nil = [(s, rows_of(PS, s)) for s in PS["seeds"]]
    txt = ("The reviewer's reading was right, and the sentence was wrong. (1) Table S18 now gives, per split, the excess loss, "
           "∂L/∂p, validation and test MRR of every converged fit at p ≥ 0, and the right panel of Figure 2 now plots the merged "
           "grid including p > 0 (the previous figure drew only the first grid). (2) We extended the grid to p = 0.75 and p = 1, the "
           "Łukasiewicz t-norm itself, with warm starts from the converged fit at the product. At p = 1 the converged fits reached "
           "a test MRR of " + joinand([f3(rr[1]["test_MRR"]) for _, rr in nil if 1 in rr]) + " against "
           + joinand([f3(rr[0]["test_MRR"]) for _, rr in nil if 1 in rr]) + " for the product, and a validation MRR of "
           + joinand([f3(rr[1]["valid_MRR"]) for _, rr in nil if 1 in rr]) + " against "
           + joinand([f3(rr[0]["valid_MRR"]) for _, rr in nil if 1 in rr]) + ". The loss was now lowest at p = "
           + joinand([ptxt(x["p_argmin"]) for x in iv]) + "; "
           + (lambda ii, ee: (joinand([f"split {iv[k]['seed']}" for k in ii]) + (" has" if len(ii) == 1 else " have")
                               + " an interior minimum" if ii else "no split has an interior minimum")
              + ("" if not ee else ", and " + joinand([f"split {iv[k]['seed']}" for k in ee]) + " still has its minimum at an end of "
                 "the grid"))([k for k in range(len(iv)) if PS["interior"][k]], [k for k in range(len(iv)) if not PS["interior"][k]])
           + ". (3) The Łukasiewicz t-norm trained from the default initialisation lost "
           f"{sg(-pr('hkg_DpS', 'dnfr', 'luk')['delta_MRR'])[1:]} in MRR, while the warm-started fit at the same operator was close "
           "to the product. The failure is therefore one of initialisation (every path starts at strength zero and passes no "
           "gradient), not a property of the data. Section 4.4 now says so, and adds that in the nilpotent range the loss and "
           "held-out accuracy again pointed in different directions: the loss was lowest at p > 0 in "
           f"{num(sum(1 for x in iv if x['p_argmin'] > 0))} of the three splits, while test MRR at p > 0 was not higher than at the "
           "product.")
    r(d, txt)
    ch(d, "Section 4.4 (the paragraph on the nilpotent end, rewritten); Figure 2 (merged grid, p = 1 marked); Supplementary Tables "
          "S7, S8 and S18 (new).")

    # M2
    heading(d, "M2. Hetionet DpS: the minimum at the lower grid edge, and the validation-selection gain", 2)
    c(d, "Extend the DpS grid to −10 and −20; report the per-split values behind the +0.016 (5/10) gain of the validation-selected "
         "p; limit the claim that selecting p on validation helps to the graded graph, and attribute the DpS gain to optimisation "
         "variability.")
    ivd = PSd["intervals"]
    txt = ("(1) We extended the profile on Hetionet disease–symptom to p = −10 and −20 (warm starts in both directions). The loss "
           "was now lowest at p = " + joinand([ptxt(x["p_argmin"]) for x in ivd]) + " in the three splits; over the whole grid it "
           "varied by " + joinand([f"{x:.4f}" for x in PSd["loss_range"]]) + " nats, and test MRR by at most "
           f"{max(x['test_range'] for x in ivd):.3f}. "
           + ("The minimum moved below −5 in " + num(sum(1 for x in ivd if x["p_argmin"] < -5)) + " split"
              + ("s" if sum(1 for x in ivd if x["p_argmin"] < -5) != 1 else "") + ", towards the minimum t-norm, but the profile "
              "stays flat: the loss does not locate p on this graph either. " if any(x["p_argmin"] < -5 for x in ivd) else
              "The minimum did not move below −5, so the earlier boundary minima were not an artefact of the grid. "))
    fv = r3p(H, "fixp_val", "product"); fr = (R3.get(H) or {}).get("fixp_val_rows") or []
    if fv:
        chd = (R3.get(H) or {}).get("fixp_val_choice") or {}
        txt += ("(2) Supplementary Table S17 gives the per-split values. The differences were "
                + ", ".join(sg(x) for x in fv["per_split"]) + f". Validation chose the product itself on {num(chd.get('product', 0))} "
                f"splits, so {num(sum(1 for x in fv['per_split'] if abs(x) < 1e-9))} differences are exactly zero, and the 5/10 "
                "count reflects these ties rather than a split into gains and losses. The gain came from the splits on which the "
                "early-stopped product run ended low. On the three profiled splits the product fitted to convergence reached "
                + joinand([f3(rows_of(PSd, s)[0]["test_MRR"]) for s in PSd["seeds"]]) + " against "
                + joinand([f3(x["product"]) for x in fr if x["seed"] in PSd["seeds"]]) + " for the early-stopped run. We agree "
                "with the reviewer: on this graph the gain reflects the variability of early-stopped training, not a better value of "
                "p. (3) Section 4.3 now says this; the claim that validation selection helps is limited to the graded graph in "
                "Section 4.3, Section 4.4, the Discussion, the Conclusion and the Statement of significance, and the Conclusion of "
                "the Abstract now reads \"the composition operator is better treated as a hyperparameter than learned\".")
    r(d, txt)
    ch(d, "Sections 4.3 (new paragraph), 4.4, 5.1, 5.2, 6; Abstract (Conclusion); Statement of significance; Supplementary Tables "
          "S15–S17.")

    # M3
    heading(d, "M3. The Hamacher split: optimisation decides the result on CtD", 2)
    c(d, "Restart the Hamacher run on split 3 from two further seeds or initial memberships; state that the identifiability "
         "conclusion is shown for the two disease–symptom tasks, and that on CtD it rests on restarts in a landscape with at least "
         "one poor optimum.")
    H4 = R4.get("hamacher_ctd") or []; hm = min(H4, key=lambda h: h["hamacher"] - h["product"]) if H4 else None
    HR = [h for h in (R4.get("hamacher_restarts") or []) if hm and h["seed"] == hm["seed"]]
    txt = ("Training is deterministic given the initial values (full-batch gradients, constant initial memberships), so a "
           "different random seed alone would reproduce the same run; we therefore restarted from different initial memberships. ")
    if HR:
        txt += (f"With the degree discount initialised at b = c = " + joinand([f"{h['b_init']:g}" for h in HR])
                + f" instead of 0.3, the Hamacher run on split {hm['seed']} reached a test MRR of " + joinand([f3(h["mrr"]) for h in HR])
                + ("" if any(h.get("loss") is None for h in HR) else
                   " with final training losses of " + joinand([f"{h['loss']:.2f}" for h in HR]))
                + f", against {f3(hm['hamacher'])} (loss {hm['loss_hamacher']:.2f}) from the default start and {f3(hm['product'])} "
                f"(loss {hm['loss_product']:.2f}) for the product. "
                + ("None of the restarts reached the product's loss, so the poorer optimum belongs to the Hamacher product on this "
                   "split rather than to one initialisation. " if all(h["mrr"] < hm["product"] - 0.05 for h in HR) else
                   "The restarts reached the product's level, so the poorer optimum came from the default initialisation. "
                   if all(h["mrr"] >= hm["product"] - 0.05 for h in HR) else
                   "One restart reached the product's level and one did not. "))
    else:
        txt += pending("Hamacher restarts on split 3")
    txt += ("Section 4.2 reports the restarts, and Section 5.5 now states that the identifiability conclusion is shown for the two "
            "disease–symptom tasks, that on drug repurposing it rests on restarts, the unbounded parameterisation and the "
            "frozen-membership experiment, and that the loss on compound–treats–disease has at least one poor optimum into which a "
            "fixed operator can fall. We agree with the reviewer that a loss that is flat in p at convergence on one graph need not "
            "be benign for every optimiser state on another.")
    r(d, txt)
    ch(d, "Sections 4.2 and 5.5; Supplementary Table S13.")

    # M4
    heading(d, "M4. The target-relation ablation should also remove palliates", 2)
    c(d, "Report the share of the remaining evidence on CpD metapaths after the treats ablation, and add a variant without both "
         "treats and palliates metapaths.")
    EV = R4.get("evidence_ablation") or {}; NP = R4.get("notgtp"); NT = R4.get("notgt") or {}
    e1, e2 = EV.get("_notgt"), EV.get("_notgtp")
    txt = ""
    if e1 and e1.get("share_mean"):
        txt += (f"(i) After the treats ablation, metapaths through CpD carried {pct1(e1['share_mean']['cpd'])} of DNFR's remaining "
                f"evidence for its correct top-10 predictions on the seed-0 split (mean; median {pct1(e1['share_median']['cpd'])}; "
                f"{e1['n_hits']} pairs), and gene-only metapaths {pct1(e1['share_mean']['gene'])}. ")
    else:
        txt += pending("CpD share after the treats ablation")
    if NP:
        q2, mn2 = NP["paired"], NP["means"]; qa = q2.get("dnfr_notgtp__dnfr_notgt")
        TL = summ(CTD, "dnfr", "_tail")["MRR"]["mean"]
        ds2 = (mn2["dnfr"]["mean"] - mn2["dnfr_notgtp"]["mean"]) / max(mn2["dnfr"]["mean"] - TL, 1e-9)
        txt += ("(ii) Without the metapaths that contain treats or palliates"
                + (f" ({e2['n_dropped_metapaths']} of {e2['n_metapaths']})" if e2 else "")
                + f", over {qa['J'] if qa else mn2['dnfr_notgtp']['n']} random splits, DNFR reached {f3(mn2['dnfr_notgtp']['mean'])}, "
                f"against {f3(mn2['dnfr_notgt']['mean'])} without treats only"
                + (f" ({dci3(qa)}, {vt(qa)})" if qa else "")
                + (f"; DWPC-LR reached {f3(mn2['dwpc_lr_notgtp']['mean'])} and the fixed product {f3(mn2['product_notgtp']['mean'])}"
                   if mn2.get("dwpc_lr_notgtp") else "")
                + f". That is {pct(min(max(ds2, 0), 1))} of the way from the random-split accuracy ({f3(mn2['dnfr']['mean'])}) to "
                f"the disease-centred accuracy ({f3(TL)}). ")
        if e2 and e2.get("share_mean"):
            txt += (f"With both relations removed, gene-only metapaths carried {pct1(e2['share_mean']['gene'])} of DNFR's evidence "
                    f"(median {pct1(e2['share_median']['gene'])}). ")
        txt += "Section 4.5 and Table 5C report both results, and the text draws the conclusion that the numbers support."
    else:
        txt += pending("treats-and-palliates ablation")
    r(d, txt)
    ch(d, "Section 4.5 (new paragraph); Table 5 (panel C).")

    # M5
    heading(d, "M5. Sign error for R1 on the fixed product", 2)
    c(d, "Table 4 reports +0.019 for R1 added to the fixed product on DpS, while Section 4.3 says the penalty cost 0.019.")
    po = {t: r3p(t, "product_overlap", "product") for t in R3T}
    r(d, "The table was right and the text was wrong, as it already was in the previous version. The sentence was generated by "
         "code that printed the absolute value of the difference, and so dropped the sign. Section 4.3 now gives the signed values: "
         "added to the fixed product, R1 changed MRR by "
         + joinand([f"{dci3(po[t])} on {SH(t)} ({po[t]['wins']}/{po[t]['J']} splits, {vt(po[t])})" for t in R3T if po[t]])
         + ". The sentence that R1 \"lowered it when the memberships could not compensate\" is replaced by the statement that with "
         "the fixed product its estimate was negative on the graded graph and positive on Hetionet disease–symptom, and that "
         "neither difference was supported under our rule. We checked the five other places where the build script prints a "
         "magnitude in the same way; in each the quantity has the sign the text assumes.")
    ch(d, "Section 4.3.")

    # M6
    heading(d, "M6. The clinical assessment: scope, expertise and independence", 2)
    c(d, "State the limitation of a rater trained in clinical psychology, or add a physician rater; rate a subsample with a second "
         "rater and an equal number of incorrect predictions mixed in blind, and report κ and discrimination; present the "
         "low-membership observation as an observation.")
    unsure = lambda t: pct(1 - XR[t]["relations_correct"]["share"] - XR[t]["relations_wrong"]["share"])
    txt = ("(1) Section 4.7 and Section 5.5 now state that the first rater is trained in clinical psychology, not in medicine, that "
           "judging disease–anatomy–disease and disease–disease–symptom paths calls for medical training, and that the share of "
           f"paths she could not judge ({unsure(H)} and {unsure(G)}) reflects this. (2) ")
    if X2:
        a2, g1, g0 = X2["agreement"], X2["correct"], X2["incorrect"]
        kp = lambda k: "—" if a2[k]["kappa"] is None else f"{a2[k]['kappa']:.2f}"
        txt += (f"We added a second rater, {X2['rater_desc']}, who is not an author, did not know the hypothesis and did not use AI "
                f"tools. The sheet held {g1['n_pairs']} of the first rater's correct pairs, drawn with a fixed seed and shown with "
                f"the same paths, and {g0['n_pairs']} incorrect top-10 predictions, shuffled and renumbered so that neither order nor "
                f"identifier revealed which pairs were correct. Agreement between the raters on the shared items was "
                f"κ = {kp('relations_correct')} for whether the relations on a path are correct ({a2['relations_correct']['n']} "
                f"paths), κ = {kp('rationale')} for the kind of rationale and κ = {kp('pair_plausible')} for plausibility "
                f"({a2['pair_plausible']['n']} pairs). The second rater judged {pct(g1['plausible'])} of the correct and "
                f"{pct(g0['plausible'])} of the incorrect predictions plausible, with a mean usefulness of "
                f"{g1['usefulness_mean']:.1f} and {g0['usefulness_mean']:.1f} of 5"
                + ("" if X2.get("auc_usefulness") is None else f" (AUC {X2['auc_usefulness']:.2f})")
                + ". Sections 3.6 and 4.7 describe the design and the results; the sheet, the key and both sets of ratings are "
                "released. ")
    else:
        txt += pending("second rater: agreement and discrimination")
    txt += ("(3) The sentence on the low-membership second edge now reads that with one rater and so few paths this is an "
            "observation, not evidence that low memberships flag wrong edges.")
    r(d, txt)
    ch(d, "Sections 3.6, 4.7, 5.5; Acknowledgements; released rating sheets.")

    # ------------------------------------------------------------------ minor
    heading(d, "Minor comments", 1)
    nbd = r3p(H, "nbfnet", "dnfr")
    MINOR = [
        ("Highlights.", "The fifth highlight now reads \"Drug-repurposing accuracy on Hetionet rested on paths through known "
                        "indications.\""),
        ("NBFNet in the Discussion.", "Section 5.1 now says that DNFR did not differ from the compact NBFNet under our rule, "
         + (f"though NBFNet was ahead on {nbd['wins']} of {nbd['J']} splits on Hetionet disease–symptom and had a higher mean MRR "
            f"there than DWPC-LR ({f3(r3sum(H, 'nbfnet')['mean'])} against {f3(summ(H, 'dwpc_lr')['MRR']['mean'])}), " if nbd else "")
         + "and that NBFNet was tuned on the graded graph only, so tuning it on Hetionet might widen that gap."),
        ("Profile intervals as evidence.", "The caveat that the χ²/(2N) threshold treats queries as independent now stands in the "
         "same sentence as the description of the narrow intervals (Section 4.4)."),
        ("Curvature at grid ends.", "Curvature is now reported only at interior minima. Where the minimum lies at an end of the "
         "grid, Section 4.4 says so and Table S8 gives the slope ∂L/∂p there instead."),
        ("Abstract, Results.", "The sentence on the validation-selected t-norm keeps to the graded graph (M2)."),
        ("Computational cost.", "Section 4.8 now gives both the time per epoch on splits 0–2 and on the shared-machine splits, and "
         "the run time on splits 0–2 alone as the better estimate of the cost of the method."),
        ("Decision-rule history.", "The sentence on the earlier rule moved from Section 3.6 to the caption of Supplementary Table "
         "S12, which is now titled as a sensitivity analysis of the verdicts to the decision rule."),
        ("Table 6.", "The caption now says that the number in parentheses is the number of pairs in the top 10 of both models (and, "
         "in the first row, the number of top-10 hits)."),
        ("Mechanistic share.", "Table 6 now gives the median next to the mean for the evidence shares by metapath class and for "
         "the share on metapaths through the treats relation."),
        ("Supplementary material.", "The supplementary file (Tables S1–S18, Figures S1–S3) accompanies the resubmission."),
    ]
    for i, (a, b) in enumerate(MINOR, 1):
        para(d, f"**{i}. {a}** {b}")

    # ------------------------------------------------------------------ questions
    heading(d, "Questions", 1)
    Q = [("Test MRR of the warm-started converged fits at p = 0.1–0.5 on the graded graph?",
          "Per split, at p = 0.1, 0.25 and 0.5: "
          + "; ".join(f"split {s}: " + ", ".join(f3(rr[p]["test_MRR"]) for p in (0.1, 0.25, 0.5) if p in rr)
                      + f" (product {f3(rr[0]['test_MRR'])})" for s, rr in nil) + ". All values, with p = 0.75 and 1, are in Table S18 (M1)."),
         ("Does the DpS profile minimum move below −5 when the grid is extended?",
          "The minimum lay at p = " + joinand([ptxt(x["p_argmin"]) for x in ivd]) + " on the grid extended to −20 (M2)."),
         ("Per-split values behind the +0.016 (5/10) gain on DpS?",
          ("" if not fv else ", ".join(sg(x) for x in fv["per_split"])) + "; four zeros are splits on which validation chose the "
          "product (Table S17, M2)."),
         ("How much of the evidence remaining after the treats ablation runs through palliates?",
          (f"{pct1(e1['share_mean']['cpd'])} on average (median {pct1(e1['share_median']['cpd'])}) (M4)." if e1 and e1.get("share_mean")
           else pending("Q4"))),
         ("Is the R1 sign on the fixed product in Table 4 or in Section 4.3 correct?",
          "Table 4 was correct; the text dropped the sign and is corrected (M5).")]
    for i, (a, b) in enumerate(Q, 1):
        para(d, f"**Q{i}. {a}** {b}")

    heading(d, "Other changes", 1)
    bullets(d, ["The author list gives the corresponding author's name as Hai Bang Truong.",
                "Supplementary Tables S17 (validation-selected p on Hetionet disease–symptom, per split) and S18 (converged fits at "
                "p ≥ 0 on the graded graph) are new.",
                f"Code and data of this version: release {MT.RELEASE} of https://github.com/Bangth2026/dnfr-medical-kg, archived "
                "at Zenodo (" + MT.ZENODO_VERSION + "all versions: https://doi.org/10.5281/zenodo.23879352)."])
    para(d, "We hope that the revised manuscript addresses the reviewer's and the editor's remaining concerns.", space_after=6)
    para(d, "Hai Bang Truong, on behalf of the authors", align="left")
    out = os.path.join(outdir, "Response_to_reviewer_rev5.docx"); d.save(out)
    print("saved", out, "; pending:", PENDING or "none")


if __name__ == "__main__":
    build(sys.argv[1] if len(sys.argv) > 1 else os.path.join(ROOT, "manuscript3", "out"))
