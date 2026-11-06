"""Response to the reviewer and the editor (fourth round, report on the third revision). Every number is read from the
same statistics files as the manuscript, so the letter and the article cannot disagree.
Usage: python3 manuscript3/build_response4.py [out_dir]"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from ms_common import *
from docx_helpers import new_document, para, heading, bullets
from ms_results import r3p, r3sum, mean_any, R3T
import ms_text as MT

SH, joinand, num, Num = MT.SH, MT.joinand, MT.num, MT.Num
CTD = "het_CtD"


def c(d, text):
    para(d, f"*Comment.* {text}", size=10.5)


def r(d, text):
    para(d, f"**Response.** {text}")


def ch(d, text):
    para(d, f"*Changes:* {text}", size=10.5)


def dci3(p):
    return "—" if not p else f"{sg(p['delta'])} [{sg(p['bs95'][0])}, {sg(p['bs95'][1])}]"


def vt(p):
    return {"different": "different under our rule", "equivalent": "equivalent within ±0.01"}.get((p or {}).get("verdict"), "inconclusive")


def nrun(t, v, ref="dnfr"):
    """Number of splits on which variant v and its reference were both run (Tables 2–4)."""
    p = r3p(t, v, ref)
    if p:
        return p["J"]
    q = pr(t, ref, v)
    return q["seeds"] if q else 0


def ptxt(x):
    return f"{x:g}".replace("-", "−")


def itv(x):
    return f"[{ptxt(x['p_lo'])}, {ptxt(x['p_hi'])}]" if x["p_lo"] != x["p_hi"] else f"{{{ptxt(x['p_lo'])}}}"


def third_split_note(PS):
    """What the extended grid shows for the third split of the graded-graph profile (computed from the converged rows)."""
    rows = sorted(PS["rows"][2], key=lambda x: x["p"])
    pos = [x for x in rows if x["p"] > 0 and x.get("dLdp") is not None]
    flip = [(a, b) for a, b in zip(pos, pos[1:]) if a["dLdp"] < 0 < b["dLdp"]]
    best_t = max(rows, key=lambda x: x["test_MRR"]); at = [x for x in rows if x["p"] == PS["intervals"][2]["p_argmin"]][0]
    txt = (f"The gradient ∂L/∂p changed sign between p = {ptxt(flip[0][0]['p'])} and {ptxt(flip[0][1]['p'])}, so the loss has "
           "a minimum close to the end of the grid, inside the nilpotent range. " if flip else
           "The loss was still decreasing at the end of the grid, so this split has no interior minimum on it. ")
    return txt + (f"Test MRR there was {f3(at['test_MRR'])}, against {f3(best_t['test_MRR'])} at p = {ptxt(best_t['p'])}, which "
                  "is another instance of the training loss and held-out accuracy pointing in different directions. ")


def build(outdir):
    d = new_document()
    PS = R3.get("profile_summary") or {}; PSd = R3.get("profile_summary_dps") or {}
    E3 = R3.get("explain3", {}); XD = R4.get("explain_dwpc", {}); NT = R4.get("notgt") or {}
    rc = MT.refine_counts()
    para(d, "**Response to the reviewer and the editor**", align="left", size=14)
    para(d, f"Manuscript: *{MT.TITLE}* (fourth version; the previous title ended \"…, with four refinements (DNFR 2.0)\")",
         align="left")
    para(d, "We thank the reviewer for a third careful report and the editor for the note on authorship. We ran the analyses the "
            "reviewer asked for, on the same shared two-core CPU as before, and say below where a request was met only in part. "
            "Reviewer comments are summarised in italics. Section, table and figure numbers refer to the revised manuscript; "
            "S-numbers refer to the supplementary file. All numbers in this letter are generated from the released result files by "
            "the script that builds the manuscript.")

    # ------------------------------------------------------------------ editor
    heading(d, "Note to the editor: the second author", 1)
    XR = XREV
    para(d, "The editor asked whether the second author's contribution, previously described as a check that the nodes and edges "
            "of Table 8 of the previous version exist in the source graphs, meets the journal's criteria for authorship. We agree that the earlier CRediT "
            "statement undersold her role and did not by itself justify authorship. Van An Phan Thi is a doctoral candidate in "
            "clinical psychology. Her contribution is now stated in full, and part of it is new work for this version:")
    items = ["Conceptualization: the clinical framing of the prediction tasks, the disease-centred split, and the question of what "
             "an explanation should let a clinician check;",
             "Validation: a check that the relations on the explanation paths agree with biomedical knowledge, not only that they "
             "exist in the source graphs;"]
    if XR:
        H, G = "het_DpS", "hkg_DpS"
        items.append(f"Investigation: a new clinical assessment of DNFR's explanations (Sections 3.6 and 4.7). She rated "
                     f"{XR[H]['n_paths'] + XR[G]['n_paths']} paths of {XR[H]['n_pairs'] + XR[G]['n_pairs']} correctly ranked "
                     "disease–symptom pairs drawn with a fixed seed, without seeing ranks, scores, memberships or shares, and "
                     "without AI tools. For each path she recorded whether its relations agree with biomedical knowledge and "
                     "whether it offers a mechanistic, an associative or no rationale; for each pair, whether it is clinically "
                     "plausible and how useful the paths would be to a clinician (1–5). The sheet, the key and her ratings are released.")
    items.append("Writing – original draft of the clinical interpretation in Sections 5.3–5.5, and review and editing of the whole "
                 "manuscript.")
    bullets(d, items)
    para(d, "Drug repurposing was not rated, because it lies outside her expertise, and the article says so. It also states the "
            "limits of a single rater who is a co-author and knew the aims of the study (Section 5.5). We believe that "
            "these contributions meet the ICMJE and Elsevier criteria (substantial contribution to conception and to the "
            "acquisition and interpretation of data, drafting and revision, approval, accountability), and we leave the judgement "
            "to the editor.")

    # ------------------------------------------------------------------ title and framing
    heading(d, "Title and framing (M5, minor 13)", 1)
    para(d, f"Following M5, \"DNFR 2.0\" was removed from the title, which is now *{MT.TITLE}*. The sentence \"The design has four "
            "weaknesses that a careful reader will see at once\" was removed; the Introduction now describes the four simplifying "
            "assumptions and their standard alternatives as motivation. The refinements are one subsection (4.3) with one table "
            "(Table 4). The former Figure 3, which duplicated Table 4, and the per-refinement details moved to Supplementary "
            "Section S4 and Figure S3.")

    # ------------------------------------------------------------------ major
    heading(d, "Major comments", 1)
    heading(d, "M1. Mixed numbers of splits on compound–treats–disease, and the Hamacher split", 2)
    c(d, "Several CtD cells use 3–5 splits; the Hamacher rows suggest one failed split; run every CtD variant on 10 splits; "
         "correct the sentence in §5.5.")
    H4 = R4.get("hamacher_ctd") or []
    hm = min(H4, key=lambda h: h["hamacher"] - h["product"]) if H4 else None
    low = [h for h in H4 if h["hamacher"] - h["product"] < -0.05]
    txt = ""
    if hm:
        txt += (f"(i) Supplementary Table S13 gives the per-split values for the fixed Hamacher product, now on {num(len(H4))} "
                f"splits. One split stands out, as the reviewer inferred: on split {hm['seed']} the Hamacher run reached "
                f"{f3(hm['hamacher'])} against {f3(hm['product'])} for the product")
        if hm.get("loss_hamacher") is not None:
            txt += (f". Its final training loss was {hm['loss_hamacher']:.2f} against {hm['loss_product']:.2f}, and the run "
                    "finished without numerical errors, so it did not fail; it ended in a poorer optimum")
        hi = [h for h in H4 if h not in low]
        dh = lambda S, a, b: sum(h[a] - h[b] for h in S) / len(S)
        txt += (f". {Num(len(low))} of the {num(len(H4))} splits had a gap larger than 0.05. We kept the run, because excluding "
                "it would select results after seeing them. Section 4.2 now reports how much it weighs: over all splits the "
                f"Hamacher product was {abs(dh(H4, 'hamacher', 'product')):.3f} below the product, and without "
                + ("that split" if len(low) == 1 else "those splits")
                + f" {abs(dh(hi, 'hamacher', 'product')):.3f} below. " if low and hi else
                f". {Num(len(low))} of the {num(len(H4))} splits had a gap larger than 0.05. ")
    vs = [("hamacher", "Hamacher"), ("dnfr_fixdeg", "fixed degree terms"), ("dnfr_shared", "shared p"),
          ("dnfr_frozen", "frozen memberships with learned p"), ("product_frozen", "frozen memberships with the product"),
          ("dnfr_anneal", "R4"), ("fixp_val", "validation-selected p"), ("dnfr2", "DNFR 2.0")]
    cnt = [(lab, nrun(CTD, v)) for v, lab in vs]
    txt += ("(ii) We re-ran the compound–treats–disease variants on the remaining splits. They now have "
            + joinand([f"{n} ({lab})" for lab, n in cnt if n]) + " splits. ")
    ab = r3sum(CTD, "anyburl")
    ov = nrun(CTD, "dnfr_overlap")
    txt += ("The add-one refinements R1–R3 and their leave-one-out versions inside DNFR 2.0 kept "
            f"{ov} splits on this task. Each run takes 20 minutes or more with one thread on our hardware, and the largest point "
            "estimate of any of the three on any task was "
            + sg(max(r3p(t, v, "dnfr")["delta"] for t in R3T for v in ("dnfr_overlap", "dnfr_gumbel", "dnfr_ic")
                     if r3p(t, v, "dnfr")))
            + ", so we gave priority to R4 and to the variants on which the Highlights and the "
            "Conclusion rest. "
            + (f"AnyBURL now has {ab['n']} splits. " if ab else "")
            + "Table 2 still marks every cell whose number of splits differs from that of DNFR. (iii) The sentence in Section 5.5 "
            f"now reads: \"The comparisons of Tables 2–4 used {num(nseeds(CTD, 'dnfr'))} splits on compound–treats–disease"
            + MT.ctd_split_note() + "…\". It now agrees with the tables.")
    r(d, txt)
    ch(d, "Section 4.2 (new paragraph on the Hamacher split), Section 5.5; Tables 2–4; Supplementary Tables S13 and S14.")

    heading(d, "M2. Is DNFR's distinctive contribution the resolution of its explanations?", 2)
    c(d, "Add path- and fact-level attributions for DWPC-LR and its fact-level counterfactual, or soften the claim; state plainly "
         "if the recommendation is 'use DWPC-LR'.")
    if XD and all(t in XD for t in R3T):
        w = lambda t, k: XD[t]["dwpc"][k]
        r(d, "We did both. Section 3.6 now attributes DWPC-LR to paths and facts: the metapath terms add up exactly, each term is "
             "split over the paths of its metapath in proportion to their degree-weighted weights (exact within a metapath, "
             "proportional across the arcsinh), and each path's share is split equally among its facts. The same fact-level "
             "counterfactual recomputes the DWPC features without the removed fact and re-scores every candidate of the head "
             "(Table 6, new rows; Section 4.7). Removing DWPC-LR's leading fact left "
             + joinand([f"{pct(w(t, 'stay_top10_cf_top'))} of its correct top-10 predictions on {SH(t)}" for t in R3T])
             + " in the top 10, against "
             + joinand([pct(E3[t]["stay_top10_edge_top_sh"]) for t in R3T if t in E3]) + " for DNFR. "
             "The two models named the same leading fact for "
             + joinand([f"{pct(w(t, 'same_lead_as_dnfr'))}" for t in R3T]) + " of the pairs that both ranked in the top 10. "
             "The fact-level counterfactuals were therefore similarly faithful, and we no longer claim that DNFR's explanations "
             "have a higher resolution or are more faithful. What differs is their form. DNFR's attribution is exact and "
             "non-negative at the level of paths and facts. DWPC-LR's is exact only at the level of metapaths, and it mixes signs: "
             f"on compound–treats–disease {pct(w(CTD, 'neg_share_median'))} of its positive attribution was offset by metapaths "
             "with negative coefficients (median). Section 5.1 now states the recommendation the reviewer anticipated: for a "
             "developer who needs only accuracy on graphs like these, DWPC-LR is the better choice. It is simpler and faster, "
             "and at least as accurate. A noisy-OR path model is worth its cost where an exactly additive, non-negative attribution "
             "to facts is required. Whether that form helps clinicians more is a question for a user study, and the paper says so.")
    ch(d, "Sections 3.6, 4.7, 5.1, 5.4 (item 4), Abstract (Conclusion), Contribution 4; Table 6.")

    heading(d, "M3. Identifiability: the grid boundary, what the intervals show, and a single task", 2)
    c(d, "Extend the profile to p > 0 with warm starts; describe within- vs between-split evidence; report curvature for all "
         "splits; run a converged profile on Hetionet DpS, or at least name the graded graph in the Abstract and Conclusion.")
    txt = ""
    if PS:
        iv = PS["intervals"]
        txt += ("(1) The grid now extends to p = 0.1, 0.25 and 0.5, fitted with warm starts from the converged fit at the product. "
                "On the graded graph the minimiser moved to p = " + joinand([ptxt(x["p_argmin"]) for x in iv])
                + " in the three splits, and the intervals became " + joinand([itv(x) for x in iv])
                + ". " + (f"The third split, whose earlier interval was {{0}}, now has an interior minimum at p = {ptxt(iv[2]['p_argmin'])}. "
                   if PS["interior"][2] else
                   f"The third split, whose earlier interval was {{0}}, has its minimum at the new upper end of the grid, p = "
                   f"{ptxt(iv[2]['p_argmin'])}. " + third_split_note(PS))
                + "(2) Curvature is reported for all three splits ("
                + joinand([f"{cv:.3f}" + ("" if it else " one-sided") for cv, it in zip(PS["curvature"], PS["interior"])])
                + "). Section 4.4 now describes the result in the terms the reviewer proposed. Within a split the profile was flat "
                "over most of the grid in one split and narrow in the others. The non-identifiability is otherwise the instability "
                "of the minimiser between random splits, and the χ²/(2N) threshold, which treats dependent queries as independent, "
                "makes the narrow intervals if anything too narrow. ")
    if PSd:
        ivd = PSd["intervals"]
        txt += ("(3) We ran the converged profile on Hetionet disease–symptom (17 metapaths, 80 parameters, "
                f"{num(len(PSd['seeds']))} splits). Over p from −5 to 0 the loss varied by "
                + joinand([f"{x:.4f}" for x in PSd["loss_range_nonpos"]]) + " nats, the minimiser was at p = "
                + joinand([ptxt(x["p_argmin"]) for x in ivd]) + " and the intervals were " + joinand([itv(x) for x in ivd])
                + " (Supplementary Tables S15–S16). ")
        edge = [x for x in ivd if x["p_argmin"] <= min(PSd["grid"][0])]
        if edge:
            txt += (f"In {num(len(edge))} of the {num(len(ivd))} splits the minimum lay at the lower end of the grid, so the "
                    "minimiser may lie further towards the minimum t-norm, and the narrow intervals there reflect the small "
                    "threshold rather than a well-located optimum. ")
        txt += (f"Test MRR varied by at most {max(x['test_range'] for x in ivd):.3f} across the grid in any split. ")
        txt += ("The finding therefore holds on a multi-relational graph as well, and the Abstract, Highlights and Conclusion "
                "name both graphs."
                if MT.profile_flat("profile_summary_dps") is not None and MT.profile_flat("profile_summary_dps") < 0.01 else
                "The loss was less flat there, so the Abstract, Highlights and Conclusion state the finding for the graded graph "
                "only.")
    else:
        txt += ("(3) The converged profile on Hetionet disease–symptom could not be completed for this version; the Abstract, "
                "Highlights and Conclusion now state the finding for the graded graph only.")
    r(d, txt)
    ch(d, "Sections 3.4 (iv), 4.4, 5.5, Abstract, Highlights, Conclusion; Figure 2; Supplementary Tables S7–S8"
         + (", S15–S16." if PSd else "."))

    heading(d, "M4. 0% mechanistic: a finding about the task", 2)
    c(d, "Report the share of evidence by metapath category; ablate all metapaths containing the target relation and report "
         "DNFR, the product and DWPC-LR.")
    XC = (XD.get(CTD) or {}).get("dnfr")
    q = NT.get("paired") or {}; mn = NT.get("means") or {}
    qd, qw = q.get("dnfr_notgt__dnfr"), q.get("dwpc_lr_notgt__dwpc_lr")
    txt = "We agree with the reviewer's reading, and both analyses confirm it. "
    if XC:
        txt += (f"(i) Across all paths of the correct top-10 predictions on compound–treats–disease, {XC['n_target_metapaths']} "
                f"of the {XC['n_metapaths']} metapaths contain the treats relation and carried {pct(XC['share_target_mean'])} "
                f"of DNFR's evidence on average; the mechanistic metapaths carried {100 * XC['share_mean']['gene']:.1f}% "
                f"(median {100 * XC['share_median']['gene']:.2f}%). A mechanistic signal is present, but it is small. ")
    if qd:
        txt += (f"(ii) With these metapaths removed ({qd['J']} random splits), DNFR fell from {f3(mn['dnfr']['mean'])} to "
                f"{f3(mn['dnfr_notgt']['mean'])} ({dci3(qd)}, {vt(qd)})")
        if qw:
            txt += f", DWPC-LR from {f3(mn['dwpc_lr']['mean'])} to {f3(mn['dwpc_lr_notgt']['mean'])} ({vt(qw)})"
        if mn.get("product_notgt"):
            txt += f" and the fixed product from {f3(mn['product']['mean'])} to {f3(mn['product_notgt']['mean'])}"
        txt += (". That is a large part of the way to the disease-centred level. Section 4.5 now says plainly that on curated "
                "graphs the random-split accuracy of all path methods on drug repurposing reflects guilt by association. "
                "Section 5.3 adds this as a feature of medical graphs, and Section 5.4 turns it into a recommendation (item 5). "
                "The Highlights and the Abstract report it.")
    r(d, txt)
    ch(d, "Sections 3.6, 4.5 (new paragraph and Table 5B), 4.7, 5.3, 5.4, 6; Highlights; Abstract; Table 6.")

    heading(d, "M5. Refinements: power for R4, a check of R3, the reading of β in R1, and framing", 2)
    c(d, "Run R4 on 10 splits on DpS and CtD; check R3 for a rescaling problem; temper the R1 interpretation; remove DNFR 2.0 "
         "from the title and compact the section.")
    an = {t: r3p(t, "dnfr_anneal", "dnfr") for t in R3T}
    isd = R3.get("hkg_DpS", {}).get("paired", {}); sd1, sd2 = isd.get("dnfr_ic_sd__dnfr"), isd.get("dnfr_ic_sd__dnfr_ic")
    ic = r3p("hkg_DpS", "dnfr_ic", "dnfr")
    txt = ("*R4.* Added to DNFR, membership annealing changed MRR by "
           + joinand([f"{dci3(an[t])} on {SH(t)} ({an[t]['wins']}/{an[t]['J']} splits, {an[t]['verdict']})" for t in R3T if an[t]])
           + ". We agree that an untested positive signal is not a negative result. The text no longer says that R4 addresses a "
           "problem the models barely have. It reports the estimates and says that we neither recommend R4 nor rule it out. ")
    if sd1 and ic:
        txt += ("*R3.* The reviewer was right to suspect the rescaling. The feature log(1 + n(v)) was matched to the degree "
                "feature in mean only, and on the graded graph it had about half its spread, so the same coefficients discounted "
                f"hubs about half as much. Matched in standard deviation as well, R3 changed MRR by {dci3(sd1)} relative to DNFR, "
                f"against {sg(ic['delta'])} for the mean-matched version, a recovery of {sg(sd2['delta'])}. Section 4.3 now "
                "attributes the earlier loss to the scale of the feature. It is no longer reported as a property of information "
                "content. ")
    txt += ("*R1.* We agree that a loss that hardly constrains p does not support reading much into the learned β either. Section "
            "4.3 now says that discounting paths through a shared node did not improve held-out accuracy, and that whether such "
            "paths add evidence or repeat it cannot be decided from the value of β. *Framing.* See \"Title and framing\" above.")
    r(d, txt)
    ch(d, "Title; Sections 1, 3.3, 4.3, 5.1; Table 4; Supplementary Section S4, Figure S3.")

    heading(d, "M6. The decision rule changed between versions", 2)
    c(d, "Say that the rule was revised and why; give verdicts under both rules for the comparisons behind the Highlights and "
         "Conclusion; use one Holm family definition.")
    r(d, "(i) Section 3.6 now states that an earlier version of the analysis called a difference supported when the unadjusted "
         "95% corrected-t interval excluded zero, and that we replaced this rule because, with three splits, it accepted "
         "differences whose bootstrap interval contained zero and did not adjust for multiplicity. The phrase \"fixed before the "
         "analyses added in this revision\" was removed. (ii) Supplementary Table S12 gives the verdicts under both rules for "
         "every comparison on which the Highlights and the Conclusion rest. (iii) Holm families are now defined once, by research "
         "question, in Section 3.6, and the captions of Tables 3–5 refer to that definition. The families are the baselines "
         "against DNFR, the learned-operator baselines, the components of DNFR (all panels of Table 3), the refinements together "
         "with the validation-selected t-norm (Table 4), and the ablation of metapaths through the target relation.")
    ch(d, "Section 3.6; captions of Tables 3–5; Supplementary Table S12.")

    heading(d, "M7. NBFNet: 'matched' is not supported", 2)
    c(d, "Rephrase as 'did not differ detectably'; run DpS on 10 splits; add a 2 × 2 tuning grid on the graded graph.")
    nb = {t: r3p(t, "nbfnet", "dnfr") for t in ("hkg_DpS", "het_DpS")}
    nbs = {t: r3sum(t, "nbfnet") for t in ("hkg_DpS", "het_DpS")}
    tu = r3p("hkg_DpS", "nbfnet_tuned", "dnfr"); tuw = r3p("hkg_DpS", "nbfnet_tuned", "dwpc_lr")
    nw = {t: r3p(t, "nbfnet", "dwpc_lr") for t in ("hkg_DpS", "het_DpS")}
    ahead = [t for t in nb if nb[t] and nb[t]["delta"] > 0 and nb[t]["wins"] >= 0.8 * nb[t]["J"] and nb[t].get("verdict") != "different"]
    txt = ("(i) The Highlights no longer mention NBFNet. Section 4.1 and the Discussion say that NBFNet did not differ "
           "detectably from DNFR. (ii) NBFNet now has "
           + joinand([f"{nbs[t]['n']} splits on {SH(t)} (minus DNFR {dci3(nb[t])}, {vt(nb[t])}; minus DWPC-LR "
                      f"{sg(nw[t]['delta'])}, {vt(nw[t])})" for t in nb if nb[t]]) + ". ")
    if ahead:
        txt += ("With ten splits, NBFNet was ahead of DNFR on " + joinand([f"{nb[t]['wins']} of {nb[t]['J']} splits on {SH(t)}"
                                                                          for t in ahead])
                + ", but the Holm-corrected test did not reach significance; Section 4.1 reports this rather than calling the two "
                "equivalent. ")
    if tu:
        txt += ("(iii) On the graded graph we ran the 2 × 2 grid of learning rate (5 × 10⁻³, 10⁻²) and dimension (32, 64) and "
                f"kept the configuration with the best validation MRR on each split. Tuned NBFNet minus DNFR was {dci3(tu)} "
                f"({vt(tu)}), and tuned NBFNet minus DWPC-LR was {sg(tuw['delta'])} ({vt(tuw)}). ")
    txt += ("Each Hetionet disease–symptom run takes about 40 minutes with one thread, so we did not tune NBFNet on that graph. "
            "Section 5.5 says so.")
    r(d, txt)
    ch(d, "Highlights; Sections 3.4, 4.1, 5.1, 5.5; Table 2.")

    # ------------------------------------------------------------------ minor
    heading(d, "Minor comments", 1)
    spt = {k: summ("het_DpS", k, "_spt0.25") for k in ("dnfr", "softimpute", "dwpc_lr")}
    MINOR = [
        ("References to the revision process.", "Removed from Contribution 1, Section 3.6, the caption of Table 2, Section 5.5 "
         "(the bullet is now \"PrimeKG covered only partly\") and the supplementary captions. The code is cited by release tag "
         f"({MT.RELEASE}), not by branch name."),
        ("Abstract.", "The profile result names the graph (M3), and hub discounting \"had the same sign on every split\"."),
        ("Highlights.", "\"Hub discounting had the same sign on every split of every task tested.\""),
        ("Ethics.", "The NHANES clause was removed; the statement lists only Hetionet, PrimeKG and the published graded graph."),
        ("Table 6 (formerly 7), two criteria for the leading fact.", "The summed path share and the Shapley value picked the same "
         "leading fact for " + joinand([pct(XD[t]["dnfr"]["same_shapley_summed"]) for t in R3T if t in XD])
         + " of the pairs, which Section 4.7 now reports. Table 6 gives this as a new row, and the counterfactual is reported for the Shapley criterion only."
         if XD and all(t in XD for t in R3T) else "[pending]"),
        ("Equal division in the Shapley attribution.", "Section 3.6 now says that equal division credits a weak and a strong "
         "edge of the same path alike. The Shapley value measures how much the score depends on the presence of a fact. The "
         "strength the model gives the fact is its membership μ, which the explanations show next to it. For a clinician the two "
         "answer different questions: which facts the prediction rests on, and how much weight each is given."),
        ("Computational cost.", "The mean CtD run time mixed two conditions: the splits added in the last two versions ran with one "
         "thread while a second job shared the machine. Time per epoch was about "
         + f"{np.mean([e['s_per_epoch'] for e in R4.get('time_per_epoch_ctd', []) if e['seed'] < 3]):.1f} s and "
         + f"{np.mean([e['s_per_epoch'] for e in R4.get('time_per_epoch_ctd', []) if e['seed'] >= 3]):.1f} s, and the number "
         "of epochs did not differ systematically (Section 4.8, Supplementary Table S14)."),
        ("Statement of significance.", "\"at least as accurate on three of four tasks\"."),
        ("R2 reproducibility as counts.", "Supplementary Section S4 now gives counts: the number of splits on the graded graph "
         "and of metapath–split pairs on Hetionet disease–symptom on which three restarts chose the same operator."),
        ("Reference to BioPathNet.", "Updated to the journal version (Nat. Biomed. Eng., 2026) with its DOI. We could not find a "
         "volume, pages or article number for it in the sources we checked, and we will add them at the proof stage."),
        ("Thinning only non-target relations.", ("Section 4.6 now qualifies the first sentence (\"path methods stayed ahead of "
         "completion in that setting\") and reports that, with target edges intact, Soft-Impute ("
         + f3(spt["softimpute"]["MRR"]["mean"]) + ") and DWPC-LR (" + f3(spt["dwpc_lr"]["MRR"]["mean"]) + ") were ahead of DNFR ("
         + f3(spt["dnfr"]["MRR"]["mean"]) + ").") if spt["dnfr"] else "Section 4.6 qualified."),
        ("Independent verification.", ("We added a script (scripts/verify_clean.sh) that clones the release into an empty "
         "directory, rebuilds the graded graph from the released edge file, and re-trains DNFR, DNFR with the strongest path only "
         "and DNFR without degree terms on the first split, then compares the result with the released file. It was run in a "
         "fresh environment (Python " + VC.get("python", "3") + ", PyTorch " + VC.get("torch", "") + ") and reproduced the "
         "released test MRR to four decimals ("
         + ", ".join(f"{VC['rows'][k]['rerun']:.4f}" for k in ("dnfr", "dnfr_max", "dnfr_nodeg")) + "). Section 7 reports this. "
         "We note, to be exact, that this re-run was also executed with the AI-assisted tooling used for the analyses. The "
         "script is released so that a reader can repeat it, and the transcription check by the corresponding author was done "
         "without the AI system.") if VC.get("rows") else "[pending]"),
        ("Length.", "The refinement details and the former Figure 3 moved to the supplement (M5). Table 5 now holds the two "
         "harder settings on drug repurposing (disease-centred split and the ablation of metapaths through the target "
         "relation), which carry the M4 result, so we kept it in the main text."),
    ]
    for i, (a, b) in enumerate(MINOR, 1):
        para(d, f"**{i}. {a}** {b}")

    # ------------------------------------------------------------------ questions
    heading(d, "Questions", 1)
    Q = [("What happened on the fourth CtD split for the fixed Hamacher product?",
          (f"Split {hm['seed']}: {f3(hm['hamacher'])} against {f3(hm['product'])} for the product. Training was numerically "
           "regular and ended in a poorer optimum (M1, Table S13).") if hm else "See M1."),
         ("Interior minimum in split 3 with p > 0?",
          ("With the grid extended to p = 0.5 and warm starts, the minima on the graded graph were at p = "
           + joinand([ptxt(x["p_argmin"]) for x in PS["intervals"]]) + "; "
           + ("the third split has an interior minimum" if PS["interior"][2] else
              "on the third split the grid minimum is at the new upper end. " + third_split_note(PS).split(". ")[0]) + " (M3).") if PS else "See M3."),
         ("Share of mechanistic CtD evidence; accuracy without the target relation?",
          (f"{100 * XC['share_mean']['gene']:.1f}% mechanistic on average, {pct(XC['share_target_mean'])} through metapaths "
           f"with the treats relation; without them DNFR fell from {f3(mn['dnfr']['mean'])} to {f3(mn['dnfr_notgt']['mean'])} "
           "(M4).") if XC and qd else "See M4."),
         ("Does R4 keep its positive sign with 10 splits?",
          joinand([f"{sg(an[t]['delta'])} on {SH(t)} ({an[t]['wins']}/{an[t]['J']} splits positive)" for t in R3T if an[t]])
          + " (M5)."),
         ("Do DWPC-LR explanations select the same leading facts as DNFR's Shapley attribution?",
          ("For " + joinand([pct(XD[t]["dwpc"]["same_lead_as_dnfr"]) for t in R3T]) + " of pairs ranked in the top 10 by both "
           "models (M2).") if XD and all(t in XD for t in R3T) else "See M2.")]
    for i, (a, b) in enumerate(Q, 1):
        para(d, f"**Q{i}. {a}** {b}")

    heading(d, "Other changes", 1)
    bullets(d, [("A clinical assessment of explanations by the second author (Sections 3.6, 4.7, 5.5; see the note to the "
                 "editor). " + (f"She judged {pct(XR['het_DpS']['pair_plausible']['share'])} and "
                                f"{pct(XR['hkg_DpS']['pair_plausible']['share'])} of the rated predictions clinically plausible, "
                                f"and gave the explanations a median usefulness of {XR['het_DpS']['usefulness']['median']} of 5. "
                                "The paper reads this as showing that the explanations were often a reason to look further "
                                "rather than a reason to believe.")) if XR else "",
                "Supplementary Tables S12–S16 are new (verdicts under both rules, the Hamacher split, run times, the profile on "
                "Hetionet disease–symptom).",
                f"Code and data of this version: release {MT.RELEASE} of https://github.com/Bangth2026/dnfr-medical-kg, archived "
                "at Zenodo (" + MT.ZENODO_VERSION + "all versions: https://doi.org/10.5281/zenodo.23879352)."])
    para(d, "We hope that the revised manuscript addresses the reviewer's and the editor's concerns.", space_after=6)
    para(d, "Truong Hai Bang, on behalf of the authors", align="left")
    out = os.path.join(outdir, "Response_to_reviewer_rev4.docx"); d.save(out)
    print("saved", out)


if __name__ == "__main__":
    build(sys.argv[1] if len(sys.argv) > 1 else os.path.join(ROOT, "manuscript3", "out"))
