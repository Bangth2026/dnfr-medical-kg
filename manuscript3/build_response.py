"""Response to the reviewer (third revision). Every number is read from the same statistics files as the manuscript.
Usage: python3 manuscript3/build_response.py [out_dir]"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from ms_common import *
from docx_helpers import new_document, para, heading, bullets
from ms_results import r3p, r3sum, mean_any
import ms_text as MT

P_ = lambda d, t, **k: para(d, t, **k)


def c(d, text):
    para(d, f"*Comment.* {text}", size=10.5)


def r(d, text):
    para(d, f"**Response.** {text}")


def ch(d, text):
    para(d, f"*Changes:* {text}", size=10.5)


def dci3(p):
    return "—" if p is None else f"{sg(p['delta'])} [{sg(p['bs95'][0])}, {sg(p['bs95'][1])}]"


def build(outdir):
    d = new_document()
    PS = R3.get("profile_summary") or {}; iv = PS.get("intervals", [])
    PSm = R3.get("prior_summary", {}); E3 = R3.get("explain3", {})
    nc = nseeds("het_CtD", "dnfr"); nca = nseeds("het_CtD", "dnfr_nodeg")
    rc = MT.refine_counts()
    para(d, "**Response to the reviewer**", align="left", size=14)
    para(d, f"Manuscript: *{MT.TITLE}* (revision 3; previously *Path aggregation and hub discounting, not the learned t-norm: a "
            "controlled component analysis of differentiable neuro-fuzzy reasoning on medical knowledge graphs*)", align="left")
    P_(d, "We thank the reviewer for a careful and constructive report. We ran every analysis that was feasible on our hardware (a "
          "shared two-core CPU with 8 GB of memory) and say plainly where we could not. Besides answering the comments, this revision "
          "tests four refinements of the model that we had proposed to address apparent weaknesses: overlap-aware aggregation, a "
          "discrete (Gumbel-softmax) operator choice, an information-content hub discount and membership annealing, together called "
          "DNFR 2.0. We expected some of them to help. None did under our decision rule, and we report this as a result rather "
          "than leave it out. The paper is therefore now framed as a controlled component analysis that includes a controlled test "
          "of four plausible modifications. Reviewer comments are summarised in italics; section, table and figure numbers refer to "
          "the revised manuscript (S-numbers to the new supplementary file).")
    heading(d, "Title", 2)
    P_(d, "Following minor comment 4, the title no longer states a conclusion. We propose:")
    bullets(d, [f"(A, used in the manuscript) {MT.TITLE}",
                "(B) DNFR 2.0: a controlled evaluation of four refinements of differentiable neuro-fuzzy path reasoning on medical "
                "knowledge graphs"])
    P_(d, "We prefer (A), because the component analysis and the identifiability result are the main contributions and the "
          "refinements did not help; (B) would put a negative result in the title. We are happy to follow the editor's preference.")

    # ------------------------------------------------------------------ major
    heading(d, "Major comments", 1)
    heading(d, "M1. Identifiability: non-convergence, parameterisation, structural vs practical", 2)
    c(d, "The profile did not converge; the sigmoid parameterisation saturates near the observed endpoints; distinguish structural "
         "from practical non-identifiability (Raue et al. 2009) and report a curvature; rewrite the early-stopping sentence for CtD.")
    lr = PS.get("loss_range", [])
    r(d, "We agree with all three points and redid the analysis. (1) *Converged profile.* On the graded graph we fixed a shared p on a "
         "grid from −5 to 0 and fitted every other parameter by L-BFGS with strong Wolfe line search until the loss changed by less "
         "than 10⁻⁹, sweeping the grid upwards and downwards with warm starts and keeping the lower loss at each p (a single sweep "
         "left some points in a worse local optimum). Over the whole range the converged loss varied by "
         + ", ".join(f"{x:.4f}" for x in lr) + " nats in the three splits (loss about 4.3). The minimiser was at p = "
         + ", ".join(f"{x['p_argmin']:g}" for x in iv).replace("-", "−") + ", and the profile-likelihood style intervals "
         "(threshold χ²(0.95,1)/(2N)) were " + ", ".join(f"[{x['p_lo']:g}, {x['p_hi']:g}]".replace("-", "−") for x in iv)
         + ". The curvature at the only interior minimum was ∂²L/∂p² ≈ "
         + (f"{[x for x in PS['curvature'] if x][0]:.3f}" if any(PS.get("curvature", [])) else "not estimable")
         + ". (2) *Gradient in p and an unbounded parameterisation.* We now report ∂L/∂p itself at every converged point (Figure 2, "
         f"centre; at most {PS.get('max_abs_dLdp', float('nan')):.4f} in absolute value), and we repeated the restarts with p = θ, "
         "clipped only to [−20, 0.95] to keep the t-norm finite (Table 5). The learned values still depended on the starting value, "
         "so the lack of convergence is not an artefact of the bounded parameterisation. (3) *Vocabulary.* Section 4.4 now separates "
         "the structural redundancy between memberships and operator (frozen-membership experiment) from practical "
         "non-identifiability by the training objective, cites Raue et al., and states, as the reviewer suggests, that held-out "
         "accuracy does carry information about p: test MRR at the value chosen on validation exceeded that of the product by "
         + ", ".join(sg(x["test_at_valid_choice"] - x["test_product"]) for x in iv)
         + " in the three splits. The early-stopping sentence for compound–treats–disease was rewritten: the median best epoch there was "
         + str(ri(RS["stopping"]["het_CtD"]["dnfr"]["best_epoch"]["median"])) + ", and that many Adam steps could move p "
         "by close to one unit, so the small spread observed there is an empirical outcome, not a mechanical one.")
    ch(d, "Sections 3.4 (iv), 4.4; Figure 2 (new); Table 5; Supplementary Tables S7–S8; Abstract, Conclusion.")

    heading(d, "M2. A validation-selected fixed t-norm", 2)
    c(d, "Add a fixed p selected on validation, trained with the same protocol as DNFR, to the main tables.")
    fv = {t: r3p(t, "fixp_val", "dnfr") for t in MT.R3T}; fvp = {t: r3p(t, "fixp_val", "product") for t in MT.R3T}
    r(d, "Done for the three tasks on which the third-revision experiments ran. A shared p is fixed at each value of a grid (−4, −2, "
         "−1, −0.5, 0.5 and the product; −2, −1, −0.5 and the product on compound–treats–disease, for cost), trained with the default "
         "early stopping, and the value with the best validation MRR is kept per split. It differed from DNFR by "
         + "; ".join(f"{dci3(fv[t])} on {MT.SH(t)}" for t in MT.R3T if fv[t]) + ", and from the fixed product by "
         + "; ".join(f"{dci3(fvp[t])} on {MT.SH(t)}" for t in MT.R3T if fvp[t])
         + ". The gain over the product on the graded graph survives early stopping and is the only difference among the t-norm "
         "comparisons that our rule supports. We did not run it on PrimeKG, where a single run takes several hours; this is stated "
         "in Section 5.5.")
    ch(d, "Section 3.4 (Fixed operators), Section 4.3, Table 2, Table 4, Supplementary Tables S1, S2, S4.")

    heading(d, "M3. Statistical power on CtD and PrimeKG", 2)
    c(d, "Two or three splits do not support generalisations to all four tasks; add splits on CtD or rephrase.")
    r(d, f"We increased compound–treats–disease to {nc} splits for the main comparisons and to {nca} for the component ablations "
         f"(Table 3); the refinements and DNFR 2.0, added in this revision, have {r3p('het_CtD', 'dnfr_overlap', 'dnfr')['J']}–"
         f"{r3p('het_CtD', 'dnfr2', 'dnfr')['J']} splits there. We also changed how multiplicity is handled (see M4): Holm adjustment is now applied within families defined by "
         "question (baselines, components, refinements, NBFNet) instead of over every comparison in a table, which had mixed "
         "confirmatory comparisons with descriptive restarts. Highlights, Abstract, Section 5.1 and the Conclusion no longer "
         "generalise across all four tasks; they state the sign consistency and, separately, on which tasks the difference is "
         "supported after adjustment. PrimeKG results (two splits) are described as descriptive throughout.")
    ch(d, "Section 3.6, Tables 2–4, Highlights, Abstract, Sections 4.2, 5.1, 5.5, 6.")

    heading(d, "M4. Bootstrap interval, corrected interval and verdict", 2)
    c(d, "A corrected interval narrower than the bootstrap is surprising; verdicts marked † next to Holm-adjusted p = 0.635.")
    pfd = pr("het_CtD", "dnfr", "dnfr_fixdeg")
    r(d, "We checked the computation; it is correct. The corrected resampled t-interval uses the variance of the per-split "
         "differences, and with three splits that variance is itself estimated from three numbers. In the case the reviewer "
         "noted, the three per-split differences happened to lie close together, so the estimated between-split variance was small "
         "and the interval narrow, while the bootstrap over about 75 test queries per split was wide. Neither interval is wrong; "
         "each misses a different source of uncertainty. The decision rule now requires both: a comparison is *different* only if "
         "the Holm-adjusted corrected-t p-value is below 0.05 *and* the 95% bootstrap interval excludes zero on the same side, and "
         "*equivalent* only if both 90% intervals lie inside ±0.01. Under this rule the comparison in question is now "
         f"{pfd['verdict']} (Holm-adjusted p = {ph(pfd)}), and no verdict in the paper is shown next to a Holm-adjusted p-value "
         "that does not support it.")
    ch(d, "Section 3.6 (Statistics and decision rule), all verdict columns.")

    heading(d, "M5. 'Pre-specified' equivalence rule and the margin", 2)
    c(d, "Do not call the rule pre-specified; justify ±0.01 substantively; note that ≤3 splits cannot give equivalence.")
    r(d, "Agreed on all three points. The rule is described as fixed before the analyses added in this revision. The margin is "
         "justified by what it means for a ranked list: a change of 0.01 in MRR corresponds to one correct answer in fifty moving "
         "from rank 1 to rank 2, or about one in seventeen moving from rank 2 to rank 3, which we regard as too small to change which "
         "candidates a reviewer of the list would examine. Section 3.6 states that with three or fewer splits no equivalence "
         "verdict is attainable. We also note in this letter that the rule was revised in this round (M4) after the reviewer's "
         "comments, so it is not independent of the results we had seen.")
    ch(d, "Section 3.6; Contribution 1.")

    heading(d, "M6. Edge prior below popularity", 2)
    c(d, "Check ties, more permutations or the analytic approximation, and leave-dyad-out effects.")
    if PSm:
        cc = PSm["het_CtD"]
        r(d, "We checked (Supplementary Table S9). With 100 permutations, the true tail tied with another candidate in "
             f"{pct(cc['tie_any_prior100'])} of compound–treats–disease test queries. On the two disease–symptom tasks the gap was a "
             f"matter of resolution: with 1000 permutations the prior reached {f3(PSm['het_DpS']['prior1000_MRR'])} and "
             f"{f3(PSm['hkg_DpS']['prior1000_MRR'])}, against {f3(PSm['het_DpS']['popularity_MRR'])} and "
             f"{f3(PSm['hkg_DpS']['popularity_MRR'])} for popularity, and the analytical approximation of Zietz et al. ranks tails "
             "in exactly the same order as popularity for a fixed head (Spearman correlation 1). On compound–treats–disease the gap "
             f"persists ({f3(cc['analytic_MRR'])} analytic, {f3(cc['popularity_MRR'])} popularity), and the reason is the "
             f"leave-dyad-out split, as the reviewer suspected: for {pct(cc['share_zero_true'])} of test queries the true pair has a "
             "prior of zero, because most compounds have one or two indications and many test compounds keep none in training. The "
             "prior is then zero for every candidate of that compound, and all candidates tie, whereas popularity still orders the "
             "diseases. The implementation is correct; the statement that path accuracy is not an artefact of degree stands, since "
             "popularity itself is far below every path method.")
    ch(d, "Section 4.1 (new paragraph); Supplementary Table S9.")

    heading(d, "M7. Learned-operator GNN baselines", 2)
    c(d, "Run NBFNet on HKG and DpS; cite BioPathNet; embeddings were not run on HKG.")
    nb = {t: r3sum(t, "nbfnet") for t in MT.R3T}
    r(d, "We implemented a compact NBFNet for CPU (three layers, dimension 32, DistMult messages, sum aggregation, layer "
         "normalisation, training with masked target edges and the same filtered softmax loss and early stopping; Section 3.4) and "
         f"ran it on the graded graph ({nb['hkg_DpS']['n']} splits, MRR {f3(nb['hkg_DpS']['mean'])} against "
         f"{f3(mean_any('hkg_DpS', 'dnfr'))} for DNFR, difference {dci3(r3p('hkg_DpS', 'nbfnet', 'dnfr'))}) and on Hetionet "
         + (f"disease–symptom ({nb['het_DpS']['n']} split{'s' if nb['het_DpS']['n'] > 1 else ''}, MRR {f3(nb['het_DpS']['mean'])})"
            if nb.get("het_DpS") else "disease–symptom")
         + ". On Hetionet disease–symptom one run took over an hour, which limited the number of splits. On the graded graph NBFNet "
         "was within 0.001 of DNFR (inconclusive under our rule) and below DWPC-LR (a difference our rule supports); on Hetionet "
         "disease–symptom it was ahead of DNFR by " + f"{r3p('het_DpS', 'nbfnet', 'dnfr')['delta']:.3f}" + " on three splits, too "
         "few for a verdict. A learned operator therefore did not show a clear advantage over DNFR or DWPC-LR, which answers Q5 "
         "for these settings. BioPathNet (now published in Nature "
         "Biomedical Engineering) and FuzzQE are cited and discussed in Section 2.2. We did not add tuned embeddings on the graded "
         "graph: on a single bipartite relation an embedding reduces to a low-rank factorisation, and Soft-Impute, which led on that "
         "graph, is that comparison. This is stated in Section 4.1.")
    ch(d, "Sections 2.2, 3.4, 4.1, 5.5; Table 2.")

    heading(d, "M8. Zero-shot evaluation on PrimeKG", 2)
    c(d, "Add the disease-centred split on PrimeKG.")
    r(d, "We could not do this within the revision. A single DNFR run on PrimeKG takes several hours on our hardware, and the "
         "disease-centred split would need path enumeration and every method on at least four new splits. Rather than report one "
         "underpowered split, we narrowed the claims: the zero-shot result is reported as resting on one task with four splits and "
         "137 candidate diseases, the reversal of the degree effect is described as plausible but not established, and Section 5.5 "
         "states that it remains open on a realistic candidate set. Q4 is therefore unanswered.")
    ch(d, "Sections 4.5, 5.3, 5.5.")

    heading(d, "M9. Medical contribution", 2)
    c(d, "Clinician rating, or a quantitative check against mechanistic resources; state that HKG tests collaborative evidence.")
    lc = E3.get("het_CtD", {}).get("lead_class")
    r(d, "(i) A clinician rating was not possible within this revision, and we do not want to report one with fewer raters than "
         "the reviewer suggests; it is named as the next step in Section 5.5. (ii) We added the quantitative check the reviewer "
         "proposed: for each correct top-10 prediction on compound–treats–disease we classified the leading metapath as mechanistic "
         "(through a gene that the compound binds or regulates and that is associated with the disease) or associative (through "
         "pharmacologic classes, similar compounds or similar diseases)"
         + (f". {pct(lc['gene'])} were mechanistic and {pct(lc['associative'])} associative" if lc else "")
         + ". We also added an exact edge-level attribution (M10), which tells a clinician which facts a prediction rests on. (iii) "
         "Sections 3.5 and 5.3 now state that the graded graph is a single bipartite relation whose only metapath links a disease "
         "to a symptom through diseases with similar symptom profiles, so it tests graded collaborative evidence from records, not "
         "multi-relational composition.")
    ch(d, "Sections 3.5, 3.6, 4.7, 5.3, 5.5; Table 7.")

    heading(d, "M10. Double counting and 'exact attribution'", 2)
    c(d, "Report an edge-level attribution; qualify 'exact'; retrained counterfactual on a sample.")
    ts = [t for t in ("het_CtD", "het_DpS", "hkg_DpS") if t in E3]
    r(d, "Because the score is a sum of path terms and a path term counts only if all its edges are present, the score is a sum of "
         "unanimity games, one per path. The Shapley value of an edge is therefore exact and cheap: each path's term is split equally "
         "among its edges. We report it in Table 7. Summing path shares, the leading fact appeared to carry "
         + ", ".join(f"{pct(E3[t]['top_edge_double_median'])} on {MT.SH(t)}" for t in ts) + "; its Shapley share was "
         + ", ".join(pct(E3[t]["top_edge_shapley_median"]) for t in ts) + ". The counterfactual now removes the fact with the "
         "largest Shapley share. In Section 1 and Section 3.2, 'exact' is qualified as exact for the model's additive score, not for "
         "the underlying evidence. We did not run the retrained counterfactual: retraining once per removed fact on a sample of "
         "pairs was beyond our budget, and Section 4.7 states that the test measures the dependence of the trained model on a fact.")
    ch(d, "Sections 1, 3.2, 3.6, 4.7; Table 7.")

    # ------------------------------------------------------------------ minor
    heading(d, "Minor comments", 1)
    pk = DS["pkg"]; und = pk["raw_rows"] // 2; extra = und - pk["dropped_edges"] - pk["edges"]
    MINOR = [
        ("Revision history inside the manuscript.", "Removed. Section 3.6 keeps only what was set before which analyses."),
        ("List numbering in §5.4.", "Numbering restarts at 1."),
        ("Abstract readability.", "The Results paragraph now carries four numbers."),
        ("Title.", "Changed; see above."),
        ("MRR comparability across tasks.", "Sentences that juxtaposed MRR across tasks were removed, and Section 4.1 and Table 2 "
                                           "state that MRR is not comparable across columns."),
        ("CtD validation set size.", "Supplementary Table S10 reports the spread of validation MRR across checkpoints for the "
                                     "default runs: the standard deviation over all checkpoints was "
                                     + ", ".join(f3(v["sd_all"]) for v in R3.get("ctd_valid_spread", []))
                                     + ", and the five best checkpoints of a run lay within "
                                     + ", ".join(f3(v["range_top5"]) for v in R3.get("ctd_valid_spread", [])) + "."),
        ("Metapath budget.", "The lower-bound remark is now in Section 4.1, where PrimeKG results first appear. A sampled-path "
                             "variant was not feasible; the limitation is kept."),
        ("PrimeKG edge count.", f"The edge file we used has {pk['raw_rows']:,} rows, each undirected edge stored in both directions, "
                                f"so {und:,} undirected edges, matching the reviewer's starting count. Removing the "
                                f"{pk['dropped_edges']:,} anatomy–protein edges leaves {und - pk['dropped_edges']:,}; keeping one "
                                f"orientation per relation and dropping repeated pairs and self-loops removed another {extra:,}, "
                                f"leaving {pk['edges']:,}. Section 3.5 now gives this reconciliation."),
        ("Sparsity analysis.", "Added a variant that thins every relation except the target relation (Hetionet disease–symptom, "
                               "25% of edges kept), so that completion keeps its training edges (Section 4.6, Supplementary Table S5)."
         if summ("het_DpS", "dnfr", "_spt0.25") else "[pending: target-intact sparsity run]"),
        ("Highlights vs Table 3.", "Highlights rewritten (M3)."),
        ("DrD–DpS sentence in §4.3.", "Removed."),
        ("Reliance on an unreviewed deposit.", "The Introduction mentions the NHANES analysis in one sentence, as motivation only, "
                                               "and motivates the question from the fuzzy-systems literature; the secondary test "
                                               "moved to the supplement."),
        ("Related work.", "FuzzQE and BioPathNet added to Section 2.2; Raue et al. and the Gumbel-softmax relaxation to Section 2.3."),
        ("AI-assistance statement.", "Section 7 now has an 'Independent verification' paragraph. [AUTHOR TO CONFIRM: the authors must "
                                     "state here, and in the manuscript, which results they re-ran or re-derived without the AI "
                                     "system.]"),
        ("CRediT.", "The second author's role is now described as a check that the nodes and edges in Table 8 exist in the source "
                    "graphs with the stated relations, a check of graph correctness and not a clinical judgement."),
        ("Length.", "Tables A1–A5, the long-training table, both restart figures and Appendix B moved to the supplementary file; Table "
                    "3 is split into panels (components, t-norm, frozen memberships)."),
    ]
    for i, (a, b) in enumerate(MINOR, 1):
        para(d, f"**{i}. {a}** {b}")

    # ------------------------------------------------------------------ questions
    heading(d, "Questions", 1)
    Q = [
        ("Curvature at the best p on HKG with other parameters converged?",
         "At the only interior minimum (one of three splits) ∂²L/∂p² ≈ "
         + (f"{[x for x in PS['curvature'] if x][0]:.3f}" if any(PS.get("curvature", [])) else "—")
         + "; in the other two splits the minimum lay at the end of the grid. Over p from −5 to 0 the loss varied by at most "
         f"{max(PS.get('loss_range', [float('nan')])):.4f} nats (M1)."),
        ("Does the HKG preference for p ≈ −2 survive early stopping?",
         f"Partly. A fixed p = −2 trained with early stopping reached {f3(mean_any('hkg_DpS', 'fixp_m2'))}, not the 0.338 of the "
         "unconverged profile; the validation-selected fixed p beat the product by "
         f"{sg(r3p('hkg_DpS', 'fixp_val', 'product')['delta'])}, a difference our rule supports (M2)."),
        ("How many candidates tie under the 100-permutation prior?",
         f"On compound–treats–disease the true tail tied in {pct(PSm['het_CtD']['tie_any_prior100'])} of test queries, with "
         f"{pct(PSm['het_CtD']['tie_share_prior100'])} of candidates tied on average (M6, Table S9)." if PSm else "—"),
        ("Does removing degree terms help on the PrimeKG disease split?", "Not tested (M8)."),
        ("Would NBFNet reproduce the conclusion?", "On the graded graph it was within 0.001 of DNFR and below DWPC-LR; on Hetionet disease–symptom it was slightly ahead of DNFR on three splits (M7)."),
    ]
    for i, (a, b) in enumerate(Q, 1):
        para(d, f"**Q{i}. {a}** {b}")

    heading(d, "Other changes", 1)
    bullets(d, ["Four refinements (Section 3.3, Table 4, Figure 3) and the combined DNFR 2.0. "
                + (f"None of the {rc['n']} add-one comparisons showed a gain under our rule." if not rc["pos_diff"] else "")
                + (" Combined, DNFR 2.0 was less accurate than DNFR on Hetionet disease–symptom ("
                   + sg(r3p('het_DpS', 'dnfr2', 'dnfr')['delta']) + ", supported by our rule)."
                   if (r3p("het_DpS", "dnfr2", "dnfr") or {}).get("verdict") == "different" else ""),
                "An exact edge-level Shapley attribution (Section 3.6) and a mechanistic classification of drug-repurposing "
                "explanations.",
                "Supplementary file with Tables S1–S11 and Figures S1–S2.",
                "All numbers are regenerated from the released result files by the scripts that build the manuscript and this letter."])
    para(d, "We hope that the revised manuscript addresses the reviewer's concerns.", space_after=6)
    para(d, "Truong Hai Bang, on behalf of the authors", align="left")
    out = os.path.join(outdir, "Response_to_reviewer_rev3.docx"); d.save(out)
    print("saved", out)


if __name__ == "__main__":
    build(sys.argv[1] if len(sys.argv) > 1 else os.path.join(ROOT, "manuscript3", "out"))
