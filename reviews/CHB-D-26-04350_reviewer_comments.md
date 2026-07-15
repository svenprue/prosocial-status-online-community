# Reviewer Comments — CHB-D-26-04350

**Manuscript title:** Help Converts Newcomers, Not Veterans: Generalized Reciprocity and Platform Engagement on Stack Overflow
**Journal:** Computers in Human Behavior
**Decision:** Major revision
**Associate Editor:** Catalina L. Toma, PhD
**Revision due:** 2026-07-24

> Verbatim reviewer comments as received. Editor's cover letter (submission logistics, highlights/graphical-abstract/source-file requirements, APC-scam boilerplate) omitted. Our point-by-point responses live in `response_to_reviewers.tex`.

---

## Reviewer #1

### Q1 — Objectives and rationale
The objectives are generally clear, but the rationale could be stated more modestly. The paper should clarify that its main contribution is not to establish a general theory of generalized reciprocity, but to test a short-run behavioral response to receiving an answer on Stack Overflow.

The motivation would be stronger if the authors positioned the paper more directly against prior empirical work on online reciprocity, knowledge sharing, and platform contribution. At present, the paper sometimes makes the contribution sound broader than what the design can support.

The authors should separate the empirical question from the proposed psychological mechanism. The data can show whether users answer more after receiving help, but they do not directly measure gratitude, moral obligation, or incentive displacement.

### Q2 — Replicability / reproducibility — **No**
The paper provides useful detail on the data source and matching strategy, but the construction of the estimation sample is not fully transparent.

The authors should clearly distinguish questions, matched observations, interval rows, users, and help events. The reported sample sizes and event counts in the main tables are difficult to reconcile.

The paper should explain more clearly how controls are duplicated or reused in matching, and how repeated observations from the same user are handled.

The replication repository is withheld for review, which is understandable, but the paper should still provide enough detail for readers to understand the data pipeline and estimation sample.

### Q3 — Statistical analyses / inference — **No**
The manuscript would benefit from additional statistical review.

The identification strategy still leaves substantial room for selection. Answered and unanswered questions may differ in unobserved quality, clarity, difficulty, posting time, user reputation, and topic specificity. These differences may also predict later helping.

Matching on activity history and tag-level answer rates helps, but the authors should add stronger robustness checks, such as controls for posting hour/day, question length, number of tags, user reputation, and text-based measures of question quality or complexity.

Standard errors are a concern. The appendix states that robust or clustered standard errors are not used. Given repeated observations by users, repeated controls, matched pairs, and recurrent helping events, the current inference may be too optimistic.

The authors should report robust standard errors clustered at least by user or matched pair, or use a bootstrap that preserves the matched structure.

The response-time analysis should be modeled more flexibly. The results appear closer to an inverted-U pattern than to a simple "faster is stronger" effect.

### Q4 — Tables and figures
Table 5 should report hazard ratios and confidence intervals for both specifications, not only for the main model.

The paper should include clearer reporting of the estimation sample in each table: number of users, questions, matched observations, interval rows, and events where relevant.

Figure 4 is useful, but the text should treat it as evidence of a non-linear response-time pattern rather than as simple support for faster responses.

The authors should translate hazard ratios into absolute effect sizes, such as additional answers per 1,000 treated questions.

### Q5 — Interpretation and conclusions — **No**
The empirical results are interesting, but the paper overinterprets them in places.

The claim that reciprocity is displaced by platform-specific incentives is plausible but not directly tested. The authors do not observe user motives, reputation salience, gratitude, or moral obligation.

The "re-engagement window" interpretation is also plausible, but the paper does not observe sessions, exits, returns, or notification behavior. This should be presented as a possible explanation rather than as an established mechanism.

H3 should be rewritten or toned down. The strongest effect occurs in the 30–60 minute bin, not among the fastest responses. This does not cleanly support the original hypothesis.

The conclusion should emphasize the narrower finding: receiving an answer is associated with a small short-run increase in helping, mainly among newer users.

### Q6 — Strengths
The authors should emphasize the strongest parts of the paper: the large behavioral dataset, the attempt to move beyond simple correlations, and the focus on heterogeneity by user tenure.

The paper is most convincing when it presents itself as a careful empirical study of short-term contribution behavior on Stack Overflow.

The tenure result is potentially the most interesting contribution and should receive more emphasis than the broader claims about moral reciprocity.

### Q7 — Limitations
The paper should more clearly acknowledge that answer receipt is not random and may reflect unobserved question quality or user characteristics.

The authors should acknowledge that they do not directly observe psychological mechanisms such as gratitude, obligation, or status motivation.

The response-time mechanism depends on unobserved session behavior, so it should be treated as speculative.

The results are from Stack Overflow, a highly specific technical Q&A community, and may not generalize to other online communities.

The small effect sizes should be discussed more explicitly.

### Q8 — Structure, flow, writing
The paper is generally readable, but it could be tightened.

The theory section should be shortened and more directly connected to the empirical design.

The response-time section should be reorganized so that the non-linear result is presented first, followed by a more cautious interpretation.

The methods section should include a clearer explanation of sample construction and event counting.

Some implementation details currently in the appendix should be moved or summarized in the main text because they affect interpretation of the results.

### Q9 — Language editing
Yes.

### Additional overall comments
This is an interesting paper with a useful empirical setting. The question of whether receiving help turns users into later contributors is worth studying, and the large Stack Overflow dataset is a clear strength.

My main concern is that the paper currently makes stronger causal and theoretical claims than the evidence can support. Receiving an answer is not random, and answered questions may differ from unanswered questions in quality, clarity, difficulty, posting time, user reputation, and topic specificity. These differences could also affect later helping behavior.

The paper should be more modest about the mechanism. The results may be consistent with gratitude or generalized reciprocity, but the study does not directly measure gratitude, moral obligation, user motivation, session behavior, or incentive displacement.

The sample and event reporting need to be clarified. The numbers of observations, users, interval rows, and help events are difficult to reconcile across the text, tables, and appendix.

I am also concerned about inference. The paper should report robust or clustered standard errors, given repeated observations by users, matched pairs, reused controls, and recurrent helping events.

The response-time finding is interesting but should be reframed. The strongest effect appears for answers arriving after 30–60 minutes, not for the fastest answers. This looks more like a non-linear pattern than simple support for the hypothesis that faster answers produce stronger reciprocity.

Overall, I see promise in the paper, but it needs a clearer contribution, cleaner reporting, stronger robustness checks, and more cautious interpretation.

---

## Reviewer #2

(Q1, Q4, Q5 marked N/A; Q6–Q8 "Please see my comments." Q9: Yes.)

This manuscript presents a methodologically ambitious attempt to causally identify generalized reciprocity using a matched difference-in-differences survival framework on 21M Stack Overflow questions. The empirical study by combining PSM, staggered DiD, and time-varying Cox models is largely appropriate for the research question. The main finding theoretically plausible and empirically supported. The problem is that the topic is out-dated in the era of GAI. That is, few people use GAI nowadays. This makes the topic not interesting. Several consequential flaws undermine the precision and external validity of the claims. My concerns are as follows.

1. The operationalisation of "help" is too narrow (answers only), excluding upvotes, edits, and comments—forms of low-cost reciprocity that newcomers often use. The striking inverted-U response-time effect (peak at 30–60 minutes) conflates answer speed with answer quality; faster answers may be terse or incorrect, while 30-minute answers tend to be more thorough, a confound not tested. The analysis suffers from left-truncation bias: users who asked and never received an answer, then permanently churned, are excluded, potentially underestimating the true effect.

2. The practical magnitude (HR ≈ 1.06–1.09) is statistically significant only due to the astronomical sample size, but the paper overstates its operational value without reporting an NNT. The authors overgeneralise findings to all knowledge-sharing platforms, ignoring that Stack Overflow is uniquely technical, anonymous, and status-driven and critically, its ecosystem has been structurally disrupted by generative AI since 2023, which the discussion barely acknowledges.

3. The analytical sample only includes users who remained active long enough to post a question and be observed. Users who posted a question, received no answer, and immediately churned are excluded. Since this exclusion is likely correlated with the treatment (no answer) and the outcome (no future helping), the estimated treatment effect (HR≈1.09) is probably downward-biased. The authors should conduct a sensitivity analysis using Heckman-type selection correction or, at minimum, explicitly bound the possible bias. Without this, the claim that reciprocity "recruits contributors" misses the fraction of newcomers who are lost precisely because they never experienced reciprocity.

4. The outcome variable counts only "posting an answer" as a helping event. On Stack Overflow, users frequently reciprocate through upvoting, editing, commenting, or accepting an answer. This ehaviours require far less effort and are especially common among novices who lack expertise. By excluding these actions, the study systematically undercounts reciprocal behaviour and misclassifies many reciprocating users as non-responders. The authors should re-estimate the model using a composite helping indicator (including upvotes and comments) or justify why only answers capture the theoretically relevant construct, with reference to prior reciprocity literature.

5. The paper interprets the peak reciprocity at 30–60 minutes as a "re-engagement window" effect. However, a simpler alternative is answer quality: extremely fast answers (<15 min) are often terse, link-only, or low-scoring, while 30–60-minute answers are typically more detailed, with code snippets and explanations, generating greater gratitude. The authors do not control for answer length, vote score, or whether the answer was accepted. Without testing this confound (e.g., by conditioning on answer quality metrics), the structural "session-dynamics" interpretation remains speculative and should be toned down.

6. The platform has undergone massive cultural, technical, and policy shifts from 2008 to 2025—including reputation inflation, spam filters, tag evolution, and most critically, the rise of ChatGPT post-2022. While calendar year is used as an exact matching variable, the tenure-stratified analysis pools newcomers from 2008 with those from 2024 as if they share the same "newcomer" experience. This ignores cohort heterogeneity. The authors should include year-of-entry fixed effects or conduct robustness checks on pre-2020 subsamples to isolate the pure human-only reciprocity mechanism.

7. The hazard ratios of 1.06 to 1.09 are statistically significant due to N > 20 million, but in absolute terms, they represent only a ~6–9% increase in the instantaneous helping rate. The paper does not report the Number Needed to Treat (NNT)—i.e., how many newcomers must receive an answer to generate one additional helping event. Without this, platform designers cannot assess whether investing in faster response systems yields meaningful returns. The authors should calculate and report NNT, and revise their "Implications for Design" section to reflect modest effect sizes.

8. Posting a question increases a user's profile visibility: other members click on their profile, view their history, and may trigger a sense of social obligation or reputation concern. This "exposure effect" is not equivalent to gratitude for receiving an answer, yet it operates precisely during the post-answer period and is not separately identified. The waiting-period covariate absorbs baseline question-asking effects, but it does not capture the increased visibility that persists after an answer arrives. A placebo test using questions that received no answers but high views could help rule out this confound; currently, it is not attempted.

9. The discussion repeatedly generalises findings to "online knowledge-sharing communities" without acknowledging that Stack Overflow is a uniquely technical, anonymous, and gamified environment. Reciprocity dynamics in smaller, enterprise, or relation-based platforms (e.g., internal Slack, academic peer review) likely differ substantially. Moreover, the platform's active user base has sharply declined since 2023 due to LLM substitution. The paper treats its conclusions as timeless, but they are, at best, a pre-generative-AI baseline. The authors must add a dedicated limitations paragraph explicitly stating that their findings are a historical snapshot, not a prescriptive guide for contemporary or future platform design.
