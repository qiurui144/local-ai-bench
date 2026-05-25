# Interview Master Bank — 120 questions across 6 categories

Each section contains 20 questions. Answers are intentionally brief; the
intent is to seed discussion, not to provide a closed-book answer key.

## Table of contents

1. [Retrieval and indexing (1-20)](#1-retrieval-and-indexing)
2. [Generation, grounding, and citation (21-40)](#2-generation-grounding-and-citation)
3. [LLM-as-judge and human evaluation (41-60)](#3-llm-as-judge-and-human-evaluation)
4. [Statistical rigor (61-80)](#4-statistical-rigor)
5. [Production, canary, drift, rollback (81-100)](#5-production-canary-drift-rollback)
6. [Failure modes, attacks, ethics (101-120)](#6-failure-modes-attacks-ethics)

---

## 1. Retrieval and indexing

1. **What does Recall@k actually measure?** Fraction of queries whose
   relevant set is fully present in top-k. Not the same as "got the
   right answer," but a necessary upstream condition.
2. **When does MRR diverge from NDCG?** When a query has multiple
   relevant docs at different grades; MRR only cares about rank of
   first relevant; NDCG rewards graded gain.
3. **Why is bpref preferred over precision when judgments are
   incomplete?** Bpref ignores unjudged docs; precision assumes
   anything not in the relevant set is non-relevant, which inflates
   the apparent error rate.
4. **Cosine vs Euclidean for embeddings - why cosine?** Embedding
   magnitudes are usually unstable across documents; angle is the
   signal that matters.
5. **HNSW vs IVFPQ tradeoffs?** HNSW: better recall, larger memory,
   harder to update. IVFPQ: lower memory, lossy quantization, easier
   to retrain.
6. **What is the bias of NDCG?** Logarithmic discount means rank 1
   matters massively; the metric is insensitive to swaps at deep
   ranks.
7. **When is BM25 still relevant?** Sparse vocabulary, rare-term
   queries (long codes, proper nouns), and as a fast lexical
   complement to dense retrieval in hybrid stacks.
8. **Why hybrid retrieval?** Dense catches paraphrase, sparse catches
   exact-match. Each is wrong on a different query type; ensembling
   beats either alone.
9. **How to evaluate retrieval without graded judgments?** Use binary
   recall/precision and report bpref. For unjudged depths, do not
   penalize.
10. **What is the catastrophic failure mode of dense retrieval?**
    Out-of-distribution queries where the embedder maps them to the
    wrong region. Sanity-check with bucketed recall.
11. **Why is chunk size a parameter to tune?** Too small: answer is
    split across chunks; recall drops. Too large: signal diluted by
    noise; reranker harder. Sweet spot per corpus.
12. **What is reciprocal rank fusion?** Combine multiple rankers by
    summing `1/(k + rank_r(d))` per ranker; rank-only, no score
    calibration needed.
13. **When does query expansion hurt?** When the corpus lacks the
    paraphrased terms; expansion increases noise without recall gain.
14. **What is graded relevance?** Multi-level (e.g. 0-4) judgment; used
    by NDCG, ERR. Better signal than binary for medium-quality matches.
15. **Why is recall@1000 useful even when you only show top 10?**
    Reranker quality is bounded by candidate-set quality; recall@1000
    is the ceiling.
16. **What is the role of a CrossEncoder reranker?** Re-score (query,
    doc) jointly with bidirectional attention; more accurate than
    bi-encoder, slower.
17. **How to detect index staleness?** New documents indexed at time T
    must appear in top-k for queries that target them; smoke test
    weekly.
18. **What is the purpose of MMR?** Maximum Marginal Relevance:
    penalize selecting a doc similar to one already selected; trades
    relevance for diversity.
19. **Difference between R-Precision and P@k?** R-Precision adapts k
    per query (k = number of relevant); P@k is fixed. R-Precision is
    a robust single-number summary.
20. **How does RBP encode user behavior?** Each rank visited has
    probability p of continuing to the next; metric is the expected
    relevance encountered before user gives up.

## 2. Generation, grounding, and citation

21. **What is groundedness?** Each claim entailed by the cited
    evidence; the anti-hallucination property.
22. **Why claim-level rather than answer-level groundedness?** A
    10-claim answer with one unsupported claim is 90% grounded, not
    "ungrounded"; granularity preserves signal.
23. **What is strict (RAGAS-style) faithfulness?** Any unsupported
    claim drops the answer to 0; binary safety view.
24. **Why is "must-not-say" a critical metric?** Substring forbidden
    by regulation or policy (a competitor name, an unsafe value);
    any occurrence is a release blocker.
25. **Difference between over-refusal and under-refusal?** Over: model
    refuses when evidence is sufficient. Under: model answers when it
    should have refused.
26. **Why is over-refusal painful in production?** Users perceive the
    bot as useless; engagement craters even though correctness is
    nominally high.
27. **What is citation precision vs recall?** Precision: of cites
    given, fraction correct. Recall: of correct evidence, fraction
    cited.
28. **What is the difference between perplexity and quality?**
    Perplexity rewards likelihood; quality requires correctness.
    Two models can have similar perplexity yet differ on factual
    accuracy.
29. **Why are short answers harder to judge?** Less signal per
    answer; statistical power for win-rate comparisons drops.
30. **What is the role of system-prompt context length?** Longer
    context = more retrieved evidence = potentially higher recall but
    risk of distractor-induced hallucination.
31. **How to evaluate cited-but-wrong vs uncited-but-correct?** Both
    are failures: the first violates citation precision; the second
    fails citation recall and user verifiability.
32. **Why decompose claims via LLM?** Sentence splits are noisy;
    nested claims need finer parsing. LLM decomposition is the
    accuracy ceiling.
33. **What is partial credit scoring?** Per-claim coverage in [0, 1]
    rather than binary; rewards answers that get most facts right.
34. **What is the role of temperature in evaluation?** Temperature > 0
    introduces sampling variance. For judging, temperature = 0;
    for generation, varies by domain.
35. **Why include both BLEU and ROUGE in a relevance battery?** Each
    has known biases (BLEU: precision-leaning; ROUGE: recall-leaning).
    Reporting both surfaces disagreement that warrants attention.
36. **What is BERTScore?** Pairwise contextual embedding similarity
    between reference and candidate tokens; smoother than BLEU on
    paraphrase.
37. **How to handle multi-doc answers (synthesis)?** Each claim should
    cite a doc; the answer's overall groundedness is the average
    across claims.
38. **Why bullet-point answers can fool judges?** Bullets atomize
    claims and look more "organized"; judges over-credit structure
    independent of substance. Control for this in calibration.
39. **What is intent satisfaction?** A judge-rated boolean/score: does
    the answer match the user's intent independent of any other
    criterion?
40. **Why is "did the model answer the question I asked" worth
    rating separately from "did it tell the truth"?** Truthfully
    answering the wrong question is still failure.

## 3. LLM-as-judge and human evaluation

41. **Why use an LLM judge at all?** Cost. Human judges are 10-100x
    more expensive per item. LLM judges scale but require calibration.
42. **What is position bias in pairwise judging?** The judge prefers
    whichever answer is shown first; standard fix is order-swapping.
43. **What is verbosity bias?** Judges prefer longer answers regardless
    of substance. Detect via length-correlation in calibration.
44. **What is self-preference?** A judge from family X over-rates
    answers from family X. Use a different family as judge or pool
    multiple families.
45. **Why is Cohen's kappa preferred over raw agreement?** Kappa
    corrects for chance agreement; raw agreement looks great on
    skewed distributions even when judges aren't really agreeing.
46. **When to use Fleiss vs Krippendorff?** Fleiss: fixed number of
    raters per item, categorical. Krippendorff: variable raters,
    missing data, mixed scales.
47. **Why include human references in any judge audit?** Human is the
    ground truth for "is the judge calibrated to humans"; without it
    you're tuning to a synthetic reference.
48. **What is the variance-budget pattern?** Run the judge N times
    per item; aggregate via median; report consistency rate
    (fraction of items where all N agree).
49. **What is tiered judging?** Cheap weak judge first; escalate to
    strong judge only on weak-tier disagreement; cost / quality
    tradeoff.
50. **What is G-Eval style prompting?** Chain-of-thought, ask the
    judge to reason internally, output structured JSON.
51. **Why JSON-only output for judges?** Parseability. Free-text
    judges break in CI when 0.1% have a stray token.
52. **What is the role of few-shot examples in judge prompts?**
    Anchors the judge to known-good labels; reduces variance.
    Caution: examples must be balanced (don't seed bias).
53. **Why aren't ROUGE/BLEU enough?** They measure n-gram overlap, not
    correctness; an answer with the right structure but wrong content
    scores high. They're a sanity check, not a headline metric.
54. **What is the role of expert reviewers?** Sample-based audit of
    judge labels; surfaces systematic judge errors that calibration
    can't catch.
55. **How often should judge calibration run?** Daily at minimum for
    high-stakes deployments; weekly for low-stakes. Judges drift.
56. **What is a probabilistic judge?** Emits a confidence score in
    [0, 1] rather than a hard label; allows ECE/Brier audit.
57. **Why human-in-the-loop on the calibration set?** A small,
    expert-reviewed gold pair set is the bedrock; the LLM judge is
    measured against it.
58. **What is judge inter-run consistency?** Same input, multiple
    judge invocations: should produce the same verdict at low
    temperature.
59. **What is the failure mode of using the same family for both
    generator and judge?** Self-preference inflates win rate.
60. **What is pairwise vs absolute scoring?** Pairwise: A vs B
    comparison; absolute: rate A on 1-5. Pairwise is more reliable
    for relative ranking; absolute is needed for monitoring.

## 4. Statistical rigor

61. **Why is a single-seed comparison untrustworthy?** Seeds
    introduce noise comparable to typical effect sizes; one seed is
    a coin flip on small benchmarks.
62. **What is the 2-sigma rule?** Improvement < 2 * typical_std
    across seeds doesn't count as improvement.
63. **Why Wilcoxon over paired-t for retrieval metrics?** Retrieval
    scores are bounded and often non-normal; non-parametric is
    safer.
64. **What is Bonferroni correction and when to use it?** Multiply
    p-values by number of comparisons; conservative; use when FWER
    matters and you have <20 tests.
65. **What is Benjamini-Hochberg?** FDR control; lets some false
    positives through to gain power; preferred when you have many
    tests.
66. **What is bootstrap CI?** Resample with replacement N times;
    compute the metric each time; percentile interval. Works for
    metrics with no closed-form distribution.
67. **What is Cohen's d?** Standardized mean difference (mean gap
    / pooled SD). Sized in SD units, comparable across studies.
68. **When to prefer Hedges' g over Cohen's d?** Small samples
    (n < 50); g applies a bias correction.
69. **What is Cliff's delta?** Non-parametric effect size for
    ordinal data; range [-1, 1].
70. **Why power analysis?** Determines sample size to detect an
    effect of operationally important size; avoids underpowered
    "no effect found" claims.
71. **What is post-hoc power?** Power computed after the fact from
    observed effect. Critiqued because it's a function of the
    p-value; useful only as a diagnostic.
72. **What is MDE?** Minimum detectable effect: the smallest gap
    your design can find at the requested power.
73. **What is ECE?** Expected calibration error: weighted average
    gap between predicted probability and observed frequency.
74. **What is Brier score?** Mean squared error between predicted
    probability and 0/1 outcome; decomposes into reliability,
    resolution, uncertainty.
75. **Platt vs isotonic calibration?** Platt: logistic; assumes
    monotone S-shape miscalibration. Isotonic: monotone but more
    flexible; needs more data.
76. **What is PSI?** Population Stability Index; industry standard
    for drift detection on continuous variables. < 0.10 stable,
    > 0.25 retrain.
77. **What is the Kolmogorov-Smirnov test?** Compares two empirical
    CDFs; rejects equality if their maximum gap is too large.
78. **What is repeated k-fold?** Multiple independent k-fold splits;
    reduces variance from a single random split.
79. **What is nested CV?** Outer loop estimates generalization;
    inner loop selects hyperparameters; standard for honest
    hyperparameter-selection error estimation.
80. **Why does single-test-set tuning bias generalization?** You
    select on the test set; performance estimate is no longer
    unbiased. Use held-out validation or nested CV.

## 5. Production, canary, drift, rollback

81. **What is a shadow run?** Send every request to candidate in
    parallel without returning its output to user; offline compare.
82. **What is a canary?** Route a small fraction of real traffic to
    candidate; gate on rolling-window quality and latency before
    promoting.
83. **Why is per-percentile latency monitoring required?** Means
    hide tails; users feel P95, not the mean.
84. **What is consecutive-breach rollback?** Rollback only after N
    consecutive bad windows; prevents flip-flops on noise.
85. **What is the cooldown after rollback?** A minimum window of no
    re-promotion attempts after a rollback; prevents loops.
86. **How does traffic split work?** Hash-based on request id; same
    user / request gets the same arm across retries.
87. **What is offline-online alignment?** Spearman / KS comparison
    between offline benchmark metrics and online observed metrics on
    the same systems; alignment > 0.7 expected.
88. **What is query distribution drift?** Today's queries differ from
    the reference distribution; detect via KS / PSI on length and
    embedding centroid.
89. **What is embedding drift?** The embedding model's outputs
    shift between versions; silent dep upgrades cause this.
90. **What is temporal performance drift?** Per-week cohort metric
    degrades even when inputs look stationary.
91. **What is auto-curation?** Mine low-confidence or
    high-disagreement production cases to add to the next golden
    set; closes the offline-online loop.
92. **Why version-pin embedding libraries?** Minor releases can
    change dimension or output; pin to exact patch.
93. **What is the role of reproducibility snapshots?** Code SHA, pip
    freeze, hardware, dataset SHA256; lets future-you recreate the
    run.
94. **What is the cost-of-a-false-positive in a rollback alarm?**
    Engineering disruption + lost time. Set thresholds high
    enough that flake doesn't trigger; that's why we have
    consecutive-breach.
95. **What is the cost-of-a-false-negative?** Bad release ships;
    users see degraded service. Adjust thresholds accordingly.
96. **What is the role of feature flags here?** They are the cheap
    rollback mechanism: candidate behind a flag is one config flip
    away from being disabled.
97. **What is the difference between deploying a model and
    releasing it?** Deployment: artifact moves to production
    infrastructure. Release: traffic starts flowing to the artifact.
    Canary lives between the two.
98. **What is the role of regression CI relative to canary?** CI
    catches known regressions before deploy. Canary catches
    unknown unknowns in real traffic.
99. **What is shadow vs A/B vs canary?** Shadow: candidate
    invisibly co-served. A/B: real users split, sample stats
    on outcomes. Canary: progressive rollout with gates.
100. **What is a graduated rollout?** 1% -> 5% -> 25% -> 50% ->
    100%; each step gated on canary green.

## 6. Failure modes, attacks, ethics

101. **What is prompt injection?** Adversarial input that smuggles
    instructions to the LLM; can come via the query or via
    retrieved evidence.
102. **What is the difference between direct and indirect prompt
    injection?** Direct: user types it. Indirect: appears in a
    document the model retrieves.
103. **Why is the corpus a threat surface?** Anything indexed can be
    used to attack a judge or generator that treats retrieved text
    as authoritative.
104. **What is jailbreaking?** Coaxing the model past safety
    filters with creative prompts.
105. **What is ground-truth leakage?** The candidate answer copies
    the expected answer verbatim because the model saw the gold set
    in training. Detect via long-substring overlap.
106. **What is data contamination?** Training data overlaps test
    data; inflates apparent quality. Audit via SHA256 of test items
    against training shard manifests.
107. **What is must-not-say?** Substrings forbidden by policy /
    regulation; any occurrence is a release blocker.
108. **What is PII leakage risk?** Generated answer reveals personal
    data not in query/evidence. Detect via named entity scan
    against a sensitive entity list.
109. **What is the role of an audit log?** Every promotion /
    rollback / config change recorded with operator id; needed
    for regulatory and post-mortem.
110. **What is the role of red-teaming?** Adversarial humans try to
    break the system before users do; complements automated
    attack detectors.
111. **What is alignment in this context?** Whether the model's
    behavior matches the designer's intent; sometimes a fine line
    from safety.
112. **Why is "looks correct" the worst metric?** It's a recipe for
    confident hallucinations. Insist on grounding + citation.
113. **What is the role of refusal training?** Train the model to
    abstain when evidence is insufficient; measured via under-
    refusal rate.
114. **How to handle adversarial paraphrase attacks?** Add
    paraphrase perturbations to the judge stability suite;
    require stability rate > 0.95.
115. **What is the threat model assumption for judges?** The
    candidate answer can be adversarial; the evidence corpus can
    be adversarial; only the system prompt and golden set are
    trusted.
116. **What ethics constraints apply to user data?** Hash any user
    identifier in traces; never log full PII; comply with
    region-specific retention rules.
117. **What is the role of a public benchmark?** Comparability across
    teams. But: benchmark contamination is a real risk; treat
    public sets as known-leaked.
118. **What is fairness auditing in RAG?** Per-demographic / per-
    domain subgroup performance breakouts; flag any subgroup that
    materially underperforms the overall.
119. **Why publish a model card?** Documents intended use,
    out-of-scope use, known biases, and operational constraints;
    reduces misuse.
120. **What is the closing principle for a RAG validation
    framework?** Trust no single metric; require gated
    multi-metric agreement; force calibration of every automated
    rater; preserve a per-stage trace for forensic attribution.
