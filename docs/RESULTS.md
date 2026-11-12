# Results

Everything recorded across the project's working sessions (June–November 2026). Where two sessions report different numbers for the same thing, both are shown and flagged.

Models: Qwen2-7B (`Qwen/Qwen2-7B`), Mistral-7B (`mistralai/Mistral-7B-v0.1`), Gemma-7B (`google/gemma-7b`). Embedding instrument: all-MiniLM-L6-v2.

Permutation p-values written as "< 0.001" (1,000 permutations; fingerprints) or "< 0.0002" (5,000; idiom and corpus tests) are floors: no shuffled run beat the real one. Earlier drafts wrote these as "p = 0.0000".

---

## 1. Broad predicate fingerprints

Does a predicate reorganize object geometry more than a random partition would?

| Model | Verb | Ratio | p | Cohen's d |
|---|---|---|---|---|
| Qwen2-7B | love | 1.255 | < 0.001 | 1.200 |
| Qwen2-7B | care | 1.301 | < 0.001 | 0.898 |
| Qwen2-7B | fear | 1.238 | < 0.001 | 0.945 |
| Qwen2-7B | hate | 1.272 | < 0.001 | 1.392 |
| Qwen2-7B | own | 1.247 | < 0.001 | 1.154 |
| Qwen2-7B | eat | 1.262 | < 0.001 | 0.884 |
| Mistral-7B | love | 1.281 | < 0.001 | 1.310 |
| Mistral-7B | care | 1.253 | < 0.001 | 1.411 |
| Mistral-7B | fear | 1.242 | < 0.001 | 1.006 |
| Mistral-7B | hate | 1.290 | < 0.001 | 1.156 |
| Gemma-7B | love | 1.296 | < 0.001 | 1.328 |
| Gemma-7B | care | 1.284 | < 0.001 | 1.208 |
| Gemma-7B | fear | 1.243 | < 0.001 | 1.078 |
| Gemma-7B | hate | 1.290 | < 0.001 | 1.320 |

Verified against `results/fingerprint_*/fingerprint_results.json`. All 9 verbs (see, love, need, fear, eat, own, remember, care, hate) beat random partitions in all three models. d ranges: Qwen 0.77–1.39, Mistral 0.83–1.41, Gemma 0.97–1.33. Fingerprints were read at block 26 of 28 (Qwen, Gemma) and block 30 of 32 (Mistral).

> **Correction:** earlier summaries listed Mistral's love and fear d as 1.328 and 1.078. Those are Gemma's numbers; the Mistral file says 1.310 and 1.006. Earlier summaries also gave Mistral's range as 0.989–1.411; the file's lowest is `eat` at 0.828.

Qwen-derived *predictions* about specific fingerprint structure failed 4 of 8 on Mistral. The geometry exists across model families, but its exact shape is model-specific.

## 2. Residual fingerprints

Definitions are in [METHODS.md §3](METHODS.md). Two stages:

- $F_V$ subtracts the neutral *see* baseline and the overall expansion or compression.
- $\tilde F_V$ also subtracts the mean fingerprint of all eight predicates, leaving what is specific to $V$. Against *see* alone, Mistral shows nearly the same pattern for every predicate (pull companion/hedonic, push companion/moral), so this second step is what separates one predicate from another.

Strongest pull / push, recomputed from `results/fingerprint_*/fingerprint_results.json`:

| Predicate | Model | $F_V$ (vs. *see*) | $\tilde F_V$ (predicate-specific) |
|:--|:--|:--|:--|
| fear | Qwen | companion/companion · moral/moral | **companion/hedonic** · covenantal/covenantal |
| fear | Mistral | companion/hedonic · companion/moral | **companion/hedonic** · covenantal/covenantal |
| fear | Gemma | companion/hedonic · companion/moral | **companion/hedonic** · companion/companion |
| care | Qwen | companion/companion · hedonic/hedonic | moral/moral · covenantal/hedonic |
| care | Mistral | companion/hedonic · companion/moral | companion/hedonic · companion/companion |
| care | Gemma | hedonic/moral · companion/hedonic | moral/moral · covenantal/hedonic |
| love | Qwen | companion/companion · hedonic/moral | companion/companion · hedonic/moral |
| love | Mistral | companion/hedonic · companion/moral | moral/moral · covenantal/moral |
| love | Gemma | companion/companion · companion/moral | companion/companion · companion/moral |

- **Fear:** on $\tilde F$, the strongest pull is companion/hedonic in all three models, and Qwen and Mistral also share the strongest push. On $F$ alone, Mistral and Gemma match exactly and Qwen differs.
- **Care:** Qwen and Gemma match exactly on $\tilde F$; Mistral differs.
- **Love:** Qwen and Gemma pull companion/companion; Mistral differs.

Earlier write-ups described this as "subtracting the neutral *see* baseline". The cross-model comparison actually uses $\tilde F$, which also removes the mean over predicates. The paper's §4.2 defines $F$ only, so it should add the second step.

Qwen's raw distances run about 10× larger than Mistral's and Gemma's (a scaling artifact in how the Qwen values were stored). Compare directions across models, not magnitudes.

## 3. Predicate-licensed affordance fields (Experiment B), Qwen2-7B

| Predicate | Ratio | p | d | Result |
|---|---|---|---|---|
| fear | 1.108 | < 0.001 | 0.634 | confirmed |
| care | 1.223 | 0.017 | 0.569 | confirmed |
| love | 1.220 | < 0.001 | 0.953 | confirmed |
| own | 1.238 | < 0.001 | 0.690 | confounded by prompt likelihood (NLL) |
| eat | 1.088 | 0.101 | 0.229 | not confirmed |

Love, fear and care classes beat random partitions at z = 3.8–8.4 and are not explained by prompt likelihood. The `eat` failure was reframed in review as a likely cultural-heterogeneity finding, not a dead end.

**Qwen only.** Exp B has not been run on Mistral or Gemma.

## 4. Semiotic accuracy: geometry predicts generation

Win condition: AUC > 0.70, Mann-Whitney p < 0.05, positive logistic coefficient.

| Model | Accuracy | AUC | Mann-Whitney p | Logistic coef |
|---|---|---|---|---|
| Qwen2-7B | 69.2% | 0.778 | 4.2e-8 | +0.176 |
| Mistral-7B | 69.5% | 0.700 | 0.000154 | +1.295 |
| Gemma-7B | 67.2% | 0.668 | 0.000699 | +1.062 |

Layers used: Mistral layer 11 of 32, Gemma layer 10 of 28 (from the summary files). n scored: Mistral 128, Gemma 137; 12 generations per prompt.

> **Qwen run recovered (Nov 12, 2026).** The script once wrote the same filenames for every model, and the Mistral run overwrote the Qwen output. A surviving copy of the Qwen generations CSV has been restored as `semiotic_accuracy_generations_qwen.csv`, and the summary recomputed from its rows: 146 scored, accuracy 69.2%, AUC 0.7779, logistic coefficient +0.1765, mean margin 8.43 (correct) vs 1.98 (wrong). The same recomputation reproduces the Mistral and Gemma summaries exactly. The JSONL was rebuilt from the CSV. The layer isn't recorded in the CSV; by the script's rule it was hidden_states[10] (Mistral 11, Gemma 10). Qwen margins are about 10× the others' (the known scaling artifact), which is why its logistic coefficient is small.

- Gemma points the right way but sits under the 0.70 bar.
- **Qwen p-value, settled:** Mann-Whitney p = 4.2e-8 (one-sided, recomputed from the recovered rows). Paper v0.93's "~0.0002" is wrong for Qwen; correct it. The early session note of 4.2e-8 was right.
- **Qwen AUC, settled:** 0.778 is the semiotic test (`semiotic_accuracy_prediction.py`, full probe set). The 0.686 "Gen AUC" at layer 6 is a different experiment: the OrdVec layer-role sweep, 12 probes, "I love X because" only.
- **Problem in the layer-role Gen AUC column.** In `hsg_ordvec_layer_role_test_2.py`, `geo_margin()` returns the gap between the two nearest centroids without checking which class is expected, even though its docstring says "correct centroid minus nearest wrong." That makes it a confidence score, not a correctness margin. In `results/hsg_ordvec_layer_role_results.json`, 26 of 29 Gen AUC values are below 0.5 (mean 0.34), which is what you'd expect from that. The layer 8 value of 0.200 and the layer 6 "peak" of 0.686 both come from this. The main semiotic test computes its margin correctly (expected class, leave-one-out), so the cross-model AUCs aren't affected. The fix is one line plus a GPU rerun; until then, the Gen AUC column and the "early layers are most predictive of generation" reading shouldn't be used.
- **The Gen AUC column also doesn't reproduce.** Two recovered Qwen runs of the layer-role test (`results/provenance/`) give identical Q_full and Q_ordvec at every layer, but Gen AUC shifts by 0.25 per layer on average (layer 9: 0.806 in one run, 0.200 in the other). It rests on 12 sampled continuations, so it's mostly sampling noise. Even after the margin fix, it needs many generations per probe (the main semiotic test uses 12 per prompt over the full probe set) before it can say anything about layers. The Q_full and Q_ordvec columns are solid.
- Scored by keyword classifier. Human annotation pending; the blind sheet can now be exported correctly (no GPU needed).

## 5. Idiom vs. literal

Identical phrases forced into idiomatic or literal readings by context.

| Model | Δ | Permutation p |
|---|---|---|
| Qwen2-7B (block 21) | +0.1413 | < 0.0002 (Mann-Whitney p = 9.0e-6) |
| Mistral-7B (block 24) | +0.1817 | < 0.0002 (Mann-Whitney p = 8.6e-6) |
| Gemma-7B (block 21) | +0.1571 | < 0.0002 (Mann-Whitney p = 3.8e-5) |

Verified against `results/idiom_vs_literal_*.json`.

By phrase (Qwen): "break the ice" +0.20, "hit the road" +0.21, "spill the beans" +0.012. "Spill the beans" is near zero in all three models, so two phrases carry the result. Expanding to 10–15 phrases would make this solid.

The Gemma and Mistral runs were re-verified on the hardened suite and matched the paper exactly.

Related finding: for unseen (novel) idioms, a paragraph is the minimum context at which idiomatic and literal readings separate geometrically.

## 6. Context distance / affordance capture

Morally loaded context captures covenantal/companion geometry before the object arrives. Neutral vs. loaded: p = 0.037.

## 7. Prefill patch / next-token logit intervention

Moral → hedonic patching worked cleanly; the other directions were partial. Layer sweeps for patching are still pending.

## 8. Layer sweep (Qwen2-7B only)

Love peaks around layer 5, care around 8, fear around 12. Full-vector class membership tops out at layer 8; ordinal agreement peaks at layer 22. An earlier session placed the cross-model convergence around layers 21–22 of 28.

Open question: does the mid-network peak hold for Mistral and Gemma? The layers each test actually used are now recorded above (fingerprints, semiotic and idiom sections); the paper should state them.

## 9. Directional margin pilot (Qwen2-7B, same model for geometry and generation)

Five everyday scenarios (leaving a long-held job, exhaustion after triple shifts, a meeting that fell apart, a sudden new idea, avoiding responsibility). Each has a correct-relation anchor and a plausible distractor. Anchor separation: 0.15–0.38 cosine distance in every scenario. Five conditions each (clean, repaired, emotional mismatch, topic drift, relation substitution) for 25 cases.

- Margin toward the correct anchor: **39%** of continuations kept the relation.
- Margin toward the distractor: **14%**.
- Mean margin +0.18 for clean prompts, −0.18 under relation substitution.

Pilot only: 25 single-generation cases, model-based judge. Next step is 20+ scenarios with human annotation. The n per group and test statistic still need to go in the paper.

The 2×2 distractor design behind it separates relational framing from figurative instruction as independent variables.

## 10. Matched-pairs syntax control (Jason's possessive confound)

Relation dominates possessive syntax at 5 of 5 layers. Syntax swaps account for only 13–18% of natural relational variation.

## 11. Corpus-linguistic corroboration (no language model)

Wikipedia corpus. Relation windows around love/care/like compared with matched control word sets from the same corpus, measured by Jensen-Shannon divergence against baseline category fields.

- Relation-vs-control ablation: love Δ = +0.224, care Δ = +0.235, like Δ = +0.214, combined Δ = +0.180, all p = 0.0002 (from paper v0.93 §9).
- Wiki affordance field test: done. Its output, and the relation-vs-control ablation's output, aren't in this repo's `results/` (only `wiki_results/training_log.jsonl` is). Copy them over from EC2.

Labelled "Preliminary Corpus-Linguistic Corroboration" in the paper.

## 12. From-scratch training (44.9M-parameter model)

Relational geometry emerges in a model trained from scratch on a 288-article Wikipedia corpus: wife and dog cluster closer than pizza under "love," across four independent training runs. Seeds showing all-zero distances are runs that hadn't reached geometric differentiation yet, so any claim needs the qualifier "in seeds that reached geometric differentiation."

## 13. Ordinal Quotient Preservation (OrdVec, Nelson Spence's method)

OrdVec throws away magnitudes and keeps only the ranking of activations. In the toy runs, about 95% of relational structure survived, which suggests the relational signal lives in the ranking structure, not the magnitudes. That's what would make a runtime monitor cheap enough to deploy.

Paper's summary line: *"The OrdVec signal is real, but it is not magic: it depends on meaningful labels, real activations, the predicate frame, and the right layer."*

The 95% figure comes from a sweep and carries a sweep-selection caveat.

> **Known issue:** in the working paper the text says violations climb at K = 256, but the table shows 5 (the same as K = 64), and K = 180 shows 1.00 agreement next to 10 violations.

## 14. Dimension saturation (Nelson's hypothesis)

- MiniLM (384 dimensions) showed roughly 10× higher efficiency per dimension than Qwen2-7B on short trust/grieve stimuli.
- Invariant-subspace split: near-zero placeholders in small models, stable non-zero "unactivated fibers" in large ones. Replicated across two stimulus sets.
- Content-type test (n = 15 per group): MiniLM resolves vocabulary-marked relational contrasts about 3× more cleanly than narrative-embedded ones at matched length, p = 0.000009, r = −0.956.

This reframed the hypothesis from character count to content structure.

## 15. Sentence encoders under relation flips

Sentence-embedding encoders separate broad narrative classes but fail when the scene is held constant and only the governing relation flips (for example, leaving a job as loss vs. leaving a job as freedom). Leave-one-out accuracy goes as low as 0.04. For systems engineers, that's an integration risk in any retrieval pipeline built on embeddings.

## 16. Alignment Curvature Shift and the n−2 invariant

Survivors of the original curvature theory:

- Residual-stream curvature peaks 2–3 layers before output, across four architectures.
- Alignment Curvature Shift: d = 0.050, p < 0.0001. Tiny but real.

## 17. The original curvature theory (null result)

Meaning as trajectory curvature, torsion as a double-entendre index, utterance-level curvature separating pragmatic categories (irony, defensiveness, faint praise). Six metrics at 0.5B scale, all p > 0.20. The end-of-sequence spike dominated everything. This is now an appendix and a sentence in the SysCon introduction.

## 18. Reverse-inference reconstruction (Jason's design), partial

Designed and partly run in June 2026. Qwen can guess a plausible source question from an answer alone, but its guesses run generic and templated, which limits it as an instrument for fine-grained ablation.
