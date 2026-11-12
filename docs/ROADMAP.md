# Roadmap and known issues

Status as of November 2026. Open items are listed honestly so anyone replicating this work knows where the soft spots are.

## Next

- [ ] **Blind human annotation.** The kit is ready and needs no GPU: `python3 run_suite.py --component annotation` exports 150 balanced items (50 per model, no shared text). Report Krippendorff's alpha and Cohen's kappa. Until this is done, the behavioral AUCs rest on a keyword classifier.
- [ ] **Fix the layer-role generation score.** `geo_margin()` in `scripts/hsg_ordvec_layer_role_test_2.py` measures the gap between the two nearest class centroids without checking which class is expected. It needs to be similarity to the expected centroid minus the best wrong one. After the fix, rerun with several generations per probe; two recovered runs show the 12-sample score swings by 0.25 per layer from sampling alone (`results/provenance/`). The Q_full and Q_ordvec columns aren't affected.
- [ ] **Exp B (predicate-licensed fields) on Mistral and Gemma.** Qwen only so far.
- [ ] **Layer sweep on Mistral and Gemma.** Does the mid-network peak seen in Qwen generalize?
- [ ] **Bigger idiom set.** Three phrases today, and "spill the beans" is near zero in every model. Ten to fifteen would make the result solid.
- [ ] **Directional margin at scale.** The 39% vs 14% pilot has 25 cases. The next step is 20+ scenarios with human annotation.
- [ ] **Reverse-inference reconstruction.** Designed and partly run.

## Known issues in working paper v0.9.3

- §4.2 defines the residual fingerprint against *see* only. The cross-model fear result uses an extra step that also subtracts the mean over predicates ([METHODS.md §3](METHODS.md)); the paper should say so.
- Fingerprint permutation tests use 1,000 permutations, so report p < 0.001 there, not p < 0.0002.

- The Qwen semiotic Mann-Whitney p is reported as "~0.0002". Recomputed from the recovered rows, it's 4.2e-8.
- The §8.3 layer table's Gen AUC column and the "three processing regimes" paragraph rest on the layer-role score above. Don't rely on them yet.
- The OrdVec K-sweep text and table disagree at K = 256 and K = 180.

## Data recovery notes

- The Qwen semiotic run was overwritten by a later Mistral run (the script used one filename for every model) and was recovered from a surviving CSV in November 2026. The summary was recomputed from its rows; the same method reproduces the Mistral and Gemma summaries exactly. The script now tags output by model.
- The wiki affordance field test and the relation-vs-control ablation outputs aren't in `results/`. Their numbers are reported in the paper.
- The directional-margin, matched-pairs and dimension-saturation experiments (June 20, 2026) aren't in this suite yet.
