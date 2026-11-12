# How it got here

## The idea that failed (spring 2026)

It started with a guitar. When you bend a note on a Strat, you're not playing where you land. You're playing the movement.

The first theory, called HSG (Hidden State Geometry), said language works the same way: meaning lives in the curve a sentence traces through a model's hidden states. Torsion would mark a double meaning. Curvature would separate irony from praise.

We built the math and ran it. Six metrics at 0.5B scale came back flat, all p > 0.20. The end-of-sequence spike swamped everything.

A few things did survive: the n−2 layer invariant, a tiny but real Alignment Curvature Shift, and relational geometry showing up in a model trained from scratch (wife and dog closer than pizza under "love"). That last one pointed somewhere new.

## Predicate-conditioned affordance geometry (June 2026)

The new question was whether a predicate (love, fear, care) changes how a model arranges the things it could apply to. This was called PAG, Predicate-Conditioned Affordance Geometry, and the paper was "Words Are Not Containers," v0.7.

The validation suite was built on an EC2 instance (Ubuntu, A10G GPU):

1. Broad predicate fingerprints: done
2. Predicate-licensed affordance classes: done
3. Geometry predicts generation: Qwen AUC 0.778
4. Context distance / affordance capture: done (p = 0.037)
5. Idiom vs. literal: run later

Around June 4–5: cross-model replication on Mistral-7B and Gemma-7B. Fingerprints held in all three. Fear survived the neutral-baseline subtraction in all three. Semiotic accuracy replicated on Mistral (0.700) and Gemma (0.668). The relation-vs-control corpus ablation came in at p = 0.0002 for all three relation words.

## RMG and the reproducibility suite (June 8, 2026)

PAG was renamed **Relational Meaning Geometry (RMG)** throughout. The paper went from v0.9.1 to v0.9.5 in one session:

- Idiom-vs-literal expanded to all three models.
- OrdVec section rewritten with Nelson's v2 fiber sweep, K-sweep and ablation tables; test renamed "Ordinal Quotient Preservation."
- "p = 0.0000" corrected to "p < 0.0002" everywhere.
- "Three regimes" softened to "possible division of representational roles."
- References went from 8 to 26.

The reproducibility package `rmg_suite_v093.tar.gz` was built and smoke-tested on EC2: portable paths, all output to `results/`, `run_suite.py` orchestrator, the three missing OrdVec scripts added, MIT license, CITATION.cff, and a human-annotation pipeline with Krippendorff's alpha.

## The arXiv wall (June 2026)

The paper stalled at arXiv: no endorsement came through, and a later submission was rejected.

## Peer review rounds (mid to late June 2026)

v0.96 working paper. A condensed short paper went through v0.4, v0.5 and v0.6 with referee-style reviews. Reviews from Gemini, ChatGPT and a deep-research tool were cross-checked. Gemini's claim about the byline couldn't be verified, and two of the AI reviewers dropped the "seeds that reached geometric differentiation" qualifier. Hernandez et al. (linear relational embeddings) was confirmed as a real citation. Christ et al. was flagged as a live alternative explanation.

June 20 was one long afternoon:

- Larry asked why AI doesn't talk in idioms. That led to the generation test.
- Jason said to fix the input instead of the output. That led to the distractor design.
- Jason flagged the possessive confound. That led to matched pairs, clean at 5 of 5 layers.
- The directional-margin idea came out of a v3 failure, where relation substitution had high geometry pointing the wrong way.
- Result: 39% vs 14% relation preservation, same model.

Nelson's dimension-saturation work landed the same session.

## Toward a systems framing (autumn 2026)

The work was reframed for an engineering audience: RMG as a pre-generation assurance signal, a check that runs inside a system and flags when a language-model component has lost the relation a prompt expressed, before anything goes out. The directional-margin pilot and the matched-pairs control were folded in, and the failed curvature idea kept its place in the story.

## The repository (November 2026)

The suite, results and docs were brought together in this repository. Along the way the lost Qwen semiotic run was recovered from a surviving CSV, a filename bug that had caused the loss was fixed, and a scoring bug in the layer-role test was found and documented.
