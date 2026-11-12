> Archived README from the v0.9.3 suite (June 2026). Kept for its component-to-file map. The current README is at the repo root.

# Relational Meaning Geometry — Reproducibility Suite v0.9.3

## Words Are Not Containers: Relational Meaning Geometry and Semiotic Accuracy in Language Models

**Mike Singleton**  
Working paper v0.9.3 · June 2026  

---

## Overview

This package contains all scripts, results, and the paper draft for the
Relational Meaning Geometry (RMG) research program as of v0.9.3. Every
number in the paper can be regenerated from the scripts listed here.

### Research program components

| # | Component | Primary script(s) | Result files |
|---|-----------|-------------------|-------------|
| 1 | Training dynamics (small model) | `hsg_wiki_train.py` | `results/wiki_results/training_log.jsonl` |
| 2 | Affordance geometry — Exp A | `hsg_affordance_test.py` | `results/affordance_results/` |
| 3 | Affordance geometry — Exp B | `hsg_exp_b.py` | `results/exp_b_results/` |
| 4 | Affordance ablations | `hsg_ablations.py` | `results/ablation_results/` |
| 5 | Cross-model fingerprints | `hsg_fingerprint_{qwen,mistral,gemma}.py` | `results/fingerprint_*/` |
| 6 | Semiotic accuracy prediction | `semiotic_accuracy_prediction.py`, `hsg_semiotic_gemma.py` | `results/semiotic_accuracy_results*/` |
| 7 | Idiom-vs-literal separation | `hsg_idiom_vs_literal.py` | `results/idiom_vs_literal_*.json` |
| 8 | Ordinal Quotient Preservation sweep | `hsg_ordvec_fiber_test_v2.py` | `results/hsg_ordvec_fiber_v2_results.json` |
| 9 | Layer role analysis | `hsg_ordvec_layer_role_test_2.py` | `results/hsg_ordvec_layer_role_results.json` |
| 10 | OQP ablation suite | `hsg_ordvec_ablations.py` | `results/hsg_ordvec_ablations_results.json` |
| 11 | Corpus-linguistic corroboration | `wiki_affordance_field_test.py`, `wiki_relation_vs_control_ablation.py` | `results/wiki_results/` |

---

## Package layout

```
rmg_suite_v093/
├── README.md
├── LICENSE
├── CITATION.cff
├── requirements.txt
├── run_suite.py              ← unified runner
├── .gitignore
│
├── scripts/                  ← all runnable scripts
│   ├── hsg_ordvec_fiber_test_v2.py     ← OQP sweep (primary)
│   ├── hsg_ordvec_fiber_test_v3.py     ← OQP sweep (mean-overlap variant)
│   ├── hsg_ordvec_layer_role_test_2.py ← layer role analysis
│   ├── hsg_ordvec_ablations.py         ← OQP ablation suite
│   ├── hsg_affordance_test.py
│   ├── hsg_exp_b.py
│   ├── hsg_ablations.py                ← affordance ablations (layer sweep, NLL, random)
│   ├── hsg_fingerprint_{qwen,mistral,gemma}.py
│   ├── semiotic_accuracy_prediction.py
│   ├── hsg_semiotic_gemma.py
│   ├── hsg_idiom_vs_literal.py
│   ├── hsg_wiki_train.py
│   ├── wiki_affordance_field_test.py
│   ├── wiki_relation_vs_control_ablation.py
│   └── [supporting scripts — see below]
│
├── results/
│   ├── hsg_ordvec_fiber_v2_results.json      ← OQP K-sweep, layers 20-27
│   ├── hsg_ordvec_layer_role_results.json    ← layer role analysis, all 27 blocks
│   ├── hsg_ordvec_ablations_results.json     ← 4 OQP ablations
│   ├── idiom_vs_literal_results_qwen.json    ← Qwen2-7B, block 21
│   ├── idiom_vs_literal_results_mistral.json ← Mistral-7B, block 24
│   ├── idiom_vs_literal_gemma.json           ← Gemma-7B, block 21
│   ├── affordance_results/
│   ├── exp_b_results/
│   ├── ablation_results/
│   ├── fingerprint_{qwen,mistral,gemma}7b/
│   ├── semiotic_accuracy_results/
│   ├── semiotic_accuracy_results_gemma/
│   ├── idiom_results/
│   ├── generation_results/
│   ├── prefill_patch_results/
│   ├── steering_results/
│   └── wiki_results/
│
├── corpus/
│   ├── wiki_corpus/corpus.jsonl          ← 288 Wikipedia articles (relational domains)
│   └── internet_corpus/wikitext_raw.txt  ← raw wikitext for corpus ablation (~9MB)
│
├── paper/
│   └── Words_Are_Not_Containers_v093.docx
│
└── legacy/
    ├── original_sources/     ← earlier script versions (provenance only)
    └── production_dfm/       ← deprecated flow-matching pipeline
```

---

## Environment

**Compute:** AWS EC2 A10G recommended (g5.xlarge or larger)  
**Model cache:** `/opt/dlami/nvme` (EC2 NVMe instance store — fast)  
**OS:** Ubuntu 22.04 (Deep Learning AMI)  
**Python:** 3.10+  
**CUDA:** 11.8+

```bash
pip install -r requirements.txt
export HF_TOKEN=your_token_here   # required for google/gemma-7b (gated)
```

**Models used:**

| Model | HuggingFace ID | Access |
|-------|---------------|--------|
| Qwen2-7B | `Qwen/Qwen2-7B` | public |
| Mistral-7B | `mistralai/Mistral-7B-v0.1` | public |
| Gemma-7B | `google/gemma-7b` | gated (requires HF approval) |

All 7B models loaded in 8-bit quantization (`bitsandbytes`). Each fits on a
single A10G (24GB VRAM) with room to spare.

---

## Quick start

```bash
# Dry run — see all commands without executing
python3 run_suite.py --dry-run

# Run a single component
python3 run_suite.py --component ordvec

# Full suite (~6-8 hours on A10G)
python3 run_suite.py
```

---

## Script reference

### Ordinal Quotient Preservation (new in v0.9.3)

**`hsg_ordvec_fiber_test_v2.py`** — *Primary OQP sweep*  
K-sweep (K=64, 128, 180, 256, 512) × layer sweep (layers 20–27) on
Qwen2-7B. Three quotient maps: Q_popcount (quartile baseline), Q_ordvec
(top-K activation bitmap, majority vote), Q_full (LOO cosine centroid).
Formally grounded in `FiniteQuotientSearch.lean` and `BitmapNull.lean`.
Writes `hsg_ordvec_fiber_v2_results.json`.

```bash
python3 scripts/hsg_ordvec_fiber_test_v2.py
```

**`hsg_ordvec_fiber_test_v3.py`** — *OQP sweep, mean-overlap variant*  
Same sweep as v2 but Q_ordvec uses mean overlap per class rather than
majority vote. Produces a complementary check on aggregation sensitivity.

```bash
python3 scripts/hsg_ordvec_fiber_test_v3.py
```

**`hsg_ordvec_layer_role_test_2.py`** — *Layer role analysis*  
Runs three measurements at every layer (1–27): Q_full accuracy, Q_ordvec
agreement with Q_full, and generative AUC (does the geometry margin predict
whether the generated completion matches the expected affordance register?).
Reveals the possible three-regime dissociation: gen-predictive signal peaks
early (~layer 6), Q_full ceiling reached at layer 8, Q_ordvec precision
peaks at layer 21–22. Writes `hsg_ordvec_layer_role_results.json`.

```bash
python3 scripts/hsg_ordvec_layer_role_test_2.py
```

**`hsg_ordvec_ablations.py`** — *OQP ablation suite*  
Four ablations falsifying the v2 result (layer=21, K=64, Q_ordvec 95%):  
1. Random class labels → mean top-1 = 0.262, p=0.000  
2. Random bitmaps → mean top-1 = 0.262, p=0.000  
3. Wrong predicate ("bought") → top-1 = 0.80, violations=21  
4. Shallow control (hidden_states[1]) → top-1 = 0.65, violations=30  
Writes `hsg_ordvec_ablations_results.json`.

```bash
python3 scripts/hsg_ordvec_ablations.py
```

---

### Core pipeline

**`hsg_wiki_train.py`**  
Trains a 44.9M GPT-2-style model from scratch on 288 Wikipedia articles.
Probes hidden states at every 300 steps for wife/dog/pizza. Produces the
training dynamics result (Section 5). The stable ordinal finding —
wife/dog < min(wife/pizza, pizza/dog) — emerges without supervision.

```bash
python3 scripts/hsg_wiki_train.py
```

**`hsg_affordance_test.py`**  
Experiment A: 20 objects × 9 predicates on Qwen2-7B, averaged across four
templates. Computes within-vs-between class distances and residual
fingerprints against the `see` baseline. Sections 6–7.

```bash
python3 scripts/hsg_affordance_test.py
```

**`hsg_exp_b.py`**  
Experiment B: predicate-licensed object fields. Tests internal affordance
subclass recovery for fear, own, care, love, eat. Section 7.

```bash
python3 scripts/hsg_exp_b.py
```

**`hsg_ablations.py`**  
Affordance ablations (distinct from OQP ablations): layer sweep across
predicates, random partition test (500 shuffles), NLL control. Section 8.
Note: this script does *not* contain the OQP ablations — those are in
`hsg_ordvec_ablations.py`.

```bash
python3 scripts/hsg_ablations.py --ablation all
python3 scripts/hsg_ablations.py --ablation layer_sweep
python3 scripts/hsg_ablations.py --ablation random_partition
python3 scripts/hsg_ablations.py --ablation nll_control
```

**`hsg_fingerprint_qwen.py`** / **`hsg_fingerprint_mistral.py`** / **`hsg_fingerprint_gemma.py`**  
Cross-model fingerprint replication. All three return p < 0.001. Section 10
(cross-model context). ~20 minutes each on A10G.

```bash
python3 scripts/hsg_fingerprint_qwen.py
python3 scripts/hsg_fingerprint_mistral.py
python3 scripts/hsg_fingerprint_gemma.py
```

**`semiotic_accuracy_prediction.py`**  
Semiotic accuracy prediction for Qwen2-7B and Mistral-7B. Reports accuracy,
AUC, Mann-Whitney p, Spearman rho, logistic coefficient. Section 10.

```bash
python3 scripts/semiotic_accuracy_prediction.py
```

**`hsg_semiotic_gemma.py`**  
Same as above, configured for Gemma-7B. Run separately.

```bash
python3 scripts/hsg_semiotic_gemma.py
```

**`hsg_idiom_vs_literal.py`**  
Idiom-vs-literal geometry separation. Three phrases (spill the beans /
break the ice / hit the road), three models. Section 11. Note: "spill the
beans" shows near-zero separation in all models — this is expected and
documented in the paper.

```bash
python3 scripts/hsg_idiom_vs_literal.py
python3 scripts/hsg_idiom_vs_literal.py --model mistralai/Mistral-7B-v0.1
python3 scripts/hsg_idiom_vs_literal.py --model google/gemma-7b
```

**`wiki_affordance_field_test.py`**  
Corpus-linguistic corroboration. Jensen-Shannon divergence over ±10-token
windows around love/care/like in Wikipedia. No GPU required. Section 9.

```bash
python3 scripts/wiki_affordance_field_test.py
```

**`wiki_relation_vs_control_ablation.py`**  
Ablation of corpus result: compares relation-word separation to matched
control-word windows. Rules out "any local window separates" explanation.

```bash
python3 scripts/wiki_relation_vs_control_ablation.py
```

---

### Supporting scripts

| Script | Purpose | Status |
|--------|---------|--------|
| `hsg_probe.py` | Single-file RMG probe, importable | active |
| `hsg_see_residual.py` | Cross-model see-residual fingerprint analysis | active |
| `hsg_generation.py` | Geometry-predicts-generation loop | active |
| `hsg_statistical_suite.py` | 30-seed bootstrap validation | active |
| `hsg_validation_package.py` | Validation harness | active |
| `affordance_steering_test.py` | Centroid steering (partial effect only) | active |
| `prefill_patch_logit_test.py` | Prefill patching (partial intervention) | active |
| `hsg_rigorous_diagnostic.py` | Diagnostic harness | active |
| `hsg_audit.py` / `hsg_coupled_audit.py` | Internal audit | active |
| `hsg_clarity.py` | Visualization helpers | active |
| `hsg_manifold_stress.py` | Manifold stress test | exploratory |
| `hsg_identical_target_context_test.py` | Context sensitivity | exploratory |
| `hsg_micro_llm_conflict_geometry.py` | Conflict geometry | exploratory |
| `hsg_hidden_state_contradiction_ladder.py` | Contradiction geometry | exploratory |
| `context_distance_stress_test.py` | Context distance under stress | exploratory |
| `context_type_stress_test.py` | Context type geometry | exploratory |
| `contextual_affordance_geometry_test.py` | Early affordance prototype | exploratory |
| `killer_prompt_audit.py` | Adversarial prompt audit | exploratory |
| `ultimate_diagnostic.py` | Full diagnostic sweep | exploratory |
| `run_continuous_eval.py` | Continuous eval harness | exploratory |
| `hsg_pragmatic_pair_test.py` | Pragmatic pair curvature | null result (see Appendix A) |
| `hsg_minimal_pair_curvature_test.py` | Minimal pair curvature | null result (see Appendix A) |
| `hsg_conditioned_suite.py` | Conditional flow architecture | deprecated |
| `hsg_production_pipeline.py` | Trajectory auditor | deprecated |
| `hsg_statistical_suite.py` | 30-seed bootstrap | deprecated path |
| `dirichlet-flow-matching.py` | DFM experiment | deprecated |
| `toy_flow_test.py` | CFM toy experiment | deprecated |
| `run_toy_cfm_experiment.py` | CFM runner | deprecated |
| `run_exp4.py` | Misc experiment | deprecated |
| `ordvec_overlap_test.py` | *Older sarcasm/pragmatic overlap test* — **not the OQP sweep** | superseded by `hsg_ordvec_fiber_test_v2.py` |

> ⚠️ **Note on `ordvec_overlap_test.py`**: This is an earlier script testing
> pragmatic pair overlap (sarcasm, faint praise, defensiveness). It is *not*
> the Ordinal Quotient Preservation sweep. Use `hsg_ordvec_fiber_test_v2.py`
> for the results in Section 16 of the paper.

---

## Corpus files

| File | Size | Source | Required for |
|------|------|--------|-------------|
| `corpus/wiki_corpus/corpus.jsonl` | ~2MB | Wikipedia (7 relational domains) | `hsg_wiki_train.py`, `hsg_affordance_test.py` |
| `corpus/internet_corpus/wikitext_raw.txt` | ~9MB | WikiText-103 subset | `wiki_affordance_field_test.py`, `wiki_relation_vs_control_ablation.py` |

The Wikipedia corpus (`corpus.jsonl`) covers: marriage, pets, food, religion,
family, friendship, philosophy. Assembled from Wikipedia plain-text extracts
using standard preprocessing (sentence splitting, stopword filtering).

The WikiText raw file is used only for the corpus-linguistic corroboration in
Section 9. It is not required for any GPU-based experiment.

---

## Execution order (full reproduction)

```bash
# 0. Set up environment
pip install -r requirements.txt
export HF_TOKEN=your_token_here

# 1. Corpus corroboration (no GPU, ~5 min)
python3 scripts/wiki_affordance_field_test.py
python3 scripts/wiki_relation_vs_control_ablation.py

# 2. Train small model (~2h on CPU or ~20min on GPU)
python3 scripts/hsg_wiki_train.py

# 3. Affordance geometry — Exp A and B (~1h)
python3 scripts/hsg_affordance_test.py
python3 scripts/hsg_exp_b.py
python3 scripts/hsg_ablations.py --ablation all

# 4. Cross-model fingerprints (~20min each)
python3 scripts/hsg_fingerprint_qwen.py
python3 scripts/hsg_fingerprint_mistral.py
python3 scripts/hsg_fingerprint_gemma.py

# 5. Semiotic accuracy prediction (~45min)
python3 scripts/hsg_generation.py
python3 scripts/semiotic_accuracy_prediction.py
python3 scripts/hsg_semiotic_gemma.py

# 6. Idiom-vs-literal (~20min each)
python3 scripts/hsg_idiom_vs_literal.py
python3 scripts/hsg_idiom_vs_literal.py --model mistralai/Mistral-7B-v0.1
python3 scripts/hsg_idiom_vs_literal.py --model google/gemma-7b

# 7. Ordinal Quotient Preservation (~1h)
python3 scripts/hsg_ordvec_fiber_test_v2.py
python3 scripts/hsg_ordvec_layer_role_test_2.py
python3 scripts/hsg_ordvec_ablations.py

# Or: run everything via the unified runner
python3 run_suite.py
```

Estimated total GPU time on A10G: **6–8 hours** for the full suite.

---

## Key numbers (v0.9.3)

| Claim | Value | Source file |
|-------|-------|-------------|
| Cross-model fingerprint p (all 3 models) | p < 0.001 | `results/fingerprint_*/` |
| Semiotic accuracy AUC range | 0.668–0.778 | `results/semiotic_accuracy_results*/` |
| Idiom separation Δ — Qwen2-7B | +0.1413, p=8.98e-06 | `results/idiom_vs_literal_results_qwen.json` |
| Idiom separation Δ — Mistral-7B | +0.1817, p=8.62e-06 | `results/idiom_vs_literal_results_mistral.json` |
| Idiom separation Δ — Gemma-7B | +0.1571, p=3.84e-05 | `results/idiom_vs_literal_gemma.json` |
| OQP Q_ordvec top-1 (K=64, layer 21) | 95% | `results/hsg_ordvec_fiber_v2_results.json` |
| Q_full (LOO) accuracy | 90% | `results/hsg_ordvec_layer_role_results.json` |
| OQP violations at optimal config | 5 | `results/hsg_ordvec_fiber_v2_results.json` |
| OQP random-label ablation mean top-1 | 0.262, p < 0.001 | `results/hsg_ordvec_ablations_results.json` |
| OQP wrong-predicate ablation top-1 | 0.80 | `results/hsg_ordvec_ablations_results.json` |
| OQP shallow-control (hidden_states[1]) top-1 | 0.65 | `results/hsg_ordvec_ablations_results.json` |
| Gen-AUC peak layer | 6, AUC=0.686 | `results/hsg_ordvec_layer_role_results.json` |
| Q_full ceiling layer | 8 | `results/hsg_ordvec_layer_role_results.json` |
| Q_ordvec agreement peak layer | 22, agree=0.95 | `results/hsg_ordvec_layer_role_results.json` |
| Corpus Δ — love | +0.224, p=0.0002 | `results/wiki_results/` |

---

## Legacy directory

`legacy/original_sources/` contains earlier versions of core scripts kept
for provenance. Do not use for reproduction — use the top-level scripts in
`scripts/` instead.

`legacy/production_dfm/` contains the deprecated Dirichlet flow-matching
pipeline from an earlier research direction. The associated curvature
experiments produced null results, documented in Appendix A of the paper.

---

## Citation

```bibtex
@misc{singleton2026rmg,
  title   = {Words Are Not Containers: Relational Meaning Geometry
             and Semiotic Accuracy in Language Models},
  author  = {Singleton, Mike},
  year    = {2026},
  note    = {Working paper v0.9.3},
  url     = {https://github.com/singleton2787/words-are-not-containers}
}
```

See `CITATION.cff` for machine-readable citation metadata.

---

*Mike Singleton · June 2026*
