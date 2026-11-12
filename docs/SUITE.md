# Reproducibility suite (rmg_suite_v093)

Run with `run_suite.py` from the repo root. All scripts write to `<repo>/results/`.

## Components

| Component | What it tests | Main scripts |
|---|---|---|
| fingerprints | Broad predicate fingerprints, cross-model | `hsg_fingerprint_qwen.py`, `hsg_fingerprint_mistral.py`, `hsg_fingerprint_gemma.py` |
| affordance | Predicate-licensed affordance fields (Exp B) | `hsg_affordance_test.py`, `hsg_exp_b.py` |
| semiotic | Geometry margin predicts generation | `semiotic_accuracy_prediction.py`, `hsg_semiotic_gemma.py`, `hsg_generation.py` |
| idiom | Idiom vs. literal, cross-model (`--model` flag) | `hsg_idiom_vs_literal.py` |
| ordvec | Ordinal Quotient Preservation (Nelson Spence) | `hsg_ordvec_fiber_test_v2.py`, `hsg_ordvec_fiber_test_v3.py`, `hsg_ordvec_ablations.py`, `hsg_ordvec_layer_role_test_2.py` |
| ablations | Relation vs. control | `hsg_ablations.py` |
| training | From-scratch training dynamics | `hsg_wiki_train.py` |
| corpus | Corpus-linguistic corroboration (no GPU) | `wiki_affordance_field_test.py` and the relation-vs-control ablation |
| annotation | Blind human annotation study | `make_annotation_sheet.py`, `analyze_human_annotations.py` |

The annotation component isn't part of the default full run; call it with `--component annotation`.

Supporting and exploratory: `context_distance_stress_test.py`, `context_type_stress_test.py`, `prefill_patch_logit_test.py`, `affordance_steering_test.py`, `run_exp4.py`, `hsg_pragmatic_pair_test.py`, `hsg_identical_target_context_test.py`, `hsg_clarity.py`. Superseded scripts go in `legacy/` (for example `ordvec_overlap_test.py`, `ultimate_diagnostic.py`).

## Shared helpers (added by Nelson Spence)

- `scripts/stats.py`: deterministic bootstrap CIs, empirical p-values that can't come out as zero, paired sign-flip tests, paired t, BH-FDR. Active scripts use it in place of their own copies.
- `scripts/hf_cache.py`: points model loading at the real Hugging Face hub cache (`HF_HUB_CACHE`, or `$HF_HOME/hub`, or `~/.cache/huggingface/hub`).

See [`SUITE_CHANGELOG.md`](SUITE_CHANGELOG.md) for the full change log. He changed no experimental design, data, thresholds, metrics, prompts or models.

## Setup

```bash
pip install -r requirements.txt
huggingface-cli login          # Gemma-7B is gated
export HF_HOME=/path/to/large/disk/.cache/huggingface   # the models need ~15 GB each
```

## Smoke test (about 20 minutes)

```bash
python3 run_suite.py --dry-run
python3 run_suite.py --component corpus
python3 run_suite.py --component ordvec
```

If those land files in `results/`, the rest follow the same pattern.

## Human annotation

```bash
python3 run_suite.py --component annotation
```

Run it only after all three semiotic runs exist (`*_qwen`, `*_mistral`, and the Gemma folder). It produces:

- `annotation_form_items.csv`: blind sheet, 150 rows balanced across models
- `annotation_key_private.csv`: private key (model, margin, keyword label). Keep this out of git and away from annotators.
- `annotation_instructions.txt`: rubric, with guidance on hard cases (companion vs. covenantal, moral vs. companion)
- `training_examples.txt`: 8 worked examples, excluded from analysis

Give the form, instructions and examples to two or three annotators. Then:

```bash
python3 scripts/analyze_human_annotations.py --responses R01.csv R02.csv R03.csv
```

Report the **per-model** AUCs from its output. Qwen's margins run about 10× the others', so the pooled AUC mixes scales and isn't meaningful on its own.

It reports Krippendorff's alpha (ordinal) as the main agreement metric, pairwise Cohen's kappa, class-level agreement, and a per-model breakdown. Thresholds: `median` (default), `strict` or `lenient`. It also suggests paper sentences for each outcome.

## Before you push

- Search for `/home/ubuntu`, `/opt/dlami` and `~/hsg` and fix any left in live code. In June, the last ones were in exploratory scripts and two docstrings.
- Don't commit model weights, Hugging Face tokens, or the private annotation key.
- Keep result JSON files under `results/`. They're small and they're the evidence.
