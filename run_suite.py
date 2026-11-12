#!/usr/bin/env python3
"""
run_suite.py — Relational Meaning Geometry unified runner
==========================================================
Mike Singleton — v0.9.3

Runs the full RMG validation suite in the correct order, or individual
components by name. Every number in the v0.9.3 paper can be reproduced
by running this script end-to-end.

Usage:
    python3 run_suite.py                    # full suite
    python3 run_suite.py --component all    # same as above
    python3 run_suite.py --component corpus
    python3 run_suite.py --component train
    python3 run_suite.py --component affordance
    python3 run_suite.py --component fingerprints
    python3 run_suite.py --component semiotic
    python3 run_suite.py --component idiom
    python3 run_suite.py --component ordvec
    python3 run_suite.py --dry-run          # print commands, don't run

Estimated total GPU time on A10G (g5.xlarge): 6-8 hours
Models required: Qwen/Qwen2-7B, mistralai/Mistral-7B-v0.1, google/gemma-7b

Environment:
    pip install -r requirements.txt
    export HF_TOKEN=your_token_here   # required for Gemma
"""

import argparse
import subprocess
import sys
import os
from pathlib import Path

SCRIPTS = Path(__file__).parent / "scripts"

COMPONENTS = {
    "corpus": {
        "description": "Corpus-linguistic corroboration (no GPU needed)",
        "steps": [
            (
                "wiki_affordance_field_test.py",
                "Wikipedia affordance field test — Jensen-Shannon divergence",
            ),
            (
                "wiki_relation_vs_control_ablation.py",
                "Relation-vs-control ablation — rules out topic confound",
            ),
        ],
    },
    "train": {
        "description": "Train small model from scratch, extract training dynamics",
        "steps": [
            (
                "hsg_wiki_train.py --all",
                "Train 44.9M GPT-2 on 288 Wikipedia articles, probe every 300 steps",
            ),
        ],
    },
    "affordance": {
        "description": "Experiment A (broad anchor) and Experiment B (licensed fields)",
        "steps": [
            (
                "hsg_affordance_test.py --model results/wiki_model",
                "Exp A — 20 objects × 9 predicates, clustering and residual fingerprints",
            ),
            ("hsg_exp_b.py", "Exp B — predicate-licensed affordance subclasses"),
            (
                "hsg_ablations.py --ablation all",
                "Ablations — layer sweep, random partition, NLL control",
            ),
        ],
    },
    "fingerprints": {
        "description": "Cross-model predicate fingerprint replication",
        "steps": [
            (
                "hsg_fingerprint_qwen.py",
                "Fingerprint replication — Qwen2-7B (p < 0.001)",
            ),
            (
                "hsg_fingerprint_mistral.py",
                "Fingerprint replication — Mistral-7B (p < 0.001)",
            ),
            (
                "hsg_fingerprint_gemma.py",
                "Fingerprint replication — Gemma-7B (p < 0.001)",
            ),
            ("hsg_see_residual.py", "See-residual fingerprint analysis"),
        ],
    },
    "semiotic": {
        "description": "Semiotic accuracy prediction across three model families",
        "steps": [
            ("hsg_generation.py", "Geometry-predicts-generation — Qwen2-7B"),
            (
                "semiotic_accuracy_prediction.py",
                "Semiotic accuracy prediction — Qwen2-7B and Mistral-7B",
            ),
            ("hsg_semiotic_gemma.py", "Semiotic accuracy prediction — Gemma-7B"),
        ],
    },
    "idiom": {
        "description": "Idiom-vs-literal geometry separation, all three models",
        "steps": [
            (
                "hsg_idiom_vs_literal.py --output-dir results/idiom_vs_literal_qwen",
                "Idiom-vs-literal — Qwen2-7B (block 21, Δ=+0.1413)",
            ),
            (
                "hsg_idiom_vs_literal.py --model mistralai/Mistral-7B-v0.1 --output-dir results/idiom_vs_literal_mistral",
                "Idiom-vs-literal — Mistral-7B (block 24, Δ=+0.1817)",
            ),
            (
                "hsg_idiom_vs_literal.py --model google/gemma-7b --output-dir results/idiom_vs_literal_gemma",
                "Idiom-vs-literal — Gemma-7B (block 21, Δ=+0.1571)",
            ),
        ],
    },
    "annotation": {
        "description": "Human annotation study: export blind sheet and analyze results",
        "steps": [
            (
                "make_annotation_sheet.py",
                "Export blind annotation CSV (150 rows, balanced across classes)",
            ),
            (
                "analyze_human_annotations.py",
                "Analyze completed annotations (AUC, kappa, keyword comparison)",
            ),
        ],
    },
    "ordvec": {
        "description": "Ordinal Quotient Preservation — v2 fiber sweep, layer role, ablations",
        "steps": [
            (
                "hsg_ordvec_fiber_test_v2.py",
                "OQP v2 sweep — K=64-512, layers 20-27 (primary results)",
            ),
            (
                "hsg_ordvec_fiber_test_v3.py",
                "OQP v3 sweep — mean-overlap aggregation variant",
            ),
            (
                "hsg_ordvec_layer_role_test_2.py",
                "Layer role analysis — Q_full, Q_ordvec, gen AUC, all 27 layers",
            ),
            (
                "hsg_ordvec_ablations.py",
                "OQP ablations — random labels, random bitmaps, wrong predicate, shallow layer",
            ),
        ],
    },
}

ORDER = ["corpus", "train", "affordance", "fingerprints", "semiotic", "idiom", "ordvec"]


def run_component(name, dry_run=False):
    comp = COMPONENTS[name]
    print(f"\n{'═' * 70}")
    print(f"  {name.upper()}: {comp['description']}")
    print(f"{'═' * 70}")

    for script_args, description in comp["steps"]:
        parts = script_args.split()
        script = parts[0]
        extra_args = parts[1:] if len(parts) > 1 else []
        script_path = SCRIPTS / script

        if not script_path.exists():
            print(f"  ⚠  MISSING: {script} — skipping")
            continue

        cmd = [sys.executable, str(script_path)] + extra_args
        print(f"\n  ▶  {description}")
        print(f"     {' '.join(cmd)}")

        if not dry_run:
            result = subprocess.run(cmd, cwd=Path(__file__).parent)
            if result.returncode != 0:
                print(f"\n  ✗  {script} exited with code {result.returncode}")
                print("     Continuing with remaining steps...")
            else:
                print("  ✓  Done")


def main():
    parser = argparse.ArgumentParser(
        description="RMG validation suite runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Components (run in order):
  corpus       Wikipedia corpus-linguistic corroboration (no GPU)
  train        Train small model from scratch (~2h)
  affordance   Exp A, Exp B, ablations (~1h)
  fingerprints Cross-model fingerprint replication (~1h each)
  semiotic     Semiotic accuracy prediction (~1h)
  idiom        Idiom-vs-literal, all three models (~30min each)
  ordvec       Ordinal Quotient Preservation sweep + ablations (~1h)
  annotation   Blind human annotation: export sheet / analyze responses (not in default run)
        """,
    )
    parser.add_argument(
        "--component",
        default="all",
        choices=["all"] + ORDER,
        help="Which component to run (default: all)",
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="Print commands without executing"
    )
    args = parser.parse_args()

    if args.dry_run:
        print("DRY RUN — commands will be printed but not executed\n")

    components = ORDER if args.component == "all" else [args.component]

    print("\nRMG Suite v0.9.3 — Relational Meaning Geometry")
    print("Mike Singleton")
    print(f"\nRunning: {', '.join(components)}")
    if not args.dry_run:
        hf = os.environ.get("HF_TOKEN", "")
        if not hf:
            print("\n  ⚠  HF_TOKEN not set — Gemma may fail (gated model)")
        print(f"\n  Scripts: {SCRIPTS}")

    for name in components:
        run_component(name, dry_run=args.dry_run)

    print(f"\n{'═' * 70}")
    print("  Suite complete.")
    print(f"  Results written to: {Path(__file__).parent / 'results'}")
    print(f"{'═' * 70}\n")


if __name__ == "__main__":
    main()
