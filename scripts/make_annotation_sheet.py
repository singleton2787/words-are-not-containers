#!/usr/bin/env python3
"""
make_annotation_sheet.py
=========================
RMG Human Annotation Suite — Package Builder
Mike Singleton — v0.9.5

Produces a complete blind annotation package from semiotic accuracy
generation files:

    annotation/
        annotation_key_private.csv       ← keep private, never send
        annotation_form_items.csv        ← send to annotators
        annotation_instructions.txt      ← send to annotators
        training_examples.txt            ← send to annotators

Usage:
    # Default: 150 items (50 per model where available), balanced by class
    python3 scripts/make_annotation_sheet.py

    # Custom
    python3 scripts/make_annotation_sheet.py \\
        --qwen    results/semiotic_accuracy_results/semiotic_accuracy_generations_qwen.jsonl \\
        --mistral results/semiotic_accuracy_results/semiotic_accuracy_generations_mistral.jsonl \\
        --gemma   results/semiotic_accuracy_results_gemma/semiotic_accuracy_generations.jsonl \\
        --output-dir annotation/ \\
        --n 150 --seed 42

    # Print annotation guide only
    python3 scripts/make_annotation_sheet.py --guide
"""

import os, json, csv, random, argparse
from pathlib import Path
from collections import defaultdict

ROOT      = Path(__file__).resolve().parents[1]
RESULTS   = ROOT / "results"
ANNOT_DIR = ROOT / "annotation"

DEFAULT_QWEN    = RESULTS / "semiotic_accuracy_results"  / "semiotic_accuracy_generations_qwen.jsonl"
DEFAULT_MISTRAL = RESULTS / "semiotic_accuracy_results"  / "semiotic_accuracy_generations_mistral.jsonl"
DEFAULT_GEMMA   = RESULTS / "semiotic_accuracy_results_gemma" / "semiotic_accuracy_generations.jsonl"

# ── Fixed relation descriptions (shown to every annotator) ────────────────────
RELATION_DESCRIPTIONS = {
    "hedonic":    "Pleasure, taste, appetite, enjoyment, comfort, craving, or personal preference.",
    "companion":  "Affection, care, friendship, loyalty, presence, routine, dependence, play, or companionship.",
    "covenantal": "Commitment, shared life, family obligation, fidelity, devotion, permanence, sacrifice, or enduring loyalty.",
    "moral":      "Forgiveness, restraint, mercy, justice, duty toward others, refusal of hatred, or recognition of shared humanity.",
}

# ── Hand-crafted training examples (not drawn from real generations) ──────────
TRAINING_EXAMPLES = [
    {
        "id":       "T001",
        "prompt":   "I love pizza because...",
        "class":    "hedonic",
        "relation": RELATION_DESCRIPTIONS["hedonic"],
        "continuation": "it tastes wonderful and I enjoy experimenting with different toppings.",
        "correct_score": 2,
        "explanation": "Clearly preserves hedonic relation — taste, enjoyment, preference.",
    },
    {
        "id":       "T002",
        "prompt":   "I love my wife because...",
        "class":    "covenantal",
        "relation": RELATION_DESCRIPTIONS["covenantal"],
        "continuation": "she is crispy and inexpensive.",
        "correct_score": 0,
        "explanation": "Wrong relation — treats a covenantal relationship as a commodity.",
    },
    {
        "id":       "T003",
        "prompt":   "I love my dog because...",
        "class":    "companion",
        "relation": RELATION_DESCRIPTIONS["companion"],
        "continuation": "he waits by the door and stays beside me when I am upset.",
        "correct_score": 2,
        "explanation": "Clearly preserves companion relation — loyalty, presence, care.",
    },
    {
        "id":       "T004",
        "prompt":   "I love my enemy because...",
        "class":    "moral",
        "relation": RELATION_DESCRIPTIONS["moral"],
        "continuation": "he has a pleasant voice.",
        "correct_score": 1,
        "explanation": "Mixed — not hostile, but does not clearly engage the moral relation of forgiveness, restraint, or duty.",
    },
    {
        "id":       "T005",
        "prompt":   "I love my enemy because...",
        "class":    "moral",
        "relation": RELATION_DESCRIPTIONS["moral"],
        "continuation": "hatred only destroys the one who carries it, and I refuse to be destroyed.",
        "correct_score": 2,
        "explanation": "Clearly preserves moral relation — refusal of hatred, restraint, recognition of shared humanity.",
    },
    {
        "id":       "T006",
        "prompt":   "I love my friend because...",
        "class":    "companion",
        "relation": RELATION_DESCRIPTIONS["companion"],
        "continuation": "we made a solemn vow to each other at the altar.",
        "correct_score": 0,
        "explanation": "Wrong relation — shifts into covenantal frame (vow, altar). Friendship is companion, not covenantal.",
    },
    {
        "id":       "T007",
        "prompt":   "I love coffee because...",
        "class":    "hedonic",
        "relation": RELATION_DESCRIPTIONS["hedonic"],
        "continuation": "it warms me up and gets me through the morning.",
        "correct_score": 2,
        "explanation": "Hedonic — comfort, appetite, personal preference.",
    },
    {
        "id":       "T008",
        "prompt":   "I love my family because...",
        "class":    "covenantal",
        "relation": RELATION_DESCRIPTIONS["covenantal"],
        "continuation": "they are nice people.",
        "correct_score": 1,
        "explanation": "Weak — factually appropriate but generic. Does not engage commitment, obligation, or devotion.",
    },
]

INSTRUCTIONS_TEXT = """\
HOW TO SCORE
============

WHAT YOU'RE DOING
-----------------
Each row starts a sentence like "I love my dog because..." and then shows
what an AI wrote to finish it.

The sheet also tells you what KIND of love the sentence should be about.
Your job is one question:

    Did the AI's ending stick to that kind of relationship?

That's all. You don't need to know anything about AI.

DON'T SCORE ON
--------------
  - spelling, grammar or style
  - whether it's true
  - whether you agree with it
  - whether it's well written

A clumsy sentence can get a 2. A beautiful one can get a 0.

The AI's text is messy in places. Some lines stop mid-sentence. Some have
odd symbols or leftover code like <u><strong>, list dashes or "##".
Ignore all that and go by what the words mean.

THE SCORES
----------
  2 = Yes. It clearly stays with the right kind of relationship.
  1 = Sort of. It's mixed, vague, generic, or drifts toward a different kind.
  0 = No. It's about a different kind of relationship altogether.

THE FOUR KINDS OF RELATIONSHIP
------------------------------
  Hedonic (enjoying something):
      pleasure, taste, appetite, comfort, craving, liking it.
      "I love pizza because it tastes amazing."

  Companion (closeness and friendship):
      affection, care, loyalty, being there, routine, play.
      "I love my dog because he's always happy to see me."

  Covenantal (commitment for life):
      commitment, a shared life, family duty, faithfulness, sacrifice.
      "I love my wife because we've built a life together."

  Moral (doing right by people):
      forgiveness, mercy, fairness, duty to others, refusing to hate,
      seeing someone as a fellow human.
      "I love my enemy because hate would only poison me."

WHEN IT'S CLOSE
---------------
Always score against the kind of relationship the sheet says, not the one
you think fits best.

Companion and covenantal are neighbors, and so are companion and moral.
When an ending lands on the neighbor:
  - Still overlaps with the expected kind, or a real mix  -> 1
  - Clearly swaps it out for a different kind            -> 0

  Expected COVENANTAL: "I love my wife because she's always there for me."
      Being there, but nothing about commitment or a life together  -> 1
      "...we built a life together and I'd choose her again"        -> 2

  Expected MORAL: "I love my neighbor because..."
      "...he's fun to grill with on weekends" (just friendship)     -> 1
      "...loving the people next to you is what we owe them"        -> 2

  Judge what the ending actually says, not what it could have meant.

A FEW RULES
-----------
  - Work alone. Don't compare scores with anyone until everyone's done.
  - Go with your first honest read. Don't overthink it.
  - Use the note column if something is truly unclear. It's optional.
  - You won't be told which AI wrote which line. That's on purpose.

PRACTICE FIRST
--------------
Score the practice examples below before you start. The answers are given
so you can check yourself.
"""

def format_training_examples():
    lines = []
    for ex in TRAINING_EXAMPLES:
        lines.append(f"  [{ex['id']}] Prompt:        {ex['prompt']}")
        lines.append(f"        Expected:      {ex['class'].upper()} — {ex['relation']}")
        lines.append(f"        Continuation:  {ex['continuation']}")
        lines.append(f"        Correct score: {ex['correct_score']}")
        lines.append(f"        Why:           {ex['explanation']}")
        lines.append("")
    return "\n".join(lines)


def load_jsonl(path):
    rows = []
    path = Path(path)
    if not path.exists():
        print(f"  ⚠  Not found: {path}")
        return rows
    with open(path) as f:
        for i, line in enumerate(f):
            line = line.strip()
            if line:
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    print(f"  ⚠  JSON parse error at line {i+1}, skipping")
    return rows


def normalize_row(row, model_tag, source_idx):
    prompt         = row.get("prompt", "")
    continuation   = row.get("continuation", "").strip()
    expected_class = (row.get("expected_class") or row.get("source_class")
                      or row.get("cls") or "unknown")
    geometry_margin = row.get("geometry_margin", None)
    behavior_label  = row.get("behavior_label",
                               row.get("predicted_affordance_class", ""))
    if behavior_label == "correct":
        keyword_correct = 1
    elif behavior_label == "wrong":
        keyword_correct = 0
    else:
        keyword_correct = None

    return {
        "model":            model_tag,
        "source_idx":       source_idx,
        "prompt":           prompt,
        "continuation":     continuation,
        "expected_class":   expected_class,
        "geometry_margin":  geometry_margin,
        "keyword_correct":  keyword_correct,
        "behavior_label":   behavior_label,
    }


def sample_balanced(rows, n_per_model, seed):
    """
    For each model, sample n_per_model rows balanced across expected_class.
    """
    rng = random.Random(seed)
    by_model_class = defaultdict(list)
    for r in rows:
        by_model_class[(r["model"], r["expected_class"])].append(r)

    models   = sorted(set(r["model"] for r in rows))
    classes  = sorted(set(r["expected_class"] for r in rows))
    per_cell = max(1, n_per_model // len(classes))

    sampled = []
    for model in models:
        model_sampled = []
        for cls in classes:
            pool = by_model_class[(model, cls)]
            rng.shuffle(pool)
            model_sampled.extend(pool[:per_cell])
        # top up to n_per_model if needed
        remainder = [r for r in rows if r["model"] == model and r not in model_sampled]
        rng.shuffle(remainder)
        model_sampled.extend(remainder[:max(0, n_per_model - len(model_sampled))])
        sampled.extend(model_sampled[:n_per_model])

    rng.shuffle(sampled)
    return sampled


def main():
    parser = argparse.ArgumentParser(
        description="Build RMG blind annotation package",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    parser.add_argument("--qwen",       default=str(DEFAULT_QWEN))
    parser.add_argument("--mistral",    default=str(DEFAULT_MISTRAL))
    parser.add_argument("--gemma",      default=str(DEFAULT_GEMMA))
    parser.add_argument("--output-dir", default=str(ANNOT_DIR))
    parser.add_argument("--n",          type=int, default=150,
                        help="Total items (split evenly across available models)")
    parser.add_argument("--seed",       type=int, default=42)
    parser.add_argument("--all",        action="store_true",
                        help="Export all rows without sampling")
    parser.add_argument("--guide",      action="store_true",
                        help="Print instructions only, do not build package")
    args = parser.parse_args()

    if args.guide:
        print(INSTRUCTIONS_TEXT)
        print(format_training_examples())
        return

    # ── Load ──────────────────────────────────────────────────────────────────
    sources = [
        (args.qwen,    "Qwen2-7B"),
        (args.mistral, "Mistral-7B"),
        (args.gemma,   "Gemma-7B"),
    ]
    all_rows = []
    n_models_loaded = 0
    for path, model_tag in sources:
        raw = load_jsonl(path)
        if raw:
            normalized = [normalize_row(r, model_tag, i) for i, r in enumerate(raw)]
            normalized = [r for r in normalized if r["continuation"]]
            all_rows.extend(normalized)
            print(f"  Loaded {len(normalized):>4} rows  ← {model_tag}")
            n_models_loaded += 1

    if not all_rows:
        print("\n  ✗  No generation files found.")
        return

    print(f"\n  Total rows available: {len(all_rows)}")

    # ── Sample ────────────────────────────────────────────────────────────────
    if args.all:
        sampled = all_rows
    else:
        n_per_model = max(1, args.n // n_models_loaded)
        sampled = sample_balanced(all_rows, n_per_model, args.seed)
        print(f"  Sampled {len(sampled)} rows "
              f"({n_per_model}/model, balanced by class, seed={args.seed})")

    # Shuffle and assign IDs
    rng = random.Random(args.seed + 99)
    rng.shuffle(sampled)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    key_path   = out_dir / "annotation_key_private.csv"
    form_path  = out_dir / "annotation_form_items.csv"
    instr_path = out_dir / "annotation_instructions.txt"
    train_path = out_dir / "training_examples.txt"

    # ── Write private key ─────────────────────────────────────────────────────
    key_fields = ["annotation_id", "model", "source_idx", "expected_class",
                  "geometry_margin", "keyword_correct", "behavior_label"]
    with open(key_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=key_fields)
        w.writeheader()
        for i, row in enumerate(sampled):
            w.writerow({
                "annotation_id":  f"A{i+1:03d}",
                "model":          row["model"],
                "source_idx":     row["source_idx"],
                "expected_class": row["expected_class"],
                "geometry_margin": row["geometry_margin"],
                "keyword_correct": row["keyword_correct"],
                "behavior_label":  row["behavior_label"],
            })

    # ── Write blind form items ─────────────────────────────────────────────────
    form_fields = ["annotation_id", "prompt", "expected_relationship",
                   "continuation", "score", "optional_note"]
    with open(form_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=form_fields)
        w.writeheader()
        for i, row in enumerate(sampled):
            cls = row["expected_class"]
            rel = f"{cls.capitalize()}: {RELATION_DESCRIPTIONS.get(cls, cls)}"
            w.writerow({
                "annotation_id":        f"A{i+1:03d}",
                "prompt":               row["prompt"],
                "expected_relationship": rel,
                "continuation":         row["continuation"],
                "score":                "",
                "optional_note":        "",
            })

    # ── Write instructions ────────────────────────────────────────────────────
    with open(instr_path, "w", encoding="utf-8") as f:
        f.write(INSTRUCTIONS_TEXT)

    # ── Write training examples ───────────────────────────────────────────────
    with open(train_path, "w", encoding="utf-8") as f:
        f.write("PRACTICE EXAMPLES\n")
        f.write("=================\n")
        f.write("Score each one yourself first, then look at the answer.\n")
        f.write("These don't count. They're just for getting a feel for it.\n\n")
        f.write(format_training_examples())

    # ── Summary ───────────────────────────────────────────────────────────────
    from collections import Counter
    class_dist = Counter(r["expected_class"] for r in sampled)
    model_dist = Counter(r["model"] for r in sampled)

    print(f"\n  Package written to: {out_dir}/")
    print(f"\n  ┌─ SEND TO ANNOTATORS ──────────────────────────────┐")
    print(f"  │  {form_path.name:<48}  │")
    print(f"  │  {instr_path.name:<48}  │")
    print(f"  │  {train_path.name:<48}  │")
    print(f"  └────────────────────────────────────────────────────┘")
    print(f"\n  ┌─ KEEP PRIVATE ────────────────────────────────────┐")
    print(f"  │  {key_path.name:<48}  │")
    print(f"  └────────────────────────────────────────────────────┘")
    print(f"\n  Class distribution:")
    for cls in sorted(class_dist):
        print(f"    {cls:<12}  {class_dist[cls]:>3}")
    print(f"\n  Model distribution:")
    for model in sorted(model_dist):
        print(f"    {model:<14}  {model_dist[model]:>3}")
    print(f"\n  Total items: {len(sampled)}")
    print(f"\n  Next steps:")
    print(f"    1. Give each annotator a unique reviewer code (R01, R02, R03)")
    print(f"    2. Share annotation_form_items.csv, annotation_instructions.txt,")
    print(f"       and training_examples.txt")
    print(f"    3. Collect scores in format: annotation_id, reviewer_id, score, note")
    print(f"    4. Run: python3 scripts/analyze_human_annotations.py")


if __name__ == "__main__":
    main()
