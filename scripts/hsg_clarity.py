#!/usr/bin/env python3
"""
HSG Prompt Clarity Engine
==========================
Singleton, M.
June 2026

Detects and corrects three affordance field problems in prompts:

1. Predicate-object distance — intervening dark matter between verb and object
2. Passive voice — hides the predicate-object relationship
3. Affordance capture — semantically loaded context shifts the intended field

The tool does one thing: makes relational meaning legible to a language model.

Usage:
    python3 hsg_clarity.py "I love, despite everything, my wife."
    python3 hsg_clarity.py --file prompts.txt
    python3 hsg_clarity.py --interactive

With geometry (requires Qwen2-7B):
    python3 hsg_clarity.py --geometry "I love my wife."
"""

import re
import json
import argparse
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple


# ── Affordance field knowledge ────────────────────────────────────────────────

# Semantically loaded word categories
# These are the "dark matter" types that capture affordance fields

MORAL_LOAD = [
    "despite",
    "forgive",
    "forgiveness",
    "duty",
    "obligation",
    "sacrifice",
    "cost",
    "pain",
    "endure",
    "burden",
    "mercy",
    "command",
    "must",
    "ought",
    "principle",
    "virtue",
    "right",
    "wrong",
    "sin",
    "redemption",
    "suffer",
    "hard",
    "difficult",
    "struggle",
    "fight",
    "battle",
    "survive",
]

HEDONIC_LOAD = [
    "pleasure",
    "taste",
    "delicious",
    "comfort",
    "easy",
    "fun",
    "enjoy",
    "appetite",
    "craving",
    "hunger",
    "sweet",
    "satisfying",
    "simple",
    "reward",
    "treat",
    "indulge",
]

COMPANION_LOAD = [
    "loyal",
    "faithful",
    "always there",
    "never leaves",
    "unconditional",
    "companion",
    "friend",
    "walks",
    "play",
    "bond",
    "affection",
]

COVENANTAL_LOAD = [
    "years",
    "life",
    "together",
    "commitment",
    "vow",
    "promise",
    "family",
    "history",
    "journey",
    "build",
    "shared",
    "devoted",
    "forever",
]

# Common predicate verbs and their expected affordance fields
RELATIONAL_PREDICATES = {
    "love": ["hedonic", "companion", "covenantal", "moral"],
    "fear": ["physical", "existential", "ambient"],
    "care": ["dependent", "companion", "family"],
    "own": ["property", "vehicle", "living", "asset"],
    "eat": ["staple", "indulgence", "preference"],
    "hate": ["hedonic", "companion", "covenantal", "moral"],
    "need": ["practical", "emotional", "existential"],
    "remember": ["sensory", "emotional", "factual"],
    "trust": ["person", "institution", "belief"],
    "miss": ["person", "place", "experience"],
}

# Passive voice patterns
PASSIVE_PATTERNS = [
    r"\b(is|are|was|were|be|been|being)\s+(\w+ed)\b",
    r"\b(is|are|was|were)\s+(loved|feared|owned|needed|remembered|trusted|missed|hated|cared for)\b",
    r"\b(gets?|got)\s+(\w+ed)\b",
]

# Interrupting clause patterns — dark matter between predicate and object
INTERRUPTION_PATTERNS = [
    r",\s+[^,]{10,80},\s+",  # , long clause ,
    r",\s+(after|despite|because|although|even though|in spite of)[^,]+,",
    r",\s+(not|never|always|still|yet)[^,]+,",
]


# ── Analysis ──────────────────────────────────────────────────────────────────


@dataclass
class FieldProblem:
    type: str  # 'distance' | 'passive' | 'capture' | 'nominalization'
    severity: str  # 'high' | 'medium' | 'low'
    location: str  # what in the sentence
    description: str
    suggestion: str


@dataclass
class AnalysisResult:
    original: str
    problems: List[FieldProblem] = field(default_factory=list)
    rewritten: Optional[str] = None
    field_score: float = 1.0  # 1.0 = clean, 0.0 = fully captured
    predicate: Optional[str] = None
    object_phrase: Optional[str] = None
    dark_matter_score: float = 0.0
    intervening_tokens: int = 0


def find_predicate(sentence: str) -> Optional[Tuple[str, int]]:
    """Find the main relational predicate and its position."""
    sentence_lower = sentence.lower()
    for pred in sorted(
        RELATIONAL_PREDICATES.keys(), key=lambda x: -len(x)
    ):  # longest first
        # Look for "I {pred}" or "she {pred}s" etc.
        pattern = (
            rf"\b(i|we|she|he|they|you)\s+(don\'t\s+)?({pred}s?|{pred}ing|{pred}ed)\b"
        )
        m = re.search(pattern, sentence_lower)
        if m:
            return pred, m.start()
        # Also bare predicate
        pattern2 = rf"\b{pred}s?\b"
        m2 = re.search(pattern2, sentence_lower)
        if m2:
            return pred, m2.start()
    return None


def measure_dark_matter(sentence: str, predicate: str) -> Tuple[float, int, List[str]]:
    """
    Measure the semantic weight of language between predicate and object.
    Returns (score, token_count, detected_types).
    """
    sentence_lower = sentence.lower()
    pred_pos = sentence_lower.find(predicate)
    if pred_pos == -1:
        return 0.0, 0, []

    # Find everything after the predicate
    after_pred = sentence[pred_pos + len(predicate) :]

    # Count tokens between predicate and likely object
    # Object is usually after any commas/clauses
    interruption_match = re.search(r",(.+?),", after_pred)
    if interruption_match:
        intervening = interruption_match.group(1)
        token_count = len(intervening.split())
    else:
        token_count = 0
        intervening = ""

    # Measure semantic load
    detected = []
    score = 0.0
    text_to_check = after_pred.lower()

    for word in MORAL_LOAD:
        if word in text_to_check:
            detected.append(f"moral:{word}")
            score += 0.3

    for word in HEDONIC_LOAD:
        if word in text_to_check:
            detected.append(f"hedonic:{word}")
            score += 0.15

    for word in COMPANION_LOAD:
        if word in text_to_check:
            detected.append(f"companion:{word}")
            score += 0.15

    # Penalize for interrupting clauses
    for pattern in INTERRUPTION_PATTERNS:
        if re.search(pattern, sentence):
            score += 0.4
            detected.append("interrupting_clause")
            break

    return min(score, 1.0), token_count, detected


def detect_passive(sentence: str) -> List[FieldProblem]:
    """Detect passive voice constructions."""
    problems = []
    for pattern in PASSIVE_PATTERNS:
        matches = re.finditer(pattern, sentence, re.IGNORECASE)
        for m in matches:
            problems.append(
                FieldProblem(
                    type="passive",
                    severity="medium",
                    location=m.group(0),
                    description=f"Passive voice '{m.group(0)}' hides the predicate-object relationship.",
                    suggestion="Rewrite in active voice: who does what to what.",
                )
            )
    return problems


def detect_nominalization(sentence: str) -> List[FieldProblem]:
    """
    Detect nominalizations — actions hidden as nouns.
    'The decision was made' instead of 'X decided'.
    """
    nominalizations = [
        (r"\bthe\s+(\w+tion)\b", "action noun"),
        (r"\bthe\s+(\w+ment)\b", "action noun"),
        (r"\bthe\s+(\w+ance)\b", "action noun"),
        (r"\bthe\s+(\w+ence)\b", "action noun"),
        (r"\ba\s+(\w+tion)\b", "action noun"),
    ]
    problems = []
    for pattern, label in nominalizations:
        matches = re.finditer(pattern, sentence, re.IGNORECASE)
        for m in matches:
            word = m.group(1).lower()
            # Only flag if it's clearly an action disguised as noun
            if len(word) > 6:  # skip short words
                problems.append(
                    FieldProblem(
                        type="nominalization",
                        severity="low",
                        location=m.group(0),
                        description=f"'{m.group(0)}' may be hiding an action as a noun.",
                        suggestion=f"Consider: who is doing the {word.replace('tion', 'ing').replace('ment', 'ing')}?",
                    )
                )
    return problems


def rewrite_sentence(
    sentence: str, problems: List[FieldProblem], predicate: Optional[str] = None
) -> str:
    """
    Attempt automatic rewrite.
    Strategy:
    1. Move object closer to predicate (cut interrupting clauses, append after)
    2. Convert obvious passive constructions
    """
    result = sentence.strip()

    # Strategy 1: Move interrupting clause to end
    # Pattern: "I love, [clause], my wife [rest]."
    # Rewrite: "I love my wife [rest]. [Clause]."
    interruption = re.search(
        r"(I\s+(?:love|fear|care\s+for|own|eat|hate|need|trust|miss)\w*)"  # predicate
        r"(,\s+[^,]{5,100},\s*)"  # interrupting clause
        r"(.+?)"  # object + rest
        r"([.!?]?)$",  # end punctuation
        result,
        re.IGNORECASE,
    )
    if interruption:
        pred_part = interruption.group(1).strip()
        clause = interruption.group(2).strip().strip(",").strip()
        obj_rest = interruption.group(3).strip()
        punct = interruption.group(4) or "."
        result = f"{pred_part} {obj_rest}{punct} {clause.capitalize()}."

    # Strategy 2: Simple passive → active suggestion
    # "X is loved by Y" → "Y loves X"
    passive_by = re.search(
        r"(.+?)\s+is\s+(loved|feared|needed|trusted|missed|hated)\s+by\s+(.+?)([.!?]?)$",
        result,
        re.IGNORECASE,
    )
    if passive_by:
        obj = passive_by.group(1).strip()
        verb = passive_by.group(2).strip()
        subj = passive_by.group(3).strip()
        punct = passive_by.group(4) or "."
        # Simple present tense
        verb_active = verb.rstrip("d") if verb.endswith("ed") else verb
        result = f"{subj.capitalize()} {verb_active}s {obj}{punct}"

    return result


# ── Main analysis function ────────────────────────────────────────────────────


def analyze(sentence: str) -> AnalysisResult:
    """Analyze a sentence for affordance field problems."""
    result = AnalysisResult(original=sentence)

    # Find predicate
    pred_info = find_predicate(sentence)
    if pred_info:
        result.predicate = pred_info[0]

    # Measure dark matter
    if result.predicate:
        score, tokens, detected = measure_dark_matter(sentence, result.predicate)
        result.dark_matter_score = score
        result.intervening_tokens = tokens

        if score >= 0.6:
            result.problems.append(
                FieldProblem(
                    type="capture",
                    severity="high",
                    location=", ".join(detected[:3]),
                    description=f"High dark matter load ({score:.2f}). "
                    f"Intervening context may capture the affordance field "
                    f"before the object arrives. Detected: {', '.join(detected[:3])}",
                    suggestion="Move object immediately after predicate. "
                    "Put context in a separate sentence after.",
                )
            )
        elif score >= 0.3:
            result.problems.append(
                FieldProblem(
                    type="distance",
                    severity="medium",
                    location=", ".join(detected[:2]),
                    description=f"Moderate dark matter ({score:.2f}). "
                    f"Intervening context adds semantic load: {', '.join(detected[:2])}",
                    suggestion="Consider moving context after the predicate-object construction.",
                )
            )

        if tokens > 10:
            result.problems.append(
                FieldProblem(
                    type="distance",
                    severity="medium" if tokens > 20 else "low",
                    location=f"{tokens} tokens between predicate and object",
                    description=f"{tokens} tokens intervene between predicate and object.",
                    suggestion="Reduce to 3 tokens or fewer between predicate and object.",
                )
            )

    # Passive voice
    result.problems.extend(detect_passive(sentence))

    # Nominalization
    result.problems.extend(detect_nominalization(sentence))

    # Field score: 1.0 - dark_matter_score, adjusted for problems
    n_high = sum(1 for p in result.problems if p.severity == "high")
    n_medium = sum(1 for p in result.problems if p.severity == "medium")
    result.field_score = max(
        0.0, 1.0 - result.dark_matter_score - n_high * 0.2 - n_medium * 0.1
    )

    # Rewrite if problems found
    if result.problems:
        result.rewritten = rewrite_sentence(sentence, result.problems, result.predicate)
        if result.rewritten == sentence:
            result.rewritten = None  # no change made

    return result


# ── Geometry integration (optional, requires model) ──────────────────────────


def analyze_with_geometry(sentence: str, model_name: str = "Qwen/Qwen2-7B"):
    """
    Full analysis including actual affordance geometry measurement.
    Requires transformers + torch + the model.
    """
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from hf_cache import hf_cache_dir

        DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
        CACHE = hf_cache_dir()

        tok = AutoTokenizer.from_pretrained(model_name, cache_dir=CACHE)
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            cache_dir=CACHE,
            output_hidden_states=True,
            torch_dtype=torch.float16 if DEVICE == "cuda" else torch.float32,
            device_map="auto" if DEVICE == "cuda" else None,
            low_cpu_mem_usage=True,
        )
        model.eval()
        n_layers = model.config.num_hidden_layers
        layer_idx = max(1, int(round(n_layers * 0.35)))

        # Import HSG utils if available
        import sys

        sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
        from hsg_utils import build_love_centroids, nearest_class, geometry_margin

        centroids = build_love_centroids(model, tok, layer_idx, DEVICE)

        inp = tok(sentence, return_tensors="pt", truncation=True, max_length=128).to(
            DEVICE
        )
        with torch.no_grad():
            out = model(**inp, output_hidden_states=True)
        vec = out.hidden_states[layer_idx][0, -1, :].float().cpu().numpy()

        pred_class, dists = nearest_class(vec, centroids)
        margin = geometry_margin(vec, pred_class, centroids)[0]

        return {
            "predicted_class": pred_class,
            "margin": float(margin),
            "distances": {k: float(v) for k, v in dists.items()},
            "layer": layer_idx,
        }

    except Exception as e:
        return {"error": str(e)}


# ── Reporting ─────────────────────────────────────────────────────────────────


def print_report(result: AnalysisResult, show_geometry: dict = None):
    print(f"\n{'=' * 65}")
    print("HSG PROMPT CLARITY ANALYSIS")
    print(f"{'=' * 65}")
    print(f"\nInput:     {result.original}")
    if result.predicate:
        print(f"Predicate: {result.predicate}")
    print(
        f"Field score: {result.field_score:.2f}  "
        f"({'clean' if result.field_score > 0.7 else 'moderate concern' if result.field_score > 0.4 else 'high capture risk'})"
    )

    if result.dark_matter_score > 0:
        print(f"Dark matter: {result.dark_matter_score:.2f}")
    if result.intervening_tokens > 0:
        print(f"Tokens between predicate and object: {result.intervening_tokens}")

    if show_geometry:
        if "error" not in show_geometry:
            print("\nGeometry (Qwen2-7B block ~35%):")
            print(f"  Predicted class: {show_geometry['predicted_class']}")
            print(f"  Margin: {show_geometry['margin']:+.3f}")
            for cls, d in sorted(
                show_geometry["distances"].items(), key=lambda x: x[1]
            ):
                print(f"  {cls:<14} dist={d:.2f}")
        else:
            print(f"\nGeometry: {show_geometry['error']}")

    if result.problems:
        print(f"\nProblems found: {len(result.problems)}")
        for i, p in enumerate(result.problems, 1):
            print(f"\n  [{i}] {p.type.upper()} ({p.severity})")
            print(f"      {p.description}")
            print(f"      → {p.suggestion}")
    else:
        print("\nNo problems detected.")

    if result.rewritten:
        print("\nSuggested rewrite:")
        print(f"  {result.rewritten}")

    print(f"\n{'=' * 65}")


# ── CLI ───────────────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(
        description="HSG Prompt Clarity Engine",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python3 hsg_clarity.py "I love, despite everything, my wife."
  python3 hsg_clarity.py "The decision was made by the committee."
  python3 hsg_clarity.py "I love my enemy because of duty and obligation."
  python3 hsg_clarity.py --interactive
  python3 hsg_clarity.py --geometry "I love my wife." --model Qwen/Qwen2-7B
        """,
    )
    parser.add_argument("sentence", nargs="?", help="Sentence to analyze")
    parser.add_argument("--file", help="File of sentences, one per line")
    parser.add_argument("--interactive", action="store_true")
    parser.add_argument(
        "--geometry",
        action="store_true",
        help="Also measure actual affordance geometry (requires model)",
    )
    parser.add_argument("--model", default="Qwen/Qwen2-7B")
    parser.add_argument("--json", action="store_true", help="Output as JSON")
    args = parser.parse_args()

    sentences = []
    if args.sentence:
        sentences = [args.sentence]
    elif args.file:
        with open(args.file) as f:
            sentences = [line.strip() for line in f if line.strip()]
    elif args.interactive:
        print("HSG Prompt Clarity Engine")
        print("Enter sentences to analyze. Ctrl+C to exit.\n")
        while True:
            try:
                s = input("→ ").strip()
                if s:
                    result = analyze(s)
                    geo = (
                        analyze_with_geometry(s, args.model) if args.geometry else None
                    )
                    print_report(result, geo)
            except KeyboardInterrupt:
                print("\nDone.")
                break
    else:
        parser.print_help()
        return

    for s in sentences:
        result = analyze(s)
        geo = analyze_with_geometry(s, args.model) if args.geometry else None

        if args.json:
            import dataclasses

            out = {
                "original": result.original,
                "field_score": result.field_score,
                "dark_matter": result.dark_matter_score,
                "predicate": result.predicate,
                "problems": [dataclasses.asdict(p) for p in result.problems],
                "rewritten": result.rewritten,
            }
            if geo:
                out["geometry"] = geo
            print(json.dumps(out, indent=2))
        else:
            print_report(result, geo)


if __name__ == "__main__":
    main()
