#!/usr/bin/env python3
"""
Context Distance Stress Test

Question:
    How much intervening context can come between a predicate and its object
    before predicate-conditioned affordance geometry breaks?

Motivation:
    Prior tests showed that local prompts like:

        I love my wife because...
        I love pizza because...
        I love my enemy because...

    produce hidden-state geometry that predicts semiotic correctness.

    Jason's question:
        How complicated can context get before reaching an object?
        Can the geometry hold across distance?

Test:
    Build prompt ladders where the predicate appears early, the object appears later,
    and intervening material grows in length and complexity.

    Example:
        I love my wife because...
        I love, after everything we have been through, my wife because...
        I love, despite the years and the fights and the long silences, my wife because...

    Extract final-token hidden states before generation.
    Compute geometry margin against affordance-class centroids.
    Measure margin decay as context distance increases.

Win condition:
    Correct affordance margin remains positive as distance/context complexity increases.

Failure condition:
    Margin collapses, flips negative, or becomes dominated by distractors.

Output:
    ~/hsg/context_distance_results/context_distance_results.json
    ~/hsg/context_distance_results/context_distance_results.csv
"""

import os
import json
import random
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np
import torch
from scipy import stats
from transformers import AutoTokenizer, AutoModelForCausalLM


# =============================================================================
# Config
# =============================================================================

MODEL_NAME = os.environ.get("HSG_MODEL", "Qwen/Qwen2-7B")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

SEED = 42
TARGET_LAYER_FRACTION = 0.35

OUTPUT_DIR = (
    Path(__file__).resolve().parents[1] / "results" / "context_distance_results"
)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# =============================================================================
# Probe definitions
# =============================================================================


@dataclass
class Probe:
    obj: str
    affordance_class: str


# Anchor affordance classes for LOVE.
ANCHOR_PROBES = [
    Probe("pizza", "hedonic"),
    Probe("chocolate", "hedonic"),
    Probe("coffee", "hedonic"),
    Probe("cake", "hedonic"),
    Probe("ice cream", "hedonic"),
    Probe("my dog", "companion"),
    Probe("my cat", "companion"),
    Probe("my horse", "companion"),
    Probe("my friend", "companion"),
    Probe("my child", "companion"),
    Probe("my wife", "covenantal"),
    Probe("my husband", "covenantal"),
    Probe("my family", "covenantal"),
    Probe("my country", "covenantal"),
    Probe("God", "covenantal"),
    Probe("my enemy", "moral"),
    Probe("my rival", "moral"),
    Probe("a stranger", "moral"),
    Probe("my neighbor", "moral"),
    Probe("the other", "moral"),
]

AFFORDANCE_CLASSES = sorted(set(p.affordance_class for p in ANCHOR_PROBES))


# Stress objects.
# Keep this small first. Expand after the smoke test.
STRESS_OBJECTS = [
    Probe("pizza", "hedonic"),
    Probe("my dog", "companion"),
    Probe("my wife", "covenantal"),
    Probe("my enemy", "moral"),
]


# Intervening context ladders.
# The object will be inserted after this context:
#     "I love{intervention} {object} because"
#
# level 0 = local
# level 1 = mild insertion
# level 2 = longer emotional context
# level 3 = long conflicting context
# level 4 = distractor object
# level 5 = distractor + negation pressure
#
INTERVENTIONS = [
    {
        "level": 0,
        "kind": "local",
        "text": "",
    },
    {
        "level": 1,
        "kind": "short_intervening_context",
        "text": ", after all this time,",
    },
    {
        "level": 2,
        "kind": "medium_intervening_context",
        "text": ", after everything we have been through and everything it cost me,",
    },
    {
        "level": 3,
        "kind": "long_intervening_context",
        "text": ", despite the years, the arguments, the long silences, and the things neither of us knew how to say,",
    },
    {
        "level": 4,
        "kind": "distractor_object",
        "text": ", not the pizza, not the applause, not the easy comfort, but",
    },
    {
        "level": 5,
        "kind": "negation_and_distractor",
        "text": ", not because it was easy, not because I was hungry, not because I wanted comfort, but because of what it meant to choose",
    },
]


# =============================================================================
# Utilities
# =============================================================================


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def token_count(tok, text: str) -> int:
    return len(tok(text, add_special_tokens=False)["input_ids"])


def build_prompt(obj: str, intervention_text: str) -> str:
    if intervention_text == "":
        return f"I love {obj} because"
    return f"I love{intervention_text} {obj} because"


@torch.no_grad()
def get_hidden_vector(model, tok, text: str, layer_idx: int) -> np.ndarray:
    inp = tok(text, return_tensors="pt", truncation=True, max_length=256).to(
        model.device
    )
    out = model(**inp, output_hidden_states=True, use_cache=False)
    return out.hidden_states[layer_idx][0, -1, :].float().cpu().numpy()


def l2(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm(a - b))


def compute_anchor_vectors(model, tok, layer_idx: int) -> Dict[str, np.ndarray]:
    """
    Build clean centroids from local anchor prompts:
        I love X because
    """
    vectors = {}
    for p in ANCHOR_PROBES:
        prompt = build_prompt(p.obj, "")
        vectors[p.obj] = get_hidden_vector(model, tok, prompt, layer_idx)
    return vectors


def compute_centroids(anchor_vectors: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    by_class = {cls: [] for cls in AFFORDANCE_CLASSES}

    for p in ANCHOR_PROBES:
        by_class[p.affordance_class].append(anchor_vectors[p.obj])

    return {cls: np.mean(vecs, axis=0) for cls, vecs in by_class.items()}


def geometry_margin(
    vec: np.ndarray, expected_class: str, centroids: Dict[str, np.ndarray]
) -> Tuple[float, Dict[str, float]]:
    """
    margin = nearest wrong centroid distance - expected centroid distance

    Positive:
        closer to expected class than any wrong class.

    Negative:
        closer to a wrong class than the expected class.
    """
    distances = {cls: l2(vec, centroid) for cls, centroid in centroids.items()}

    expected_d = distances[expected_class]
    nearest_wrong = min(d for cls, d in distances.items() if cls != expected_class)

    margin = nearest_wrong - expected_d
    return float(margin), distances


def nearest_class(distances: Dict[str, float]) -> str:
    return min(distances, key=distances.get)


# =============================================================================
# Main
# =============================================================================


def main():
    set_seed(SEED)

    print("=" * 90)
    print("CONTEXT DISTANCE STRESS TEST")
    print("=" * 90)
    print(f"Model:  {MODEL_NAME}")
    print(f"Device: {DEVICE}")

    tok = AutoTokenizer.from_pretrained(MODEL_NAME)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.float16 if DEVICE == "cuda" else torch.float32,
        device_map="auto" if DEVICE == "cuda" else None,
        output_hidden_states=True,
        low_cpu_mem_usage=True,
    )
    model.eval()

    n_layers = model.config.num_hidden_layers
    layer_idx = max(1, int(round(n_layers * TARGET_LAYER_FRACTION)))

    print(f"Layers: {n_layers}")
    print(f"Using hidden_states[{layer_idx}] for geometry")
    print()

    # -------------------------------------------------------------------------
    # Build clean affordance centroids
    # -------------------------------------------------------------------------

    print("Building clean affordance centroids from local anchor prompts...")
    anchor_vectors = compute_anchor_vectors(model, tok, layer_idx)
    centroids = compute_centroids(anchor_vectors)

    print("Centroids:")
    for cls in AFFORDANCE_CLASSES:
        print(f"  {cls}")
    print()

    # -------------------------------------------------------------------------
    # Run stress ladder
    # -------------------------------------------------------------------------

    rows = []

    print("Running context distance ladder...")
    for probe in STRESS_OBJECTS:
        print(f"\nObject: {probe.obj} [{probe.affordance_class}]")

        for intervention in INTERVENTIONS:
            prompt = build_prompt(probe.obj, intervention["text"])
            vec = get_hidden_vector(model, tok, prompt, layer_idx)

            margin, distances = geometry_margin(
                vec,
                probe.affordance_class,
                centroids,
            )

            predicted_class = nearest_class(distances)
            is_correct_region = predicted_class == probe.affordance_class

            pred_obj_pos = prompt.lower().find("love")
            obj_pos = prompt.lower().find(probe.obj.lower())
            char_distance = (
                obj_pos - pred_obj_pos if obj_pos >= 0 and pred_obj_pos >= 0 else None
            )

            # Approx token distance from predicate to object.
            prefix_to_obj = prompt[:obj_pos] if obj_pos >= 0 else prompt
            prefix_to_pred = prompt[:pred_obj_pos] if pred_obj_pos >= 0 else ""
            token_distance = token_count(tok, prefix_to_obj) - token_count(
                tok, prefix_to_pred
            )

            row = {
                "object": probe.obj,
                "expected_class": probe.affordance_class,
                "level": intervention["level"],
                "kind": intervention["kind"],
                "prompt": prompt,
                "token_distance_predicate_to_object": token_distance,
                "char_distance_predicate_to_object": char_distance,
                "margin": margin,
                "predicted_class": predicted_class,
                "correct_region": is_correct_region,
                "distances": distances,
            }

            rows.append(row)

            status = "OK" if is_correct_region else "FLIP"
            print(
                f"  level={intervention['level']} "
                f"{intervention['kind']:<28} "
                f"tokens={token_distance:<3} "
                f"margin={margin:+8.3f} "
                f"nearest={predicted_class:<12} "
                f"{status}"
            )

    # -------------------------------------------------------------------------
    # Summary stats
    # -------------------------------------------------------------------------

    print("\n" + "=" * 90)
    print("SUMMARY BY OBJECT")
    print("=" * 90)

    for probe in STRESS_OBJECTS:
        rs = [r for r in rows if r["object"] == probe.obj]
        levels = np.array([r["level"] for r in rs], dtype=float)
        margins = np.array([r["margin"] for r in rs], dtype=float)

        rho, pval = stats.spearmanr(levels, margins)

        first_margin = rs[0]["margin"]
        last_margin = rs[-1]["margin"]
        decay = last_margin - first_margin

        n_correct = sum(r["correct_region"] for r in rs)

        print(f"\n{probe.obj} [{probe.affordance_class}]")
        print(f"  correct regions: {n_correct}/{len(rs)}")
        print(f"  first margin:    {first_margin:+.3f}")
        print(f"  last margin:     {last_margin:+.3f}")
        print(f"  net decay:       {decay:+.3f}")
        print(f"  Spearman level vs margin: rho={rho:+.3f}, p={pval:.4f}")

    print("\n" + "=" * 90)
    print("SUMMARY BY LEVEL")
    print("=" * 90)

    for level in sorted(set(r["level"] for r in rows)):
        rs = [r for r in rows if r["level"] == level]
        mean_margin = float(np.mean([r["margin"] for r in rs]))
        acc = float(np.mean([r["correct_region"] for r in rs]))

        kind = rs[0]["kind"]
        print(
            f"  level={level} {kind:<28} mean_margin={mean_margin:+.3f} correct_region={acc:.3f}"
        )

    # Overall trend
    levels = np.array([r["level"] for r in rows], dtype=float)
    margins = np.array([r["margin"] for r in rows], dtype=float)
    rho, pval = stats.spearmanr(levels, margins)

    print("\n" + "=" * 90)
    print("OVERALL TREND")
    print("=" * 90)
    print(f"Spearman context level vs geometry margin: rho={rho:+.4f}, p={pval:.6f}")

    # -------------------------------------------------------------------------
    # Save
    # -------------------------------------------------------------------------

    json_path = OUTPUT_DIR / "context_distance_results.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "model": MODEL_NAME,
                "layer_idx": layer_idx,
                "n_layers": n_layers,
                "rows": rows,
            },
            f,
            indent=2,
        )

    csv_path = OUTPUT_DIR / "context_distance_results.csv"
    with open(csv_path, "w", encoding="utf-8") as f:
        f.write(
            "object,expected_class,level,kind,token_distance,margin,"
            "predicted_class,correct_region,prompt\n"
        )
        for r in rows:
            prompt = r["prompt"].replace('"', '""')
            f.write(
                f'"{r["object"]}",'
                f'"{r["expected_class"]}",'
                f"{r['level']},"
                f'"{r["kind"]}",'
                f"{r['token_distance_predicate_to_object']},"
                f"{r['margin']:.6f},"
                f'"{r["predicted_class"]}",'
                f"{int(r['correct_region'])},"
                f'"{prompt}"\n'
            )

    print("\nSaved:")
    print(f"  {json_path}")
    print(f"  {csv_path}")

    print("\n" + "=" * 90)
    print("INTERPRETATION")
    print("=" * 90)
    print("""
Win condition:
    Correct affordance margin stays positive as context distance increases.

Expected degradation:
    Margin may decay as intervening context grows.

Failure boundary:
    Margin flips negative or nearest class becomes wrong.

Strong result:
    Most prompts remain in the correct affordance region even at higher levels.

Weak result:
    Margins collapse quickly with short insertions.

Interesting failure:
    Distractors capture the geometry.
    Example: hedonic distractors pull covenantal prompts toward hedonic space.

This test answers Jason's question:
    How complicated can context get before reaching an object?
""")


if __name__ == "__main__":
    main()
