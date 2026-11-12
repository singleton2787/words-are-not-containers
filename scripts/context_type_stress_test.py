#!/usr/bin/env python3
"""
D2: Context Type Stress Test

Question:
    Does affordance geometry decay because of distance alone,
    or because intervening context captures the affordance field?

Previous result:
    Context inserted between "love" and the object caused margin decay.
    But much of the inserted context was morally/obligationally loaded.

This test separates context types:
    1. local baseline
    2. neutral distance filler
    3. temporal filler
    4. moral/obligation filler
    5. hedonic distractor
    6. companion distractor
    7. contrastive negation

Core measurement:
    margin = distance to nearest wrong centroid - distance to expected centroid

Positive margin:
    hidden state is closer to expected affordance field.

Negative margin:
    hidden state is closer to a wrong affordance field.

Win:
    Neutral distance causes mild decay.
    Semantically loaded context causes class-specific capture.
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

OUTPUT_DIR = Path(__file__).resolve().parents[1] / "results" / "context_type_results"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# =============================================================================
# Data
# =============================================================================


@dataclass
class Probe:
    obj: str
    affordance_class: str


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

TEST_OBJECTS = [
    Probe("pizza", "hedonic"),
    Probe("my dog", "companion"),
    Probe("my wife", "covenantal"),
    Probe("my enemy", "moral"),
]

CONTEXT_TYPES = [
    {
        "type": "local",
        "description": "No inserted context",
        "interventions": [
            "",
        ],
    },
    {
        "type": "neutral_distance",
        "description": "Longer filler without strong affordance content",
        "interventions": [
            ", in a way that is hard to explain,",
            ", after some time and without making too much of it,",
            ", in the ordinary course of things, after many days had passed,",
            ", in a way that became clearer over time, though not all at once,",
        ],
    },
    {
        "type": "temporal_memory",
        "description": "Memory/history without explicit moral language",
        "interventions": [
            ", after all these years,",
            ", after many ordinary mornings and quiet evenings,",
            ", after years of small moments, repeated habits, and familiar rooms,",
            ", after a long stretch of time marked mostly by ordinary days,",
        ],
    },
    {
        "type": "moral_obligation",
        "description": "Duty, forgiveness, endurance, moral pressure",
        "interventions": [
            ", despite the pain and the need to forgive,",
            ", despite the years, the arguments, and the long silences,",
            ", not because it was easy, but because choosing mercy mattered,",
            ", after everything it cost me and everything I had to forgive,",
        ],
    },
    {
        "type": "hedonic_distractor",
        "description": "Food/pleasure/comfort distractors",
        "interventions": [
            ", not the pizza, not the cake, not the coffee, but",
            ", after thinking about comfort, hunger, pleasure, and taste,",
            ", not because of sweetness or appetite or anything delicious, but",
            ", with all the language of craving and pleasure pushed aside,",
        ],
    },
    {
        "type": "companion_distractor",
        "description": "Pet/friend/care distractors",
        "interventions": [
            ", not my dog, not my cat, not my closest friend, but",
            ", after thinking about loyalty, companionship, and being there every day,",
            ", not because of walks or play or the comfort of an animal, but",
            ", with all the language of friendship and companionship in the background,",
        ],
    },
    {
        "type": "contrastive_negation",
        "description": "Not-X-but-Y contrast structures",
        "interventions": [
            ", not because it was simple, but because",
            ", not because anyone expected it, but because",
            ", not because the obvious answer was enough, but because",
            ", not for comfort, not for duty, not for appearances, but because",
        ],
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


def build_prompt(obj: str, intervention: str) -> str:
    if intervention == "":
        return f"I love {obj} because"
    return f"I love{intervention} {obj} because"


def token_count(tok, text: str) -> int:
    return len(tok(text, add_special_tokens=False)["input_ids"])


@torch.no_grad()
def get_hidden_vector(model, tok, text: str, layer_idx: int) -> np.ndarray:
    inp = tok(text, return_tensors="pt", truncation=True, max_length=384).to(
        model.device
    )
    out = model(**inp, output_hidden_states=True, use_cache=False)
    return out.hidden_states[layer_idx][0, -1, :].float().cpu().numpy()


def l2(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm(a - b))


def compute_anchor_vectors(model, tok, layer_idx: int) -> Dict[str, np.ndarray]:
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
    distances = {cls: l2(vec, centroid) for cls, centroid in centroids.items()}

    expected_d = distances[expected_class]
    nearest_wrong = min(d for cls, d in distances.items() if cls != expected_class)

    margin = nearest_wrong - expected_d
    return float(margin), distances


def nearest_class(distances: Dict[str, float]) -> str:
    return min(distances, key=distances.get)


def predicate_object_token_distance(tok, prompt: str, obj: str) -> int:
    lower = prompt.lower()
    love_pos = lower.find("love")
    obj_pos = lower.find(obj.lower())

    if love_pos < 0 or obj_pos < 0:
        return -1

    before_love = prompt[:love_pos]
    before_obj = prompt[:obj_pos]

    return token_count(tok, before_obj) - token_count(tok, before_love)


# =============================================================================
# Main
# =============================================================================


def main():
    set_seed(SEED)

    print("=" * 90)
    print("D2 CONTEXT TYPE STRESS TEST")
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
    print(f"Using hidden_states[{layer_idx}]")
    print()

    print("Building clean local centroids...")
    anchor_vectors = compute_anchor_vectors(model, tok, layer_idx)
    centroids = compute_centroids(anchor_vectors)

    rows = []

    print("\nRunning stress test...")
    for probe in TEST_OBJECTS:
        print(f"\nObject: {probe.obj} [{probe.affordance_class}]")

        for ctx in CONTEXT_TYPES:
            for idx, intervention in enumerate(ctx["interventions"]):
                prompt = build_prompt(probe.obj, intervention)
                vec = get_hidden_vector(model, tok, prompt, layer_idx)
                margin, distances = geometry_margin(
                    vec, probe.affordance_class, centroids
                )
                pred_class = nearest_class(distances)
                correct = pred_class == probe.affordance_class
                tok_dist = predicate_object_token_distance(tok, prompt, probe.obj)

                row = {
                    "object": probe.obj,
                    "expected_class": probe.affordance_class,
                    "context_type": ctx["type"],
                    "context_description": ctx["description"],
                    "intervention_index": idx,
                    "intervention": intervention,
                    "prompt": prompt,
                    "token_distance": tok_dist,
                    "margin": margin,
                    "nearest_class": pred_class,
                    "correct_region": correct,
                    "distances": distances,
                }
                rows.append(row)

                status = "OK" if correct else "FLIP"
                print(
                    f"  {ctx['type']:<22} "
                    f"idx={idx:<2} "
                    f"tokens={tok_dist:<3} "
                    f"margin={margin:+8.3f} "
                    f"nearest={pred_class:<12} "
                    f"{status}"
                )

    # -------------------------------------------------------------------------
    # Summary by context type
    # -------------------------------------------------------------------------

    print("\n" + "=" * 90)
    print("SUMMARY BY CONTEXT TYPE")
    print("=" * 90)

    summary_by_type = {}

    for ctx in CONTEXT_TYPES:
        ctype = ctx["type"]
        rs = [r for r in rows if r["context_type"] == ctype]
        margins = np.array([r["margin"] for r in rs], dtype=float)
        correct = np.array([r["correct_region"] for r in rs], dtype=float)
        tok_dist = np.array([r["token_distance"] for r in rs], dtype=float)

        summary = {
            "n": len(rs),
            "mean_margin": float(np.mean(margins)),
            "median_margin": float(np.median(margins)),
            "correct_region_rate": float(np.mean(correct)),
            "mean_token_distance": float(np.mean(tok_dist)),
        }
        summary_by_type[ctype] = summary

        print(
            f"{ctype:<24} "
            f"n={summary['n']:<3} "
            f"mean_margin={summary['mean_margin']:+8.3f} "
            f"median={summary['median_margin']:+8.3f} "
            f"correct={summary['correct_region_rate']:.3f} "
            f"mean_tokens={summary['mean_token_distance']:.1f}"
        )

    # -------------------------------------------------------------------------
    # Summary by object
    # -------------------------------------------------------------------------

    print("\n" + "=" * 90)
    print("SUMMARY BY OBJECT")
    print("=" * 90)

    summary_by_object = {}

    for probe in TEST_OBJECTS:
        rs = [r for r in rows if r["object"] == probe.obj]
        margins = np.array([r["margin"] for r in rs], dtype=float)
        correct = np.array([r["correct_region"] for r in rs], dtype=float)
        tok_dist = np.array([r["token_distance"] for r in rs], dtype=float)

        rho, pval = stats.spearmanr(tok_dist, margins)

        summary = {
            "n": len(rs),
            "mean_margin": float(np.mean(margins)),
            "correct_region_rate": float(np.mean(correct)),
            "spearman_token_distance_vs_margin": float(rho),
            "spearman_p": float(pval),
        }
        summary_by_object[probe.obj] = summary

        print(
            f"{probe.obj:<16} [{probe.affordance_class:<11}] "
            f"mean_margin={summary['mean_margin']:+8.3f} "
            f"correct={summary['correct_region_rate']:.3f} "
            f"rho(tokens,margin)={rho:+.3f} p={pval:.4f}"
        )

    # -------------------------------------------------------------------------
    # Overall distance trend
    # -------------------------------------------------------------------------

    all_tok_dist = np.array([r["token_distance"] for r in rows], dtype=float)
    all_margins = np.array([r["margin"] for r in rows], dtype=float)

    rho, pval = stats.spearmanr(all_tok_dist, all_margins)

    print("\n" + "=" * 90)
    print("OVERALL DISTANCE TREND")
    print("=" * 90)
    print(f"Spearman token distance vs margin: rho={rho:+.4f}, p={pval:.6f}")

    # -------------------------------------------------------------------------
    # Compare neutral vs loaded contexts
    # -------------------------------------------------------------------------

    neutral_types = {"local", "neutral_distance", "temporal_memory"}
    loaded_types = {
        "moral_obligation",
        "hedonic_distractor",
        "companion_distractor",
        "contrastive_negation",
    }

    neutral_margins = np.array(
        [r["margin"] for r in rows if r["context_type"] in neutral_types]
    )
    loaded_margins = np.array(
        [r["margin"] for r in rows if r["context_type"] in loaded_types]
    )

    u, mw_p = stats.mannwhitneyu(neutral_margins, loaded_margins, alternative="greater")

    print("\n" + "=" * 90)
    print("NEUTRAL VS LOADED CONTEXT")
    print("=" * 90)
    print(f"Neutral mean margin: {neutral_margins.mean():+.3f}")
    print(f"Loaded mean margin:  {loaded_margins.mean():+.3f}")
    print(f"Mann-Whitney p(neutral > loaded): {mw_p:.6f}")

    # -------------------------------------------------------------------------
    # Save
    # -------------------------------------------------------------------------

    out = {
        "model": MODEL_NAME,
        "n_layers": n_layers,
        "layer_idx": layer_idx,
        "rows": rows,
        "summary_by_type": summary_by_type,
        "summary_by_object": summary_by_object,
        "overall_distance_spearman": {
            "rho": float(rho),
            "p": float(pval),
        },
        "neutral_vs_loaded": {
            "neutral_mean_margin": float(neutral_margins.mean()),
            "loaded_mean_margin": float(loaded_margins.mean()),
            "mannwhitney_p": float(mw_p),
        },
    }

    json_path = OUTPUT_DIR / "context_type_results.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)

    csv_path = OUTPUT_DIR / "context_type_results.csv"
    with open(csv_path, "w", encoding="utf-8") as f:
        f.write(
            "object,expected_class,context_type,intervention_index,token_distance,"
            "margin,nearest_class,correct_region,prompt\n"
        )
        for r in rows:
            prompt = r["prompt"].replace('"', '""')
            f.write(
                f'"{r["object"]}",'
                f'"{r["expected_class"]}",'
                f'"{r["context_type"]}",'
                f"{r['intervention_index']},"
                f"{r['token_distance']},"
                f"{r['margin']:.6f},"
                f'"{r["nearest_class"]}",'
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
This test separates pure distance from context capture.

Clean result:
    Neutral distance causes smaller margin decay than semantically loaded context.

Context-capture result:
    Loaded contexts pull the hidden state toward the affordance class implied by
    the inserted language.

Failure boundary:
    The nearest class flips before or after the object arrives.

Strong result:
    Neutral margin remains positive, but moral/hedonic/companion distractors
    produce class-specific capture.

This is the actual next boundary condition:
    Not just "how many tokens can intervene?"
    But "what kind of meaning intervenes?"
""")


if __name__ == "__main__":
    main()
