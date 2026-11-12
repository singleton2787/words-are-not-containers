#!/usr/bin/env python3
"""
HSG Validation Suite — Component 5: Idiom vs Literal Geometry
==============================================================
Test question:
    Does the model's hidden-state geometry separate idiomatic meaning
    from literal meaning when the surface words are identical?

Pairs tested:
    spill the beans  = idiom  / reveal a secret
    spill the beans  = literal / drop legumes
    break the ice    = idiom  / reduce social tension
    break the ice    = literal / fracture frozen water
    hit the road     = idiom  / leave / depart
    hit the road     = literal / strike pavement

Method:
    1. Construct minimal context pairs that force idiomatic vs literal reading.
    2. Extract hidden-state geometry at the same layer used for affordance tests.
    3. Measure cosine distance between idiomatic and literal representations.
    4. Compare within-reading vs between-reading distances.
    5. Permutation test for significance.

Falsifiable prediction:
    If geometry encodes relational/semantic meaning beyond surface form,
    idiomatic and literal readings of the same phrase should be geometrically
    separable even though the surface tokens are identical.
"""

import os
import json
import numpy as np
import torch
from pathlib import Path
from itertools import combinations
from scipy.stats import mannwhitneyu
from transformers import AutoTokenizer, AutoModelForCausalLM
from hf_cache import hf_cache_dir
from stats import empirical_p_value

# ── Config ────────────────────────────────────────────────────────────────────
import argparse as _ap

_parser = _ap.ArgumentParser(description="RMG Idiom-vs-Literal geometry test")
_parser.add_argument(
    "--model",
    default=os.environ.get("HSG_MODEL", "Qwen/Qwen2-7B"),
    help="HuggingFace model ID (default: Qwen/Qwen2-7B)",
)
_parser.add_argument(
    "--output-dir",
    default=None,
    help="Directory for results JSON (default: <repo>/results/)",
)
_parser.add_argument(
    "--n-perm", type=int, default=5000, help="Number of permutations (default: 5000)"
)
_args, _unknown = _parser.parse_known_args()

MODEL_NAME = _args.model
CACHE_DIR = hf_cache_dir()
_default_out = Path(__file__).resolve().parents[1] / "results"
OUTPUT_DIR = Path(_args.output_dir) if _args.output_dir else _default_out
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
N_PERM = _args.n_perm
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ── Stimulus pairs ────────────────────────────────────────────────────────────
# Each entry: (phrase, reading, context_sentence)
# Context sentences force the intended reading unambiguously.
STIMULI = [
    # spill the beans
    {
        "phrase": "spill the beans",
        "reading": "idiom",
        "gloss": "reveal a secret",
        "contexts": [
            "She finally decided to spill the beans about the surprise party.",
            "He spilled the beans and told everyone about the secret plan.",
            "Don't spill the beans — nobody is supposed to know yet.",
            "She spilled the beans to her sister about the engagement.",
        ],
    },
    {
        "phrase": "spill the beans",
        "reading": "literal",
        "gloss": "drop legumes",
        "contexts": [
            "She knocked over the pot and spilled the beans all over the floor.",
            "He spilled the beans from the bag while cooking dinner.",
            "Be careful not to spill the beans — they're hot from the stove.",
            "She spilled the beans onto the counter while measuring them out.",
        ],
    },
    # break the ice
    {
        "phrase": "break the ice",
        "reading": "idiom",
        "gloss": "reduce social tension",
        "contexts": [
            "He told a joke to break the ice at the start of the meeting.",
            "She brought snacks to help break the ice with the new team.",
            "A funny story can really break the ice at a first introduction.",
            "He tried to break the ice by asking everyone about their weekend.",
        ],
    },
    {
        "phrase": "break the ice",
        "reading": "literal",
        "gloss": "fracture frozen water",
        "contexts": [
            "The icebreaker ship was sent ahead to break the ice in the channel.",
            "They used a pickaxe to break the ice blocking the harbor entrance.",
            "In winter, workers come each morning to break the ice on the path.",
            "The boat's hull was reinforced to break the ice in polar waters.",
        ],
    },
    # hit the road
    {
        "phrase": "hit the road",
        "reading": "idiom",
        "gloss": "leave or depart",
        "contexts": [
            "It's getting late — we should hit the road before dark.",
            "She packed her bags and hit the road early in the morning.",
            "They decided to hit the road after saying goodbye to everyone.",
            "Let's hit the road now if we want to make it there by noon.",
        ],
    },
    {
        "phrase": "hit the road",
        "reading": "literal",
        "gloss": "strike pavement",
        "contexts": [
            "The motorcyclist lost control and hit the road hard.",
            "When he fell off his bike, he hit the road and scraped his knee.",
            "The rock bounced and hit the road with a loud crack.",
            "She tripped and hit the road, injuring her hand.",
        ],
    },
]


# ── Model loading ─────────────────────────────────────────────────────────────
def load_model(model_name, cache_dir):
    print(f"Loading {model_name}...")
    tok = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        cache_dir=cache_dir,
        output_hidden_states=True,
        torch_dtype=torch.float32,
        device_map="auto",
    )
    model.eval()
    n_layers = model.config.num_hidden_layers
    block_idx = int(n_layers * 0.75)
    print(f"Layers: {n_layers}, using hidden_states[{block_idx}]")
    return tok, model, block_idx


# ── Geometry extraction ───────────────────────────────────────────────────────
def get_phrase_vector(tok, model, block_idx, context, phrase):
    """
    Extract hidden state at the last token of the target phrase
    within the context sentence.
    """
    inputs = tok(context, return_tensors="pt").to(DEVICE)
    phrase_tokens = tok.encode(" " + phrase, add_special_tokens=False)

    with torch.no_grad():
        outputs = model(**inputs)

    hidden = outputs.hidden_states[block_idx][0]  # [seq_len, hidden]

    # Find the last token of the phrase in the sequence
    input_ids = inputs["input_ids"][0].tolist()
    phrase_len = len(phrase_tokens)
    pos = None
    for i in range(len(input_ids) - phrase_len + 1):
        if input_ids[i : i + phrase_len] == phrase_tokens:
            pos = i + phrase_len - 1

    if pos is None:
        # Fallback: use second-to-last token
        pos = hidden.shape[0] - 2

    vec = hidden[pos].cpu().float().numpy()
    return vec / (np.linalg.norm(vec) + 1e-8)


def cosine_dist(a, b):
    return float(1.0 - np.dot(a, b))


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    tok, model, block_idx = load_model(MODEL_NAME, CACHE_DIR)

    # Extract vectors for all stimuli
    print("\nExtracting geometry vectors...")
    vectors = {}
    for stim in STIMULI:
        key = f"{stim['phrase']}_{stim['reading']}"
        vecs = []
        for ctx in stim["contexts"]:
            v = get_phrase_vector(tok, model, block_idx, ctx, stim["phrase"])
            vecs.append(v)
            print(f"  [{stim['reading']:8s}] {stim['phrase']:20s} | {ctx[:60]}...")
        vectors[key] = vecs

    # Compute within-reading and between-reading distances
    print("\n" + "=" * 80)
    print("IDIOM VS LITERAL GEOMETRY SEPARATION")
    print("=" * 80)

    phrases = ["spill the beans", "break the ice", "hit the road"]
    within_dists = []
    between_dists = []
    results = []

    for phrase in phrases:
        idiom_vecs = vectors[f"{phrase}_idiom"]
        lit_vecs = vectors[f"{phrase}_literal"]

        # Within-reading distances (idiom-idiom, literal-literal)
        w = []
        for a, b in combinations(idiom_vecs, 2):
            w.append(cosine_dist(a, b))
        for a, b in combinations(lit_vecs, 2):
            w.append(cosine_dist(a, b))

        # Between-reading distances (idiom-literal)
        b_dists = []
        for iv in idiom_vecs:
            for lv in lit_vecs:
                b_dists.append(cosine_dist(iv, lv))

        within_mean = float(np.mean(w))
        between_mean = float(np.mean(b_dists))
        separation = between_mean - within_mean

        within_dists.extend(w)
        between_dists.extend(b_dists)

        print(f"\n  {phrase.upper()}")
        print(f"    within-reading  mean dist: {within_mean:.4f}")
        print(f"    between-reading mean dist: {between_mean:.4f}")
        print(
            f"    separation (between-within): {separation:+.4f}  {'✓ separated' if separation > 0 else '✗ not separated'}"
        )

        results.append(
            {
                "phrase": phrase,
                "within_mean": within_mean,
                "between_mean": between_mean,
                "separation": separation,
            }
        )

    # Global permutation test
    print("\n" + "=" * 80)
    print("GLOBAL PERMUTATION TEST")
    print("=" * 80)
    observed_delta = float(np.mean(between_dists) - np.mean(within_dists))
    all_dists = within_dists + between_dists
    labels = [0] * len(within_dists) + [1] * len(between_dists)
    rng = np.random.default_rng(42)
    perm_deltas = []
    for _ in range(N_PERM):
        shuffled = rng.permutation(labels)
        w_perm = [all_dists[i] for i, label in enumerate(shuffled) if label == 0]
        b_perm = [all_dists[i] for i, label in enumerate(shuffled) if label == 1]
        perm_deltas.append(np.mean(b_perm) - np.mean(w_perm))
    perm_p = empirical_p_value(perm_deltas, observed_delta, alternative="greater")

    mw_stat, mw_p = mannwhitneyu(between_dists, within_dists, alternative="greater")

    print(f"  Observed delta (between - within): {observed_delta:+.4f}")
    print(f"  Permutation p (n={N_PERM}):         {perm_p:.4f}")
    print(f"  Mann-Whitney p:                    {mw_p:.6f}")
    print()
    if perm_p < 0.05 and observed_delta > 0:
        print("  RESULT: Idiomatic and literal readings are geometrically separable.")
        print("          The model encodes meaning beyond surface form.")
    else:
        print("  RESULT: No significant geometric separation detected.")
        print(
            "          Geometry does not distinguish idiom from literal at this layer."
        )

    print("\n" + "=" * 80)
    print("INTERPRETATION")
    print("=" * 80)
    print("""
  Positive result (between > within, p < 0.05):
      The hidden-state geometry at the target phrase differs between
      idiomatic and literal contexts, even though surface tokens are identical.
      This supports the claim that predicate-conditioned geometry encodes
      relational/semantic meaning, not just surface form.

  Null result:
      The geometry does not separate readings at this layer.
      Possible explanations: wrong layer, insufficient context length,
      or the model does not encode this distinction geometrically.
      Does not falsify the broader HSG claim — idiom/literal separation
      may require a different probe or layer sweep.
    """)

    # Save results
    out = {
        "model": MODEL_NAME,
        "block_idx": block_idx,
        "per_phrase": results,
        "global": {
            "observed_delta": observed_delta,
            "perm_p": perm_p,
            "mannwhitney_p": float(mw_p),
            "within_mean": float(np.mean(within_dists)),
            "between_mean": float(np.mean(between_dists)),
        },
    }
    out_path = OUTPUT_DIR / "idiom_vs_literal_results.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"  Results saved to {out_path}")


if __name__ == "__main__":
    main()
