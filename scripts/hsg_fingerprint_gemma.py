#!/usr/bin/env python3
"""
HSG Predicate Fingerprint Test — Qwen2-7B
==========================================
Tests whether verb-specific affordance fingerprints appear in a pretrained
large model (no training required).

If the same fingerprints appear in Qwen2-7B as in our 45M toy model,
the effect is not a training artifact — it's a property of how language
models encode predicate semantics.

Run:
    python3 hsg_fingerprint_qwen.py

Requires: transformers, torch, scipy, numpy
~20 minutes on A10G, ~4GB VRAM (float16)
"""

import json
import numpy as np
from itertools import combinations
from pathlib import Path
from hf_cache import hf_cache_dir
from stats import empirical_p_value

# ── Config ────────────────────────────────────────────────────────────────────

MODEL_NAME = "google/gemma-7b"
OUTPUT_DIR = str(
    Path(__file__).resolve().parents[1]
    / "results"
    / Path(__file__).stem.replace("hsg_", "").replace("fingerprint_", "fingerprint_")
)
CACHE_DIR = hf_cache_dir()

# Target verbs
VERBS = ["see", "love", "need", "fear", "eat", "own", "remember", "care", "hate"]

# Affordance classes with grammatical phrases
AFFORDANCE_CLASSES = {
    "hedonic": [
        "pizza",
        "chocolate",
        "coffee",
        "music",
        "wine",
    ],
    "companion": [
        "my dog",
        "my cat",
        "my child",
        "my friend",
        "my horse",
    ],
    "covenantal": [
        "my wife",
        "my husband",
        "my country",
        "God",
        "my family",
    ],
    "moral": [
        "my enemy",
        "a stranger",
        "my neighbor",
        "the other",
        "my rival",
    ],
}

# Multiple templates — average for robustness
TEMPLATES = [
    "I {verb} {obj}.",
    "We {verb} {obj}.",
    "She {verb} {obj}.",
    "They {verb} {obj}.",
]

Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

# ── Geometry ──────────────────────────────────────────────────────────────────


def get_vector(model, tok, sentence, layer_idx, device):
    """Final token hidden state at layer_idx."""
    import torch

    inp = tok(sentence, return_tensors="pt", truncation=True, max_length=128).to(device)
    with torch.no_grad():
        out = model(**inp, output_hidden_states=True)
    return out.hidden_states[layer_idx][0, -1, :].float().cpu().numpy()


def l2(v1, v2):
    return float(np.linalg.norm(v1 - v2))


# ── Fingerprint computation ───────────────────────────────────────────────────


def compute_verb_fingerprint(model, tok, n_layers, device, verb):
    """
    For a given verb, extract averaged vectors for all objects
    and compute pairwise distances by class pair.
    """
    block_idx = max(1, n_layers - 2)

    all_objects = [
        (obj, cls) for cls, objs in AFFORDANCE_CLASSES.items() for obj in objs
    ]

    vecs = {}
    for obj, cls in all_objects:
        tvecs = []
        for tmpl in TEMPLATES:
            sentence = tmpl.format(verb=verb, obj=obj)
            tvecs.append(get_vector(model, tok, sentence, block_idx, device))
        vecs[obj] = np.mean(tvecs, axis=0)

    # Class-pair mean distances
    class_dists = {}
    for (o1, c1), (o2, c2) in combinations(all_objects, 2):
        key = "/".join(sorted([c1, c2]))
        if key not in class_dists:
            class_dists[key] = []
        class_dists[key].append(l2(vecs[o1], vecs[o2]))

    return {k: float(np.mean(v)) for k, v in class_dists.items()}


def permutation_test(vecs, n_perm=1000):
    """
    Test whether within-class < between-class using label permutation.
    Returns (observed_ratio, p_value).
    """
    all_objects = [
        (obj, cls) for cls, objs in AFFORDANCE_CLASSES.items() for obj in objs
    ]

    # Observed distances
    within, between = [], []
    for (o1, c1), (o2, c2) in combinations(all_objects, 2):
        d = l2(vecs[o1], vecs[o2])
        if c1 == c2:
            within.append(d)
        else:
            between.append(d)

    observed_ratio = np.mean(between) / np.mean(within)

    # Permute
    np.random.seed(42)
    obj_list = [o for o, _ in all_objects]
    cls_list = [c for _, c in all_objects]
    perm_ratios = []
    for _ in range(n_perm):
        perm_cls = np.random.permutation(cls_list)
        perm_map = dict(zip(obj_list, perm_cls))
        pw, pb = [], []
        for (o1, _), (o2, _) in combinations(all_objects, 2):
            d = l2(vecs[o1], vecs[o2])
            if perm_map[o1] == perm_map[o2]:
                pw.append(d)
            else:
                pb.append(d)
        if pw:
            perm_ratios.append(np.mean(pb) / np.mean(pw))

    p_val = empirical_p_value(perm_ratios, observed_ratio, alternative="greater")
    d_stat = (np.mean(between) - np.mean(within)) / np.sqrt(
        (np.std(within) ** 2 + np.std(between) ** 2) / 2
    )

    return observed_ratio, p_val, d_stat, within, between


# ── Main ──────────────────────────────────────────────────────────────────────


def main():
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {DEVICE}")
    if DEVICE == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f}GB")

    print(f"\nLoading {MODEL_NAME}...")
    tok = AutoTokenizer.from_pretrained(MODEL_NAME, cache_dir=CACHE_DIR)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        cache_dir=CACHE_DIR,
        output_hidden_states=True,
        torch_dtype=torch.float16 if DEVICE == "cuda" else torch.float32,
        device_map="auto" if DEVICE == "cuda" else None,
        low_cpu_mem_usage=True,
    )
    model.eval()
    n_layers = model.config.num_hidden_layers
    d_model = model.config.hidden_size
    block_idx = max(1, n_layers - 2)
    print(f"  {n_layers}L × d{d_model}  target_block={block_idx}")

    # ── Per-verb fingerprints ──
    print(f"\nComputing fingerprints for {len(VERBS)} verbs...")
    verb_class_dists = {}
    verb_stats = {}

    for verb in VERBS:
        print(f"  {verb}...", end="", flush=True)

        # Get all vectors for this verb
        all_objects = [
            (obj, cls) for cls, objs in AFFORDANCE_CLASSES.items() for obj in objs
        ]
        vecs = {}
        for obj, cls in all_objects:
            tvecs = [
                get_vector(
                    model, tok, tmpl.format(verb=verb, obj=obj), block_idx, DEVICE
                )
                for tmpl in TEMPLATES
            ]
            vecs[obj] = np.mean(tvecs, axis=0)

        # Class-pair distances
        class_dists = {}
        for (o1, c1), (o2, c2) in combinations(all_objects, 2):
            key = "/".join(sorted([c1, c2]))
            if key not in class_dists:
                class_dists[key] = []
            class_dists[key].append(l2(vecs[o1], vecs[o2]))
        verb_class_dists[verb] = {k: float(np.mean(v)) for k, v in class_dists.items()}

        # Permutation test
        ratio, p_val, d_stat, within, between = permutation_test(vecs)
        verb_stats[verb] = {
            "ratio": ratio,
            "p": p_val,
            "d": d_stat,
            "within_mean": float(np.mean(within)),
            "between_mean": float(np.mean(between)),
        }
        print(f" ratio={ratio:.3f} p={p_val:.4f} d={d_stat:.3f}")

    # ── Residual fingerprints vs see baseline ──
    see_dists = verb_class_dists["see"]
    class_pairs = sorted(see_dists.keys())

    fingerprints = {}
    for verb in VERBS:
        if verb == "see":
            continue
        raw_diffs = {
            cp: verb_class_dists[verb][cp] - see_dists[cp] for cp in class_pairs
        }
        global_mean = np.mean(list(raw_diffs.values()))
        fingerprints[verb] = {cp: raw_diffs[cp] - global_mean for cp in class_pairs}

    # ── Print results ──
    print(f"\n{'=' * 75}")
    print("QWEN2-7B PREDICATE FINGERPRINTS  (residual vs see baseline)")
    print(f"Model: {MODEL_NAME}  Block: {block_idx}/{n_layers}")
    print(f"{'=' * 75}")

    print(f"\n{'Class pair':<28}" + "".join(f"{v:>10}" for v in VERBS if v != "see"))
    print("-" * 75)
    for cp in class_pairs:
        row = f"{cp:<28}"
        for verb in VERBS:
            if verb == "see":
                continue
            val = fingerprints[verb][cp]
            row += f"{val:>+10.3f}"
        print(row)

    print(f"\n{'=' * 75}")
    print("FINGERPRINT SUMMARY")
    print(f"{'=' * 75}")
    for verb in VERBS:
        if verb == "see":
            continue
        fp = fingerprints[verb]
        most_neg = min(fp, key=fp.get)
        most_pos = max(fp, key=fp.get)
        s = verb_stats[verb]
        print(f"  {verb:<10} ratio={s['ratio']:.3f} p={s['p']:.4f} d={s['d']:.3f}")
        print(f"             pulls: {most_neg:<28} ({fp[most_neg]:>+.3f})")
        print(f"             pushes: {most_pos:<28} ({fp[most_pos]:>+.3f})")

    # ── Prediction checks ──
    print(f"\n{'=' * 75}")
    print("PREDICTION CHECKS")
    print(f"{'=' * 75}")
    predictions = [
        ("love", "companion/companion", "negative", "love pulls companion closer"),
        ("love", "hedonic/moral", "positive", "love pushes hedonic/moral farther"),
        ("care", "companion/companion", "negative", "care pulls companion closer"),
        ("eat", "hedonic/hedonic", "negative", "eat pulls hedonic closer"),
        ("eat", "covenantal/covenantal", "positive", "eat pushes covenantal farther"),
        (
            "fear",
            "covenantal/hedonic",
            "negative",
            "fear pulls covenantal/hedonic closer",
        ),
        ("hate", "hedonic/moral", "negative", "hate pulls hedonic/moral closer"),
        ("own", "covenantal/covenantal", "negative", "own pulls covenantal closer"),
    ]
    confirmed = 0
    for verb, cp, direction, label in predictions:
        if verb not in fingerprints or cp not in fingerprints[verb]:
            continue
        val = fingerprints[verb][cp]
        ok = (val < 0 and direction == "negative") or (
            val > 0 and direction == "positive"
        )
        confirmed += ok
        print(
            f"  {'✓' if ok else '✗'} {verb:<8} {cp:<28} {val:>+7.3f}  "
            f"{'CONFIRMED' if ok else 'FAILED'}: {label}"
        )
    print(f"\n  {confirmed}/{len(predictions)} predictions confirmed")

    # ── Cross-model comparison ──
    print(f"\n{'=' * 75}")
    print("CROSS-MODEL COMPARISON: 45M toy vs Qwen2-7B")
    print(f"{'=' * 75}")
    toy_results = {
        "love": {
            "ratio": 1.220,
            "p": 0.0000,
            "d": 0.825,
            "pulls": "companion/companion",
            "pushes": "hedonic/moral",
        },
        "care": {
            "ratio": None,
            "pulls": "companion/companion",
            "pushes": "moral/moral",
        },
        "eat": {"pulls": "companion/companion", "pushes": "covenantal/covenantal"},
        "fear": {"pulls": "covenantal/hedonic", "pushes": "moral/moral"},
        "hate": {"pulls": "hedonic/moral", "pushes": "covenantal/covenantal"},
    }
    print(f"\n  {'Verb':<10} {'Toy pulls':<28} {'7B pulls':<28} {'Match'}")
    print(f"  {'-' * 85}")
    for verb in ["love", "care", "eat", "fear", "hate"]:
        if verb not in fingerprints or verb not in toy_results:
            continue
        fp = fingerprints[verb]
        qwen_pulls = min(fp, key=fp.get)
        toy_pulls = toy_results[verb].get("pulls", "?")
        match = (
            "✓"
            if qwen_pulls == toy_pulls
            else "~"
            if any(c in qwen_pulls for c in toy_pulls.split("/"))
            else "✗"
        )
        print(f"  {verb:<10} {toy_pulls:<28} {qwen_pulls:<28} {match}")

    # ── Save ──
    results = {
        "model": MODEL_NAME,
        "n_layers": n_layers,
        "block_idx": block_idx,
        "verb_stats": verb_stats,
        "fingerprints": {
            v: {k: float(x) for k, x in fp.items()} for v, fp in fingerprints.items()
        },
        "class_pairs": class_pairs,
    }
    out_path = f"{OUTPUT_DIR}/fingerprint_results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  Results: {out_path}")
    print("\nDone.")


if __name__ == "__main__":
    main()
