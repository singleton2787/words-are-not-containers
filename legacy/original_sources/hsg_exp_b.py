#!/usr/bin/env python3
"""
HSG Experiment B — Predicate-Licensed Object Fields
=====================================================
For each predicate, test whether the model recovers internal affordance
subclasses within the predicate's natural object domain.

Question: Within objects naturally licensed by a predicate, does the model
organize them by affordance subclass?

This is distinct from Experiment A (broad anchor test) which uses the same
objects across all predicates. Here each predicate gets its own object universe.

Run:
    python3 hsg_exp_b.py
    python3 hsg_exp_b.py --model Qwen/Qwen2-0.5B   # faster, for testing
"""

import os, sys, json, argparse
import numpy as np
from itertools import combinations
from pathlib import Path
from scipy import stats as ss

OUTPUT_DIR = os.path.expanduser("~/hsg/exp_b_results")
CACHE_DIR  = os.path.expanduser("~/.cache/huggingface")

# ── Predicate-licensed object fields ─────────────────────────────────────────
# Each predicate gets its own object universe with internal subclasses.
# Objects are grammatically natural for the predicate.

PREDICATE_FIELDS = {

    "eat": {
        "templates": [
            "I eat {obj}.",
            "She eats {obj} every day.",
            "We eat {obj} together.",
            "He ate {obj} for dinner.",
        ],
        "classes": {
            "staple":     ["bread", "soup", "pasta", "tacos", "rice"],
            "indulgence": ["chocolate", "cake", "ice cream", "cookies", "pie"],
            "preference": ["pizza", "sushi", "steak", "salad", "cheese"],
        },
        "prediction": "staple/staple and indulgence/indulgence should cluster; staple/indulgence should be more distant",
    },

    "fear": {
        "templates": [
            "I fear {obj}.",
            "She fears {obj}.",
            "They fear {obj}.",
            "He is afraid of {obj}.",
        ],
        "classes": {
            "physical":    ["disease", "fire", "violence", "death", "pain"],
            "existential": ["failure", "loss", "betrayal", "loneliness", "meaninglessness"],
            "ambient":     ["darkness", "heights", "spiders", "storms", "silence"],
        },
        "prediction": "physical threats cluster together; existential threats cluster; ambient fears cluster",
    },

    "own": {
        "templates": [
            "I own {obj}.",
            "She owns {obj}.",
            "They own {obj}.",
            "He bought {obj}.",
        ],
        "classes": {
            "property":  ["a house", "land", "a boat", "a building", "a farm"],
            "vehicle":   ["a car", "a truck", "a motorcycle", "a bicycle", "a plane"],
            "living":    ["a dog", "a horse", "a cat", "cattle", "a parrot"],
            "asset":     ["a painting", "a business", "stocks", "jewelry", "a patent"],
        },
        "prediction": "property clusters with property; living things cluster together",
    },

    "care": {
        "templates": [
            "I care for {obj}.",
            "She cares for {obj}.",
            "They care for {obj}.",
            "He takes care of {obj}.",
        ],
        "classes": {
            "dependent":  ["a child", "a baby", "an elder", "a patient", "an invalid"],
            "companion":  ["a dog", "a cat", "a friend", "a partner", "a horse"],
            "family":     ["my mother", "my father", "my sister", "my brother", "my family"],
        },
        "prediction": "dependent care clusters (child/baby/elder); companion care clusters (dog/friend)",
    },

    # Anchor test — broad object universe (Experiment A style, kept for comparison)
    "love": {
        "templates": [
            "I love {obj}.",
            "She loves {obj}.",
            "They love {obj}.",
            "He deeply loves {obj}.",
        ],
        "classes": {
            "hedonic":    ["pizza", "chocolate", "coffee", "music", "wine"],
            "companion":  ["my dog", "my cat", "my child", "my friend", "my horse"],
            "covenantal": ["my wife", "my husband", "my country", "God", "my family"],
            "moral":      ["my enemy", "a stranger", "my neighbor", "the other", "my rival"],
        },
        "prediction": "hedonic clusters; covenantal clusters; hedonic most distant from moral",
    },
}

TEMPLATES_DEFAULT = [
    "I {verb} {obj}.",
    "She {verb}s {obj}.",
    "They {verb} {obj}.",
]

# ── Geometry ──────────────────────────────────────────────────────────────────

def get_vector(model, tok, sentence, layer_idx, device):
    import torch
    inp = tok(sentence, return_tensors='pt',
              truncation=True, max_length=128).to(device)
    with torch.no_grad():
        out = model(**inp, output_hidden_states=True)
    return out.hidden_states[layer_idx][0, -1, :].float().cpu().numpy()

def l2(v1, v2):
    return float(np.linalg.norm(v1 - v2))

def permutation_test(vecs_by_class, n_perm=1000):
    """
    Test whether within-class < between-class distances.
    Uses class-label permutation to handle non-independence.
    """
    all_items = [(obj, cls) for cls, objs in vecs_by_class.items()
                 for obj in objs]

    within, between = [], []
    for (o1, c1), (o2, c2) in combinations(all_items, 2):
        d = l2(vecs_by_class[c1][o1], vecs_by_class[c2][o2]
               if c1 != c2 else vecs_by_class[c1][o2])
        if c1 == c2: within.append(d)
        else: between.append(d)

    if not within or not between:
        return 1.0, 1.0, 0.0

    observed_ratio = np.mean(between) / np.mean(within)

    np.random.seed(42)
    obj_list = [o for o, _ in all_items]
    cls_list = [c for _, c in all_items]
    perm_ratios = []
    for _ in range(n_perm):
        perm_cls = np.random.permutation(cls_list)
        perm_map = dict(zip(obj_list, perm_cls))
        pw, pb = [], []
        for (o1, _), (o2, _) in combinations(all_items, 2):
            c1_p = perm_map[o1]; c2_p = perm_map[o2]
            # Recompute distance using original vecs
            orig_c1 = next(c for c, objs in vecs_by_class.items() if o1 in objs)
            orig_c2 = next(c for c, objs in vecs_by_class.items() if o2 in objs)
            d = l2(vecs_by_class[orig_c1][o1], vecs_by_class[orig_c2][o2]
                   if orig_c1 != orig_c2 else vecs_by_class[orig_c1][o2])
            if c1_p == c2_p: pw.append(d)
            else: pb.append(d)
        if pw: perm_ratios.append(np.mean(pb) / np.mean(pw))

    p_val = float(np.mean(np.array(perm_ratios) >= observed_ratio))
    d_stat = (np.mean(between) - np.mean(within)) / np.sqrt(
        (np.std(within)**2 + np.std(between)**2) / 2)

    return observed_ratio, p_val, d_stat

# ── Main ──────────────────────────────────────────────────────────────────────

def run_experiment_b(model, tok, n_layers, device):
    block_idx = max(1, n_layers - 2)
    print(f"  Block {block_idx}/{n_layers} (n-2 invariant)")

    all_results = {}

    for predicate, field in PREDICATE_FIELDS.items():
        print(f"\n  Predicate: {predicate.upper()}")
        print(f"  Prediction: {field['prediction']}")

        templates = field.get('templates', TEMPLATES_DEFAULT)
        classes   = field['classes']

        # Extract vectors — average across templates
        vecs_by_class = {}
        for cls, objects in classes.items():
            vecs_by_class[cls] = {}
            for obj in objects:
                tvecs = []
                for tmpl in templates:
                    # Handle templates with {verb} vs predicate-specific
                    if '{verb}' in tmpl:
                        sentence = tmpl.format(verb=predicate, obj=obj)
                    elif '{obj}' in tmpl:
                        sentence = tmpl.format(obj=obj)
                    else:
                        sentence = f"I {predicate} {obj}."
                    tvecs.append(get_vector(model, tok, sentence, block_idx, device))
                vecs_by_class[cls][obj] = np.mean(tvecs, axis=0)

        # Permutation test
        ratio, p_val, d_stat = permutation_test(vecs_by_class)
        print(f"  ratio={ratio:.3f}  p={p_val:.4f}  d={d_stat:.3f}  "
              f"{'CONFIRMED' if p_val < 0.05 else 'NOT CONFIRMED'}")

        # Per-class-pair distances
        class_names = list(classes.keys())
        pair_dists = {}
        for c1, c2 in combinations(class_names, 2):
            dists = []
            for o1 in classes[c1]:
                for o2 in classes[c2]:
                    dists.append(l2(vecs_by_class[c1][o1], vecs_by_class[c2][o2]))
            pair_dists[f"{c1}/{c2}"] = float(np.mean(dists))

        within_dists = {}
        for cls in class_names:
            objs = classes[cls]
            dists = [l2(vecs_by_class[cls][o1], vecs_by_class[cls][o2])
                     for o1, o2 in combinations(objs, 2)]
            within_dists[cls] = float(np.mean(dists)) if dists else 0.0

        print(f"  Within-class distances:")
        for cls, d in sorted(within_dists.items(), key=lambda x: x[1]):
            print(f"    {cls:<20} {d:.3f}")

        print(f"  Between-class distances:")
        for pair, d in sorted(pair_dists.items(), key=lambda x: x[1]):
            print(f"    {pair:<30} {d:.3f}")

        # Most similar within-class pairs
        print(f"  Tightest within-class pairs:")
        all_within = []
        for cls, objs in classes.items():
            for o1, o2 in combinations(objs, 2):
                d = l2(vecs_by_class[cls][o1], vecs_by_class[cls][o2])
                all_within.append((f"{o1}/{o2}", d, cls))
        for pair, d, cls in sorted(all_within, key=lambda x: x[1])[:5]:
            print(f"    {pair:<35} {d:.3f}  [{cls}]")

        all_results[predicate] = {
            'ratio': ratio, 'p': p_val, 'd': d_stat,
            'within_dists': within_dists,
            'pair_dists': pair_dists,
            'prediction': field['prediction'],
        }

    return all_results

def print_summary(all_results):
    print(f"\n{'='*65}")
    print("EXPERIMENT B — SUMMARY")
    print(f"{'='*65}")
    print(f"\n{'Predicate':<12} {'ratio':>8} {'p':>8} {'d':>8} {'result'}")
    print("-"*50)
    for pred, r in all_results.items():
        result = 'CONFIRMED' if r['p'] < 0.05 else 'NOT CONFIRMED'
        print(f"{pred:<12} {r['ratio']:>8.3f} {r['p']:>8.4f} "
              f"{r['d']:>8.3f} {result}")

    print(f"\nKey question: do predicate-licensed fields show internal")
    print(f"affordance structure? Within-class < between-class?")
    confirmed = sum(1 for r in all_results.values() if r['p'] < 0.05)
    print(f"\n{confirmed}/{len(all_results)} predicates show significant internal structure")

def main():
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='Qwen/Qwen2-7B')
    parser.add_argument('--output', default=OUTPUT_DIR)
    args = parser.parse_args()

    Path(args.output).mkdir(parents=True, exist_ok=True)
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

    print(f"HSG Experiment B — Predicate-Licensed Object Fields")
    print(f"Model: {args.model}")
    print(f"Device: {DEVICE}")

    tok = AutoTokenizer.from_pretrained(args.model, cache_dir=CACHE_DIR)
    if tok.pad_token is None: tok.pad_token = tok.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model, cache_dir=CACHE_DIR,
        output_hidden_states=True,
        torch_dtype=torch.float16 if DEVICE=='cuda' else torch.float32,
        device_map='auto' if DEVICE=='cuda' else None,
        low_cpu_mem_usage=True,
    )
    model.eval()
    n_layers = model.config.num_hidden_layers
    print(f"  {n_layers}L × d{model.config.hidden_size}\n")

    all_results = run_experiment_b(model, tok, n_layers, DEVICE)
    print_summary(all_results)

    # Save
    out = f"{args.output}/exp_b_results.json"
    with open(out, 'w') as f:
        json.dump({k: {kk: vv for kk, vv in v.items()
                       if isinstance(vv, (float, str, dict))}
                   for k, v in all_results.items()}, f, indent=2)
    print(f"\nResults: {out}")

if __name__ == '__main__':
    main()
