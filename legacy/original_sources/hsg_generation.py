#!/usr/bin/env python3
"""
HSG Component 3 — Geometry Predicts Generation
===============================================
Does the affordance geometry of a predicate-object construction predict
the semantic register of the model's generated completion?

The loop:
    1. Extract final-token hidden state for "I love X because"
    2. Predict affordance class from geometry (centroid proximity)
    3. Generate a completion: "I love X because ___"
    4. Score the completion against affordance-class keyword lists
    5. Compare geometric prediction to generation register

If geometry predicts generation: the representation is doing real work,
not just organizing vectors in ways that don't affect output.

If geometry does NOT predict generation: the affordance structure is
a representational artifact — real in the residual stream but not
causally connected to what the model says.

Run:
    python3 hsg_generation.py
    python3 hsg_generation.py --model Qwen/Qwen2-7B
    python3 hsg_generation.py --n_tokens 60 --temperature 0.3
"""

import os, json, argparse
import numpy as np
from pathlib import Path
from itertools import combinations

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

OUTPUT_DIR = Path(os.path.expanduser("~/hsg/generation_results"))
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
CACHE_DIR  = os.path.expanduser("~/.cache/huggingface")

# ── Affordance class keyword lists ────────────────────────────────────────────
# These are the semantic registers we expect each class to generate.
# Deliberately broad — we want recall, not precision.

AFFORDANCE_KEYWORDS = {
    "hedonic": [
        "taste", "tastes", "flavor", "delicious", "eat", "eating", "food",
        "hungry", "hunger", "enjoy", "enjoyable", "pleasure", "satisfying",
        "craving", "crave", "appetite", "comfort", "comforting", "good",
        "smell", "smells", "bite", "cooking", "cook", "meal", "dining",
        "yummy", "treat", "indulge", "indulgence", "sweet", "savory",
        "warm", "warming", "nourish", "nourishing", "simple", "easy",
    ],
    "companion": [
        "loyal", "loyalty", "faithful", "companion", "companionship",
        "always there", "there for", "never judge", "unconditional",
        "play", "walks", "walk", "together", "beside", "presence",
        "comfort", "greet", "greeting", "wag", "purr", "care",
        "depend", "dependable", "trust", "trusting", "friend", "friendship",
        "bond", "bonded", "affection", "affectionate", "gentle", "innocent",
        "joy", "simple joy", "every day", "routine", "home", "safe",
    ],
    "covenantal": [
        "commit", "commitment", "vow", "vows", "promise", "promises",
        "through", "together", "partnership", "partner", "shared",
        "life", "years", "grow", "grown", "build", "built", "family",
        "home", "sacrifice", "support", "supported", "understanding",
        "knowing", "known", "history", "journey", "chose", "choose",
        "trust", "faithful", "fidelity", "devoted", "devotion",
        "complete", "whole", "steady", "constant", "unconditional",
        "everything", "always", "forever", "us", "we", "our",
    ],
    "moral": [
        "command", "commanded", "taught", "teaching", "duty", "obligation",
        "forgive", "forgiveness", "mercy", "compassion", "overcome",
        "transcend", "beyond", "difficult", "hard", "choose", "choice",
        "right", "principle", "virtue", "humility", "humble",
        "enemy", "stranger", "other", "difficult people", "those who",
        "scripture", "faith", "believe", "meaning", "purpose",
        "greater", "higher", "spiritual", "grow", "growth", "transform",
        "peace", "reconcile", "reconciliation", "understand", "empathy",
    ],
}

# ── Probe sentences ───────────────────────────────────────────────────────────

PROBE_OBJECTS = [
    # (object, expected_class, description)
    ("pizza",       "hedonic",    "hedonic — food preference"),
    ("chocolate",   "hedonic",    "hedonic — sweet indulgence"),
    ("coffee",      "hedonic",    "hedonic — daily pleasure"),
    ("my dog",      "companion",  "companion — loyal animal"),
    ("my cat",      "companion",  "companion — independent animal"),
    ("my friend",   "companion",  "companion — human friendship"),
    ("my wife",     "covenantal", "covenantal — marriage commitment"),
    ("my husband",  "covenantal", "covenantal — marriage commitment"),
    ("my family",   "covenantal", "covenantal — family bond"),
    ("my country",  "covenantal", "covenantal — civic devotion"),
    ("my enemy",    "moral",      "moral — commanded love"),
    ("a stranger",  "moral",      "moral — universal compassion"),
    ("my neighbor", "moral",      "moral — civic/ethical duty"),
]

# Anchor probes for building centroids (same as other scripts)
ANCHOR_OBJECTS = [
    ("pizza",       "hedonic"),
    ("chocolate",   "hedonic"),
    ("coffee",      "hedonic"),
    ("cake",        "hedonic"),
    ("ice cream",   "hedonic"),
    ("my dog",      "companion"),
    ("my cat",      "companion"),
    ("my horse",    "companion"),
    ("my friend",   "companion"),
    ("my child",    "companion"),
    ("my wife",     "covenantal"),
    ("my husband",  "covenantal"),
    ("my family",   "covenantal"),
    ("my country",  "covenantal"),
    ("God",         "covenantal"),
    ("my enemy",    "moral"),
    ("my rival",    "moral"),
    ("a stranger",  "moral"),
    ("my neighbor", "moral"),
    ("the other",   "moral"),
]

AFFORDANCE_CLASSES = ["hedonic", "companion", "covenantal", "moral"]

# ── Geometry helpers ──────────────────────────────────────────────────────────

def l2(v1, v2):
    return float(np.linalg.norm(v1 - v2))

@torch.no_grad()
def get_vector(model, tok, sentence, layer_idx, device):
    inp = tok(sentence, return_tensors='pt',
              truncation=True, max_length=128).to(device)
    out = model(**inp, output_hidden_states=True)
    return out.hidden_states[layer_idx][0, -1, :].float().cpu().numpy()

def build_centroids(model, tok, layer_idx, device):
    centroids = {cls: [] for cls in AFFORDANCE_CLASSES}
    for obj, cls in ANCHOR_OBJECTS:
        prompt = f"I love {obj} because"
        vec = get_vector(model, tok, prompt, layer_idx, device)
        centroids[cls].append(vec)
    return {cls: np.mean(vecs, axis=0) for cls, vecs in centroids.items()}

def geometric_class(vec, centroids):
    dists = {cls: l2(vec, c) for cls, c in centroids.items()}
    return min(dists, key=dists.get), dists

# ── Generation ────────────────────────────────────────────────────────────────

@torch.no_grad()
def generate_completion(model, tok, prompt, n_tokens, temperature, device):
    inp = tok(prompt, return_tensors='pt').to(device)
    input_len = inp['input_ids'].shape[1]

    out = model.generate(
        **inp,
        max_new_tokens=n_tokens,
        temperature=temperature,
        do_sample=temperature > 0,
        top_p=0.92,
        repetition_penalty=1.1,
        pad_token_id=tok.eos_token_id,
    )
    new_tokens = out[0][input_len:]
    return tok.decode(new_tokens, skip_special_tokens=True).strip()

# ── Scoring ───────────────────────────────────────────────────────────────────

def score_completion(text, keywords_by_class):
    """
    Score a completion against each affordance class keyword list.
    Returns dict of {class: score} where score = keyword hit count.
    Also returns the winning class and total hits.
    """
    text_lower = text.lower()
    scores = {}
    for cls, keywords in keywords_by_class.items():
        hits = sum(1 for kw in keywords if kw in text_lower)
        scores[cls] = hits

    total = sum(scores.values())
    if total == 0:
        return scores, None, 0  # no signal

    winner = max(scores, key=scores.get)
    # Check for tie
    max_score = scores[winner]
    winners = [c for c, s in scores.items() if s == max_score]
    if len(winners) > 1:
        winner = None  # tie — no clear register

    return scores, winner, total

def keyword_margin(scores, expected_class):
    """
    Keyword margin = expected class score - nearest wrong class score.
    Analogous to geometry margin.
    """
    expected = scores.get(expected_class, 0)
    others = [s for cls, s in scores.items() if cls != expected_class]
    nearest_wrong = max(others) if others else 0
    return expected - nearest_wrong

# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description='HSG Component 3: Geometry Predicts Generation')
    parser.add_argument('--model',       default='Qwen/Qwen2-7B')
    parser.add_argument('--n_tokens',    type=int,   default=50)
    parser.add_argument('--temperature', type=float, default=0.3)
    parser.add_argument('--n_samples',   type=int,   default=3,
                        help='Completions per prompt (averaged for robustness)')
    parser.add_argument('--output',      default=str(OUTPUT_DIR))
    args = parser.parse_args()

    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    Path(args.output).mkdir(exist_ok=True)

    print("="*70)
    print("HSG COMPONENT 3 — GEOMETRY PREDICTS GENERATION")
    print("="*70)
    print(f"Model:       {args.model}")
    print(f"Device:      {DEVICE}")
    print(f"n_tokens:    {args.n_tokens}")
    print(f"temperature: {args.temperature}")
    print(f"n_samples:   {args.n_samples}")

    # Load model
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.model, cache_dir=CACHE_DIR)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model, cache_dir=CACHE_DIR,
        output_hidden_states=True,
        torch_dtype=torch.float16 if DEVICE=='cuda' else torch.float32,
        device_map='auto' if DEVICE=='cuda' else None,
        low_cpu_mem_usage=True,
    )
    model.eval()

    n_layers  = model.config.num_hidden_layers
    layer_idx = max(1, int(round(n_layers * 0.35)))
    print(f"Layers:      {n_layers}  (using block {layer_idx})")

    # Build centroids
    print("\nBuilding affordance centroids...")
    centroids = build_centroids(model, tok, layer_idx, DEVICE)

    # Run probes
    print("\nRunning probes...\n")
    rows = []
    match_count = 0
    total_count = 0
    no_signal   = 0

    print(f"{'Object':<16} {'Expected':>11} {'Geom pred':>11} {'Gen pred':>11} {'Match':>7} {'Completion preview'}")
    print("-"*90)

    for obj, expected_cls, desc in PROBE_OBJECTS:
        prompt = f"I love {obj} because"

        # Geometry prediction
        vec = get_vector(model, tok, prompt, layer_idx, DEVICE)
        geo_pred, geo_dists = geometric_class(vec, centroids)
        geo_margin = geo_dists[geo_pred] - min(
            d for c, d in geo_dists.items() if c != geo_pred)
        geo_correct = geo_pred == expected_cls

        # Generate multiple completions and aggregate scores
        all_scores = {cls: 0 for cls in AFFORDANCE_CLASSES}
        completions = []

        for _ in range(args.n_samples):
            completion = generate_completion(
                model, tok, prompt, args.n_tokens, args.temperature, DEVICE)
            completions.append(completion)
            scores, _, _ = score_completion(completion, AFFORDANCE_KEYWORDS)
            for cls in AFFORDANCE_CLASSES:
                all_scores[cls] += scores[cls]

        # Winner from aggregated scores
        total_hits = sum(all_scores.values())
        if total_hits == 0:
            gen_pred = None
            no_signal += 1
        else:
            gen_pred = max(all_scores, key=all_scores.get)
            max_score = all_scores[gen_pred]
            ties = [c for c, s in all_scores.items() if s == max_score]
            if len(ties) > 1:
                gen_pred = None  # tie

        gen_correct  = gen_pred == expected_cls
        gen_margin   = keyword_margin(all_scores, expected_cls)

        # Agreement: does geometry predict generation?
        geo_gen_agree = (geo_pred == gen_pred) if gen_pred else None

        # Match = both correct
        both_correct = geo_correct and gen_correct
        if gen_pred is not None:
            total_count += 1
            if both_correct:
                match_count += 1

        preview = completions[0][:55].replace('\n', ' ') + "..."

        match_str = "✓✓" if both_correct else ("✓✗" if geo_correct else ("✗✓" if gen_correct else "✗✗"))
        print(f"{obj:<16} {expected_cls:>11} {(geo_pred or '?'):>11} {(gen_pred or 'tie/none'):>11} {match_str:>7}  {preview}")

        rows.append({
            'object':           obj,
            'expected_class':   expected_cls,
            'description':      desc,
            'prompt':           prompt,
            'geo_predicted':    geo_pred,
            'geo_correct':      geo_correct,
            'geo_margin':       float(geo_margin) if geo_margin else 0,
            'geo_distances':    {k: float(v) for k, v in geo_dists.items()},
            'gen_predicted':    gen_pred,
            'gen_correct':      gen_correct,
            'gen_scores':       {k: int(v) for k, v in all_scores.items()},
            'gen_margin':       int(gen_margin),
            'geo_gen_agree':    geo_gen_agree,
            'completions':      completions,
        })

    # ── Summary ────────────────────────────────────────────────────────────────
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)

    geo_correct_total = sum(1 for r in rows if r['geo_correct'])
    gen_correct_total = sum(1 for r in rows if r['gen_correct'])
    agree_rows        = [r for r in rows if r['geo_gen_agree'] is not None]
    agree_count       = sum(1 for r in agree_rows if r['geo_gen_agree'])
    n                 = len(rows)

    print(f"\n  Total probes:              {n}")
    print(f"  Geometry correct:          {geo_correct_total}/{n} ({geo_correct_total/n*100:.0f}%)")
    print(f"  Generation correct:        {gen_correct_total}/{n} ({gen_correct_total/n*100:.0f}%)")
    print(f"  No generation signal:      {no_signal}")
    if agree_rows:
        print(f"  Geo-gen agreement:         {agree_count}/{len(agree_rows)} ({agree_count/len(agree_rows)*100:.0f}%)")

    print(f"\n  Key question: does geometry predict generation?")
    if agree_rows:
        agree_rate = agree_count / len(agree_rows)
        if agree_rate >= 0.75:
            print(f"  YES — agreement rate {agree_rate*100:.0f}% — geometry is predictive of output register")
        elif agree_rate >= 0.5:
            print(f"  PARTIAL — agreement rate {agree_rate*100:.0f}% — geometry partially predictive")
        else:
            print(f"  NO — agreement rate {agree_rate*100:.0f}% — geometry does not predict generation register")

    # Per-class breakdown
    print(f"\n  Per-class results:")
    for cls in AFFORDANCE_CLASSES:
        cls_rows = [r for r in rows if r['expected_class'] == cls]
        g_ok = sum(1 for r in cls_rows if r['geo_correct'])
        c_ok = sum(1 for r in cls_rows if r['gen_correct'])
        print(f"  {cls:<12} geometry {g_ok}/{len(cls_rows)}  generation {c_ok}/{len(cls_rows)}")

    # Show completions for one example per class
    print(f"\n  Sample completions:")
    shown = set()
    for r in rows:
        if r['expected_class'] not in shown and r['completions']:
            shown.add(r['expected_class'])
            print(f"\n  [{r['expected_class']}] I love {r['object']} because...")
            print(f"  → {r['completions'][0][:200]}")

    # ── Verdict ────────────────────────────────────────────────────────────────
    print(f"\n" + "="*70)
    print("VERDICT")
    print("="*70)
    print("""
If geometry predicts generation (agreement >= 75%):
    The affordance structure in the residual stream is causally connected
    to what the model says. The representation is doing real work.

If geometry does NOT predict generation:
    The affordance structure is a representational artifact — measurable
    in the residual stream but not connected to output behavior.
    This would be an important negative result.

Either outcome is informative. The claim 'predicates are geometric operators'
requires that the geometry connects to behavior, not just representation.
""")

    # ── Save ───────────────────────────────────────────────────────────────────
    out_path = Path(args.output) / 'generation_results.json'
    with open(out_path, 'w') as f:
        json.dump({
            'model':    args.model,
            'n_tokens': args.n_tokens,
            'temperature': args.temperature,
            'layer_idx': layer_idx,
            'n_layers':  n_layers,
            'summary': {
                'n': n,
                'geo_correct': geo_correct_total,
                'gen_correct': gen_correct_total,
                'no_signal':   no_signal,
                'agree_count': agree_count,
                'agree_total': len(agree_rows),
            },
            'rows': rows,
        }, f, indent=2)
    print(f"  Results: {out_path}")

if __name__ == '__main__':
    main()
