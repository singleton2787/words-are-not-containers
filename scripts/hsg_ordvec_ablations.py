"""
OrdVec Fiber Ablation Suite
============================
Mike Singleton — OrdVec / Ordinal Quotient Preservation tests

Four ablations to falsify the v2 result (layer=21, K=64, Q_ordvec 95% top-1).
A result that can't be falsified isn't a result.

ABLATION DESIGN
----------------
The null hypothesis for each: Q_ordvec performance is NOT due to semantic
structure in the residual stream. Each ablation destroys one specific source
of signal. If performance collapses, that source was load-bearing.

ABLATION 1: Random class labels
  Destroy: label structure
  Method:  Shuffle affordance labels randomly across items.
  Formal:  Tests whether SampleConsistent violations are label-dependent.
           If Q_ordvec still gets ~95% under random labels, the fiber
           boundaries are spurious — not QuotientCompatible with anything real.
  Expected: collapse to ~25% (4-class chance)

ABLATION 2: Random bitmaps
  Destroy: bitmap signal
  Method:  Replace hidden state bitmaps with random K-active bitmaps.
           Same weight K, same dimension D, pure noise.
  Formal:  Tests whether bitmapOverlap carries semantic information.
           Under random bitmaps, expected overlap = K²/D (the null floor).
           No signal above null → no discrimination → chance performance.
  Expected: collapse to ~25%

ABLATION 3: Wrong predicate
  Destroy: predicate conditioning
  Method:  Replace "I love X" with "I bought X".
           Affordance classes (hedonic/companion/covenantal/moral) are
           defined for the "love" predicate specifically.
  Formal:  HSG claim: predicate conditions the affordance manifold.
           Under a different predicate, the geometric fracture shouldn't hold.
           Tests ObservationFamilyTargetInvariant across predicate shift.
  Expected: degraded performance — classes bleed across boundaries

ABLATION 4: Layer 1 (embedding layer)
  Destroy: learned contextual representations
  Method:  Run Q_ordvec on layer 1 (token embeddings, no attention processing).
           The semantic structure should be absent before the network builds it.
  Formal:  Tests whether the signal is a property of deep representations
           or already present in surface token statistics.
  Expected: collapse to ~25% or near-chance

FORMAL GROUNDING (ordvec-formalization)
----------------------------------------
- SampleConsistent, QuotientCompatible: FiniteQuotientSearch.lean
- bitmapOverlap, constantWeightBitmapSpace: BitmapNull.lean
- ObservationFamilyTargetInvariant: FiniteObservationWindow.lean
- bitmapUniformPMF_overlapTail_prob (null floor): BitmapNull.lean
"""

# ruff: noqa: E402

# Portable path: insert repo root so shared modules are importable
import sys as _sys
from pathlib import Path as _Path

_REPO_ROOT = _Path(__file__).resolve().parents[1]
_sys.path.insert(0, str(_REPO_ROOT))
import torch
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from collections import defaultdict
import json
import random
from itertools import combinations
from stats import empirical_p_value

# ── Model ──────────────────────────────────────────────────────────────────────
MODEL_NAME = "Qwen/Qwen2-7B"
print(f"Loading {MODEL_NAME}...")
bnb_config = BitsAndBytesConfig(load_in_8bit=True)
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    device_map="auto",
    quantization_config=bnb_config,
    output_hidden_states=True,
)

# ── Parameters ─────────────────────────────────────────────────────────────────
D_TOTAL = 3584
CLASSES = ["hedonic", "companion", "covenantal", "moral"]
K = 64  # best config from v2
LAYER = 21  # best config from v2
LAYER_EMB = 1  # ablation 4: embedding layer
N_SHUFFLE = 1000  # random label shuffles for ablation 1

# ── Canonical sample ───────────────────────────────────────────────────────────
SAMPLE = [
    {"text": "I love pizza", "label": "hedonic"},
    {"text": "I love chocolate", "label": "hedonic"},
    {"text": "I love coffee", "label": "hedonic"},
    {"text": "I love music", "label": "hedonic"},
    {"text": "I love wine", "label": "hedonic"},
    {"text": "I love my dog", "label": "companion"},
    {"text": "I love my cat", "label": "companion"},
    {"text": "I love my child", "label": "companion"},
    {"text": "I love my friend", "label": "companion"},
    {"text": "I love my horse", "label": "companion"},
    {"text": "I love my wife", "label": "covenantal"},
    {"text": "I love my husband", "label": "covenantal"},
    {"text": "I love my country", "label": "covenantal"},
    {"text": "I love God", "label": "covenantal"},
    {"text": "I love my family", "label": "covenantal"},
    {"text": "I love my enemy", "label": "moral"},
    {"text": "I love a stranger", "label": "moral"},
    {"text": "I love my neighbor", "label": "moral"},
    {"text": "I love the other", "label": "moral"},
    {"text": "I love my rival", "label": "moral"},
]

# Ablation 3: wrong predicate — same objects, different verb
SAMPLE_BOUGHT = [
    {
        "text": f"I bought {s['text'].replace('I love ', '').replace('I love', '').strip()}",
        "label": s["label"],
    }
    for s in SAMPLE
]
# Fix edge cases
SAMPLE_BOUGHT = [
    {"text": "I bought pizza", "label": "hedonic"},
    {"text": "I bought chocolate", "label": "hedonic"},
    {"text": "I bought coffee", "label": "hedonic"},
    {"text": "I bought music", "label": "hedonic"},
    {"text": "I bought wine", "label": "hedonic"},
    {"text": "I bought my dog", "label": "companion"},
    {"text": "I bought my cat", "label": "companion"},
    {"text": "I bought my child", "label": "companion"},
    {"text": "I bought my friend", "label": "companion"},
    {"text": "I bought my horse", "label": "companion"},
    {"text": "I bought my wife", "label": "covenantal"},
    {"text": "I bought my husband", "label": "covenantal"},
    {"text": "I bought my country", "label": "covenantal"},
    {"text": "I bought God", "label": "covenantal"},
    {"text": "I bought my family", "label": "covenantal"},
    {"text": "I bought my enemy", "label": "moral"},
    {"text": "I bought a stranger", "label": "moral"},
    {"text": "I bought my neighbor", "label": "moral"},
    {"text": "I bought the other", "label": "moral"},
    {"text": "I bought my rival", "label": "moral"},
]
N = len(SAMPLE)


# ── Core functions ─────────────────────────────────────────────────────────────
def get_hidden_state(text, layer):
    inputs = tokenizer(text, return_tensors="pt").to("cuda")
    with torch.no_grad():
        outputs = model(**inputs)
    return outputs.hidden_states[layer][0, -1].float().cpu().numpy()


def to_bitmap(hs, k=K):
    """
    constantWeightBitmapSpace D K (BitmapNull.lean)
    Top-k activations -> K-active bitmap.
    """
    bm = np.zeros(D_TOTAL, dtype=np.uint8)
    bm[np.argsort(hs)[-k:]] = 1
    return bm


def random_bitmap(k=K):
    """
    Ablation 2: uniform random K-active bitmap.
    Expected overlap with any other random K-active bitmap = K²/D (null floor).
    Nelson: bitmapUniformPMF — the uniform law over constantWeightBitmapSpace.
    """
    bm = np.zeros(D_TOTAL, dtype=np.uint8)
    bm[np.random.choice(D_TOTAL, k, replace=False)] = 1
    return bm


def bitmap_overlap(q, d):
    """bitmapOverlap q d := (q ∩ d).card (BitmapNull.lean)"""
    return int(np.dot(q, d))


def q_ordvec_run(sample, bitmaps):
    """
    Q_ordvec: mean overlap per class across all other prototypes.
    Returns per-item predictions and accuracy.
    bitmaps: dict text -> bitmap (allows ablation 2 to inject random bitmaps)
    """
    correct_top1, correct_top2 = 0, 0
    items_out = []
    for i, item in enumerate(sample):
        bm_test = bitmaps[item["text"]]
        class_overlaps = defaultdict(list)
        for j, proto in enumerate(sample):
            if i == j:
                continue
            ov = bitmap_overlap(bm_test, bitmaps[proto["text"]])
            class_overlaps[proto["label"]].append(ov)
        scores = {cls: np.mean(v) if v else 0.0 for cls, v in class_overlaps.items()}
        for cls in CLASSES:
            if cls not in scores:
                scores[cls] = 0.0
        ranked = sorted(CLASSES, key=lambda c: scores[c], reverse=True)
        pred = ranked[0]
        top2 = item["label"] in ranked[:2]
        if pred == item["label"]:
            correct_top1 += 1
        if top2:
            correct_top2 += 1
        items_out.append(
            {
                "text": item["text"],
                "label": item["label"],
                "pred": pred,
                "correct": pred == item["label"],
            }
        )
    return correct_top1 / N, correct_top2 / N, items_out


def fiber_violations(sample, pred_key="pred"):
    """
    SampleConsistent check: same predicted bucket, different true label.
    Nelson: no_compatible_target_of_sample_collision (FiniteQuotientSearch.lean)
    """
    buckets = defaultdict(list)
    for item in sample:
        buckets[item[pred_key]].append(item)
    viols, total = 0, 0
    for members in buckets.values():
        for a, b in combinations(members, 2):
            total += 1
            if a["label"] != b["label"]:
                viols += 1
    return viols, total


def null_floor(k=K):
    return k * k / D_TOTAL


# ══════════════════════════════════════════════════════════════════════════════
# BASELINE — replicate v2 best config
# ══════════════════════════════════════════════════════════════════════════════
print(f"\nExtracting hidden states (layer={LAYER}, layer_emb={LAYER_EMB})...")
hs_love = {}  # layer 21, "I love X"
hs_emb = {}  # layer 1,  "I love X"
hs_bought = {}  # layer 21, "I bought X"

all_texts = (
    [(s["text"], LAYER, hs_love) for s in SAMPLE]
    + [(s["text"], LAYER_EMB, hs_emb) for s in SAMPLE]
    + [(s["text"], LAYER, hs_bought) for s in SAMPLE_BOUGHT]
)

for text, layer, store in all_texts:
    if text not in store:
        store[text] = get_hidden_state(text, layer)
        print(f"  [{layer}] {text}")

bitmaps_love = {s["text"]: to_bitmap(hs_love[s["text"]]) for s in SAMPLE}
bitmaps_emb = {s["text"]: to_bitmap(hs_emb[s["text"]]) for s in SAMPLE}
bitmaps_bought = {s["text"]: to_bitmap(hs_bought[s["text"]]) for s in SAMPLE_BOUGHT}

print("\nBaseline (v2 replication, layer=21, K=64):")
base_top1, base_top2, base_items = q_ordvec_run(SAMPLE, bitmaps_love)
base_viols, base_pairs = fiber_violations(base_items)
print(
    f"  top-1={base_top1 * 100:.0f}%  top-2={base_top2 * 100:.0f}%  "
    f"fiber violations={base_viols}/{base_pairs}"
)


# ══════════════════════════════════════════════════════════════════════════════
# ABLATION 1: Random class labels
# ══════════════════════════════════════════════════════════════════════════════
print(f"\n{'═' * 68}")
print(f"ABLATION 1: Random class labels ({N_SHUFFLE} shuffles)")
print("Destroys label structure. Expected: collapse to ~25%.")
print(f"{'═' * 68}")

shuf_top1_scores = []
labels = [s["label"] for s in SAMPLE]
for _ in range(N_SHUFFLE):
    shuffled = random.sample(labels, len(labels))
    shuf_sample = [
        {"text": sample["text"], "label": label}
        for sample, label in zip(SAMPLE, shuffled)
    ]
    t1, t2, _ = q_ordvec_run(shuf_sample, bitmaps_love)
    shuf_top1_scores.append(t1)

print(f"  Baseline top-1:         {base_top1 * 100:.0f}%")
print(f"  Shuffled mean top-1:    {np.mean(shuf_top1_scores) * 100:.1f}%")
print(f"  Shuffled max top-1:     {np.max(shuf_top1_scores) * 100:.0f}%")
print(f"  Shuffled std:           {np.std(shuf_top1_scores) * 100:.1f}%")
print("  Chance level (4-class): 25%")
shuf_empirical_p = empirical_p_value(shuf_top1_scores, base_top1, alternative="greater")
pct_above_base = shuf_empirical_p * 100
print(f"  Shuffles >= baseline:   {pct_above_base:.1f}%  (empirical p-value)")
abl1_verdict = "PASS" if np.mean(shuf_top1_scores) < 0.40 else "FAIL"
print(
    f"  Verdict: {abl1_verdict}  ({'signal is label-dependent' if abl1_verdict == 'PASS' else 'WARNING: signal persists under random labels'})"
)


# ══════════════════════════════════════════════════════════════════════════════
# ABLATION 2: Random bitmaps
# ══════════════════════════════════════════════════════════════════════════════
print(f"\n{'═' * 68}")
print(f"ABLATION 2: Random bitmaps ({N_SHUFFLE} runs)")
print("Destroys bitmap signal. Expected: collapse to ~25%.")
print(f"Null floor K²/D = {null_floor():.2f}")
print(f"{'═' * 68}")

rand_top1_scores = []
rand_overlap_same, rand_overlap_other = [], []

for trial in range(N_SHUFFLE):
    rand_bitmaps = {s["text"]: random_bitmap() for s in SAMPLE}
    t1, t2, ritems = q_ordvec_run(SAMPLE, rand_bitmaps)
    rand_top1_scores.append(t1)
    # Spot-check overlap distribution on first trial
    if trial == 0:
        for i, item in enumerate(SAMPLE):
            bm_i = rand_bitmaps[item["text"]]
            for j, other in enumerate(SAMPLE):
                if i == j:
                    continue
                ov = bitmap_overlap(bm_i, rand_bitmaps[other["text"]])
                if item["label"] == other["label"]:
                    rand_overlap_same.append(ov)
                else:
                    rand_overlap_other.append(ov)

print(f"  Baseline top-1:         {base_top1 * 100:.0f}%")
print(f"  Random mean top-1:      {np.mean(rand_top1_scores) * 100:.1f}%")
print(f"  Random max top-1:       {np.max(rand_top1_scores) * 100:.0f}%")
print(f"  Random std:             {np.std(rand_top1_scores) * 100:.1f}%")
print("  Chance level (4-class): 25%")
print(
    f"  Random same-class overlap (trial 0):  {np.mean(rand_overlap_same):.2f}  (null floor: {null_floor():.2f})"
)
print(f"  Random cross-class overlap (trial 0): {np.mean(rand_overlap_other):.2f}")
rand_empirical_p = empirical_p_value(rand_top1_scores, base_top1, alternative="greater")
pct_above_base2 = rand_empirical_p * 100
print(f"  Random runs >= baseline: {pct_above_base2:.1f}%  (empirical p-value)")
abl2_verdict = "PASS" if np.mean(rand_top1_scores) < 0.40 else "FAIL"
print(
    f"  Verdict: {abl2_verdict}  ({'signal requires real bitmaps' if abl2_verdict == 'PASS' else 'WARNING: signal persists with random bitmaps'})"
)


# ══════════════════════════════════════════════════════════════════════════════
# ABLATION 3: Wrong predicate ("I bought X")
# ══════════════════════════════════════════════════════════════════════════════
print(f"\n{'═' * 68}")
print("ABLATION 3: Wrong predicate ('I bought X' vs 'I love X')")
print("Destroys predicate conditioning. Expected: degraded performance.")
print("Tests HSG claim: predicate conditions the affordance manifold.")
print(f"{'═' * 68}")

bought_top1, bought_top2, bought_items = q_ordvec_run(SAMPLE_BOUGHT, bitmaps_bought)
bought_viols, bought_pairs = fiber_violations(bought_items)

# Also measure overlap separation for "bought"
bought_same, bought_other = [], []
for i, item in enumerate(SAMPLE_BOUGHT):
    bm_i = bitmaps_bought[item["text"]]
    for j, other in enumerate(SAMPLE_BOUGHT):
        if i == j:
            continue
        ov = bitmap_overlap(bm_i, bitmaps_bought[other["text"]])
        if item["label"] == other["label"]:
            bought_same.append(ov)
        else:
            bought_other.append(ov)

love_same, love_other = [], []
for i, item in enumerate(SAMPLE):
    bm_i = bitmaps_love[item["text"]]
    for j, other in enumerate(SAMPLE):
        if i == j:
            continue
        ov = bitmap_overlap(bm_i, bitmaps_love[other["text"]])
        if item["label"] == other["label"]:
            love_same.append(ov)
        else:
            love_other.append(ov)

print(
    f"  'I love X'   top-1={base_top1 * 100:.0f}%  top-2={base_top2 * 100:.0f}%  "
    f"fiber violations={base_viols}/{base_pairs}"
)
print(
    f"  'I bought X' top-1={bought_top1 * 100:.0f}%  top-2={bought_top2 * 100:.0f}%  "
    f"fiber violations={bought_viols}/{bought_pairs}"
)
print("")
print("  Overlap separation (same vs cross class):")
print(
    f"    'I love X':   same={np.mean(love_same):.1f}  other={np.mean(love_other):.1f}  "
    f"delta={np.mean(love_same) - np.mean(love_other):.1f}"
)
print(
    f"    'I bought X': same={np.mean(bought_same):.1f}  other={np.mean(bought_other):.1f}  "
    f"delta={np.mean(bought_same) - np.mean(bought_other):.1f}"
)
print(f"    Null floor: {null_floor():.2f}")
delta_love = np.mean(love_same) - np.mean(love_other)
delta_bought = np.mean(bought_same) - np.mean(bought_other)
print(f"  Predicate effect on separation: {delta_love:.1f} vs {delta_bought:.1f}")

print("\n  Per-item 'I bought X' predictions:")
print(f"  {'Item':<30} {'True':<12} {'Pred':<12} {'Match'}")
print(f"  {'─' * 30} {'─' * 12} {'─' * 12} {'─' * 5}")
for r in bought_items:
    print(
        f"  {r['text']:<30} {r['label']:<12} {r['pred']:<12} "
        f"{'✓' if r['correct'] else '✗'}"
    )

abl3_verdict = "PASS" if bought_top1 < base_top1 else "INCONCLUSIVE"
print(
    f"\n  Verdict: {abl3_verdict}  "
    f"({'predicate conditioning matters' if abl3_verdict == 'PASS' else 'signal persists — classes may be object-driven not predicate-driven'})"
)


# ══════════════════════════════════════════════════════════════════════════════
# ABLATION 4: Embedding layer (layer 1)
# ══════════════════════════════════════════════════════════════════════════════
print(f"\n{'═' * 68}")
print(f"ABLATION 4: Embedding layer (layer {LAYER_EMB})")
print("Destroys learned contextual representations.")
print("Expected: collapse toward chance — structure is built by the network.")
print(f"{'═' * 68}")

emb_top1, emb_top2, emb_items = q_ordvec_run(SAMPLE, bitmaps_emb)
emb_viols, emb_pairs = fiber_violations(emb_items)

emb_same, emb_other = [], []
for i, item in enumerate(SAMPLE):
    bm_i = bitmaps_emb[item["text"]]
    for j, other in enumerate(SAMPLE):
        if i == j:
            continue
        ov = bitmap_overlap(bm_i, bitmaps_emb[other["text"]])
        if item["label"] == other["label"]:
            emb_same.append(ov)
        else:
            emb_other.append(ov)

print(
    f"  Layer {LAYER_EMB} (embedding): top-1={emb_top1 * 100:.0f}%  top-2={emb_top2 * 100:.0f}%  "
    f"fiber violations={emb_viols}/{emb_pairs}"
)
print(
    f"  Layer {LAYER}  (best):      top-1={base_top1 * 100:.0f}%  top-2={base_top2 * 100:.0f}%  "
    f"fiber violations={base_viols}/{base_pairs}"
)
print("")
print("  Overlap separation at embedding layer:")
print(f"    same-class:   {np.mean(emb_same):.1f}")
print(f"    cross-class:  {np.mean(emb_other):.1f}")
print(f"    delta:        {np.mean(emb_same) - np.mean(emb_other):.1f}")
print(f"    null floor:   {null_floor():.2f}")
abl4_verdict = "PASS" if emb_top1 < base_top1 - 0.15 else "INCONCLUSIVE"
print(
    f"  Verdict: {abl4_verdict}  "
    f"({'structure is network-learned' if abl4_verdict == 'PASS' else 'structure present at embedding layer — may be token-level'})"
)


# ══════════════════════════════════════════════════════════════════════════════
# SUMMARY TABLE
# ══════════════════════════════════════════════════════════════════════════════
print(f"\n{'═' * 68}")
print("ABLATION SUMMARY")
print(f"{'═' * 68}")
print(f"{'Condition':<35} {'top-1':>6} {'top-2':>6} {'Viols':>8} {'Verdict'}")
print(f"{'─' * 35} {'─' * 6} {'─' * 6} {'─' * 8} {'─' * 12}")
rows = [
    (
        "Baseline (layer=21, K=64, I love)",
        base_top1,
        base_top2,
        base_viols,
        base_pairs,
        "—",
    ),
    (
        "Abl 1: random labels (mean)",
        np.mean(shuf_top1_scores),
        None,
        None,
        None,
        abl1_verdict,
    ),
    (
        "Abl 2: random bitmaps (mean)",
        np.mean(rand_top1_scores),
        None,
        None,
        None,
        abl2_verdict,
    ),
    (
        "Abl 3: wrong predicate (I bought)",
        bought_top1,
        bought_top2,
        bought_viols,
        bought_pairs,
        abl3_verdict,
    ),
    (
        "Abl 4: embedding layer (layer=1)",
        emb_top1,
        emb_top2,
        emb_viols,
        emb_pairs,
        abl4_verdict,
    ),
]
for name, t1, t2, v, p, verdict in rows:
    t2_str = f"{t2 * 100:.0f}%" if t2 is not None else "  —  "
    v_str = f"{v}/{p}" if v is not None else "  —  "
    print(f"{name:<35} {t1 * 100:>5.0f}% {t2_str:>6} {v_str:>8}  {verdict}")

print("\nChance level (4-class): 25%")
print(f"Null floor K²/D = {null_floor():.2f}")

# ── Save ──────────────────────────────────────────────────────────────────────
results = {
    "model": MODEL_NAME,
    "best_layer": LAYER,
    "K": K,
    "n_shuffles": N_SHUFFLE,
    "null_floor": null_floor(),
    "baseline": {"top1": base_top1, "top2": base_top2, "violations": base_viols},
    "ablation1_random_labels": {
        "mean_top1": float(np.mean(shuf_top1_scores)),
        "max_top1": float(np.max(shuf_top1_scores)),
        "std_top1": float(np.std(shuf_top1_scores)),
        "empirical_p": float(shuf_empirical_p),
        "verdict": abl1_verdict,
    },
    "ablation2_random_bitmaps": {
        "mean_top1": float(np.mean(rand_top1_scores)),
        "max_top1": float(np.max(rand_top1_scores)),
        "std_top1": float(np.std(rand_top1_scores)),
        "empirical_p": float(rand_empirical_p),
        "null_overlap_same": float(np.mean(rand_overlap_same)),
        "null_overlap_other": float(np.mean(rand_overlap_other)),
        "verdict": abl2_verdict,
    },
    "ablation3_wrong_predicate": {
        "top1": bought_top1,
        "top2": bought_top2,
        "violations": bought_viols,
        "overlap_delta_love": float(delta_love),
        "overlap_delta_bought": float(delta_bought),
        "verdict": abl3_verdict,
        "per_item": bought_items,
    },
    "ablation4_embedding_layer": {
        "top1": emb_top1,
        "top2": emb_top2,
        "violations": emb_viols,
        "overlap_same": float(np.mean(emb_same)),
        "overlap_other": float(np.mean(emb_other)),
        "verdict": abl4_verdict,
    },
}
_OUT = (
    _Path(__file__).resolve().parents[1]
    / "results"
    / "hsg_ordvec_ablations_results.json"
)
_OUT.parent.mkdir(parents=True, exist_ok=True)
with open(_OUT, "w") as f:
    json.dump(results, f, indent=2)
print("\nResults -> hsg_ordvec_ablations_results.json")
