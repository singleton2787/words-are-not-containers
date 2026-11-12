"""
HSG x OrdVec Layer Role Test
==============================
Mike Singleton — OrdVec / Ordinal Quotient Preservation tests

QUESTION
---------
Layer 21 gives best Q_ordvec agreement with full geometry (95%).
Layer 26 is where HSG curvature peaks.
Are they doing different things, or the same thing at different resolutions?

THREE MEASUREMENTS PER LAYER (layers 1-27)
--------------------------------------------
1. Q_full accuracy (LOO)
   Does full geometry classify the affordance correctly at this layer?
   Measures: is representational structure present?

2. Q_ordvec agreement with Q_full
   Does bitmap overlap (cheap, ordinal) recover the full geometry decision?
   Measures: is the ordinal signal sufficient to approximate geometry?
   Nelson: ObservationFamilyTargetInvariant — does the overlap family
   separate the target as well as the full observation? (FiniteObservationWindow.lean)

3. Generation prediction AUC
   Does the geometry margin at this layer predict whether the model's
   generated completion matches the expected affordance register?
   Measures: is the representation causally connected to output?
   Higher AUC = this layer's geometry is output-aligned.

EXPECTED OUTCOMES
------------------
If layer 21 has best Q_ordvec agreement AND layer 26 has best generation AUC:
  → Layer 21 = clean relational representation layer
  → Layer 26 = behavior/output aligned layer
  → Different roles, different tools needed

If both peak at same layer:
  → One affordance-decision layer doing both jobs
  → HSG curvature peak and ordvec signal peak are the same phenomenon

If Q_full stable but Q_ordvec varies across layers:
  → Full geometry is robust; compression (bitmap) quality degrades away from peak
  → Issue is ordinal approximation quality, not representation quality

FORMAL GROUNDING
-----------------
Q_ordvec / fiber structure:    FiniteQuotientSearch.lean, FiniteObservationWindow.lean
bitmapOverlap, null floor:     BitmapNull.lean
Generation AUC:                empirical causal test — does geometry predict output?
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
from sklearn.metrics import roc_auc_score
from collections import defaultdict
import json
from itertools import combinations

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
K = 64  # best from v2 sweep
CLASSES = ["hedonic", "companion", "covenantal", "moral"]
LAYER_SWEEP = list(range(1, 28))  # all layers 1-27

# Generation parameters
N_TOKENS = 50
TEMPERATURE = 0.1  # low temp for reproducibility

# ── Sample ────────────────────────────────────────────────────────────────────
# Probe objects for generation (from hsg_generation.py)
PROBE_OBJECTS = [
    ("pizza", "hedonic"),
    ("chocolate", "hedonic"),
    ("coffee", "hedonic"),
    ("my dog", "companion"),
    ("my cat", "companion"),
    ("my friend", "companion"),
    ("my wife", "covenantal"),
    ("my husband", "covenantal"),
    ("my country", "covenantal"),
    ("my enemy", "moral"),
    ("a stranger", "moral"),
    ("my neighbor", "moral"),
]

# Full sample for Q_full / Q_ordvec measurements
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
N = len(SAMPLE)

# Keyword lists (from hsg_generation.py — exact copy)
KEYWORDS = {
    "hedonic": [
        "taste",
        "tastes",
        "flavor",
        "delicious",
        "eat",
        "eating",
        "food",
        "hungry",
        "hunger",
        "enjoy",
        "enjoyable",
        "pleasure",
        "satisfying",
        "craving",
        "crave",
        "appetite",
        "comfort",
        "comforting",
        "good",
        "smell",
        "smells",
        "bite",
        "cooking",
        "cook",
        "meal",
        "dining",
        "yummy",
        "treat",
        "indulge",
        "indulgence",
        "sweet",
        "savory",
        "warm",
        "warming",
        "nourish",
        "nourishing",
        "simple",
        "easy",
    ],
    "companion": [
        "loyal",
        "loyalty",
        "faithful",
        "companion",
        "companionship",
        "always there",
        "there for",
        "never judge",
        "unconditional",
        "play",
        "walks",
        "walk",
        "together",
        "beside",
        "presence",
        "comfort",
        "greet",
        "greeting",
        "wag",
        "purr",
        "care",
        "depend",
        "dependable",
        "trust",
        "trusting",
        "friend",
        "friendship",
        "bond",
        "bonded",
        "affection",
        "affectionate",
        "gentle",
        "innocent",
        "joy",
        "simple joy",
        "every day",
        "routine",
        "home",
        "safe",
    ],
    "covenantal": [
        "commit",
        "commitment",
        "vow",
        "vows",
        "promise",
        "promises",
        "through",
        "together",
        "partnership",
        "partner",
        "shared",
        "life",
        "years",
        "grow",
        "grown",
        "build",
        "built",
        "family",
        "home",
        "sacrifice",
        "support",
        "supported",
        "understanding",
        "knowing",
        "known",
        "history",
        "journey",
        "chose",
        "choose",
        "trust",
        "faithful",
        "fidelity",
        "devoted",
        "devotion",
        "complete",
        "whole",
        "steady",
        "constant",
        "unconditional",
        "everything",
        "always",
        "forever",
        "us",
        "we",
        "our",
    ],
    "moral": [
        "command",
        "commanded",
        "taught",
        "teaching",
        "duty",
        "obligation",
        "forgive",
        "forgiveness",
        "mercy",
        "compassion",
        "overcome",
        "transcend",
        "beyond",
        "difficult",
        "hard",
        "choose",
        "choice",
        "right",
        "principle",
        "virtue",
        "humility",
        "humble",
        "enemy",
        "stranger",
        "other",
        "difficult people",
        "those who",
        "scripture",
        "faith",
        "believe",
        "meaning",
        "purpose",
        "greater",
        "higher",
        "spiritual",
        "grow",
        "growth",
        "transform",
        "peace",
        "reconcile",
        "reconciliation",
        "understand",
        "empathy",
    ],
}


# ── Core functions ─────────────────────────────────────────────────────────────
def get_hidden_state(text, layer):
    inputs = tokenizer(text, return_tensors="pt").to("cuda")
    with torch.no_grad():
        out = model(**inputs)
    return out.hidden_states[layer][0, -1].float().cpu().numpy()


def to_bitmap(hs):
    bm = np.zeros(D_TOTAL, dtype=np.uint8)
    bm[np.argsort(hs)[-K:]] = 1
    return bm


def bitmap_overlap(q, d):
    """bitmapOverlap q d := (q ∩ d).card  (BitmapNull.lean)"""
    return int(np.dot(q, d))


def cosine_sim(a, b):
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))


def score_completion(text):
    """Keyword hit count per class. Returns scores dict and margin."""
    tl = text.lower()
    scores = {cls: sum(1 for kw in kws if kw in tl) for cls, kws in KEYWORDS.items()}
    return scores


def geo_margin(hs, centroids):
    """
    Geometry margin = similarity to correct centroid - similarity to nearest wrong.
    Higher = representation more clearly in the right affordance region.
    """
    sims = {cls: cosine_sim(hs, c) for cls, c in centroids.items()}
    ranked = sorted(CLASSES, key=lambda c: sims[c], reverse=True)
    return ranked[0], sims[ranked[0]] - sims[ranked[1]], sims


@torch.no_grad()
def generate(prompt):
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    ilen = inputs["input_ids"].shape[1]
    out = model.generate(
        **inputs,
        max_new_tokens=N_TOKENS,
        temperature=TEMPERATURE,
        do_sample=TEMPERATURE > 0,
        top_p=0.92,
        repetition_penalty=1.1,
        pad_token_id=tokenizer.eos_token_id,
    )
    return tokenizer.decode(out[0][ilen:], skip_special_tokens=True).strip()


# ══════════════════════════════════════════════════════════════════════════════
# STEP 1: Generate completions ONCE (layer-independent)
# ══════════════════════════════════════════════════════════════════════════════
print("\nGenerating completions (once, layer-independent)...")
generations = {}
for obj, cls in PROBE_OBJECTS:
    prompt = f"I love {obj} because"
    gen = generate(prompt)
    scores = score_completion(gen)
    total = sum(scores.values())
    winner = max(scores, key=scores.get) if total > 0 else None
    gen_correct = winner == cls
    gen_margin = scores.get(cls, 0) - max(
        (s for c, s in scores.items() if c != cls), default=0
    )
    generations[obj] = {
        "cls": cls,
        "prompt": prompt,
        "gen": gen,
        "scores": scores,
        "winner": winner,
        "gen_correct": gen_correct,
        "gen_margin": gen_margin,
    }
    marker = "✓" if gen_correct else "✗"
    print(f"  {marker} [{cls}] I love {obj} because: {gen[:60]}...")


# ══════════════════════════════════════════════════════════════════════════════
# STEP 2: Extract hidden states for ALL layers at once
# ══════════════════════════════════════════════════════════════════════════════
print(f"\nExtracting hidden states across {len(LAYER_SWEEP)} layers...")
# all_hs[text][layer] = vector
all_hs = defaultdict(dict)

# Texts needed: sample + probe prompts
probe_prompts = {f"I love {obj} because": cls for obj, cls in PROBE_OBJECTS}
all_texts = list(set([s["text"] for s in SAMPLE] + list(probe_prompts.keys())))

for text in all_texts:
    inputs = tokenizer(text, return_tensors="pt").to("cuda")
    with torch.no_grad():
        out = model(**inputs)
    for layer in LAYER_SWEEP:
        all_hs[text][layer] = out.hidden_states[layer][0, -1].float().cpu().numpy()
    print(f"  {text}")

print("Done.")


# ══════════════════════════════════════════════════════════════════════════════
# STEP 3: Layer sweep — three measurements per layer
# ══════════════════════════════════════════════════════════════════════════════
print(f"\n{'═' * 76}")
print(f"LAYER ROLE TEST — K={K}")
print(f"{'═' * 76}")
print(
    f"{'Layer':>6} | {'QF_acc':>7} | {'QO_agree':>9} | {'QO_viols':>9} | "
    f"{'Gen_AUC':>8} | {'Geo_AUC':>8}"
)
print(f"{'─' * 6}   {'─' * 7}   {'─' * 9}   {'─' * 9}   {'─' * 8}   {'─' * 8}")

sweep = []

for layer in LAYER_SWEEP:
    # ── Q_full LOO accuracy ──────────────────────────────────────────────────
    qf_correct = 0
    for i, item in enumerate(SAMPLE):
        hs = all_hs[item["text"]][layer]
        # LOO centroid
        centroids = {}
        for cls in CLASSES:
            vecs = [
                all_hs[SAMPLE[j]["text"]][layer]
                for j in range(N)
                if SAMPLE[j]["label"] == cls and j != i
            ]
            centroids[cls] = np.mean(vecs, axis=0) if vecs else np.zeros(D_TOTAL)
        pred, _, _ = geo_margin(hs, centroids)
        if pred == item["label"]:
            qf_correct += 1
    qf_acc = qf_correct / N

    # ── Q_ordvec agreement with Q_full (multi-prototype, LOO) ────────────────
    qo_agree_count = 0
    qo_items = []
    for i, item in enumerate(SAMPLE):
        hs = all_hs[item["text"]][layer]
        bm = to_bitmap(hs)

        # Q_full LOO prediction
        centroids = {}
        for cls in CLASSES:
            vecs = [
                all_hs[SAMPLE[j]["text"]][layer]
                for j in range(N)
                if SAMPLE[j]["label"] == cls and j != i
            ]
            centroids[cls] = np.mean(vecs, axis=0) if vecs else np.zeros(D_TOTAL)
        qf_pred, _, _ = geo_margin(hs, centroids)

        # Q_ordvec multi-prototype LOO
        class_ovs = defaultdict(list)
        for j, proto in enumerate(SAMPLE):
            if j == i:
                continue
            bm_p = to_bitmap(all_hs[proto["text"]][layer])
            class_ovs[proto["label"]].append(bitmap_overlap(bm, bm_p))
        scores = {cls: np.mean(v) if v else 0.0 for cls, v in class_ovs.items()}
        for cls in CLASSES:
            scores.setdefault(cls, 0.0)
        qo_pred = max(CLASSES, key=lambda c: scores[c])

        agree = qf_pred == qo_pred
        if agree:
            qo_agree_count += 1
        qo_items.append(
            {
                "label": item["label"],
                "q_full": qf_pred,
                "q_ordvec": qo_pred,
                "agree": agree,
            }
        )

    qo_agree = qo_agree_count / N

    # Fiber violations for Q_ordvec at this layer
    buckets = defaultdict(list)
    for it in qo_items:
        buckets[it["q_ordvec"]].append(it)
    qo_viols, qo_pairs = 0, 0
    for members in buckets.values():
        for a, b in combinations(members, 2):
            qo_pairs += 1
            if a["label"] != b["label"]:
                qo_viols += 1

    # ── Generation AUC ────────────────────────────────────────────────────────
    # For each probe object: get geometry margin at this layer
    # AUC = does geo_margin predict gen_correct?
    geo_margins, gen_corrects, gen_margins_list = [], [], []

    # Build full centroids (not LOO — probe objects may not be in sample)
    full_centroids = {}
    for cls in CLASSES:
        vecs = [all_hs[s["text"]][layer] for s in SAMPLE if s["label"] == cls]
        full_centroids[cls] = np.mean(vecs, axis=0)

    for obj, cls in PROBE_OBJECTS:
        prompt = f"I love {obj} because"
        hs = all_hs[prompt][layer]
        _, margin, _ = geo_margin(hs, full_centroids)
        gen_data = generations[obj]
        geo_margins.append(margin)
        gen_corrects.append(int(gen_data["gen_correct"]))
        gen_margins_list.append(gen_data["gen_margin"])

    # AUC: does geo_margin predict generation correctness?
    try:
        if len(set(gen_corrects)) > 1:
            gen_auc = roc_auc_score(gen_corrects, geo_margins)
        else:
            gen_auc = float("nan")
    except Exception:
        gen_auc = float("nan")

    # AUC: does geo_margin correlate with generation margin?
    try:
        if len(set(gen_margins_list)) > 1:
            geo_auc = roc_auc_score(
                [1 if m > 0 else 0 for m in gen_margins_list], geo_margins
            )
        else:
            geo_auc = float("nan")
    except Exception:
        geo_auc = float("nan")

    gen_auc_str = f"{gen_auc:.3f}" if not np.isnan(gen_auc) else "  nan"
    geo_auc_str = f"{geo_auc:.3f}" if not np.isnan(geo_auc) else "  nan"

    # Mark notable layers
    marker = ""
    if layer == 21:
        marker = " ← OrdVec peak (v2)"
    if layer == 26:
        marker = " ← HSG curvature peak"

    print(
        f"{layer:>6} | {qf_acc * 100:>6.0f}% | {qo_agree * 100:>8.0f}% | "
        f"{qo_viols:>4}/{qo_pairs:<4} | {gen_auc_str:>8} | "
        f"{geo_auc_str:>8}{marker}"
    )

    sweep.append(
        dict(
            layer=layer,
            qf_acc=qf_acc,
            qo_agree=qo_agree,
            qo_violations=qo_viols,
            qo_pairs=qo_pairs,
            gen_auc=None if np.isnan(gen_auc) else gen_auc,
            geo_auc=None if np.isnan(geo_auc) else geo_auc,
            geo_margins=geo_margins,
            gen_corrects=gen_corrects,
        )
    )


# ── Summary ────────────────────────────────────────────────────────────────────
valid_gen = [(r["layer"], r["gen_auc"]) for r in sweep if r["gen_auc"] is not None]
valid_qo = [(r["layer"], r["qo_agree"]) for r in sweep]
valid_qf = [(r["layer"], r["qf_acc"]) for r in sweep]

best_gen_layer, best_gen_auc = (
    max(valid_gen, key=lambda x: x[1]) if valid_gen else (None, None)
)
best_qo_layer, best_qo_agree = max(valid_qo, key=lambda x: x[1])
best_qf_layer, best_qf_acc = max(valid_qf, key=lambda x: x[1])

print(f"\n{'═' * 76}")
print("LAYER ROLE SUMMARY")
print(f"{'═' * 76}")
print(
    f"  Best Q_full accuracy:       layer {best_qf_layer:>2}  ({best_qf_acc * 100:.0f}%)"
)
print(
    f"  Best Q_ordvec agreement:    layer {best_qo_layer:>2}  ({best_qo_agree * 100:.0f}%)"
)
if best_gen_layer:
    print(
        f"  Best generation AUC:        layer {best_gen_layer:>2}  ({best_gen_auc:.3f})"
    )

print("\n  Layer 21 (OrdVec peak from v2):")
r21 = next(r for r in sweep if r["layer"] == 21)
g21 = f"{r21['gen_auc']:.3f}" if r21["gen_auc"] is not None else "nan"
print(
    f"    Q_full={r21['qf_acc'] * 100:.0f}%  Q_ordvec={r21['qo_agree'] * 100:.0f}%  gen_AUC={g21}"
)

print("\n  Layer 26 (HSG curvature peak):")
r26 = next(r for r in sweep if r["layer"] == 26)
g26 = f"{r26['gen_auc']:.3f}" if r26["gen_auc"] is not None else "nan"
print(
    f"    Q_full={r26['qf_acc'] * 100:.0f}%  Q_ordvec={r26['qo_agree'] * 100:.0f}%  gen_AUC={g26}"
)

print("\n  INTERPRETATION:")
if best_qo_layer != best_gen_layer and best_gen_layer:
    print(f"  Layer {best_qo_layer} = best ordinal representation (relational layer)")
    print(
        f"  Layer {best_gen_layer} = best generation prediction (output-aligned layer)"
    )
    print("  → Different roles. OrdVec and HSG are measuring complementary structure.")
elif best_qo_layer == best_gen_layer:
    print(f"  Both peak at layer {best_qo_layer}.")
    print("  → One affordance-decision layer. OrdVec and geometry are the same signal.")
else:
    print("  → Inconclusive. Check full table.")

# ── Save ──────────────────────────────────────────────────────────────────────
out = {
    "model": MODEL_NAME,
    "K": K,
    "best_qf_layer": best_qf_layer,
    "best_qf_acc": best_qf_acc,
    "best_qo_layer": best_qo_layer,
    "best_qo_agree": best_qo_agree,
    "best_gen_layer": best_gen_layer,
    "best_gen_auc": best_gen_auc,
    "layer_21": {
        "qf_acc": r21["qf_acc"],
        "qo_agree": r21["qo_agree"],
        "gen_auc": r21["gen_auc"],
    },
    "layer_26": {
        "qf_acc": r26["qf_acc"],
        "qo_agree": r26["qo_agree"],
        "gen_auc": r26["gen_auc"],
    },
    "sweep": [
        {k: v for k, v in r.items() if k not in ("geo_margins", "gen_corrects")}
        for r in sweep
    ],
    "generations": {
        obj: {k: v for k, v in d.items() if k != "gen"}
        for obj, d in generations.items()
    },
}
_OUT = (
    _Path(__file__).resolve().parents[1]
    / "results"
    / "hsg_ordvec_layer_role_results.json"
)
_OUT.parent.mkdir(parents=True, exist_ok=True)
with open(_OUT, "w") as f:
    json.dump(out, f, indent=2)
print(f"\nResults -> {_OUT}")
