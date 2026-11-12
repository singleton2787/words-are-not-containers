"""
OrdVec Fiber Smoke Test v2
===========================
Mike Singleton — OrdVec / Ordinal Quotient Preservation tests

Improvements over v1:
  - Leave-one-out Q_full centroids (no data leakage)
  - Multi-prototype Q_ordvec (all class members as prototypes, majority vote)
  - Top-2 accuracy (is true label in top-2 predictions?)
  - K sweep: K=64, 128, 180, 256, 512
  - Layer sweep: layers 20-27

THREE QUOTIENT MAPS
--------------------
Q_popcount  : quartile of raw anchor overlap              (weak baseline)
Q_ordvec    : majority vote over all class prototype      (ordinal, bitmap-native)
              bitmaps — rank by overlap, vote by rank-1
Q_full      : leave-one-out nearest centroid by cosine    (full geometry, no leakage)

FORMAL GROUNDING
-----------------
SampleConsistent / local fiber falsifiers: FiniteQuotientSearch.lean
bitmapOverlap, hypergeometric null:        BitmapNull.lean
constantWeightBitmapSpace D K:             BitmapNull.lean
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
from scipy.stats import hypergeom
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

# ── Fixed parameters ───────────────────────────────────────────────────────────
D_TOTAL = 3584
CLASSES = ["hedonic", "companion", "covenantal", "moral"]
ANCHOR_TEXT = "I love"

K_SWEEP = [64, 128, 180, 256, 512]
LAYER_SWEEP = list(range(20, 28))  # layers 20-27

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


# ── Extract all hidden states once, all layers ─────────────────────────────────
print("\nExtracting hidden states (all layers, all items)...")
# shape: item -> layer -> vector
all_hs = {}
texts_to_extract = [item["text"] for item in SAMPLE] + [ANCHOR_TEXT]
for text in texts_to_extract:
    inputs = tokenizer(text, return_tensors="pt").to("cuda")
    with torch.no_grad():
        outputs = model(**inputs)
    # Store all layers as float32
    all_hs[text] = {
        layer: outputs.hidden_states[layer][0, -1].float().cpu().numpy()
        for layer in LAYER_SWEEP
    }
    print(f"  {text}")

print("Done extracting.")


# ── Core functions ─────────────────────────────────────────────────────────────
def to_bitmap(hs, K):
    bm = np.zeros(D_TOTAL, dtype=np.uint8)
    bm[np.argsort(hs)[-K:]] = 1
    return bm


def bitmap_overlap(q, d):
    return int(np.dot(q, d))


def cosine_sim(a, b):
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))


def top2_acc(pred_ranked, true_label):
    """True label appears in top-2 predicted classes."""
    return true_label in pred_ranked[:2]


# ── Q_full: leave-one-out nearest centroid ─────────────────────────────────────
def q_full_loo(item_idx, layer):
    """
    Leave-one-out: centroid for each class computed WITHOUT the test item.
    No data leakage. Returns (top1_pred, top2_pred, ranked_classes, sim_dict).
    """
    target = SAMPLE[item_idx]
    hs_test = all_hs[target["text"]][layer]

    # Build LOO centroids
    centroids = {}
    for cls in CLASSES:
        vecs = [
            all_hs[SAMPLE[i]["text"]][layer]
            for i in range(N)
            if SAMPLE[i]["label"] == cls and i != item_idx
        ]
        centroids[cls] = np.mean(vecs, axis=0) if vecs else np.zeros(D_TOTAL)

    sims = {cls: cosine_sim(hs_test, centroids[cls]) for cls in CLASSES}
    ranked = sorted(CLASSES, key=lambda c: sims[c], reverse=True)
    return ranked[0], ranked[1], ranked, sims


# ── Q_ordvec: multi-prototype majority vote ────────────────────────────────────
def q_ordvec_multi(item_idx, layer, K):
    """
    For each class prototype (all members except self if same class),
    compute overlap with test item bitmap.
    Each prototype votes for its class weighted by overlap rank.
    Returns (top1_pred, top2_pred, ranked_classes, overlap_summary).
    """
    target = SAMPLE[item_idx]
    bm_test = to_bitmap(all_hs[target["text"]][layer], K)

    # Collect overlaps: {class: [overlap scores from each prototype]}
    class_overlaps = defaultdict(list)
    for i, proto in enumerate(SAMPLE):
        if i == item_idx:
            continue  # leave self out
        bm_proto = to_bitmap(all_hs[proto["text"]][layer], K)
        ov = bitmap_overlap(bm_test, bm_proto)
        class_overlaps[proto["label"]].append(ov)

    # Aggregate: mean overlap per class
    class_scores = {cls: np.mean(v) if v else 0.0 for cls, v in class_overlaps.items()}
    # Fill missing classes
    for cls in CLASSES:
        if cls not in class_scores:
            class_scores[cls] = 0.0

    ranked = sorted(CLASSES, key=lambda c: class_scores[c], reverse=True)
    return ranked[0], ranked[1], ranked, class_scores


# ── Q_popcount ─────────────────────────────────────────────────────────────────
def make_popcount_buckets(overlaps):
    boundaries = np.percentile(overlaps, [25, 50, 75])

    def bucket(x):
        if x <= boundaries[0]:
            return "q1"
        elif x <= boundaries[1]:
            return "q2"
        elif x <= boundaries[2]:
            return "q3"
        else:
            return "q4"

    return bucket


# ── Fiber disagreements ────────────────────────────────────────────────────────
def fiber_disagreements(items, bucket_key):
    buckets = defaultdict(list)
    for item in items:
        buckets[item[bucket_key]].append(item)
    violations, total = 0, 0
    for members in buckets.values():
        for a, b in combinations(members, 2):
            total += 1
            if a["label"] != b["label"]:
                violations += 1
    return violations, total


# ══════════════════════════════════════════════════════════════════════════════
# SWEEP
# ══════════════════════════════════════════════════════════════════════════════
sweep_results = []

print(f"\n{'═' * 72}")
print(f"SWEEP: {len(LAYER_SWEEP)} layers × {len(K_SWEEP)} K values")
print(f"{'═' * 72}")
print(
    f"{'Layer':>6} {'K':>5} | {'QF_top1':>8} {'QF_top2':>8} | "
    f"{'QO_top1':>8} {'QO_top2':>8} | {'QO_agree':>9} | "
    f"{'QP_viols':>9} {'QO_viols':>9} {'QF_viols':>9}"
)
print(
    f"{'─' * 6} {'─' * 5}   {'─' * 8} {'─' * 8}   {'─' * 8} {'─' * 8}   "
    f"{'─' * 9}   {'─' * 9} {'─' * 9} {'─' * 9}"
)

best_qo_top1 = 0
best_config = None

for layer in LAYER_SWEEP:
    # Anchor bitmap for Q_popcount — one per layer/K combo
    for K in K_SWEEP:
        anchor_bm = to_bitmap(all_hs[ANCHOR_TEXT][layer], K)

        # Per-item predictions
        results_layer_k = []
        for i, item in enumerate(SAMPLE):
            # Q_full LOO
            qf1, qf2, qf_ranked, qf_sims = q_full_loo(i, layer)

            # Q_ordvec multi-prototype
            qo1, qo2, qo_ranked, qo_scores = q_ordvec_multi(i, layer, K)

            # Q_popcount
            bm_item = to_bitmap(all_hs[item["text"]][layer], K)
            anc_ov = bitmap_overlap(anchor_bm, bm_item)

            results_layer_k.append(
                {
                    "text": item["text"],
                    "label": item["label"],
                    "q_full": qf1,
                    "q_full_t2": top2_acc(qf_ranked, item["label"]),
                    "q_ordvec": qo1,
                    "q_ordvec_t2": top2_acc(qo_ranked, item["label"]),
                    "q_agree": qf1 == qo1,
                    "anchor_overlap": anc_ov,
                    "q_ordvec_scores": qo_scores,
                }
            )

        # Popcount bucket
        anc_ovs = [r["anchor_overlap"] for r in results_layer_k]
        bfn = make_popcount_buckets(anc_ovs)
        for r in results_layer_k:
            r["q_popcount"] = bfn(r["anchor_overlap"])

        # Metrics
        qf_top1 = sum(r["q_full"] == r["label"] for r in results_layer_k) / N
        qf_top2 = sum(r["q_full_t2"] for r in results_layer_k) / N
        qo_top1 = sum(r["q_ordvec"] == r["label"] for r in results_layer_k) / N
        qo_top2 = sum(r["q_ordvec_t2"] for r in results_layer_k) / N
        qo_agree = sum(r["q_agree"] for r in results_layer_k) / N

        qp_v, qp_t = fiber_disagreements(results_layer_k, "q_popcount")
        qo_v, qo_t = fiber_disagreements(
            [{**r, "q_ordvec_sig": r["q_ordvec"]} for r in results_layer_k],
            "q_ordvec_sig",
        )
        qf_v, qf_t = fiber_disagreements(results_layer_k, "q_full")

        print(
            f"{layer:>6} {K:>5} | {qf_top1 * 100:>7.0f}% {qf_top2 * 100:>7.0f}% | "
            f"{qo_top1 * 100:>7.0f}% {qo_top2 * 100:>7.0f}% | "
            f"{qo_agree * 100:>8.0f}% | "
            f"{qp_v:>4}/{qp_t:<4} {qo_v:>4}/{qo_t:<4} {qf_v:>4}/{qf_t:<4}"
        )

        row = dict(
            layer=layer,
            K=K,
            qf_top1=qf_top1,
            qf_top2=qf_top2,
            qo_top1=qo_top1,
            qo_top2=qo_top2,
            qo_agree=qo_agree,
            qp_violations=qp_v,
            qp_pairs=qp_t,
            qo_violations=qo_v,
            qo_pairs=qo_t,
            qf_violations=qf_v,
            qf_pairs=qf_t,
            per_item=results_layer_k,
        )
        sweep_results.append(row)

        if qo_top1 > best_qo_top1:
            best_qo_top1 = qo_top1
            best_config = (layer, K)

# ── Best config detail ─────────────────────────────────────────────────────────
print(f"\n{'═' * 72}")
print(
    f"BEST Q_ordvec top-1: {best_qo_top1 * 100:.0f}%  at layer={best_config[0]}, K={best_config[1]}"
)
print(f"{'═' * 72}")

best_row = next(
    r
    for r in sweep_results
    if r["layer"] == best_config[0] and r["K"] == best_config[1]
)
print(f"\nPer-item detail (layer={best_config[0]}, K={best_config[1]}):")
print(f"{'Item':<30} {'True':<12} {'Q_full':<12} {'Q_ordvec':<12} {'Agree'}")
print(f"{'─' * 30} {'─' * 12} {'─' * 12} {'─' * 12} {'─' * 5}")
for r in best_row["per_item"]:
    marker = "✓" if r["q_agree"] else "✗"
    print(
        f"{r['text']:<30} {r['label']:<12} {r['q_full']:<12} {r['q_ordvec']:<12} {marker}"
    )

# ── Hypergeometric null at best config ────────────────────────────────────────
print(f"\n{'═' * 72}")
print(f"HYPERGEOMETRIC NULL — layer={best_config[0]}, K={best_config[1]}")
print(
    f"Expected overlap under uniform null (K²/D): {best_config[1] ** 2 / D_TOTAL:.2f}"
)
print(f"{'═' * 72}")
for cls in CLASSES:
    same = [
        r["q_ordvec_scores"][cls] for r in best_row["per_item"] if r["label"] == cls
    ]
    other = [
        r["q_ordvec_scores"][cls] for r in best_row["per_item"] if r["label"] != cls
    ]
    null_exp = best_config[1] ** 2 / D_TOTAL
    hg_p = hypergeom.sf(int(np.mean(same)) - 1, D_TOTAL, best_config[1], best_config[1])
    print(
        f"  {cls:<12}: same={np.mean(same):.1f}  other={np.mean(other):.1f}  "
        f"above_null={np.mean(same) - null_exp:.1f}  p={hg_p:.2e}"
    )

# ── Save ──────────────────────────────────────────────────────────────────────
# Strip per_item bitmaps for clean JSON
clean = []
for row in sweep_results:
    r2 = {k: v for k, v in row.items() if k != "per_item"}
    r2["per_item"] = [
        {k2: v2 for k2, v2 in item.items() if k2 not in ("q_ordvec_scores",)}
        for item in row["per_item"]
    ]
    clean.append(r2)

_OUT = (
    _Path(__file__).resolve().parents[1]
    / "results"
    / "hsg_ordvec_fiber_v2_results.json"
)
_OUT.parent.mkdir(parents=True, exist_ok=True)
with open(_OUT, "w") as f:
    json.dump(clean, f, indent=2)
print(f"\nResults -> {_OUT}")
