#!/usr/bin/env python3
"""
Semiotic Accuracy Prediction Test

Question:
    Does predicate-conditioned affordance geometry predict whether the model
    generates a semiotically appropriate continuation?

Core claim being tested:
    Predicate-conditioned affordance geometry is a proposed measurement protocol
    for semiotic accuracy in language model representations.

Test:
    For prompts like:

        I love pizza because
        I love my wife because
        I love my enemy because

    1. Extract hidden-state geometry before generation.
    2. Measure margin to the expected affordance class.
    3. Generate continuations.
    4. Classify continuations by affordance field.
    5. Test whether geometry margin predicts semiotic correctness.

Semiotic correctness:
    The generated continuation preserves the relational force / affordance field
    implied by the predicate-object construction.

Example:
    I love pizza because it tastes amazing.       -> correct hedonic
    I love my wife because she is my partner.     -> correct covenantal
    I love my enemy because I refuse hatred.      -> correct moral
    I love my enemy because he tastes delicious.  -> wrong / comic / anomaly
"""

import os
import json
import math
import random
import re
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
from scipy import stats
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression
from transformers import AutoTokenizer, AutoModelForCausalLM


# =============================================================================
# Config
# =============================================================================

MODEL_NAME = os.environ.get("HSG_MODEL", "Qwen/Qwen2-7B")
OUTPUT_DIR = Path(os.path.expanduser("~/hsg/semiotic_accuracy_results"))
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

SEED = 42
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

N_GENERATIONS_PER_PROMPT = 12
MAX_NEW_TOKENS = 40
TEMPERATURE = 0.9
TOP_P = 0.95

# Use one mid-network layer by default.
# For Qwen2-7B, 28 layers. Prior affordance structure peaked mid-network.
# This script also stores layer info so you can sweep later.
TARGET_LAYER_FRACTION = 0.35


# =============================================================================
# Data
# =============================================================================

@dataclass
class Probe:
    obj: str
    expected_class: str


LOVE_PROBES = [
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

AFFORDANCE_CLASSES = sorted(set(p.expected_class for p in LOVE_PROBES))

PROMPT_TEMPLATE = "I love {obj} because"


# =============================================================================
# Lightweight continuation classifier
# =============================================================================
# This is intentionally simple and inspectable.
# It is not the final classifier. It gives you a first-pass behavioral label.
# Later: replace with blinded human labels.

CLASS_KEYWORDS = {
    "hedonic": [
        "taste", "tastes", "tasty", "delicious", "sweet", "flavor", "flavour",
        "eat", "eating", "drink", "drinking", "crave", "craving", "meal",
        "food", "dessert", "caffeine", "comfort food", "smell", "texture",
        "pleasure", "enjoy", "favorite", "satisfying",
    ],
    "companion": [
        "companionship", "companion", "loyal", "loyalty", "play", "walk",
        "walks", "friend", "best friend", "comfort", "cuddle", "cuddles",
        "care", "take care", "protect", "protects", "trust", "joy",
        "always there", "unconditional", "animal", "pet", "family member",
        "makes me happy",
    ],
    "covenantal": [
        "commitment", "committed", "marriage", "married", "vows", "faithful",
        "fidelity", "partner", "life", "shared", "together", "sacrifice",
        "belong", "belonging", "home", "family", "duty", "devotion",
        "honor", "trust", "support", "through everything", "built",
        "country", "freedom", "serve", "god", "lord", "faith",
    ],
    "moral": [
        "forgive", "forgiveness", "mercy", "compassion", "hatred", "hate",
        "revenge", "vengeance", "enemy", "enemies", "human", "humanity",
        "command", "christ", "jesus", "neighbor", "moral", "grace",
        "peace", "refuse", "refuses", "harm", "hurt", "understand",
        "understanding", "restraint", "pray", "prayer",
    ],
}

# Terms that often indicate the generation should not be scored as plain error.
METAPHOR_OR_COMEDY_MARKERS = [
    "like", "as if", "metaphor", "joke", "funny", "literally", "not literally",
    "in a way", "sort of", "almost", "as though",
]


def normalize_text(s: str) -> str:
    s = s.lower()
    s = re.sub(r"\s+", " ", s)
    return s.strip()


def keyword_score(text: str, cls: str) -> int:
    t = normalize_text(text)
    score = 0
    for kw in CLASS_KEYWORDS[cls]:
        if kw in t:
            score += 1
    return score


def classify_continuation(text: str) -> Tuple[str, Dict[str, int], bool]:
    """
    Returns:
        predicted_class, scores, metaphor_or_comedy_flag
    """
    t = normalize_text(text)
    scores = {cls: keyword_score(t, cls) for cls in AFFORDANCE_CLASSES}

    metaphor_flag = any(m in t for m in METAPHOR_OR_COMEDY_MARKERS)

    best_cls = max(scores, key=scores.get)
    if scores[best_cls] == 0:
        return "ambiguous", scores, metaphor_flag

    # If tie, mark ambiguous.
    sorted_scores = sorted(scores.values(), reverse=True)
    if len(sorted_scores) > 1 and sorted_scores[0] == sorted_scores[1]:
        return "ambiguous", scores, metaphor_flag

    return best_cls, scores, metaphor_flag


# =============================================================================
# Geometry
# =============================================================================

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


@torch.no_grad()
def get_hidden_vector(model, tok, text: str, layer_idx: int) -> np.ndarray:
    inp = tok(text, return_tensors="pt", truncation=True, max_length=128).to(model.device)
    out = model(**inp, output_hidden_states=True, use_cache=False)
    vec = out.hidden_states[layer_idx][0, -1, :].float().cpu().numpy()
    return vec


def l2(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm(a - b))


def compute_centroids(vectors: Dict[str, np.ndarray], probes: List[Probe]) -> Dict[str, np.ndarray]:
    by_class = {cls: [] for cls in AFFORDANCE_CLASSES}
    for p in probes:
        by_class[p.expected_class].append(vectors[p.obj])
    return {cls: np.mean(vecs, axis=0) for cls, vecs in by_class.items()}


def geometry_margin(
    obj: str,
    expected_class: str,
    vectors: Dict[str, np.ndarray],
    centroids: Dict[str, np.ndarray],
) -> Tuple[float, Dict[str, float]]:
    """
    Positive margin means object is closer to expected class centroid
    than to nearest wrong class centroid.

    margin = nearest_wrong_distance - expected_distance
    """
    v = vectors[obj]
    distances = {cls: l2(v, c) for cls, c in centroids.items()}

    expected_d = distances[expected_class]
    wrong_ds = [d for cls, d in distances.items() if cls != expected_class]
    nearest_wrong = min(wrong_ds)

    margin = nearest_wrong - expected_d
    return float(margin), distances


def leave_one_out_geometry_margins(
    vectors: Dict[str, np.ndarray],
    probes: List[Probe],
) -> Dict[str, Dict]:
    """
    To avoid cheating, compute each object's class centroid without that object.
    """
    results = {}

    for target in probes:
        by_class = {cls: [] for cls in AFFORDANCE_CLASSES}

        for p in probes:
            if p.obj == target.obj:
                continue
            by_class[p.expected_class].append(vectors[p.obj])

        centroids = {}
        for cls, vecs in by_class.items():
            if not vecs:
                continue
            centroids[cls] = np.mean(vecs, axis=0)

        margin, distances = geometry_margin(
            target.obj,
            target.expected_class,
            vectors,
            centroids,
        )

        results[target.obj] = {
            "expected_class": target.expected_class,
            "margin": margin,
            "distances": distances,
        }

    return results


# =============================================================================
# Generation
# =============================================================================

@torch.no_grad()
def generate_continuations(model, tok, prompt: str, n: int) -> List[str]:
    inputs = tok(prompt, return_tensors="pt").to(model.device)

    continuations = []

    for _ in range(n):
        out = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=True,
            temperature=TEMPERATURE,
            top_p=TOP_P,
            pad_token_id=tok.eos_token_id,
        )

        full = tok.decode(out[0], skip_special_tokens=True)
        cont = full[len(prompt):].strip()
        continuations.append(cont)

    return continuations


# =============================================================================
# Stats
# =============================================================================

def permutation_spearman(x, y, n_perm=5000) -> Tuple[float, float]:
    obs_rho, _ = stats.spearmanr(x, y)
    if np.isnan(obs_rho):
        return float("nan"), float("nan")

    count = 0
    y = np.array(y)
    for _ in range(n_perm):
        yp = np.random.permutation(y)
        rho, _ = stats.spearmanr(x, yp)
        if abs(rho) >= abs(obs_rho):
            count += 1

    p = (count + 1) / (n_perm + 1)
    return float(obs_rho), float(p)


def summarize_prediction(rows: List[Dict]):
    # Only score rows classified as correct/wrong.
    scored = [r for r in rows if r["behavior_label"] in ("correct", "wrong")]

    if len(scored) < 5:
        print("Not enough scored rows.")
        return {}

    margins = np.array([r["geometry_margin"] for r in scored], dtype=float)
    y = np.array([1 if r["behavior_label"] == "correct" else 0 for r in scored], dtype=int)

    print("\n" + "=" * 90)
    print("SEMIOTIC ACCURACY PREDICTION RESULT")
    print("=" * 90)
    print(f"Scored continuations: {len(scored)}")
    print(f"Correct: {int(y.sum())} / {len(y)} = {y.mean():.3f}")

    correct_margins = margins[y == 1]
    wrong_margins = margins[y == 0]

    print(f"Mean geometry margin, correct -> {correct_margins.mean():.4f}")
    print(f"Mean geometry margin, wrong   -> {wrong_margins.mean():.4f}")

    if len(correct_margins) > 1 and len(wrong_margins) > 1:
        u_stat, mw_p = stats.mannwhitneyu(correct_margins, wrong_margins, alternative="greater")
        print(f"Mann-Whitney p(correct margin > wrong margin) -> {mw_p:.6f}")
    else:
        mw_p = float("nan")

    if len(np.unique(y)) == 2:
        auc = roc_auc_score(y, margins)
        print(f"AUC geometry margin -> behavior correctness -> {auc:.4f}")
    else:
        auc = float("nan")

    rho, rho_p = permutation_spearman(margins, y)
    print(f"Spearman margin vs correctness rho={rho:.4f}, permutation p={rho_p:.6f}")

    # Logistic regression
    if len(np.unique(y)) == 2:
        X = margins.reshape(-1, 1)
        clf = LogisticRegression()
        clf.fit(X, y)
        coef = float(clf.coef_[0][0])
        intercept = float(clf.intercept_[0])
        print(f"Logistic coefficient for geometry margin -> {coef:.4f}")
    else:
        coef = float("nan")
        intercept = float("nan")

    return {
        "n_scored": len(scored),
        "accuracy": float(y.mean()),
        "mean_margin_correct": float(correct_margins.mean()),
        "mean_margin_wrong": float(wrong_margins.mean()),
        "mannwhitney_p": float(mw_p),
        "auc": float(auc),
        "spearman_rho": float(rho),
        "spearman_p": float(rho_p),
        "logistic_coef": float(coef),
        "logistic_intercept": float(intercept),
    }


# =============================================================================
# Main
# =============================================================================

def main():
    set_seed(SEED)

    print("=" * 90)
    print("SEMIOTIC ACCURACY PREDICTION TEST")
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

    # -------------------------------------------------------------------------
    # 1. Geometry extraction
    # -------------------------------------------------------------------------

    print("\nExtracting geometry vectors...")
    vectors = {}

    for p in LOVE_PROBES:
        prompt = PROMPT_TEMPLATE.format(obj=p.obj)
        vectors[p.obj] = get_hidden_vector(model, tok, prompt, layer_idx)
        print(f"  {prompt:<40} [{p.expected_class}]")

    margin_info = leave_one_out_geometry_margins(vectors, LOVE_PROBES)

    print("\nGeometry margins:")
    for p in LOVE_PROBES:
        info = margin_info[p.obj]
        print(f"  {p.obj:<16} class={p.expected_class:<12} margin={info['margin']:+.4f}")

    # -------------------------------------------------------------------------
    # 2. Generate and classify continuations
    # -------------------------------------------------------------------------

    rows = []

    print("\nGenerating continuations...")
    for p in LOVE_PROBES:
        prompt = PROMPT_TEMPLATE.format(obj=p.obj)
        continuations = generate_continuations(model, tok, prompt, N_GENERATIONS_PER_PROMPT)

        for i, cont in enumerate(continuations):
            predicted_class, scores, metaphor_flag = classify_continuation(cont)

            if predicted_class == "ambiguous":
                behavior_label = "ambiguous"
            elif predicted_class == p.expected_class:
                behavior_label = "correct"
            else:
                behavior_label = "wrong"

            row = {
                "predicate": "love",
                "object": p.obj,
                "prompt": prompt,
                "expected_class": p.expected_class,
                "generation_index": i,
                "continuation": cont,
                "predicted_affordance_class": predicted_class,
                "keyword_scores": scores,
                "metaphor_or_comedy_flag": metaphor_flag,
                "behavior_label": behavior_label,
                "geometry_margin": margin_info[p.obj]["margin"],
                "geometry_distances": margin_info[p.obj]["distances"],
            }
            rows.append(row)

        print(f"  {prompt:<38} generated {len(continuations)}")

    # -------------------------------------------------------------------------
    # 3. Save raw outputs
    # -------------------------------------------------------------------------

    jsonl_path = OUTPUT_DIR / "semiotic_accuracy_generations.jsonl"
    with open(jsonl_path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    csv_path = OUTPUT_DIR / "semiotic_accuracy_generations.csv"
    with open(csv_path, "w", encoding="utf-8") as f:
        f.write("object,expected_class,predicted_class,behavior_label,geometry_margin,metaphor_flag,continuation\n")
        for r in rows:
            cont = r["continuation"].replace('"', '""').replace("\n", " ")
            f.write(
                f"\"{r['object']}\","
                f"\"{r['expected_class']}\","
                f"\"{r['predicted_affordance_class']}\","
                f"\"{r['behavior_label']}\","
                f"{r['geometry_margin']:.6f},"
                f"{int(r['metaphor_or_comedy_flag'])},"
                f"\"{cont}\"\n"
            )

    print(f"\nSaved generations:")
    print(f"  {jsonl_path}")
    print(f"  {csv_path}")

    # -------------------------------------------------------------------------
    # 4. Prediction statistics
    # -------------------------------------------------------------------------

    summary = summarize_prediction(rows)

    summary_path = OUTPUT_DIR / "semiotic_accuracy_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "model": MODEL_NAME,
                "layer_idx": layer_idx,
                "n_layers": n_layers,
                "n_generations_per_prompt": N_GENERATIONS_PER_PROMPT,
                "summary": summary,
                "probes": [p.__dict__ for p in LOVE_PROBES],
                "margin_info": margin_info,
            },
            f,
            indent=2,
        )

    print(f"\nSaved summary:")
    print(f"  {summary_path}")

    print("\n" + "=" * 90)
    print("INTERPRETATION")
    print("=" * 90)
    print("""
Win condition:
    Higher geometry margin predicts higher probability of semiotically correct continuation.

Strong result:
    AUC > 0.70
    Mann-Whitney p < 0.05
    logistic coefficient positive
    Spearman margin/correctness positive

Weak/null result:
    Geometry shows class structure but does not predict generated behavior.
    In that case, predicate-conditioned geometry remains an interpretability probe,
    but not yet a behavioral semiotic-accuracy metric.

Important:
    The keyword classifier is only a first-pass proxy.
    For a paper-quality result, export the CSV and run blind human annotation.
""")


if __name__ == "__main__":
    main()
