#!/usr/bin/env python3
"""
Contextual Affordance Geometry Test

Original question:
    "I love pizza"
    "I love my wife"
    "I love my dog"

Same verb. Different meaning.

Hypothesis:
    A word's meaning is not a fixed vector.
    Its meaning is the continuation field / affordance field opened by context.

Test:
    Compare next-token distributions after matched prompts:

        I love pizza because
        I love my wife because
        I love my dog because

    If the model understands contextual meaning, the continuation distribution
    after "I love X because" should cluster by object category.

Ablations:
    1. love + object versus object-only
    2. love versus neutral verb "see"
    3. shuffled category labels
    4. within-category versus between-category distance
    5. same template, same verb, different object
"""

import os
import math
import random
from dataclasses import dataclass
from typing import List, Dict, Tuple

import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


# -----------------------------
# Config
# -----------------------------

MODEL_NAME = os.environ.get("CAG_MODEL", "distilgpt2")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SEED = 42

TOP_K = 5000
N_PERM = 5000


# -----------------------------
# Data
# -----------------------------


@dataclass
class Case:
    category: str
    obj: str


CASES = [
    # Food / appetite love
    Case("food", "pizza"),
    Case("food", "chocolate"),
    Case("food", "ice cream"),
    Case("food", "sushi"),
    Case("food", "coffee"),
    Case("food", "tacos"),
    Case("food", "barbecue"),
    Case("food", "fresh bread"),
    # Spousal / romantic relational love
    Case("spouse", "my wife"),
    Case("spouse", "my husband"),
    Case("spouse", "my partner"),
    Case("spouse", "my fiancee"),
    Case("spouse", "the woman I married"),
    Case("spouse", "the man I married"),
    Case("spouse", "my beloved"),
    Case("spouse", "the person I built my life with"),
    # Pet / companion love
    Case("pet", "my dog"),
    Case("pet", "my cat"),
    Case("pet", "my puppy"),
    Case("pet", "my old dog"),
    Case("pet", "my rescue dog"),
    Case("pet", "my horse"),
    Case("pet", "my parrot"),
    Case("pet", "my little terrier"),
    # Family love, distinct from romantic
    Case("family", "my daughter"),
    Case("family", "my son"),
    Case("family", "my mother"),
    Case("family", "my father"),
    Case("family", "my sister"),
    Case("family", "my brother"),
    Case("family", "my grandmother"),
    Case("family", "my child"),
    # Activity / preference love
    Case("activity", "guitar"),
    Case("activity", "fishing"),
    Case("activity", "mathematics"),
    Case("activity", "driving"),
    Case("activity", "reading"),
    Case("activity", "running"),
    Case("activity", "painting"),
    Case("activity", "cooking"),
]


TEMPLATES = {
    "love_because": "I love {obj} because",
    "object_because": "{obj} because",
    "see_because": "I see {obj} because",
    "like_because": "I like {obj} because",
    "love_period": "I love {obj}.",
}


# -----------------------------
# Reproducibility
# -----------------------------


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# -----------------------------
# Distances
# -----------------------------


def js_divergence(p: np.ndarray, q: np.ndarray, eps: float = 1e-12) -> float:
    """
    Jensen-Shannon divergence.
    Symmetric and finite.
    """
    p = np.asarray(p, dtype=np.float64)
    q = np.asarray(q, dtype=np.float64)

    p = p + eps
    q = q + eps
    p = p / p.sum()
    q = q / q.sum()

    m = 0.5 * (p + q)

    kl_pm = np.sum(p * np.log(p / m))
    kl_qm = np.sum(q * np.log(q / m))

    return float(0.5 * (kl_pm + kl_qm))


def cosine_distance(a: np.ndarray, b: np.ndarray, eps: float = 1e-12) -> float:
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    if na < eps or nb < eps:
        return 0.0
    return float(1.0 - np.dot(a, b) / (na * nb))


def pairwise_distances(vectors: List[np.ndarray], metric: str) -> np.ndarray:
    n = len(vectors)
    D = np.zeros((n, n), dtype=np.float64)

    for i in range(n):
        for j in range(i + 1, n):
            if metric == "js":
                d = js_divergence(vectors[i], vectors[j])
            elif metric == "cosine":
                d = cosine_distance(vectors[i], vectors[j])
            else:
                raise ValueError(metric)

            D[i, j] = d
            D[j, i] = d

    return D


def within_between_gap(D: np.ndarray, labels: List[str]) -> Tuple[float, float, float]:
    within = []
    between = []

    n = len(labels)
    for i in range(n):
        for j in range(i + 1, n):
            if labels[i] == labels[j]:
                within.append(D[i, j])
            else:
                between.append(D[i, j])

    within_mean = float(np.mean(within))
    between_mean = float(np.mean(between))
    gap = between_mean - within_mean

    return within_mean, between_mean, gap


def permutation_p_for_gap(
    D: np.ndarray, labels: List[str], observed_gap: float, n_perm: int
) -> float:
    labels = np.array(labels)
    count = 0

    for _ in range(n_perm):
        shuffled = np.random.permutation(labels)
        _, _, gap = within_between_gap(D, list(shuffled))
        if gap >= observed_gap:
            count += 1

    return (count + 1) / (n_perm + 1)


# -----------------------------
# Model auditor
# -----------------------------


class AffordanceAuditor:
    def __init__(self, model_name: str):
        print("=" * 90)
        print("CONTEXTUAL AFFORDANCE GEOMETRY TEST")
        print("=" * 90)
        print(f"Model:  {model_name}")
        print(f"Device: {DEVICE}")

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            output_hidden_states=True,
        ).to(DEVICE)

        self.model.eval()

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    @torch.no_grad()
    def audit_prompt(self, prompt: str) -> Dict[str, np.ndarray]:
        enc = self.tokenizer(prompt, return_tensors="pt").to(DEVICE)

        out = self.model(
            **enc,
            output_hidden_states=True,
            return_dict=True,
        )

        # Next-token distribution at final prompt position.
        logits = out.logits[0, -1, :]
        probs = torch.softmax(logits, dim=-1).detach().cpu().numpy()

        # Keep top-K mass to reduce noise and memory.
        # Re-normalize after truncation.
        if TOP_K is not None and TOP_K < len(probs):
            idx = np.argpartition(probs, -TOP_K)[-TOP_K:]
            compact = np.zeros_like(probs)
            compact[idx] = probs[idx]
            probs = compact / compact.sum()

        # Final-token hidden state from final layer.
        final_hidden = out.hidden_states[-1][0, -1, :].detach().cpu().numpy()

        # Also collect layer trajectory for optional curvature check.
        layer_vecs = []
        for h in out.hidden_states:
            layer_vecs.append(h[0, -1, :].detach().cpu().numpy())
        layer_path = np.stack(layer_vecs, axis=0)

        return {
            "probs": probs,
            "hidden": final_hidden,
            "layer_path": layer_path,
            "tokens": np.array([enc["input_ids"].shape[1]], dtype=np.float64),
        }

    def top_tokens(self, probs: np.ndarray, n: int = 12) -> List[Tuple[str, float]]:
        idx = np.argsort(probs)[::-1][:n]
        out = []
        for i in idx:
            tok = self.tokenizer.decode([int(i)])
            out.append((tok.replace("\n", "\\n"), float(probs[i])))
        return out


# -----------------------------
# Optional curvature
# -----------------------------


def integrated_curvature(path: np.ndarray) -> float:
    diffs = np.diff(path, axis=0)
    total = 0.0

    for i in range(len(diffs) - 1):
        a = diffs[i]
        b = diffs[i + 1]
        na = np.linalg.norm(a)
        nb = np.linalg.norm(b)

        if na < 1e-12 or nb < 1e-12:
            continue

        c = np.dot(a, b) / (na * nb)
        c = np.clip(c, -1.0, 1.0)
        total += math.acos(c)

    return float(total)


# -----------------------------
# Experiment
# -----------------------------


def run_condition(auditor: AffordanceAuditor, template_name: str) -> Dict:
    template = TEMPLATES[template_name]

    prompts = []
    labels = []
    objects = []
    probs = []
    hidden = []
    curvatures = []

    print("\n" + "=" * 90)
    print(f"RUNNING CONDITION: {template_name}")
    print("=" * 90)

    for case in CASES:
        prompt = template.format(obj=case.obj)
        result = auditor.audit_prompt(prompt)

        prompts.append(prompt)
        labels.append(case.category)
        objects.append(case.obj)
        probs.append(result["probs"])
        hidden.append(result["hidden"])
        curvatures.append(integrated_curvature(result["layer_path"]))

    D_js = pairwise_distances(probs, metric="js")
    D_hidden = pairwise_distances(hidden, metric="cosine")

    w_js, b_js, gap_js = within_between_gap(D_js, labels)
    p_js = permutation_p_for_gap(D_js, labels, gap_js, N_PERM)

    w_h, b_h, gap_h = within_between_gap(D_hidden, labels)
    p_h = permutation_p_for_gap(D_hidden, labels, gap_h, N_PERM)

    print("\nAffordance distribution geometry — next-token JS divergence")
    print(f"  within-category mean JS     -> {w_js:.6f}")
    print(f"  between-category mean JS    -> {b_js:.6f}")
    print(f"  gap between-within          -> {gap_js:.6f}")
    print(f"  shuffled-label p            -> {p_js:.6f}")

    print("\nFinal hidden-state geometry — cosine distance")
    print(f"  within-category mean dist   -> {w_h:.6f}")
    print(f"  between-category mean dist  -> {b_h:.6f}")
    print(f"  gap between-within          -> {gap_h:.6f}")
    print(f"  shuffled-label p            -> {p_h:.6f}")

    print("\nExample next-token affordances")
    for target in ["pizza", "my wife", "my dog", "mathematics"]:
        if target in objects:
            idx = objects.index(target)
            print(f"\nPrompt: {prompts[idx]!r}")
            for tok, pr in auditor.top_tokens(probs[idx], n=10):
                print(f"  {tok!r:<12} {pr:.5f}")

    return {
        "template": template_name,
        "labels": labels,
        "objects": objects,
        "prompts": prompts,
        "D_js": D_js,
        "D_hidden": D_hidden,
        "gap_js": gap_js,
        "p_js": p_js,
        "gap_hidden": gap_h,
        "p_hidden": p_h,
        "curvatures": np.array(curvatures),
    }


def compare_conditions(results: Dict[str, Dict]) -> None:
    print("\n" + "=" * 90)
    print("ABLATION SUMMARY")
    print("=" * 90)

    print(
        f"{'condition':<18} {'JS_gap':>12} {'JS_p':>12} {'hidden_gap':>14} {'hidden_p':>12}"
    )
    print("-" * 90)

    for name, res in results.items():
        print(
            f"{name:<18} "
            f"{res['gap_js']:>12.6f} {res['p_js']:>12.6f} "
            f"{res['gap_hidden']:>14.6f} {res['p_hidden']:>12.6f}"
        )

    print("\nInterpretation:")
    print("""
The key comparison is:

    love_because vs object_because vs see_because

If "I love X because" has a stronger category gap than "X because",
then the verb-object relation is adding contextual meaning beyond the object alone.

If "love_because" beats "see_because", then the effect is not merely noun category.
It is relation-specific.

A real foothold looks like:
    - love_because JS_gap > object_because JS_gap
    - love_because JS_gap > see_because JS_gap
    - shuffled-label p < 0.05 for love_because
    - top next-token affordances differ sensibly:
        pizza -> taste/eating
        wife  -> person/life/love/relationship
        dog   -> companion/cute/walk/loyalty

This is not hidden-intent detection.
This is a test of contextual meaning as affordance-field separation.
""")


def main() -> None:
    set_seed(SEED)

    auditor = AffordanceAuditor(MODEL_NAME)

    results = {}
    for condition in [
        "love_because",
        "object_because",
        "see_because",
        "like_because",
        "love_period",
    ]:
        results[condition] = run_condition(auditor, condition)

    compare_conditions(results)


if __name__ == "__main__":
    main()
