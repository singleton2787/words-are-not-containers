#!/usr/bin/env python3
"""
Wikipedia Relation-vs-Control Affordance Ablation

Purpose:
    Test whether relation words such as love/care/like create stronger
    category-specific affordance fields than ordinary control words.

Previous result:
    Relation-window fields around love/care/like were more category-separated
    than baseline article-category fields.

Best critic:
    "Any local word window in category-specific Wikipedia articles will separate."

This ablation:
    1. Loads the existing Wikipedia corpus.
    2. Builds relation-window fields for love/care/like.
    3. Samples matched control word sets from the same corpus.
    4. Computes category separation for relation windows and control windows.
    5. Runs empirical permutation/random-control tests.

Win condition:
    relation_delta > control_delta distribution

Where:
    delta = mean_relation_window_JS - mean_baseline_category_JS

If relation words reliably exceed controls:
    stronger support for Relational Affordance Semantics.

If not:
    prior effect may be ordinary topical lexical separation.
"""

import os
import re
import random
import numpy as np

from collections import Counter, defaultdict
from typing import Dict, List, Tuple
from scipy import stats
from stats import bootstrap_mean_ci, empirical_p_value


# =============================================================================
# Config
# =============================================================================

SEED = 42

CORPUS_FILE = "wiki_affordance_corpus.txt"

WINDOW = 12
MIN_TOKEN_LEN = 3
N_CONTROL_SETS = 5000
CONTROL_SET_SIZE = 12

TOP_N_FIELD_WORDS = 20

RELATION_TERMS = [
    "love",
    "loved",
    "loves",
    "loving",
    "care",
    "cares",
    "cared",
    "caring",
    "like",
    "liked",
    "likes",
    "liking",
]

# Avoid obvious topical category words as controls.
# Controls should be common-ish, semantically generic content words.
CONTROL_BLACKLIST = set(
    RELATION_TERMS
    + [
        "food",
        "pizza",
        "taste",
        "flavor",
        "cuisine",
        "cooking",
        "restaurant",
        "marriage",
        "wife",
        "husband",
        "spouse",
        "family",
        "wedding",
        "romance",
        "dog",
        "cat",
        "pet",
        "animal",
        "veterinarian",
        "shelter",
        "puppy",
        "country",
        "nation",
        "patriotism",
        "citizenship",
        "flag",
        "homeland",
        "enemy",
        "forgiveness",
        "ethics",
        "mercy",
        "revenge",
        "christian",
        "mathematics",
        "math",
        "algebra",
        "geometry",
        "calculus",
        "proof",
    ]
)

STOPWORDS = set(
    """
a an the and or but if while although though because since of to in on for from with without by as at into
is are was were be been being have has had do does did this that these those it its they them their he she
his her hers him we us our you your i me my mine not no yes can could may might must should would will shall
about after before during over under between among than then there here also such other more most many some any
one two three first second new known used use often including include includes called made make makes became become
which who whom whose where when why how into onto per via each both either neither same different various several
may however therefore thus also many much such since within across through between among against toward towards
""".split()
)


# =============================================================================
# Corpus parsing
# =============================================================================


def tokenize(text: str) -> List[str]:
    text = text.lower()
    text = re.sub(r"[^a-z0-9'\- ]+", " ", text)
    raw = text.split()

    out = []
    for t in raw:
        t = t.strip("-'")

        if len(t) < MIN_TOKEN_LEN:
            continue
        if t in STOPWORDS:
            continue
        if t.isdigit():
            continue

        out.append(t)

    return out


def split_category_sections(corpus: str) -> Dict[str, str]:
    category_chunks = defaultdict(list)
    current = None

    for line in corpus.splitlines():
        m = re.match(r"### CATEGORY=([a-z_]+) TITLE=", line)
        if m:
            current = m.group(1)
            continue

        if current is not None:
            category_chunks[current].append(line)

    return {k: "\n".join(v) for k, v in category_chunks.items()}


def load_category_tokens() -> Dict[str, List[str]]:
    if not os.path.exists(CORPUS_FILE):
        raise FileNotFoundError(
            f"{CORPUS_FILE} not found. Run wiki_affordance_field_test.py first."
        )

    with open(CORPUS_FILE, "r", encoding="utf-8") as f:
        corpus = f.read()

    category_text = split_category_sections(corpus)

    if len(category_text) < 3:
        raise RuntimeError(
            f"Only found categories: {sorted(category_text.keys())}. "
            "Need at least 3 categories."
        )

    return {cat: tokenize(text) for cat, text in category_text.items()}


# =============================================================================
# Field construction
# =============================================================================


def build_window_field(
    tokens: List[str], target_terms: List[str], window: int
) -> Counter:
    target_terms = set(target_terms)
    field = Counter()

    for i, tok in enumerate(tokens):
        if tok in target_terms:
            lo = max(0, i - window)
            hi = min(len(tokens), i + window + 1)

            for j in range(lo, hi):
                if j == i:
                    continue

                w = tokens[j]

                if len(w) >= MIN_TOKEN_LEN and w not in STOPWORDS:
                    field[w] += 1

    return field


def build_baseline_field(tokens: List[str]) -> Counter:
    return Counter(tokens)


def normalize_counter(
    counter: Counter, vocab: List[str], alpha: float = 1e-6
) -> np.ndarray:
    arr = np.array([counter.get(w, 0.0) for w in vocab], dtype=np.float64)
    arr += alpha
    arr /= arr.sum()
    return arr


def js_divergence(p: np.ndarray, q: np.ndarray, eps: float = 1e-12) -> float:
    p = np.asarray(p, dtype=np.float64) + eps
    q = np.asarray(q, dtype=np.float64) + eps

    p /= p.sum()
    q /= q.sum()

    m = 0.5 * (p + q)

    return float(0.5 * np.sum(p * np.log(p / m)) + 0.5 * np.sum(q * np.log(q / m)))


def pairwise_js(vectors: Dict[str, np.ndarray]) -> Dict[Tuple[str, str], float]:
    cats = sorted(vectors.keys())
    out = {}

    for i, a in enumerate(cats):
        for b in cats[i + 1 :]:
            out[(a, b)] = js_divergence(vectors[a], vectors[b])

    return out


def mean_pairwise_js(fields_by_cat: Dict[str, Counter]) -> float:
    active = {
        cat: field for cat, field in fields_by_cat.items() if sum(field.values()) > 0
    }

    if len(active) < 2:
        return float("nan")

    vocab = sorted(set(w for field in active.values() for w in field.keys()))
    vectors = {cat: normalize_counter(field, vocab) for cat, field in active.items()}

    vals = list(pairwise_js(vectors).values())

    if not vals:
        return float("nan")

    return float(np.mean(vals))


def compute_delta_for_terms(
    category_tokens: Dict[str, List[str]],
    target_terms: List[str],
    baseline_js: float,
) -> Tuple[float, float, Dict[str, Counter]]:
    fields = {
        cat: build_window_field(tokens, target_terms, WINDOW)
        for cat, tokens in category_tokens.items()
    }

    rel_js = mean_pairwise_js(fields)
    delta = rel_js - baseline_js

    return rel_js, delta, fields


# =============================================================================
# Control sampling
# =============================================================================


def corpus_frequency(category_tokens: Dict[str, List[str]]) -> Counter:
    c = Counter()
    for toks in category_tokens.values():
        c.update(toks)
    return c


def eligible_control_words(
    freq: Counter, min_count: int = 20, max_count: int = 5000
) -> List[str]:
    words = []

    for w, n in freq.items():
        if n < min_count:
            continue
        if n > max_count:
            continue
        if w in STOPWORDS:
            continue
        if w in CONTROL_BLACKLIST:
            continue
        if len(w) < MIN_TOKEN_LEN:
            continue
        if w.isdigit():
            continue

        words.append(w)

    return sorted(words)


def sample_control_set(words: List[str], freq: Counter, k: int) -> List[str]:
    """
    Weighted sample without replacement, using sqrt frequency.
    This biases toward useful mid-frequency words without only selecting
    the most frequent generic terms.
    """
    weights = np.array([np.sqrt(freq[w]) for w in words], dtype=np.float64)
    weights /= weights.sum()

    idx = np.random.choice(len(words), size=k, replace=False, p=weights)
    return [words[i] for i in idx]


def bootstrap_ci(data: np.ndarray, n_boot: int = 5000) -> Tuple[float, float]:
    return bootstrap_mean_ci(data, n_resamples=n_boot, seed=SEED)


# =============================================================================
# Main
# =============================================================================


def main() -> None:
    random.seed(SEED)
    np.random.seed(SEED)

    category_tokens = load_category_tokens()

    print("=" * 90)
    print("WIKIPEDIA RELATION-vs-CONTROL ABLATION")
    print("=" * 90)

    print("Loaded categories:")
    for cat, toks in sorted(category_tokens.items()):
        print(f"  {cat:<12} {len(toks):>8} tokens")

    # Baseline category separation over full category texts.
    baseline_fields = {
        cat: build_baseline_field(tokens) for cat, tokens in category_tokens.items()
    }

    baseline_js = mean_pairwise_js(baseline_fields)

    print("\nBaseline full-category separation:")
    print(f"  mean baseline JS -> {baseline_js:.6f}")

    # Relation set.
    relation_js, relation_delta, relation_fields = compute_delta_for_terms(
        category_tokens,
        RELATION_TERMS,
        baseline_js,
    )

    print("\nRelation terms:")
    print(f"  terms             -> {RELATION_TERMS}")
    print(f"  relation JS       -> {relation_js:.6f}")
    print(f"  relation delta    -> {relation_delta:.6f}")

    print("\nTop relation-window words by category:")
    for cat in sorted(relation_fields.keys()):
        hits = sum(relation_fields[cat].values())
        print(f"\n[{cat}] hits={hits}")
        print(
            ", ".join(
                [
                    f"{w}:{n}"
                    for w, n in relation_fields[cat].most_common(TOP_N_FIELD_WORDS)
                ]
            )
        )

    # Control words.
    freq = corpus_frequency(category_tokens)
    controls = eligible_control_words(freq)

    print("\nControl pool:")
    print(f"  eligible control words -> {len(controls)}")

    if len(controls) < CONTROL_SET_SIZE:
        raise RuntimeError("Not enough eligible controls.")

    control_deltas = []
    control_js_values = []
    control_examples = []

    print(f"\nSampling {N_CONTROL_SETS} matched control sets...")
    for i in range(N_CONTROL_SETS):
        control_terms = sample_control_set(controls, freq, CONTROL_SET_SIZE)
        c_js, c_delta, _ = compute_delta_for_terms(
            category_tokens, control_terms, baseline_js
        )

        if not np.isnan(c_delta):
            control_js_values.append(c_js)
            control_deltas.append(c_delta)

            if len(control_examples) < 5:
                control_examples.append((control_terms, c_js, c_delta))

        if (i + 1) % 1000 == 0:
            print(f"  sampled {i + 1}/{N_CONTROL_SETS}")

    control_deltas = np.array(control_deltas, dtype=np.float64)
    control_js_values = np.array(control_js_values, dtype=np.float64)

    # Empirical p-value: how many random controls >= relation delta?
    empirical_p = empirical_p_value(
        control_deltas, relation_delta, alternative="greater"
    )

    ci_low, ci_high = bootstrap_ci(control_deltas)

    print("\n" + "=" * 90)
    print("CONTROL ABLATION RESULT")
    print("=" * 90)

    print(f"Relation delta                     -> {relation_delta:.6f}")
    print(f"Control delta mean                 -> {np.mean(control_deltas):.6f}")
    print(f"Control delta median               -> {np.median(control_deltas):.6f}")
    print(
        f"Control delta 95% range            -> [{np.percentile(control_deltas, 2.5):.6f}, {np.percentile(control_deltas, 97.5):.6f}]"
    )
    print(f"Bootstrap CI of control mean       -> [{ci_low:.6f}, {ci_high:.6f}]")
    print(f"Empirical p(control >= relation)   -> {empirical_p:.6f}")

    z = (relation_delta - np.mean(control_deltas)) / (
        np.std(control_deltas, ddof=1) + 1e-12
    )
    print(f"Relation-vs-control z              -> {z:.3f}")

    try:
        t_stat, p_val = stats.ttest_1samp(control_deltas, relation_delta)
        # This tests whether control mean differs from relation_delta.
        print(f"Control mean vs relation t-test p  -> {p_val:.6f}")
    except Exception as e:
        print(f"t-test failed: {e}")

    print("\nExample sampled control sets:")
    for terms, js_val, delta in control_examples:
        print(f"  terms={terms}")
        print(f"    control_JS={js_val:.6f}, delta={delta:.6f}")

    print("\n" + "=" * 90)
    print("INTERPRETATION")
    print("=" * 90)
    print("""
Scientific win:
    relation_delta is larger than almost all sampled control deltas
    empirical p(control >= relation) < 0.05

This would mean:
    The relation terms love/care/like produce stronger category-specific
    local affordance fields than ordinary control words from the same corpus.

Scientific loss:
    relation_delta is typical of control deltas.

That would mean:
    The prior result was likely just a generic topical-window effect:
    any local word window in category-specific articles separates categories.

Important:
    This still does not prove a full semantic theory.
    It tests whether relation predicates are unusually strong organizers
    of local affordance fields.
""")


if __name__ == "__main__":
    main()
