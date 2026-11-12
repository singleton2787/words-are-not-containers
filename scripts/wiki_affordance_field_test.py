#!/usr/bin/env python3
"""
Wikipedia Affordance Field Test

Thesis:
    A word does not carry meaning into a sentence like a sealed container.
    It enters a relation, and the relation opens a field of possible
    continuations, inferences, actions, and obligations.

Corpus:
    Wikipedia plain-text extracts.

Question:
    Do relation words such as love/care/like open different local
    affordance fields depending on the object/domain context?

Examples:
    love + food     -> taste, eat, cuisine, flavor, dish
    love + spouse   -> marriage, relationship, partner, family, affection
    love + pet      -> animal, companion, care, owner, loyalty
    love + country  -> patriotism, nation, identity, sacrifice
    love + enemy    -> forgiveness, ethics, mercy, religion
    love + math     -> beauty, proof, logic, abstraction

This script does not use an LLM.
It uses corpus statistics:
    - local relation-window co-occurrence
    - category baseline fields
    - Jensen-Shannon divergence
    - relation-field vs baseline-category separation
    - shuffled-label permutation test

Important:
    Wikipedia may rate-limit you. This script batches requests, backs off,
    respects Retry-After when present, and caches every successful page.
"""

import json
import os
import re
import time
import random
import requests
import numpy as np

from collections import Counter, defaultdict
from typing import Dict, List, Tuple
from scipy import stats
from stats import bootstrap_mean_ci, empirical_p_value


# =============================================================================
# Config
# =============================================================================

SEED = 42
LANG = "en"

WINDOW = 12
MIN_TOKEN_LEN = 3
TOP_N_FIELD_WORDS = 30
N_PERM = 5000

CACHE_FILE = "wiki_affordance_corpus.txt"
CACHE_JSON = "wiki_affordance_pages_cache.json"

USER_AGENT = "HSG-Affordance-Field-Test/0.3 (research)"

API_SLEEP = 2.5
MAX_RETRIES = 8
BATCH_SIZE_TITLES = 5


# =============================================================================
# Page sets
# =============================================================================

PAGE_SETS = {
    "food": [
        "Pizza",
        "Food",
        "Cuisine",
        "Taste",
        "Flavor",
        "Cooking",
        "Restaurant",
        "Chocolate",
        "Ice cream",
        "Sushi",
        "Barbecue",
        "Bread",
        "Coffee",
        "Taco",
        "Eating",
        "Appetite",
    ],
    "spouse": [
        "Marriage",
        "Wife",
        "Husband",
        "Spouse",
        "Romance (love)",
        "Intimate relationship",
        "Family",
        "Wedding",
        "Monogamy",
        "Partnership",
        "Love",
        "Affection",
        "Commitment",
    ],
    "pet": [
        "Dog",
        "Cat",
        "Pet",
        "Human–animal bond",
        "Companion dog",
        "Animal shelter",
        "Veterinarian",
        "Pet food",
        "Dog training",
        "Domestication",
        "Animal welfare",
        "Puppy",
    ],
    "country": [
        "Country",
        "Nation",
        "Patriotism",
        "Nationalism",
        "Citizenship",
        "Flag",
        "Homeland",
        "National identity",
        "Civic virtue",
        "Military service",
        "Constitution",
        "Republic",
    ],
    "enemy": [
        "Enemy",
        "Forgiveness",
        "Ethics",
        "Love your enemies",
        "Sermon on the Mount",
        "Christian ethics",
        "Revenge",
        "Mercy",
        "Nonviolence",
        "Moral psychology",
        "Altruism",
    ],
    "math": [
        "Mathematics",
        "Beauty",
        "Mathematical beauty",
        "Proof (mathematics)",
        "Number theory",
        "Geometry",
        "Algebra",
        "Calculus",
        "Mathematician",
        "Logic",
        "Symmetry",
    ],
}

RELATION_SETS = {
    "love": ["love", "loved", "loves", "loving"],
    "care": ["care", "cares", "cared", "caring"],
    "like": ["like", "liked", "likes", "liking"],
    "love_care_like": [
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
    ],
}

STOPWORDS = set(
    """
a an the and or but if while although though because since of to in on for from with without by as at into
is are was were be been being have has had do does did this that these those it its they them their he she
his her hers him we us our you your i me my mine not no yes can could may might must should would will shall
about after before during over under between among than then there here also such other more most many some any
one two three first second new known used use often including include includes called made make makes became become
which who whom whose where when why how into onto per via each both either neither same different various several
""".split()
)


# =============================================================================
# Cache helpers
# =============================================================================


def load_page_cache() -> Dict[str, str]:
    if os.path.exists(CACHE_JSON):
        with open(CACHE_JSON, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def save_page_cache(cache: Dict[str, str]) -> None:
    with open(CACHE_JSON, "w", encoding="utf-8") as f:
        json.dump(cache, f, ensure_ascii=False, indent=2)


def chunks(xs: List[str], n: int):
    for i in range(0, len(xs), n):
        yield xs[i : i + n]


# =============================================================================
# Wikipedia fetcher
# =============================================================================


def fetch_extract_batch(titles: List[str]) -> Dict[str, str]:
    """
    Fetch several page extracts in one MediaWiki Action API request.
    Handles 429 with Retry-After / exponential backoff.
    """
    url = f"https://{LANG}.wikipedia.org/w/api.php"

    params = {
        "action": "query",
        "format": "json",
        "prop": "extracts",
        "explaintext": "1",
        "exsectionformat": "plain",
        "redirects": "1",
        "titles": "|".join(titles),
        "maxlag": "5",
    }

    headers = {
        "User-Agent": USER_AGENT,
        "Accept": "application/json",
    }

    delay = API_SLEEP

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            r = requests.get(url, params=params, headers=headers, timeout=30)

            if r.status_code == 429:
                retry_after = r.headers.get("Retry-After")
                wait = float(retry_after) if retry_after else delay
                print(
                    f"    429 rate limit. Waiting {wait:.1f}s before retry {attempt}/{MAX_RETRIES}."
                )
                time.sleep(wait)
                delay = min(delay * 2.0, 180.0)
                continue

            if r.status_code in (503, 504):
                print(f"    server lag/status {r.status_code}. Waiting {delay:.1f}s.")
                time.sleep(delay)
                delay = min(delay * 2.0, 180.0)
                continue

            r.raise_for_status()
            data = r.json()

            if "error" in data:
                code = data["error"].get("code", "")
                info = data["error"].get("info", "")

                if code == "maxlag":
                    print(f"    maxlag: {info}. Waiting {delay:.1f}s.")
                    time.sleep(delay)
                    delay = min(delay * 2.0, 180.0)
                    continue

                raise RuntimeError(f"MediaWiki API error: {data['error']}")

            pages = data.get("query", {}).get("pages", {})
            results = {}

            for _, page in pages.items():
                title = page.get("title", "")
                extract = page.get("extract", "")
                if title and extract:
                    results[title] = extract

            time.sleep(API_SLEEP)
            return results

        except Exception as e:
            print(f"    fetch attempt {attempt}/{MAX_RETRIES} failed: {e}")
            time.sleep(delay)
            delay = min(delay * 2.0, 180.0)

    return {}


def build_or_load_corpus(force_rebuild: bool = False) -> Tuple[str, Dict[str, str]]:
    """
    Build corpus from cached page extracts and/or Wikipedia API.

    The page cache is JSON and persists each page.
    The corpus file is regenerated from the page cache.
    """
    page_cache = load_page_cache()

    if force_rebuild and os.path.exists(CACHE_FILE):
        os.remove(CACHE_FILE)

    category_texts = {}
    all_text = []

    for category, titles in PAGE_SETS.items():
        print(f"\nCategory: {category}")

        chunks_for_category = []
        missing = []

        for title in titles:
            key = f"{category}::{title}"

            if key in page_cache and page_cache[key].strip():
                chunks_for_category.append(
                    f"\n\n### CATEGORY={category} TITLE={title}\n{page_cache[key]}\n"
                )
            else:
                missing.append(title)

        print(f"  cached pages: {len(titles) - len(missing)}/{len(titles)}")

        if missing:
            print(f"  missing pages: {len(missing)}")

        for batch in chunks(missing, BATCH_SIZE_TITLES):
            print(f"  fetching batch: {batch}")
            results = fetch_extract_batch(batch)

            used_return_titles = set()

            for title in batch:
                matched_text = None

                # Exact title match.
                if title in results:
                    matched_text = results[title]
                    used_return_titles.add(title)
                else:
                    # Redirects may return a different normalized title.
                    # Use a conservative fallback if only one or few results returned.
                    for returned_title, text in results.items():
                        if returned_title not in used_return_titles and text:
                            matched_text = text
                            used_return_titles.add(returned_title)
                            break

                if matched_text:
                    key = f"{category}::{title}"
                    page_cache[key] = matched_text
                    save_page_cache(page_cache)

                    chunks_for_category.append(
                        f"\n\n### CATEGORY={category} TITLE={title}\n{matched_text}\n"
                    )
                    print(f"    ok: {title} ({len(matched_text):,} chars)")
                else:
                    print(f"    failed/no extract: {title}")

        category_texts[category] = "\n".join(chunks_for_category)

        if category_texts[category].strip():
            all_text.append(category_texts[category])

    corpus = "\n".join(all_text)

    with open(CACHE_FILE, "w", encoding="utf-8") as f:
        f.write(corpus)

    print(f"\nSaved corpus: {CACHE_FILE} ({len(corpus):,} chars)")
    print(f"Saved page cache: {CACHE_JSON} ({len(page_cache)} pages)")

    return corpus, category_texts


# =============================================================================
# Tokenization/statistics
# =============================================================================


def tokenize(text: str) -> List[str]:
    text = text.lower()

    # Keep apostrophes/hyphens inside words, but simplify everything else.
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


def build_field_for_category(
    tokens: List[str], relation_terms: List[str], window: int
) -> Counter:
    field = Counter()

    for i, tok in enumerate(tokens):
        if tok in relation_terms:
            lo = max(0, i - window)
            hi = min(len(tokens), i + window + 1)

            for j in range(lo, hi):
                if j == i:
                    continue

                w = tokens[j]
                if w not in STOPWORDS and len(w) >= MIN_TOKEN_LEN:
                    field[w] += 1

    return field


def build_baseline_field(tokens: List[str]) -> Counter:
    return Counter(tokens)


def normalize_counter(
    c: Counter, vocabulary: List[str], alpha: float = 1e-6
) -> np.ndarray:
    arr = np.array([c.get(w, 0.0) for w in vocabulary], dtype=np.float64)
    arr += alpha
    total = arr.sum()

    if total <= 0:
        return np.ones(len(vocabulary), dtype=np.float64) / max(1, len(vocabulary))

    return arr / total


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


def top_words(counter: Counter, n: int = 25) -> List[Tuple[str, int]]:
    return counter.most_common(n)


def mean_pairwise_distance(js_dict: Dict[Tuple[str, str], float]) -> float:
    vals = list(js_dict.values())
    if not vals:
        return float("nan")
    return float(np.mean(vals))


def permutation_category_separation(
    fields_by_cat: Dict[str, Counter], n_perm: int = N_PERM
) -> float:
    """
    Shuffled-label null:

    Pool all relation-window words, then randomly redistribute to categories
    preserving field sizes. Compare mean pairwise JS against observed.

    Null:
        relation-window tokens are exchangeable across category labels.
    """
    categories = sorted(fields_by_cat.keys())

    vocab = sorted(set(w for c in fields_by_cat.values() for w in c.keys()))
    if len(vocab) == 0 or len(categories) < 2:
        return float("nan")

    observed_vecs = {
        cat: normalize_counter(fields_by_cat[cat], vocab) for cat in categories
    }

    observed_mean = mean_pairwise_distance(pairwise_js(observed_vecs))

    pool = []
    sizes = {}

    for cat in categories:
        words = []
        for w, n in fields_by_cat[cat].items():
            words.extend([w] * n)

        sizes[cat] = len(words)
        pool.extend(words)

    if len(pool) == 0:
        return float("nan")

    perm_means = []

    for _ in range(n_perm):
        random.shuffle(pool)

        idx = 0
        perm_fields = {}

        for cat in categories:
            n = sizes[cat]
            perm_fields[cat] = Counter(pool[idx : idx + n])
            idx += n

        perm_vecs = {
            cat: normalize_counter(perm_fields[cat], vocab) for cat in categories
        }

        perm_mean = mean_pairwise_distance(pairwise_js(perm_vecs))
        perm_means.append(perm_mean)

    return empirical_p_value(perm_means, observed_mean, alternative="greater")


def bootstrap_ci(data: List[float], n_boot: int = 5000) -> Tuple[float, float]:
    return bootstrap_mean_ci(data, n_resamples=n_boot, seed=SEED)


# =============================================================================
# Analysis
# =============================================================================


def analyze_relation_set(
    relation_name: str,
    relation_terms: List[str],
    category_tokens: Dict[str, List[str]],
) -> Dict:
    print("\n" + "=" * 90)
    print(f"RELATION SET: {relation_name} -> {relation_terms}")
    print("=" * 90)

    fields = {}
    baselines = {}

    for cat, toks in category_tokens.items():
        fields[cat] = build_field_for_category(toks, relation_terms, WINDOW)
        baselines[cat] = build_baseline_field(toks)

    # Drop categories with zero relation hits for this specific relation set.
    active_cats = [
        cat for cat in sorted(fields.keys()) if sum(fields[cat].values()) > 0
    ]

    print(f"Active categories with relation hits: {active_cats}")

    if len(active_cats) < 2:
        print("Not enough active categories for pairwise comparison.")
        return {
            "relation_name": relation_name,
            "active_categories": active_cats,
            "mean_relation_js": float("nan"),
            "mean_baseline_js": float("nan"),
            "mean_delta": float("nan"),
            "p_perm": float("nan"),
            "deltas": [],
        }

    fields = {cat: fields[cat] for cat in active_cats}
    baselines = {cat: baselines[cat] for cat in active_cats}

    field_vocab = sorted(set(w for c in fields.values() for w in c.keys()))
    base_vocab = sorted(set(w for c in baselines.values() for w in c.keys()))

    field_vecs = {cat: normalize_counter(fields[cat], field_vocab) for cat in fields}

    base_vecs = {
        cat: normalize_counter(baselines[cat], base_vocab) for cat in baselines
    }

    field_js = pairwise_js(field_vecs)
    base_js = pairwise_js(base_vecs)

    print("\nFIELD WORDS AROUND RELATION TERMS")
    for cat in sorted(fields.keys()):
        hits = sum(fields[cat].values())
        print(f"\n[{cat}] relation-window hits={hits}")
        print(
            ", ".join(
                [f"{w}:{n}" for w, n in top_words(fields[cat], TOP_N_FIELD_WORDS)]
            )
        )

    print("\nPAIRWISE CATEGORY SEPARATION")
    print(f"{'pair':<28} {'relation_JS':>14} {'baseline_JS':>14} {'delta':>14}")
    print("-" * 90)

    deltas = []

    for pair in sorted(field_js.keys()):
        rel = field_js[pair]
        base = base_js[pair]
        delta = rel - base
        deltas.append(delta)

        pair_name = f"{pair[0]} vs {pair[1]}"
        print(f"{pair_name:<28} {rel:>14.6f} {base:>14.6f} {delta:>14.6f}")

    mean_relation_js = mean_pairwise_distance(field_js)
    mean_baseline_js = mean_pairwise_distance(base_js)
    mean_delta = float(np.mean(deltas)) if deltas else float("nan")
    ci_low, ci_high = bootstrap_ci(deltas)
    p_perm = permutation_category_separation(fields)

    print("\nSUMMARY")
    print(f"Mean relation-field JS        -> {mean_relation_js:.6f}")
    print(f"Mean baseline-category JS     -> {mean_baseline_js:.6f}")
    print(f"Mean delta relation-baseline  -> {mean_delta:.6f}")
    print(f"Delta bootstrap 95% CI        -> [{ci_low:.6f}, {ci_high:.6f}]")

    if len(deltas) >= 2:
        try:
            t_stat, p_two = stats.ttest_1samp(deltas, 0.0)
            p_one = p_two / 2.0 if t_stat > 0 else 1.0 - (p_two / 2.0)
            print(f"One-sample t-test delta>0 p   -> {p_one:.6f}")
        except Exception as e:
            print(f"t-test failed: {e}")

    print(f"Shuffled field-label p        -> {p_perm:.6f}")

    return {
        "relation_name": relation_name,
        "active_categories": active_cats,
        "mean_relation_js": mean_relation_js,
        "mean_baseline_js": mean_baseline_js,
        "mean_delta": mean_delta,
        "p_perm": p_perm,
        "deltas": deltas,
    }


def main() -> None:
    random.seed(SEED)
    np.random.seed(SEED)

    corpus, _ = build_or_load_corpus(force_rebuild=True)
    category_text = split_category_sections(corpus)

    print("\n" + "=" * 90)
    print("WIKIPEDIA AFFORDANCE FIELD TEST")
    print("=" * 90)
    print(f"Loaded categories: {sorted(category_text.keys())}")

    if len(category_text) < 3:
        print("\nNOT ENOUGH CATEGORIES LOADED.")
        print(
            "The previous run likely hit Wikipedia rate limits before enough pages downloaded."
        )
        print(
            "Wait a few minutes and rerun. The JSON page cache will preserve successful pages."
        )
        print("You need at least 3 categories for this analysis.")
        raise SystemExit(1)

    category_tokens = {cat: tokenize(text) for cat, text in category_text.items()}

    print("\nCATEGORY TOKEN COUNTS")
    for cat, toks in sorted(category_tokens.items()):
        print(f"{cat:<12} {len(toks):>8} tokens")

    results = []

    for relation_name, relation_terms in RELATION_SETS.items():
        result = analyze_relation_set(relation_name, relation_terms, category_tokens)
        results.append(result)

    print("\n" + "=" * 90)
    print("FINAL ABLATION SUMMARY")
    print("=" * 90)
    print(
        f"{'relation':<18} {'active_cats':>12} {'rel_JS':>12} {'base_JS':>12} {'delta':>12} {'perm_p':>12}"
    )
    print("-" * 90)

    for r in results:
        print(
            f"{r['relation_name']:<18} "
            f"{len(r['active_categories']):>12} "
            f"{r['mean_relation_js']:>12.6f} "
            f"{r['mean_baseline_js']:>12.6f} "
            f"{r['mean_delta']:>12.6f} "
            f"{r['p_perm']:>12.6f}"
        )

    print("\nINTERPRETATION")
    print("""
This is a corpus-linguistic test of Relational Affordance Semantics.

Weak but real support:
    relation-window fields differ by category more than shuffled-label null.

Stronger support:
    relation-window separation exceeds baseline-category separation:
        mean_delta = mean_relation_JS - mean_baseline_JS > 0

That stronger result would suggest the relation terms are not merely sitting
inside different topics. They are selecting and reorganizing local affordance
fields inside those topics.

If relation JS is significant but delta <= 0:
    The categories differ, but we have not shown that the relation word itself
    adds structure beyond topic/domain.

If only love works:
    The effect may be relation-specific.

If love/care/like all work:
    The effect may be broader: relation terms generally open category-specific
    affordance fields.

This is not an LLM claim.
This is a linguistics/corpus-semantics claim.
""")


if __name__ == "__main__":
    main()
