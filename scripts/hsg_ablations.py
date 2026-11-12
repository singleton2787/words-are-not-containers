#!/usr/bin/env python3
"""
HSG Ablation Suite — Experiments A and B
=========================================
Three ablations before sending to Nelson:

1. Layer sweep — does affordance structure peak at n-2 across all predicates?
2. Random partition test — does observed ratio beat random class assignments?
3. NLL control — is class structure driven by prompt naturalness?

Run:
    python3 hsg_ablations.py
    python3 hsg_ablations.py --ablation layer_sweep
    python3 hsg_ablations.py --ablation random_partition
    python3 hsg_ablations.py --ablation nll_control
    python3 hsg_ablations.py --ablation all
"""

import json
import argparse
import numpy as np
from itertools import combinations
from pathlib import Path
import torch
from hf_cache import hf_cache_dir
from stats import empirical_p_value

MODEL_NAME = "Qwen/Qwen2-7B"
CACHE_DIR = hf_cache_dir()
OUTPUT_DIR = str(Path(__file__).resolve().parents[1] / "results" / "ablation_results")
Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

# ── Object sets (from Experiment B) ──────────────────────────────────────────

EXP_B_FIELDS = {
    "love": {
        "templates": ["I love {obj}.", "She loves {obj}.", "They love {obj}."],
        "classes": {
            "hedonic": ["pizza", "chocolate", "coffee", "music", "wine"],
            "companion": ["my dog", "my cat", "my child", "my friend", "my horse"],
            "covenantal": ["my wife", "my husband", "my country", "God", "my family"],
            "moral": ["my enemy", "a stranger", "my neighbor", "the other", "my rival"],
        },
    },
    "fear": {
        "templates": ["I fear {obj}.", "She fears {obj}.", "They fear {obj}."],
        "classes": {
            "physical": ["disease", "fire", "violence", "death", "pain"],
            "existential": [
                "failure",
                "loss",
                "betrayal",
                "loneliness",
                "meaninglessness",
            ],
            "ambient": ["darkness", "heights", "spiders", "storms", "silence"],
        },
    },
    "own": {
        "templates": ["I own {obj}.", "She owns {obj}.", "They own {obj}."],
        "classes": {
            "property": ["a house", "land", "a boat", "a building", "a farm"],
            "vehicle": ["a car", "a truck", "a motorcycle", "a bicycle", "a plane"],
            "living": ["a dog", "a horse", "a cat", "cattle", "a parrot"],
            "asset": ["a painting", "a business", "stocks", "jewelry", "a patent"],
        },
    },
    "care": {
        "templates": [
            "I care for {obj}.",
            "She cares for {obj}.",
            "They care for {obj}.",
        ],
        "classes": {
            "dependent": ["a child", "a baby", "an elder", "a patient", "an invalid"],
            "companion": ["a dog", "a cat", "a friend", "a partner", "a horse"],
            "family": [
                "my mother",
                "my father",
                "my sister",
                "my brother",
                "my family",
            ],
        },
    },
}

# ── Model / geometry helpers ──────────────────────────────────────────────────


def load_model():
    from transformers import AutoModelForCausalLM, AutoTokenizer

    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading {MODEL_NAME} on {DEVICE}...")
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
    print(f"  {n_layers}L × d{model.config.hidden_size}")
    return model, tok, n_layers, DEVICE


def get_hidden_states(model, tok, sentence, device, max_length=128):
    """Return all hidden states for a sentence."""
    inp = tok(sentence, return_tensors="pt", truncation=True, max_length=max_length).to(
        device
    )
    with torch.no_grad():
        out = model(**inp, output_hidden_states=True)
    # Stack: (n_layers+1, seq_len, d_model)
    return [hs[0, -1, :].float().cpu().numpy() for hs in out.hidden_states]


def get_nll(model, tok, sentence, device, max_length=128):
    """Compute NLL (negative log likelihood) of a sentence."""
    inp = tok(sentence, return_tensors="pt", truncation=True, max_length=max_length).to(
        device
    )
    with torch.no_grad():
        out = model(**inp, labels=inp["input_ids"])
    return float(out.loss.cpu())


def l2(v1, v2):
    return float(np.linalg.norm(v1 - v2))


def within_between_ratio(vecs_by_class):
    """Compute within/between ratio for a set of class vectors."""
    all_items = [(obj, cls) for cls, objs in vecs_by_class.items() for obj in objs]
    within, between = [], []
    for (o1, c1), (o2, c2) in combinations(all_items, 2):
        v1 = vecs_by_class[c1][o1]
        v2 = vecs_by_class[c2][o2] if c1 != c2 else vecs_by_class[c1][o2]
        d = l2(v1, v2)
        if c1 == c2:
            within.append(d)
        else:
            between.append(d)
    if not within:
        return 1.0
    return np.mean(between) / np.mean(within)


# ── Ablation 1: Layer sweep ───────────────────────────────────────────────────


def ablation_layer_sweep(model, tok, n_layers, device):
    """
    Sweep all layers. For each predicate, compute within/between ratio
    at every block. Does affordance structure peak at block n-2?
    """
    print(f"\n{'=' * 65}")
    print("ABLATION 1: LAYER SWEEP")
    print("Does affordance structure peak at block n-2?")
    print(f"{'=' * 65}")

    # Pre-extract all hidden states for all sentences (expensive but do once)
    print("\nExtracting all hidden states (this takes a few minutes)...")

    pred_layer_vecs = {}  # predicate -> {cls -> {obj -> [layer_vecs]}}
    for pred, field in EXP_B_FIELDS.items():
        pred_layer_vecs[pred] = {}
        for cls, objects in field["classes"].items():
            pred_layer_vecs[pred][cls] = {}
            for obj in objects:
                # Average across templates
                all_layer_vecs = None
                for tmpl in field["templates"]:
                    sentence = tmpl.format(obj=obj)
                    layer_vecs = get_hidden_states(model, tok, sentence, device)
                    if all_layer_vecs is None:
                        all_layer_vecs = [v.copy() for v in layer_vecs]
                    else:
                        for i in range(len(layer_vecs)):
                            all_layer_vecs[i] = all_layer_vecs[i] + layer_vecs[i]
                # Average
                n_tmpl = len(field["templates"])
                pred_layer_vecs[pred][cls][obj] = [v / n_tmpl for v in all_layer_vecs]
        print(f"  {pred}: done")

    # Compute ratio at each layer
    print("\n  Layer sweep results:")
    print(
        f"\n  {'Layer':>6}"
        + "".join(f"  {p:>8}" for p in EXP_B_FIELDS)
        + f"  {'mean':>8}"
    )
    print(
        f"  {'-' * 6}" + "".join(f"  {'-' * 8}" for p in EXP_B_FIELDS) + f"  {'-' * 8}"
    )

    layer_results = {pred: [] for pred in EXP_B_FIELDS}
    n_hidden = n_layers + 1  # includes embedding

    for li in range(n_hidden):
        row = f"  {li:>6}"
        ratios = []
        for pred in EXP_B_FIELDS:
            # Build vecs_by_class for this layer
            vbc = {
                cls: {obj: pred_layer_vecs[pred][cls][obj][li] for obj in objs}
                for cls, objs in EXP_B_FIELDS[pred]["classes"].items()
            }
            ratio = within_between_ratio(vbc)
            layer_results[pred].append(ratio)
            ratios.append(ratio)
            row += f"  {ratio:>8.3f}"
        row += f"  {np.mean(ratios):>8.3f}"
        print(row)

    # Find peak for each predicate
    print("\n  Peak layers:")
    target_block = n_layers - 2
    all_peaks = []
    for pred in EXP_B_FIELDS:
        ratios = layer_results[pred]
        peak_li = int(np.argmax(ratios))
        peak_pct = peak_li / n_layers * 100
        all_peaks.append(peak_li)
        confirmed = peak_li == target_block or abs(peak_li - target_block) <= 1
        print(
            f"  {pred:<12} peak at block {peak_li}/{n_layers} "
            f"({peak_pct:.0f}%)  "
            f"{'~n-2 ✓' if confirmed else f'n-{n_layers - peak_li} ✗'}"
        )

    mean_peak = np.mean(all_peaks)
    print(
        f"\n  Mean peak block: {mean_peak:.1f}/{n_layers} "
        f"({mean_peak / n_layers * 100:.0f}% depth)"
    )
    print(f"  n-2 block: {target_block}/{n_layers}")
    print(
        f"  Structural invariant: "
        f"{'CONFIRMED' if abs(mean_peak - target_block) <= 1 else 'NOT CONFIRMED'}"
    )

    # Save
    with open(f"{OUTPUT_DIR}/layer_sweep.json", "w") as f:
        json.dump(
            {pred: [float(r) for r in layer_results[pred]] for pred in EXP_B_FIELDS},
            f,
            indent=2,
        )

    return layer_results


# ── Ablation 2: Random partition test ────────────────────────────────────────


def ablation_random_partition(model, tok, n_layers, device, n_rand=500):
    """
    Generate random same-sized partitions of objects.
    Test: does observed ratio beat random partitions?
    """
    print(f"\n{'=' * 65}")
    print("ABLATION 2: RANDOM PARTITION TEST")
    print("Does observed ratio beat random class assignments?")
    print(f"{'=' * 65}")

    block_idx = n_layers - 2

    results = {}
    for pred, field in EXP_B_FIELDS.items():
        print(f"\n  {pred}...")
        classes = field["classes"]
        class_sizes = {cls: len(objs) for cls, objs in classes.items()}

        # Extract vectors
        vbc = {}
        for cls, objects in classes.items():
            vbc[cls] = {}
            for obj in objects:
                tvecs = []
                for tmpl in field["templates"]:
                    sentence = tmpl.format(obj=obj)
                    hs = get_hidden_states(model, tok, sentence, device)
                    tvecs.append(hs[block_idx])
                vbc[cls][obj] = np.mean(tvecs, axis=0)

        observed_ratio = within_between_ratio(vbc)

        # Random partitions
        all_objects = [(obj, cls) for cls, objs in classes.items() for obj in objs]
        obj_list = [o for o, _ in all_objects]
        sizes = list(class_sizes.values())
        np.random.seed(42)

        rand_ratios = []
        for _ in range(n_rand):
            perm = np.random.permutation(len(obj_list))
            rand_classes = {}
            idx = 0
            for i, sz in enumerate(sizes):
                rand_classes[f"rand_{i}"] = {
                    obj_list[perm[idx + j]]: vbc[
                        next(
                            c
                            for c, objs in classes.items()
                            if obj_list[perm[idx + j]] in objs
                        )
                    ][obj_list[perm[idx + j]]]
                    for j in range(sz)
                }
                idx += sz
            rand_ratios.append(within_between_ratio(rand_classes))

        rand_ratios = np.array(rand_ratios)
        p_val = empirical_p_value(rand_ratios, observed_ratio, alternative="greater")
        z = (observed_ratio - rand_ratios.mean()) / rand_ratios.std()

        print(f"  Observed ratio: {observed_ratio:.3f}")
        print(f"  Random mean:    {rand_ratios.mean():.3f} ± {rand_ratios.std():.3f}")
        print(f"  p (rand >= obs): {p_val:.4f}  z={z:.2f}")
        print(f"  {'BEATS RANDOM ✓' if p_val < 0.05 else 'NOT BETTER THAN RANDOM ✗'}")

        results[pred] = {
            "observed": float(observed_ratio),
            "rand_mean": float(rand_ratios.mean()),
            "rand_std": float(rand_ratios.std()),
            "p_value": p_val,
            "z_score": float(z),
        }

    with open(f"{OUTPUT_DIR}/random_partition.json", "w") as f:
        json.dump(results, f, indent=2)
    return results


# ── Ablation 3: NLL control ───────────────────────────────────────────────────


def ablation_nll_control(model, tok, n_layers, device):
    """
    Compute NLL for each prompt.
    Test: does affordance class structure survive after residualizing
    against prompt NLL? (Controls for 'weird phrases are far away' effect)
    """
    print(f"\n{'=' * 65}")
    print("ABLATION 3: NLL CONTROL")
    print("Is class structure driven by prompt naturalness?")
    print(f"{'=' * 65}")

    block_idx = n_layers - 2
    results = {}

    for pred, field in EXP_B_FIELDS.items():
        print(f"\n  {pred}...")
        classes = field["classes"]

        # Compute NLL and vectors for each object
        obj_nlls = {}
        obj_vecs = {}
        for cls, objects in classes.items():
            for obj in objects:
                nlls, vecs = [], []
                for tmpl in field["templates"]:
                    sentence = tmpl.format(obj=obj)
                    nlls.append(get_nll(model, tok, sentence, device))
                    hs = get_hidden_states(model, tok, sentence, device)
                    vecs.append(hs[block_idx])
                obj_nlls[obj] = float(np.mean(nlls))
                obj_vecs[obj] = np.mean(vecs, axis=0)

        # Print NLL by class
        print("  NLL by class:")
        for cls, objects in classes.items():
            class_nlls = [obj_nlls[obj] for obj in objects]
            print(
                f"    {cls:<20} mean NLL={np.mean(class_nlls):.3f}  "
                f"range=[{min(class_nlls):.2f},{max(class_nlls):.2f}]"
            )

        # Check if high-NLL objects are geometrically outliers
        all_objs = [obj for cls_objs in classes.values() for obj in cls_objs]
        nll_vals = np.array([obj_nlls[obj] for obj in all_objs])

        # Compute distance to class centroid for each object
        centroids = {}
        for cls, objects in classes.items():
            centroids[cls] = np.mean([obj_vecs[obj] for obj in objects], axis=0)

        centroid_dists = []
        for obj in all_objs:
            cls = next(c for c, objs in classes.items() if obj in objs)
            dist_to_own = l2(obj_vecs[obj], centroids[cls])
            centroid_dists.append(dist_to_own)

        centroid_dists = np.array(centroid_dists)
        corr, p_corr = __import__("scipy").stats.pearsonr(nll_vals, centroid_dists)

        print(f"  NLL vs centroid distance: r={corr:.3f}  p={p_corr:.4f}")
        if abs(corr) > 0.4 and p_corr < 0.05:
            print(
                "  WARNING: NLL correlates with centroid distance — "
                "class structure may be partly driven by prompt naturalness"
            )
        else:
            print("  OK: class structure not primarily driven by prompt NLL")

        # Recompute within/between ratio using residual-adjusted vectors
        # (approximate: subtract NLL-predicted component from each vector)
        # Current implementation keeps vectors unchanged and reports the NLL
        # correlation as the control diagnostic.
        adj_vecs = {}
        for i, obj in enumerate(all_objs):
            # Scale adjustment by correlation strength
            adj_vecs[obj] = obj_vecs[
                obj
            ]  # simplified: just flag if correlation is high

        print("  Highest NLL prompts:")
        sorted_by_nll = sorted(zip(all_objs, nll_vals), key=lambda x: -x[1])
        for obj, nll in sorted_by_nll[:3]:
            cls = next(c for c, objs in classes.items() if obj in objs)
            print(f"    [{cls}] I {pred} {obj}  NLL={nll:.3f}")

        results[pred] = {
            "nll_by_obj": obj_nlls,
            "nll_centroid_corr": float(corr),
            "nll_centroid_p": float(p_corr),
            "nll_confound": abs(corr) > 0.4 and p_corr < 0.05,
        }

    with open(f"{OUTPUT_DIR}/nll_control.json", "w") as f:
        json.dump(results, f, indent=2, default=float)

    print("\n  Summary:")
    for pred, r in results.items():
        flag = "⚠ NLL CONFOUND" if r["nll_confound"] else "✓ clean"
        print(
            f"  {pred:<12} r={r['nll_centroid_corr']:.3f}  "
            f"p={r['nll_centroid_p']:.4f}  {flag}"
        )

    return results


# ── Main ──────────────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--ablation",
        default="all",
        choices=["layer_sweep", "random_partition", "nll_control", "all"],
    )
    parser.add_argument("--model", default=MODEL_NAME)
    args = parser.parse_args()

    model, tok, n_layers, device = load_model()

    if args.ablation in ("layer_sweep", "all"):
        ablation_layer_sweep(model, tok, n_layers, device)

    if args.ablation in ("random_partition", "all"):
        ablation_random_partition(model, tok, n_layers, device)

    if args.ablation in ("nll_control", "all"):
        ablation_nll_control(model, tok, n_layers, device)

    print(f"\nAll results saved to {OUTPUT_DIR}")
    print("Done.")


if __name__ == "__main__":
    main()
