#!/usr/bin/env python3
"""
HSG Affordance Class Clustering Test
=====================================
Tests whether "love" organizes by affordance class rather than object type.

Prediction: within-class distances < between-class distances.

Four affordance classes:
  hedonic:     pizza, chocolate, coffee, music, wine
  companion:   dog, cat, child, friend, horse
  covenantal:  wife, husband, country, God, family
  moral:       enemy, stranger, neighbor, other, rival

Run on AWS instance with existing model:
    python3 hsg_affordance_test.py --model ~/hsg/wiki_model/checkpoint_final

Or retrain:
    python3 hsg_affordance_test.py --corpus ~/hsg/wiki_corpus/corpus.jsonl
"""

import os
import sys
import json
import argparse
import numpy as np
from pathlib import Path
from itertools import combinations
from stats import empirical_p_value

DEFAULT_OUTPUT_DIR = (
    Path(__file__).resolve().parents[1] / "results" / "affordance_results"
)

# ── Affordance classes ────────────────────────────────────────────────────────

AFFORDANCE_CLASSES = {
    "hedonic": [
        "pizza",
        "chocolate",
        "coffee",
        "music",
        "wine",
    ],
    "companion": [
        "my dog",
        "my cat",
        "my child",
        "my friend",
        "my horse",
    ],
    "covenantal": [
        "my wife",
        "my husband",
        "my country",
        "God",
        "my family",
    ],
    "moral": [
        "my enemy",
        "a stranger",
        "my neighbor",
        "the other",
        "my rival",
    ],
}

TARGET_WORD = "love"

# Multiple templates — average across these for robustness
TEMPLATES = [
    "I {verb} {obj}.",
    "We {verb} {obj}.",
    "They {verb} {obj}.",
    "She {verb} {obj} deeply.",
]

# Neutral verb ablation — if love clusters stronger than see,
# the effect is relational, not just object semantics
ABLATION_VERBS = ["see", "need", "fear"]

CLASS_COLORS = {
    "hedonic": "#F4A261",
    "companion": "#2E86AB",
    "covenantal": "#6A4C93",
    "moral": "#E84855",
}

CLASS_DESCRIPTIONS = {
    "hedonic": "hedonic preference (taste, craving, pleasure)",
    "companion": "companionate care (affection, routine, dependence)",
    "covenantal": "covenantal/devotional (duty, sacrifice, loyalty)",
    "moral": "moral obligation (forgiveness, paradox, command)",
}

# ── Geometry helpers ──────────────────────────────────────────────────────────


def final_token_vector(model, tok, sentence, layer_idx, device):
    import torch

    inp = tok(sentence, return_tensors="pt", truncation=True, max_length=128)
    inp = {k: v.to(device) for k, v in inp.items()}
    with torch.no_grad():
        out = model(**inp, output_hidden_states=True)
    hs = out.hidden_states
    idx = layer_idx % len(hs)
    return hs[idx][0, -1, :].float().cpu().numpy()


def l2(v1, v2):
    return float(np.linalg.norm(v1 - v2))


def cos(v1, v2):
    n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
    return float(np.dot(v1, v2) / (n1 * n2)) if n1 > 1e-10 and n2 > 1e-10 else 0.0


# ── Model loading ─────────────────────────────────────────────────────────────


def load_model(model_dir):
    import torch
    from transformers import GPT2LMHeadModel, GPT2Tokenizer

    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading model from {model_dir}...")
    tok = GPT2Tokenizer.from_pretrained(model_dir)
    model = GPT2LMHeadModel.from_pretrained(model_dir, output_hidden_states=True).to(
        DEVICE
    )
    model.eval()
    n_layers = model.config.n_layer
    print(f"  {n_layers}L × d{model.config.n_embd}  device={DEVICE}")
    return model, tok, n_layers, DEVICE


def train_model(corpus_path, output_dir, epochs=5):
    import torch
    import torch.nn as nn
    from torch.utils.data import Dataset, DataLoader
    from transformers import GPT2Config, GPT2LMHeadModel, GPT2Tokenizer

    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    with open(corpus_path) as f:
        if corpus_path.endswith(".jsonl"):
            texts = [json.loads(line).get("text", "") for line in f]
        else:
            texts = [f.read()]
    corpus = "\n\n".join(texts)

    tok = GPT2Tokenizer.from_pretrained("gpt2", cache_dir=f"{output_dir}/cache")
    tok.pad_token = tok.eos_token

    config = GPT2Config(
        vocab_size=tok.vocab_size,
        n_positions=512,
        n_embd=512,
        n_layer=6,
        n_head=8,
        n_inner=2048,
        resid_pdrop=0.1,
        attn_pdrop=0.1,
        embd_pdrop=0.1,
    )
    model = GPT2LMHeadModel(config).to(DEVICE)
    print(f"  {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M params")

    class D(Dataset):
        def __init__(self):
            tokens = tok.encode(corpus)
            self.chunks = []
            for i in range(0, len(tokens) - 512, 256):
                c = tokens[i : i + 512]
                if len(c) == 512:
                    import torch as t

                    self.chunks.append(t.tensor(c, dtype=t.long))

        def __len__(self):
            return len(self.chunks)

        def __getitem__(self, i):
            return self.chunks[i]

    loader = DataLoader(D(), batch_size=8, shuffle=True)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4)
    print(f"  Training {epochs} epochs...")
    for ep in range(epochs):
        losses = []
        for batch in loader:
            batch = batch.to(DEVICE)
            loss = model(input_ids=batch, labels=batch.clone()).loss
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            losses.append(loss.item())
        print(f"  Epoch {ep + 1}: loss={np.mean(losses):.4f}")

    model.save_pretrained(output_dir)
    tok.save_pretrained(output_dir)
    return model, tok, config.n_layer, DEVICE


# ── Main test ─────────────────────────────────────────────────────────────────


def run_affordance_test(model, tok, n_layers, device):
    """
    Core test: extract vectors for all objects, compute pairwise distances,
    test whether within-class < between-class.
    """
    # hidden_states[0] = embedding, hidden_states[k] = after block k
    # n-2 invariant: use block output at n_layers - 2
    block_idx = max(1, n_layers - 2)
    hidden_idx = block_idx  # hidden_states[block_idx] = after block block_idx

    # Extract vectors for all objects
    print(
        f"\nExtracting vectors after transformer block {block_idx}/{n_layers} (n-2 invariant)..."
    )
    all_objects = []
    for cls, objects in AFFORDANCE_CLASSES.items():
        all_objects.extend([(obj, cls) for obj in objects])

    vecs = {}
    for obj, cls in all_objects:
        sentence = f"I {TARGET_WORD} {obj}."
        vecs[obj] = final_token_vector(model, tok, sentence, hidden_idx, device)
    print(f"  {len(vecs)} vectors extracted")

    # Compute all pairwise distances
    distances = {}
    for (obj1, cls1), (obj2, cls2) in combinations(all_objects, 2):
        key = f"{obj1}/{obj2}"
        distances[key] = {
            "dist": l2(vecs[obj1], vecs[obj2]),
            "cos": cos(vecs[obj1], vecs[obj2]),
            "class1": cls1,
            "class2": cls2,
            "same_class": cls1 == cls2,
        }

    # Separate within-class and between-class distances
    within = [v["dist"] for v in distances.values() if v["same_class"]]
    between = [v["dist"] for v in distances.values() if not v["same_class"]]

    print(f"\n  Within-class pairs:   n={len(within)}")
    print(f"  Between-class pairs:  n={len(between)}")

    # Permutation test over class labels (not Mann-Whitney — pairwise distances
    # are not independent since each object appears in multiple pairs)
    observed_ratio = np.mean(between) / np.mean(within) if np.mean(within) > 0 else 1.0

    # Permute class labels 1000 times, recompute ratio
    np.random.seed(42)
    n_perm = 1000
    perm_ratios = []
    obj_list = [obj for obj, cls in all_objects]
    cls_list = [cls for obj, cls in all_objects]

    for _ in range(n_perm):
        perm_cls = np.random.permutation(cls_list)
        perm_map = dict(zip(obj_list, perm_cls))
        perm_within, perm_between = [], []
        for (o1, _), (o2, _) in combinations(all_objects, 2):
            d_val = distances[f"{o1}/{o2}"]["dist"]
            if perm_map[o1] == perm_map[o2]:
                perm_within.append(d_val)
            else:
                perm_between.append(d_val)
        if perm_within:
            perm_ratios.append(np.mean(perm_between) / np.mean(perm_within))

    p_val = empirical_p_value(perm_ratios, observed_ratio, alternative="greater")
    d = (np.mean(between) - np.mean(within)) / np.sqrt(
        (np.std(within) ** 2 + np.std(between) ** 2) / 2
    )
    u_stat = observed_ratio  # reuse slot for ratio

    return vecs, distances, within, between, u_stat, p_val, d


def print_results(distances, within, between, u_stat, p_val, d):
    """Print full results."""

    print(f"\n{'=' * 65}")
    print(f"HSG AFFORDANCE CLASS TEST  \u2014  target word: '{TARGET_WORD}'")
    print(f"{'=' * 65}")

    # Per-class mean distances
    print("\n  Mean within-class distances:")
    for cls in AFFORDANCE_CLASSES:
        pairs = [
            v["dist"]
            for v in distances.values()
            if v["same_class"] and v["class1"] == cls
        ]
        if pairs:
            print(
                f"    {cls:<15} mean={np.mean(pairs):.3f}  ({CLASS_DESCRIPTIONS[cls]})"
            )

    print("\n  Mean between-class distances:")
    for cls1, cls2 in combinations(AFFORDANCE_CLASSES.keys(), 2):
        pairs = [
            v["dist"]
            for v in distances.values()
            if not v["same_class"] and {v["class1"], v["class2"]} == {cls1, cls2}
        ]
        if pairs:
            print(f"    {cls1}/{cls2:<20} mean={np.mean(pairs):.3f}")

    print("\n  SUMMARY:")
    print(f"    Within-class  mean={np.mean(within):.3f}  sd={np.std(within):.3f}")
    print(f"    Between-class mean={np.mean(between):.3f}  sd={np.std(between):.3f}")
    print(
        f"    Ratio: {np.mean(between) / np.mean(within):.3f}× "
        f"(>1 = within closer than between)"
    )
    print(
        f"\n    Permutation test: observed ratio={u_stat:.3f}  p={p_val:.4f}  (n_perm=1000)"
    )
    print(f"    Cohen's d={d:.3f}")

    if p_val < 0.05 and d > 0:
        verdict = "SUPPORTED: within-class distances significantly smaller"
        detail = "The model organizes 'love' by affordance class."
    elif p_val < 0.1:
        verdict = "DIRECTIONAL: trend present, marginal significance"
        detail = "Weak evidence for affordance-class organization."
    else:
        verdict = "NOT SUPPORTED at \u03b1=0.05"
        detail = "No clear affordance-class clustering at this scale."

    print(f"\n    Verdict: {verdict}")
    print(f"    {detail}")

    # Show most revealing pairs
    print("\n  Most within-class similar:")
    within_pairs = [(k, v) for k, v in distances.items() if v["same_class"]]
    for k, v in sorted(within_pairs, key=lambda x: x[1]["dist"])[:5]:
        print(f"    {k:<25} {v['dist']:.3f}  [{v['class1']}]")

    print("\n  Most between-class distant:")
    between_pairs = [(k, v) for k, v in distances.items() if not v["same_class"]]
    for k, v in sorted(between_pairs, key=lambda x: -x[1]["dist"])[:5]:
        print(f"    {k:<25} {v['dist']:.3f}  [{v['class1']}] vs [{v['class2']}]")

    print(f"\n{'=' * 65}")


def make_figure(vecs, distances, output_path):
    """Visualize affordance class geometry."""
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from sklearn.decomposition import PCA
        from sklearn.preprocessing import StandardScaler

        fig, axes = plt.subplots(1, 2, figsize=(14, 6))

        # (a) PCA of all vectors
        ax1 = axes[0]
        all_objs = []
        all_vecs = []
        all_cls = []
        for cls, objects in AFFORDANCE_CLASSES.items():
            for obj in objects:
                if obj in vecs:
                    all_objs.append(obj)
                    all_vecs.append(vecs[obj])
                    all_cls.append(cls)

        V = np.array(all_vecs)
        V_scaled = StandardScaler().fit_transform(V)
        pca = PCA(n_components=2)
        V2 = pca.fit_transform(V_scaled)

        for cls in AFFORDANCE_CLASSES:
            idx = [i for i, c in enumerate(all_cls) if c == cls]
            ax1.scatter(
                V2[idx, 0],
                V2[idx, 1],
                color=CLASS_COLORS[cls],
                s=120,
                label=cls,
                edgecolors="white",
                linewidth=1.5,
                zorder=5,
            )
            for i in idx:
                ax1.annotate(
                    all_objs[i],
                    (V2[i, 0], V2[i, 1]),
                    textcoords="offset points",
                    xytext=(5, 3),
                    fontsize=8,
                    color=CLASS_COLORS[cls],
                )

        ax1.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0] * 100:.0f}%)")
        ax1.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1] * 100:.0f}%)")
        ax1.set_title(
            '"love" affordance geometry\n(PCA of final-token vectors)',
            fontweight="bold",
        )
        ax1.legend(fontsize=9)
        ax1.grid(alpha=0.3)

        # (b) Within vs between class distance distributions
        ax2 = axes[1]
        within_d = [v["dist"] for v in distances.values() if v["same_class"]]
        between_d = [v["dist"] for v in distances.values() if not v["same_class"]]
        vp = ax2.violinplot(
            [within_d, between_d], positions=[0, 1], showmedians=True, widths=0.6
        )
        for pc, col in zip(vp["bodies"], ["#2E86AB", "#E84855"]):
            pc.set_facecolor(col)
            pc.set_alpha(0.65)
        ax2.set_xticks([0, 1])
        ax2.set_xticklabels(
            ["Within class\n(same affordance)", "Between class\n(different affordance)"]
        )
        ax2.set_ylabel("L2 distance")
        ax2.set_title("Within vs between-class distances", fontweight="bold")
        ax2.grid(alpha=0.3, axis="y")

        fig.suptitle(
            f"HSG Affordance Class Test  \u2014  target: '{TARGET_WORD}'\n"
            f"Singleton, M.",
            fontsize=11,
            fontweight="bold",
        )
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        print(f"  Figure: {output_path}")
    except ImportError:
        print(
            "  (sklearn not available for PCA — install with: pip install scikit-learn)"
        )
    except Exception as e:
        print(f"  Figure error: {e}")


# ── CLI ───────────────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(description="HSG Affordance Class Clustering Test")
    parser.add_argument("--model", default=None, help="Path to trained model directory")
    parser.add_argument(
        "--corpus", default=None, help="Corpus file to train on (if no model)"
    )
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--epochs", type=int, default=5)
    args = parser.parse_args()

    Path(args.output).mkdir(parents=True, exist_ok=True)

    if args.model and os.path.exists(f"{args.model}/config.json"):
        model, tok, n_layers, device = load_model(args.model)
    elif args.corpus:
        model_dir = f"{args.output}/model"
        model, tok, n_layers, device = train_model(args.corpus, model_dir, args.epochs)
    else:
        print("Need --model or --corpus")
        sys.exit(1)

    vecs, distances, within, between, u, p, d = run_affordance_test(
        model, tok, n_layers, device
    )

    print_results(distances, within, between, u, p, d)

    # Save
    # Centroid-based separation
    centroids = {}
    for cls, objects in AFFORDANCE_CLASSES.items():
        cls_vecs = [vecs[obj] for obj in objects if obj in vecs]
        if cls_vecs:
            centroids[cls] = np.mean(cls_vecs, axis=0)
    centroid_dists = {}
    for c1, c2 in combinations(centroids.keys(), 2):
        centroid_dists[f"{c1}/{c2}"] = float(l2(centroids[c1], centroids[c2]))

    print("\n  Centroid distances between classes:")
    for pair, dist in sorted(centroid_dists.items(), key=lambda x: x[1]):
        print(f"    {pair:<25} {dist:.3f}")

    results = {
        "target_word": TARGET_WORD,
        "classes": AFFORDANCE_CLASSES,
        "within_mean": float(np.mean(within)),
        "between_mean": float(np.mean(between)),
        "ratio": float(np.mean(between) / np.mean(within)),
        "permutation_p": float(p),
        "cohens_d": float(d),
        "centroid_distances": centroid_dists,
        "distances": {
            k: {kk: float(vv) if isinstance(vv, float) else vv for kk, vv in v.items()}
            for k, v in distances.items()
        },
    }
    out_path = f"{args.output}/affordance_results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"  Results: {out_path}")

    make_figure(vecs, distances, f"{args.output}/affordance_geometry.png")


if __name__ == "__main__":
    main()
