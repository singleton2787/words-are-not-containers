#!/usr/bin/env python3
"""
HSG Corpus Auditor
==================
Singleton, M.
v0.1 — June 2026

Measures the relational geometry of a target word across comparison terms
in any text corpus. Trains a small GPT from scratch on your corpus and
measures how the target word's meaning is geometrically differentiated
by its surrounding context.

Usage:
    python hsg_audit.py corpus.txt --word love --compare wife,dog,pizza,enemy,god
    python hsg_audit.py corpus.txt --word leader --compare hero,tyrant,servant,father
    python hsg_audit.py corpus.txt --word care --compare patient,profit,family,duty

    # Compare two corpora:
    python hsg_audit.py corpus1.txt corpus2.txt --word love --compare wife,dog,pizza

    # Use a pretrained model (skip training):
    python hsg_audit.py --model /path/to/checkpoint --word love --compare wife,dog,pizza

Install:
    pip install transformers torch scipy numpy matplotlib requests

The tool trains a small 45M parameter GPT on your corpus (~2-4 hours on GPU,
longer on CPU). For quick results use --epochs 2 (less accurate but faster).
"""

import os
import json
import argparse
import numpy as np
from pathlib import Path

# ── Geometry ──────────────────────────────────────────────────────────────────


def final_token_vector(model, tokenizer, sentence, layer_idx, device):
    """Extract hidden state of final token at layer_idx."""
    import torch

    inp = tokenizer(sentence, return_tensors="pt", truncation=True, max_length=128)
    inp = {k: v.to(device) for k, v in inp.items()}
    with torch.no_grad():
        out = model(**inp, output_hidden_states=True)
    hs = out.hidden_states
    idx = layer_idx % len(hs)
    return hs[idx][0, -1, :].float().cpu().numpy()


def l2_distance(v1, v2):
    return float(np.linalg.norm(v1 - v2))


def cosine_similarity(v1, v2):
    n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
    if n1 < 1e-10 or n2 < 1e-10:
        return 0.0
    return float(np.dot(v1, v2) / (n1 * n2))


# ── Corpus loading ────────────────────────────────────────────────────────────


def load_corpus(path):
    """Load text from file. Supports .txt, .jsonl (text field), plain text."""
    text = []
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        if path.endswith(".jsonl"):
            for line in f:
                try:
                    d = json.loads(line)
                    t = d.get("text", d.get("content", d.get("body", "")))
                    if t:
                        text.append(t)
                except Exception:
                    pass
        else:
            text = [f.read()]
    full = "\n\n".join(text)
    n_words = len(full.split())
    print(
        f"  Corpus: {len(text)} document(s), {n_words:,} words, {len(full) / 1e6:.1f}MB"
    )
    return full


def check_target_word(corpus_text, word):
    """Check how many times the target word appears."""
    count = corpus_text.lower().count(f" {word.lower()} ")
    print(f"  Target word '{word}': {count:,} occurrences in corpus")
    if count < 50:
        print("  WARNING: fewer than 50 occurrences. Results may be unreliable.")
        print("  Consider a larger corpus or a more common target word.")
    return count


# ── Model training ────────────────────────────────────────────────────────────


def train_model(
    corpus_text,
    output_dir,
    epochs=5,
    batch_size=8,
    n_embd=512,
    n_layer=6,
    n_head=8,
    verbose=True,
):
    """
    Train a small GPT on the corpus.
    Returns (model, tokenizer, n_layers).
    """
    import torch
    import torch.nn as nn
    from torch.utils.data import Dataset, DataLoader
    from transformers import GPT2Config, GPT2LMHeadModel, GPT2Tokenizer

    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    if verbose:
        print(f"\n  Device: {DEVICE}")
        if DEVICE == "cuda":
            print(f"  GPU: {torch.cuda.get_device_name(0)}")
            print(
                f"  VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f}GB"
            )

    Path(output_dir).mkdir(parents=True, exist_ok=True)
    cache_dir = f"{output_dir}/tokenizer_cache"

    tok = GPT2Tokenizer.from_pretrained("gpt2", cache_dir=cache_dir)
    tok.pad_token = tok.eos_token

    config = GPT2Config(
        vocab_size=tok.vocab_size,
        n_positions=512,
        n_embd=n_embd,
        n_layer=n_layer,
        n_head=n_head,
        n_inner=n_embd * 4,
        resid_pdrop=0.1,
        attn_pdrop=0.1,
        embd_pdrop=0.1,
    )
    model = GPT2LMHeadModel(config).to(DEVICE)
    n_params = sum(p.numel() for p in model.parameters())
    if verbose:
        print(f"  Model: {n_params / 1e6:.1f}M parameters, {n_layer}L \u00d7 d{n_embd}")

    class TextDataset(Dataset):
        def __init__(self):
            self.chunks = []
            tokens = tok.encode(corpus_text, add_special_tokens=True)
            for i in range(0, len(tokens) - 512, 256):
                chunk = tokens[i : i + 512]
                if len(chunk) == 512:
                    import torch as _t

                    self.chunks.append(_t.tensor(chunk, dtype=_t.long))
            if verbose:
                print(f"  Dataset: {len(self.chunks)} chunks of 512 tokens")

        def __len__(self):
            return len(self.chunks)

        def __getitem__(self, i):
            return self.chunks[i]

    dataset = TextDataset()
    if len(dataset) == 0:
        raise ValueError("Corpus too small. Need at least 512 tokens.")

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=(DEVICE == "cuda"),
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.1)
    total_steps = len(loader) * epochs
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps)

    if verbose:
        print(f"\n  Training {epochs} epochs ({total_steps} steps)...")
        print("  Progress: ", end="", flush=True)

    step = 0
    for epoch in range(epochs):
        for batch in loader:
            batch = batch.to(DEVICE)
            loss = model(input_ids=batch, labels=batch.clone()).loss
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            step += 1
            if verbose and step % max(1, total_steps // 20) == 0:
                print(".", end="", flush=True)

    if verbose:
        print(f" done (final loss: {loss.item():.3f})")

    # Save
    model.save_pretrained(output_dir)
    tok.save_pretrained(output_dir)
    with open(f"{output_dir}/config_hsg.json", "w") as f:
        json.dump({"n_layers": n_layer, "d_model": n_embd}, f)

    return model, tok, n_layer, DEVICE


def load_trained_model(model_dir):
    """Load a previously trained model."""
    import torch
    from transformers import GPT2LMHeadModel, GPT2Tokenizer

    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    tok = GPT2Tokenizer.from_pretrained(model_dir)
    model = GPT2LMHeadModel.from_pretrained(model_dir, output_hidden_states=True).to(
        DEVICE
    )
    model.eval()

    cfg_path = f"{model_dir}/config_hsg.json"
    if os.path.exists(cfg_path):
        with open(cfg_path) as f:
            cfg = json.load(f)
        n_layers = cfg["n_layers"]
    else:
        n_layers = model.config.n_layer

    print(f"  Loaded model from {model_dir} ({n_layers}L)")
    return model, tok, n_layers, DEVICE


# ── Measurement ───────────────────────────────────────────────────────────────


def build_probe_sentences(target_word, compare_words):
    """
    Build probe sentences for target word with each comparison word.
    Uses the primary probe template.
    """
    sentences = {}
    for obj in compare_words:
        # Use first template as primary
        sentences[obj] = f"I {target_word} {obj}."
    return sentences


def measure_geometry(
    model, tok, target_word, compare_words, n_layers, device, verbose=True
):
    """
    Measure pairwise L2 distances between comparison word vectors
    at the peak layer (n_layers - 2).
    """
    model.eval()
    peak_layer = n_layers - 1  # index into hidden_states (0=embedding, 1..n=layers)
    # n_layers - 2 from output = index n_layers - 1 in hidden_states tuple

    probe_sentences = build_probe_sentences(target_word, compare_words)

    if verbose:
        print(
            f"\n  Extracting vectors at layer {n_layers - 2}/{n_layers} (n-2 invariant)..."
        )

    vecs = {}
    for label, sentence in probe_sentences.items():
        vec = final_token_vector(model, tok, sentence, peak_layer, device)
        vecs[label] = vec
        if verbose:
            print(f'    "{sentence}"  norm={np.linalg.norm(vec):.2f}')

    # Pairwise distances
    distances = {}
    cosines = {}
    for i, w1 in enumerate(compare_words):
        for w2 in compare_words[i + 1 :]:
            key = f"{w1}/{w2}"
            distances[key] = l2_distance(vecs[w1], vecs[w2])
            cosines[key] = cosine_similarity(vecs[w1], vecs[w2])

    return vecs, distances, cosines


def rank_by_similarity(target_word, compare_words, vecs):
    """
    For each comparison word, compute its distance from all others.
    Rank comparison words by how 'central' they are to the target word's context.
    """
    # Find which comparison word is closest to which
    rankings = {}
    for w1 in compare_words:
        dists = {}
        for w2 in compare_words:
            if w1 != w2:
                dists[w2] = l2_distance(vecs[w1], vecs[w2])
        rankings[w1] = sorted(dists.items(), key=lambda x: x[1])
    return rankings


# ── Output formatting ─────────────────────────────────────────────────────────


def print_results(
    target_word,
    compare_words,
    distances,
    cosines,
    baseline_distances=None,
    corpus_name="corpus",
):
    """Print a clean, readable results table."""

    print(f"\n{'=' * 65}")
    print(f"HSG CORPUS AUDIT  \u2014  target word: '{target_word}'")
    print(f"Corpus: {corpus_name}")
    print(f"{'=' * 65}")

    # Sort pairs by distance
    sorted_pairs = sorted(distances.items(), key=lambda x: x[1])

    print("\n  Pairwise distances (smaller = more similar relational context):\n")
    print(f"  {'Pair':<25} {'L2 distance':>12} {'Cosine sim':>12}", end="")
    if baseline_distances:
        print(f"  {'vs baseline':>12}", end="")
    print()
    print(f"  {'-' * 25} {'-' * 12} {'-' * 12}", end="")
    if baseline_distances:
        print(f"  {'-' * 12}", end="")
    print()

    for pair, dist in sorted_pairs:
        cos = cosines[pair]
        line = f"  {pair:<25} {dist:>12.3f} {cos:>12.3f}"
        if baseline_distances and pair in baseline_distances:
            delta = dist - baseline_distances[pair]
            sign = "+" if delta > 0 else ""
            line += f"  {sign}{delta:>11.3f}"
        print(line)

    # Interpretation
    print("\n  Ordering (closest \u2192 furthest from each other):")
    # For each word, find its nearest neighbor among comparison words
    words_by_centrality = []
    for pair, dist in sorted_pairs:
        words_by_centrality.append((pair, dist))

    # Show the ordering triangle
    if len(compare_words) >= 3:
        min_pair = sorted_pairs[0][0]
        max_pair = sorted_pairs[-1][0]
        print(f"  Most similar:  {min_pair}  (dist={sorted_pairs[0][1]:.3f})")
        print(f"  Most distant:  {max_pair}  (dist={sorted_pairs[-1][1]:.3f})")

    print(f"\n{'=' * 65}")


def save_results(results_dict, output_path):
    """Save results to JSON."""
    with open(output_path, "w") as f:
        json.dump(results_dict, f, indent=2, default=float)
    print(f"\n  Results saved: {output_path}")


def compare_two_corpora(results1, results2, target_word, compare_words):
    """Print comparison between two corpus audit results."""
    print(f"\n{'=' * 65}")
    print(f"CORPUS COMPARISON  \u2014  target word: '{target_word}'")
    print(f"{'=' * 65}")
    print(f"\n  {'Pair':<25} {'Corpus 1':>12} {'Corpus 2':>12} {'Delta':>10}")
    print(f"  {'-' * 25} {'-' * 12} {'-' * 12} {'-' * 10}")

    d1 = results1["distances"]
    d2 = results2["distances"]
    all_pairs = sorted(set(list(d1.keys()) + list(d2.keys())))

    for pair in sorted(all_pairs, key=lambda p: d1.get(p, 0)):
        v1 = d1.get(pair, 0)
        v2 = d2.get(pair, 0)
        delta = v2 - v1
        sign = "+" if delta > 0 else ""
        print(f"  {pair:<25} {v1:>12.3f} {v2:>12.3f} {sign}{delta:>9.3f}")

    # Check if orderings differ
    order1 = sorted(d1.items(), key=lambda x: x[1])
    order2 = sorted(d2.items(), key=lambda x: x[1])
    if [p for p, _ in order1] != [p for p, _ in order2]:
        print("\n  ORDERING DIFFERS between corpora.")
        print(f"  Corpus 1 ordering: {' < '.join(p for p, _ in order1)}")
        print(f"  Corpus 2 ordering: {' < '.join(p for p, _ in order2)}")
    else:
        print("\n  Ordering is CONSISTENT across both corpora.")
        print(f"  Stable ordering: {' < '.join(p for p, _ in order1)}")


def make_figure(results_list, output_path):
    """Generate comparison figure."""
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(
            1, len(results_list), figsize=(7 * len(results_list), 6)
        )
        if len(results_list) == 1:
            axes = [axes]

        colors = ["#2E86AB", "#E84855", "#F4A261", "#6A4C93", "#48CAE4", "#90E0EF"]

        for ax, results in zip(axes, results_list):
            pairs = list(results["distances"].keys())
            vals = [results["distances"][p] for p in pairs]
            sorted_idx = sorted(range(len(pairs)), key=lambda i: vals[i])
            pairs_s = [pairs[i] for i in sorted_idx]
            vals_s = [vals[i] for i in sorted_idx]
            bar_colors = [colors[i % len(colors)] for i in range(len(pairs_s))]
            ax.barh(pairs_s, vals_s, color=bar_colors, edgecolor="white", alpha=0.85)
            ax.set_xlabel("L2 distance (smaller = more similar)")
            ax.set_title(
                f"'{results['target_word']}' relational geometry\n{results['corpus_name']}",
                fontweight="bold",
            )
            ax.grid(alpha=0.3, axis="x")

        fig.suptitle(
            "HSG Corpus Auditor\nSingleton, M.",
            fontsize=11,
            fontweight="bold",
        )
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        print(f"  Figure: {output_path}")
    except Exception as e:
        print(f"  Figure generation failed: {e}")


# ── Main ──────────────────────────────────────────────────────────────────────


def audit(
    corpus_path,
    target_word,
    compare_words,
    model_dir=None,
    epochs=5,
    batch_size=8,
    output_dir=None,
    verbose=True,
):
    """
    Full audit pipeline.
    Returns results dict.
    """
    corpus_name = Path(corpus_path).stem if corpus_path else "unknown"

    if output_dir is None:
        output_dir = f"hsg_audit_{corpus_name}_{target_word}"
    Path(output_dir).mkdir(exist_ok=True)

    # Load corpus
    if verbose:
        print(f"\nLoading corpus: {corpus_path}")
    corpus_text = load_corpus(corpus_path)
    check_target_word(corpus_text, target_word)

    # Train or load model
    trained_model_dir = model_dir or f"{output_dir}/model"
    if model_dir and os.path.exists(f"{model_dir}/config.json"):
        if verbose:
            print("\nLoading pretrained model...")
        model, tok, n_layers, device = load_trained_model(model_dir)
    else:
        if verbose:
            print("\nTraining model on corpus...")
        model, tok, n_layers, device = train_model(
            corpus_text,
            trained_model_dir,
            epochs=epochs,
            batch_size=batch_size,
            verbose=verbose,
        )
        model.eval()

    # Measure geometry
    if verbose:
        print("\nMeasuring relational geometry...")
    vecs, distances, cosines = measure_geometry(
        model, tok, target_word, compare_words, n_layers, device, verbose
    )

    # Print results
    print_results(
        target_word, compare_words, distances, cosines, corpus_name=corpus_name
    )

    # Save
    results = {
        "corpus_path": corpus_path,
        "corpus_name": corpus_name,
        "target_word": target_word,
        "compare_words": compare_words,
        "distances": distances,
        "cosines": cosines,
        "n_layers": n_layers,
        "model_dir": trained_model_dir,
    }
    save_results(results, f"{output_dir}/results.json")
    make_figure([results], f"{output_dir}/geometry.png")

    return results


# ── CLI ───────────────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(
        description="HSG Corpus Auditor — measure relational geometry of a word in any corpus",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python hsg_audit.py corpus.txt --word love --compare wife,dog,pizza,enemy,god
  python hsg_audit.py corpus.txt --word leader --compare hero,tyrant,servant,father
  python hsg_audit.py corp1.txt corp2.txt --word care --compare patient,profit,duty

  # Skip training (use saved model):
  python hsg_audit.py corpus.txt --word love --compare wife,dog --model ./my_model

  # Quick run (less accurate):
  python hsg_audit.py corpus.txt --word love --compare wife,dog,pizza --epochs 2
        """,
    )
    parser.add_argument(
        "corpora", nargs="+", help="One or two corpus files (.txt or .jsonl)"
    )
    parser.add_argument(
        "--word", required=True, help="Target word to audit (e.g. love, care, leader)"
    )
    parser.add_argument(
        "--compare",
        required=True,
        help="Comma-separated comparison words (e.g. wife,dog,pizza)",
    )
    parser.add_argument(
        "--model", default=None, help="Path to pretrained HSG model (skip training)"
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=5,
        help="Training epochs (default 5; use 2 for quick test)",
    )
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--output", default=None, help="Output directory")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    compare_words = [w.strip() for w in args.compare.split(",") if w.strip()]
    verbose = not args.quiet

    if verbose:
        print("\nHSG Corpus Auditor  \u2014  Mike Singleton")
        print(f"Target word: '{args.word}'")
        print(f"Comparison:  {compare_words}")

    results_list = []
    for corpus_path in args.corpora[:2]:  # max 2 corpora
        results = audit(
            corpus_path=corpus_path,
            target_word=args.word,
            compare_words=compare_words,
            model_dir=args.model,
            epochs=args.epochs,
            batch_size=args.batch_size,
            output_dir=args.output,
            verbose=verbose,
        )
        results_list.append(results)

    if len(results_list) == 2:
        compare_two_corpora(results_list[0], results_list[1], args.word, compare_words)
        if args.output:
            make_figure(results_list, f"{args.output or 'hsg_audit'}/comparison.png")

    if verbose:
        print("\nDone.")


if __name__ == "__main__":
    main()
