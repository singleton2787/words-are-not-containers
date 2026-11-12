#!/usr/bin/env python3
"""
Affordance Steering / Activation Patching Test
==============================================

Purpose:
    Move from "geometry predicts behavior" to preliminary causal evidence.

Question:
    If we deliberately move a hidden state toward a different affordance field,
    does the model's generated continuation shift toward that field?

Plain English:
    Baseline:
        I love my enemy because...
        Expected: moral / forgiveness / mercy / restraint

    Steering:
        Push hidden state toward hedonic love.
        Prediction: output becomes less moral and more pleasure/preference/comfort flavored.

Important:
    This is not final proof of causality.
    But if increasing steering strength produces monotonic register shift,
    it is stronger than passive correlation.

Run:
    python3 affordance_steering_test.py --model Qwen/Qwen2-7B

Optional:
    python3 affordance_steering_test.py --model Qwen/Qwen2-0.5B --smoke
"""

import json
import argparse
import random
import re
from pathlib import Path
from typing import Dict, List, Tuple, Any

import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


# =============================================================================
# Data
# =============================================================================

AFFORDANCE_CLASSES = ["hedonic", "companion", "covenantal", "moral"]

ANCHOR_OBJECTS = [
    ("pizza", "hedonic"),
    ("chocolate", "hedonic"),
    ("coffee", "hedonic"),
    ("cake", "hedonic"),
    ("ice cream", "hedonic"),
    ("my dog", "companion"),
    ("my cat", "companion"),
    ("my horse", "companion"),
    ("my friend", "companion"),
    ("my child", "companion"),
    ("my wife", "covenantal"),
    ("my husband", "covenantal"),
    ("my family", "covenantal"),
    ("my country", "covenantal"),
    ("God", "covenantal"),
    ("my enemy", "moral"),
    ("my rival", "moral"),
    ("a stranger", "moral"),
    ("my neighbor", "moral"),
    ("the other", "moral"),
]

CLASS_KEYWORDS = {
    "hedonic": [
        "taste",
        "tastes",
        "tasty",
        "delicious",
        "sweet",
        "flavor",
        "flavour",
        "eat",
        "eating",
        "drink",
        "drinking",
        "crave",
        "craving",
        "meal",
        "food",
        "dessert",
        "caffeine",
        "comfort",
        "pleasure",
        "enjoy",
        "favorite",
        "satisfying",
        "hungry",
        "appetite",
        "good",
        "warm",
        "recipe",
        "topping",
        "snack",
    ],
    "companion": [
        "companion",
        "companionship",
        "loyal",
        "loyalty",
        "play",
        "walk",
        "walks",
        "friend",
        "comfort",
        "cuddle",
        "care",
        "protect",
        "trust",
        "joy",
        "always there",
        "unconditional",
        "animal",
        "pet",
        "bond",
        "affection",
        "gentle",
        "home",
        "beside",
        "presence",
    ],
    "covenantal": [
        "commitment",
        "committed",
        "marriage",
        "married",
        "vows",
        "faithful",
        "fidelity",
        "partner",
        "life",
        "shared",
        "together",
        "sacrifice",
        "belong",
        "home",
        "family",
        "duty",
        "devotion",
        "honor",
        "support",
        "through everything",
        "years",
        "build",
        "trust",
        "forever",
        "our",
        "we",
        "us",
        "country",
        "freedom",
        "serve",
        "God",
        "Lord",
        "faith",
    ],
    "moral": [
        "forgive",
        "forgiveness",
        "mercy",
        "compassion",
        "hatred",
        "hate",
        "revenge",
        "vengeance",
        "enemy",
        "enemies",
        "human",
        "humanity",
        "command",
        "Christ",
        "Jesus",
        "neighbor",
        "moral",
        "grace",
        "peace",
        "refuse",
        "harm",
        "understand",
        "restraint",
        "pray",
        "prayer",
        "right",
        "principle",
        "virtue",
        "spiritual",
        "reconcile",
        "wounded",
        "child of God",
        "God loves",
    ],
}

TEST_CASES = [
    {
        "name": "enemy_moral_to_hedonic",
        "prompt": "I love my enemy because",
        "source_class": "moral",
        "target_class": "hedonic",
        "expected_baseline": "moral",
        "description": "Can moral love be pushed toward hedonic/preference register?",
    },
    {
        "name": "pizza_hedonic_to_moral",
        "prompt": "I love pizza because",
        "source_class": "hedonic",
        "target_class": "moral",
        "expected_baseline": "hedonic",
        "description": "Can hedonic love be pushed toward moral/obligation register?",
    },
    {
        "name": "wife_covenantal_to_hedonic",
        "prompt": "I love my wife because",
        "source_class": "covenantal",
        "target_class": "hedonic",
        "expected_baseline": "covenantal",
        "description": "Can covenantal love be pushed toward hedonic/preference register?",
    },
    {
        "name": "dog_companion_to_moral",
        "prompt": "I love my dog because",
        "source_class": "companion",
        "target_class": "moral",
        "expected_baseline": "companion",
        "description": "Can companionate love be pushed toward moral register?",
    },
]


# =============================================================================
# Utility
# =============================================================================


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def clean_text(s: str) -> str:
    return re.sub(r"\s+", " ", s.strip())


def keyword_scores(text: str) -> Dict[str, int]:
    t = text.lower()
    scores = {}
    for cls, kws in CLASS_KEYWORDS.items():
        score = 0
        for kw in kws:
            if kw.lower() in t:
                score += 1
        scores[cls] = score
    return scores


def predicted_register(text: str) -> str:
    scores = keyword_scores(text)
    max_score = max(scores.values())
    if max_score == 0:
        return "none"

    winners = [k for k, v in scores.items() if v == max_score]
    if len(winners) > 1:
        return "tie"
    return winners[0]


def get_model_layers(model):
    """
    Supports common HF causal LM architectures:
    - Qwen/Llama/Mistral/Gemma style: model.model.layers
    - GPT2 style: model.transformer.h
    """
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return model.model.layers
    if hasattr(model, "transformer") and hasattr(model.transformer, "h"):
        return model.transformer.h
    if hasattr(model, "gpt_neox") and hasattr(model.gpt_neox, "layers"):
        return model.gpt_neox.layers
    raise ValueError("Could not locate transformer layers for this model architecture.")


@torch.no_grad()
def get_vector(model, tok, sentence: str, layer_idx: int) -> np.ndarray:
    inputs = tok(sentence, return_tensors="pt", truncation=True, max_length=256).to(
        model.device
    )
    out = model(**inputs, output_hidden_states=True, use_cache=False)
    return out.hidden_states[layer_idx][0, -1, :].float().cpu().numpy()


def l2(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm(a - b))


def build_centroids(model, tok, layer_idx: int) -> Dict[str, np.ndarray]:
    by_class = {cls: [] for cls in AFFORDANCE_CLASSES}

    for obj, cls in ANCHOR_OBJECTS:
        prompt = f"I love {obj} because"
        vec = get_vector(model, tok, prompt, layer_idx)
        by_class[cls].append(vec)

    return {cls: np.mean(vecs, axis=0) for cls, vecs in by_class.items()}


def nearest_centroid(
    vec: np.ndarray, centroids: Dict[str, np.ndarray]
) -> Tuple[str, Dict[str, float]]:
    dists = {cls: l2(vec, c) for cls, c in centroids.items()}
    return min(dists, key=dists.get), dists


def margin_to_expected(
    vec: np.ndarray, expected_class: str, centroids: Dict[str, np.ndarray]
) -> float:
    _, dists = nearest_centroid(vec, centroids)
    expected_d = dists[expected_class]
    nearest_wrong = min(v for k, v in dists.items() if k != expected_class)
    return nearest_wrong - expected_d


# =============================================================================
# Hooked generation
# =============================================================================


def make_steering_hook(
    steering_vec: torch.Tensor,
    alpha: float,
    mode: str,
    target_centroid: torch.Tensor = None,
):
    """
    mode = "add":
        hidden[:, -1, :] += alpha * steering_vec

    mode = "patch":
        hidden[:, -1, :] = (1-alpha) * hidden[:, -1, :] + alpha * target_centroid

    Notes:
        - This modifies only the last token representation at the target layer.
        - During generation with KV cache, each new token is also seq_len=1,
          so the hook applies to generation steps too.
    """

    def hook(module, inputs, output):
        if isinstance(output, tuple):
            hidden = output[0]
            rest = output[1:]
        else:
            hidden = output
            rest = None

        hidden2 = hidden.clone()

        if mode == "add":
            vec = steering_vec.to(device=hidden2.device, dtype=hidden2.dtype)
            hidden2[:, -1, :] = hidden2[:, -1, :] + alpha * vec

        elif mode == "patch":
            if target_centroid is None:
                raise ValueError("target_centroid required for patch mode.")
            target = target_centroid.to(device=hidden2.device, dtype=hidden2.dtype)
            hidden2[:, -1, :] = (1.0 - alpha) * hidden2[:, -1, :] + alpha * target

        else:
            raise ValueError(f"Unknown mode: {mode}")

        if rest is not None:
            return (hidden2,) + rest
        return hidden2

    return hook


@torch.no_grad()
def generate_with_optional_hook(
    model,
    tok,
    prompt: str,
    layer_idx: int,
    mode: str,
    alpha: float,
    steering_vec: torch.Tensor = None,
    target_centroid: torch.Tensor = None,
    max_new_tokens: int = 50,
    temperature: float = 0.8,
    top_p: float = 0.95,
) -> str:
    layers = get_model_layers(model)

    handle = None
    if mode != "baseline":
        hook = make_steering_hook(
            steering_vec=steering_vec,
            alpha=alpha,
            mode=mode,
            target_centroid=target_centroid,
        )
        handle = layers[layer_idx - 1].register_forward_hook(hook)

    try:
        inputs = tok(prompt, return_tensors="pt").to(model.device)
        out = model.generate(
            **inputs,
            do_sample=True,
            temperature=temperature,
            top_p=top_p,
            max_new_tokens=max_new_tokens,
            pad_token_id=tok.eos_token_id,
        )
        full = tok.decode(out[0], skip_special_tokens=True)
        continuation = full[len(prompt) :].strip()
        return clean_text(continuation)
    finally:
        if handle is not None:
            handle.remove()


# =============================================================================
# Main experiment
# =============================================================================


def run_case(
    model,
    tok,
    centroids: Dict[str, np.ndarray],
    layer_idx: int,
    case: Dict[str, Any],
    alphas: List[float],
    n_generations: int,
    mode: str,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
) -> Dict[str, Any]:
    prompt = case["prompt"]
    source_class = case["source_class"]
    target_class = case["target_class"]
    expected_baseline = case["expected_baseline"]

    source_centroid = centroids[source_class]
    target_centroid = centroids[target_class]

    direction = target_centroid - source_centroid

    # Normalize direction so alpha is more stable across models.
    # Then scale by average centroid norm / 8. This avoids totally wrecking activations.
    direction_norm = np.linalg.norm(direction)
    avg_norm = float(np.mean([np.linalg.norm(c) for c in centroids.values()]))
    if direction_norm > 1e-8:
        direction_unit = direction / direction_norm
    else:
        direction_unit = direction

    scaled_direction = direction_unit * (avg_norm / 8.0)

    steering_tensor = torch.tensor(scaled_direction, dtype=torch.float32)
    target_tensor = torch.tensor(target_centroid, dtype=torch.float32)

    rows = []

    print("\n" + "=" * 90)
    print(f"CASE: {case['name']}")
    print(case["description"])
    print(f"Prompt: {prompt}")
    print(f"Source -> Target: {source_class} -> {target_class}")
    print("=" * 90)

    for alpha in alphas:
        for gen_idx in range(n_generations):
            if alpha == 0:
                gen_mode = "baseline"
            else:
                gen_mode = mode

            cont = generate_with_optional_hook(
                model=model,
                tok=tok,
                prompt=prompt,
                layer_idx=layer_idx,
                mode=gen_mode,
                alpha=alpha,
                steering_vec=steering_tensor,
                target_centroid=target_tensor,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
            )

            scores = keyword_scores(cont)
            pred = predicted_register(cont)

            target_score = scores[target_class]
            source_score = scores[source_class]
            expected_score = scores[expected_baseline]

            row = {
                "case": case["name"],
                "prompt": prompt,
                "mode": gen_mode,
                "alpha": alpha,
                "generation_index": gen_idx,
                "source_class": source_class,
                "target_class": target_class,
                "expected_baseline": expected_baseline,
                "predicted_register": pred,
                "scores": scores,
                "target_score": target_score,
                "source_score": source_score,
                "expected_score": expected_score,
                "continuation": cont,
            }
            rows.append(row)

        # quick per-alpha summary
        alpha_rows = [r for r in rows if r["alpha"] == alpha]
        target_rate = np.mean(
            [r["predicted_register"] == target_class for r in alpha_rows]
        )
        source_rate = np.mean(
            [r["predicted_register"] == source_class for r in alpha_rows]
        )
        mean_target_score = np.mean([r["target_score"] for r in alpha_rows])
        mean_source_score = np.mean([r["source_score"] for r in alpha_rows])

        print(
            f"alpha={alpha:<5} "
            f"target_rate={target_rate:.2f} "
            f"source_rate={source_rate:.2f} "
            f"mean_target_score={mean_target_score:.2f} "
            f"mean_source_score={mean_source_score:.2f}"
        )

        preview = alpha_rows[0]["continuation"][:150].replace("\n", " ")
        print(f"  preview: {preview}")

    # aggregate
    summary_by_alpha = {}
    for alpha in alphas:
        alpha_rows = [r for r in rows if r["alpha"] == alpha]
        summary_by_alpha[str(alpha)] = {
            "n": len(alpha_rows),
            "target_register_rate": float(
                np.mean([r["predicted_register"] == target_class for r in alpha_rows])
            ),
            "source_register_rate": float(
                np.mean([r["predicted_register"] == source_class for r in alpha_rows])
            ),
            "mean_target_score": float(
                np.mean([r["target_score"] for r in alpha_rows])
            ),
            "mean_source_score": float(
                np.mean([r["source_score"] for r in alpha_rows])
            ),
            "mean_expected_score": float(
                np.mean([r["expected_score"] for r in alpha_rows])
            ),
        }

    return {
        "case": case,
        "mode": mode,
        "rows": rows,
        "summary_by_alpha": summary_by_alpha,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen2-7B")
    parser.add_argument(
        "--output",
        default=str(
            Path(__file__).resolve().parents[1] / "results" / "steering_results"
        ),
    )
    parser.add_argument("--mode", choices=["add", "patch"], default="add")
    parser.add_argument("--n_generations", type=int, default=6)
    parser.add_argument("--max_new_tokens", type=int, default=50)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top_p", type=float, default=0.95)
    parser.add_argument("--layer_fraction", type=float, default=0.35)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()

    if args.smoke:
        args.n_generations = 2

    set_seed(42)

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 90)
    print("AFFORDANCE STEERING / ACTIVATION PATCHING TEST")
    print("=" * 90)
    print(f"Model: {args.model}")
    print(f"Mode:  {args.mode}")

    device = "cuda" if torch.cuda.is_available() else "cpu"

    tok = AutoTokenizer.from_pretrained(args.model)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        device_map="auto" if device == "cuda" else None,
        output_hidden_states=True,
        low_cpu_mem_usage=True,
    )
    model.eval()

    n_layers = model.config.num_hidden_layers
    layer_idx = max(1, int(round(n_layers * args.layer_fraction)))

    print(f"Layers: {n_layers}")
    print(f"Using hidden_states[{layer_idx}] / module layer {layer_idx - 1}")
    print(f"Device: {device}")

    print("\nBuilding affordance centroids...")
    centroids = build_centroids(model, tok, layer_idx)

    for cls, c in centroids.items():
        print(f"  {cls:<12} norm={np.linalg.norm(c):.2f}")

    # Alpha sweep.
    # For add mode, alpha is steering strength.
    # For patch mode, alpha is interpolation amount toward target centroid.
    if args.mode == "add":
        alphas = [0.0, 0.5, 1.0, 2.0, 4.0]
    else:
        alphas = [0.0, 0.05, 0.10, 0.20, 0.35]

    test_cases = TEST_CASES if not args.smoke else TEST_CASES[:2]

    all_results = []

    for case in test_cases:
        result = run_case(
            model=model,
            tok=tok,
            centroids=centroids,
            layer_idx=layer_idx,
            case=case,
            alphas=alphas,
            n_generations=args.n_generations,
            mode=args.mode,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
        )
        all_results.append(result)

    # Save JSONL rows and summary JSON.
    rows_path = out_dir / f"steering_rows_{args.mode}.jsonl"
    with open(rows_path, "w", encoding="utf-8") as f:
        for result in all_results:
            for row in result["rows"]:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")

    summary_path = out_dir / f"steering_summary_{args.mode}.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "model": args.model,
                "mode": args.mode,
                "n_layers": n_layers,
                "layer_idx": layer_idx,
                "layer_fraction": args.layer_fraction,
                "n_generations": args.n_generations,
                "alphas": alphas,
                "results": [
                    {
                        "case": r["case"],
                        "mode": r["mode"],
                        "summary_by_alpha": r["summary_by_alpha"],
                    }
                    for r in all_results
                ],
            },
            f,
            indent=2,
        )

    print("\nSaved:")
    print(f"  {rows_path}")
    print(f"  {summary_path}")

    print("\n" + "=" * 90)
    print("INTERPRETATION")
    print("=" * 90)
    print("""
Win condition:
    As alpha increases, target-register rate or target keyword score increases.

Strong causal evidence:
    Smooth monotonic shift toward target class across alpha levels,
    without destroying fluency.

Weak evidence:
    Some shift, but noisy or only at high alpha.

Failure:
    No shift, or output collapses into garbage.

Important:
    This is an intervention test, stronger than correlation.
    But paper-grade causality still needs careful controls:
      - multiple seeds
      - human annotation
      - fluency checks
      - reverse directions
      - layer sweep
      - patch only prompt prefill vs patch all generation steps
""")


if __name__ == "__main__":
    main()
