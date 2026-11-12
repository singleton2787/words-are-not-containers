#!/usr/bin/env python3
"""
Prefill-Only Activation Patching with Next-Token Logit Readout
==============================================================

Purpose:
    Move from behavioral prediction toward causal evidence.

Question:
    If we patch the final prompt hidden state toward another affordance field,
    do next-token probabilities shift toward that field?

Why this test:
    Full generation is noisy.
    Naive steering during every generation step did not work cleanly.
    This test only patches the prompt prefill state and reads the immediate
    next-token distribution.

Core idea:
    Baseline:
        I love my enemy because

    Expected baseline affordance:
        moral

    Patch from:
        I love pizza because

    Target affordance:
        hedonic

    Readout:
        Do hedonic next-token logits rise?
        Do moral next-token logits fall?

This is not final proof of causality.
But directional logit movement after activation patching is stronger evidence
than passive correlation.
"""

import json
import argparse
import random
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


# =============================================================================
# Config / data
# =============================================================================

AFFORDANCE_TOKEN_SETS = {
    "hedonic": [
        " delicious",
        " tasty",
        " food",
        " pizza",
        " flavor",
        " taste",
        " comforting",
        " enjoyable",
        " satisfying",
        " sweet",
        " hungry",
        " eating",
        " meal",
        " favorite",
        " fun",
    ],
    "companion": [
        " loyal",
        " companion",
        " friend",
        " comforting",
        " gentle",
        " trusting",
        " playful",
        " affectionate",
        " unconditional",
        " there",
        " happy",
        " joy",
        " bond",
    ],
    "covenantal": [
        " committed",
        " commitment",
        " partner",
        " marriage",
        " family",
        " together",
        " faithful",
        " support",
        " life",
        " years",
        " trust",
        " shared",
        " devotion",
        " home",
    ],
    "moral": [
        " forgive",
        " forgiveness",
        " mercy",
        " compassion",
        " God",
        " human",
        " hatred",
        " peace",
        " grace",
        " moral",
        " right",
        " kindness",
        " understand",
        " prayer",
        " enemy",
        " love",
    ],
}

PATCH_CASES = [
    {
        "name": "enemy_moral_to_hedonic",
        "base_prompt": "I love my enemy because",
        "source_prompt": "I love pizza because",
        "base_class": "moral",
        "target_class": "hedonic",
    },
    {
        "name": "pizza_hedonic_to_moral",
        "base_prompt": "I love pizza because",
        "source_prompt": "I love my enemy because",
        "base_class": "hedonic",
        "target_class": "moral",
    },
    {
        "name": "wife_covenantal_to_companion",
        "base_prompt": "I love my wife because",
        "source_prompt": "I love my dog because",
        "base_class": "covenantal",
        "target_class": "companion",
    },
    {
        "name": "dog_companion_to_covenantal",
        "base_prompt": "I love my dog because",
        "source_prompt": "I love my wife because",
        "base_class": "companion",
        "target_class": "covenantal",
    },
]

ALPHAS = [0.0, 0.05, 0.10, 0.20, 0.35, 0.50, 0.75, 1.00]


# =============================================================================
# Helpers
# =============================================================================


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_layers(model):
    """
    Supports common HF causal LM architectures.
    """
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return model.model.layers
    if hasattr(model, "transformer") and hasattr(model.transformer, "h"):
        return model.transformer.h
    if hasattr(model, "gpt_neox") and hasattr(model.gpt_neox, "layers"):
        return model.gpt_neox.layers
    raise ValueError("Could not locate transformer layers for this model architecture.")


def token_ids_for_strings(tok, strings: List[str]) -> Dict[str, List[int]]:
    """
    Convert candidate strings to token IDs.

    We only use single-token candidates for clean next-token logit readout.
    Multi-token strings are skipped.
    """
    out = {}
    for s in strings:
        ids = tok(s, add_special_tokens=False)["input_ids"]
        if len(ids) == 1:
            out[s] = ids
    return out


def build_token_sets(tok) -> Dict[str, Dict[str, int]]:
    token_sets = {}
    for cls, strings in AFFORDANCE_TOKEN_SETS.items():
        token_sets[cls] = token_ids_for_strings(tok, strings)

    print("\nSingle-token readout sets:")
    for cls, mapping in token_sets.items():
        print(f"  {cls:<12} {len(mapping):>2} tokens: {list(mapping.keys())[:8]}")

    return token_sets


def class_logit_score(logits: torch.Tensor, token_map: Dict[str, int]) -> float:
    """
    Mean raw logit over class tokens.
    logits shape: [vocab]
    """
    if not token_map:
        return float("nan")
    ids = [v[0] for v in token_map.values()]
    return float(logits[ids].float().mean().item())


def class_prob_mass(logits: torch.Tensor, token_map: Dict[str, int]) -> float:
    """
    Total probability mass assigned to class tokens.
    """
    if not token_map:
        return float("nan")
    probs = torch.softmax(logits.float(), dim=-1)
    ids = [v[0] for v in token_map.values()]
    return float(probs[ids].sum().item())


@torch.no_grad()
def get_next_logits(model, tok, prompt: str) -> torch.Tensor:
    inputs = tok(prompt, return_tensors="pt").to(model.device)
    out = model(**inputs, use_cache=False)
    return out.logits[0, -1, :].detach().cpu()


@torch.no_grad()
def get_layer_final_activation(model, tok, prompt: str, layer_idx: int) -> torch.Tensor:
    """
    Returns final-token hidden activation at hidden_states[layer_idx].
    hidden_states[0] is embedding output.
    hidden_states[1] is after transformer block 0, etc.
    """
    inputs = tok(prompt, return_tensors="pt").to(model.device)
    out = model(**inputs, output_hidden_states=True, use_cache=False)
    return out.hidden_states[layer_idx][0, -1, :].detach()


def make_prefill_patch_hook(source_vec: torch.Tensor, alpha: float):
    """
    Patch final token activation only during prefill.

    hidden[:, -1, :] = (1-alpha) * hidden[:, -1, :] + alpha * source_vec

    This does not run generation. It only affects one forward pass for next-token logits.
    """

    def hook(module, inputs, output):
        if isinstance(output, tuple):
            hidden = output[0]
            rest = output[1:]
        else:
            hidden = output
            rest = None

        patched = hidden.clone()
        src = source_vec.to(device=patched.device, dtype=patched.dtype)
        patched[:, -1, :] = (1.0 - alpha) * patched[:, -1, :] + alpha * src

        if rest is not None:
            return (patched,) + rest
        return patched

    return hook


@torch.no_grad()
def get_patched_next_logits(
    model,
    tok,
    base_prompt: str,
    source_vec: torch.Tensor,
    layer_idx: int,
    alpha: float,
) -> torch.Tensor:
    """
    One forward pass with a hook at the selected layer.
    Reads next-token logits after patching prompt activation.
    """
    layers = get_layers(model)

    # hidden_states[layer_idx] corresponds to output after module layer_idx-1.
    module_idx = layer_idx - 1
    if module_idx < 0:
        raise ValueError("Cannot patch embedding output with this module hook.")

    hook = make_prefill_patch_hook(source_vec, alpha)
    handle = layers[module_idx].register_forward_hook(hook)

    try:
        inputs = tok(base_prompt, return_tensors="pt").to(model.device)
        out = model(**inputs, use_cache=False)
        logits = out.logits[0, -1, :].detach().cpu()
    finally:
        handle.remove()

    return logits


def summarize_logits(
    logits: torch.Tensor,
    token_sets: Dict[str, Dict[str, int]],
) -> Dict[str, Dict[str, float]]:
    summary = {}
    for cls, token_map in token_sets.items():
        summary[cls] = {
            "mean_logit": class_logit_score(logits, token_map),
            "prob_mass": class_prob_mass(logits, token_map),
        }
    return summary


def top_tokens(tok, logits: torch.Tensor, k: int = 12) -> List[Tuple[str, float]]:
    vals, ids = torch.topk(logits.float(), k)
    out = []
    for v, i in zip(vals.tolist(), ids.tolist()):
        out.append((tok.decode([i]), float(v)))
    return out


def delta(a: float, b: float) -> float:
    return float(a - b)


# =============================================================================
# Main test
# =============================================================================


def run_case(model, tok, token_sets, layer_idx, case):
    base_prompt = case["base_prompt"]
    source_prompt = case["source_prompt"]
    base_class = case["base_class"]
    target_class = case["target_class"]

    print("\n" + "=" * 90)
    print(f"CASE: {case['name']}")
    print(f"Base:   {base_prompt}   [{base_class}]")
    print(f"Source: {source_prompt}   [{target_class}]")
    print("=" * 90)

    source_vec = get_layer_final_activation(model, tok, source_prompt, layer_idx)

    baseline_logits = get_next_logits(model, tok, base_prompt)
    baseline_summary = summarize_logits(baseline_logits, token_sets)

    print("\nBaseline top tokens:")
    for t, v in top_tokens(tok, baseline_logits, 12):
        print(f"  {repr(t):<16} {v:.3f}")

    base_target_logit = baseline_summary[target_class]["mean_logit"]
    base_source_logit = baseline_summary[base_class]["mean_logit"]
    base_target_prob = baseline_summary[target_class]["prob_mass"]
    base_source_prob = baseline_summary[base_class]["prob_mass"]

    rows = []

    print("\nAlpha sweep:")
    print(
        f"{'alpha':>6} "
        f"{'target_logit':>13} {'base_logit':>12} {'Δtarget':>10} {'Δbase':>10} "
        f"{'target_prob':>12} {'base_prob':>10}"
    )
    print("-" * 82)

    for alpha in ALPHAS:
        if alpha == 0.0:
            logits = baseline_logits
        else:
            logits = get_patched_next_logits(
                model=model,
                tok=tok,
                base_prompt=base_prompt,
                source_vec=source_vec,
                layer_idx=layer_idx,
                alpha=alpha,
            )

        summary = summarize_logits(logits, token_sets)

        target_logit = summary[target_class]["mean_logit"]
        source_logit = summary[base_class]["mean_logit"]
        target_prob = summary[target_class]["prob_mass"]
        source_prob = summary[base_class]["prob_mass"]

        row = {
            "case": case["name"],
            "base_prompt": base_prompt,
            "source_prompt": source_prompt,
            "base_class": base_class,
            "target_class": target_class,
            "alpha": alpha,
            "target_mean_logit": target_logit,
            "base_mean_logit": source_logit,
            "target_prob_mass": target_prob,
            "base_prob_mass": source_prob,
            "delta_target_logit": delta(target_logit, base_target_logit),
            "delta_base_logit": delta(source_logit, base_source_logit),
            "delta_target_prob": delta(target_prob, base_target_prob),
            "delta_base_prob": delta(source_prob, base_source_prob),
            "class_summary": summary,
            "top_tokens": top_tokens(tok, logits, 10),
        }
        rows.append(row)

        print(
            f"{alpha:6.2f} "
            f"{target_logit:13.3f} {source_logit:12.3f} "
            f"{row['delta_target_logit']:10.3f} {row['delta_base_logit']:10.3f} "
            f"{target_prob:12.6f} {source_prob:10.6f}"
        )

    # Simple directional verdict.
    target_deltas = [r["delta_target_logit"] for r in rows if r["alpha"] > 0]
    base_deltas = [r["delta_base_logit"] for r in rows if r["alpha"] > 0]

    target_up = np.mean([d > 0 for d in target_deltas])
    base_down = np.mean([d < 0 for d in base_deltas])

    verdict = "negative"
    if target_up >= 0.75 and base_down >= 0.50:
        verdict = "strong_directional_shift"
    elif target_up >= 0.60:
        verdict = "partial_target_shift"

    print(f"\nVerdict: {verdict}")
    print(f"  target logit increased in {target_up:.0%} of patched alphas")
    print(f"  base logit decreased in {base_down:.0%} of patched alphas")

    return {
        "case": case,
        "baseline_summary": baseline_summary,
        "rows": rows,
        "verdict": verdict,
        "target_up_rate": float(target_up),
        "base_down_rate": float(base_down),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen2-7B")
    parser.add_argument(
        "--output",
        default=str(
            Path(__file__).resolve().parents[1] / "results" / "prefill_patch_results"
        ),
    )
    parser.add_argument("--layer_fraction", type=float, default=0.35)
    parser.add_argument("--cases", nargs="*", default=None)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()

    set_seed(42)

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 90)
    print("PREFILL-ONLY ACTIVATION PATCHING / NEXT-TOKEN LOGIT TEST")
    print("=" * 90)
    print(f"Model: {args.model}")

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

    print(f"Device: {device}")
    print(f"Layers: {n_layers}")
    print(f"Using hidden_states[{layer_idx}] / module layer {layer_idx - 1}")

    token_sets = build_token_sets(tok)

    cases = PATCH_CASES
    if args.cases:
        wanted = set(args.cases)
        cases = [c for c in cases if c["name"] in wanted]
    if args.smoke:
        cases = cases[:1]

    all_results = []

    for case in cases:
        result = run_case(model, tok, token_sets, layer_idx, case)
        all_results.append(result)

    summary = {
        "model": args.model,
        "n_layers": n_layers,
        "layer_idx": layer_idx,
        "layer_fraction": args.layer_fraction,
        "alphas": ALPHAS,
        "results": all_results,
    }

    out_path = out_dir / "prefill_patch_logit_results.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    # Also write a compact CSV.
    csv_path = out_dir / "prefill_patch_logit_results.csv"
    with open(csv_path, "w", encoding="utf-8") as f:
        f.write(
            "case,base_class,target_class,alpha,"
            "target_mean_logit,base_mean_logit,delta_target_logit,delta_base_logit,"
            "target_prob_mass,base_prob_mass,verdict\n"
        )
        for res in all_results:
            verdict = res["verdict"]
            for r in res["rows"]:
                f.write(
                    f"{r['case']},"
                    f"{r['base_class']},"
                    f"{r['target_class']},"
                    f"{r['alpha']},"
                    f"{r['target_mean_logit']:.6f},"
                    f"{r['base_mean_logit']:.6f},"
                    f"{r['delta_target_logit']:.6f},"
                    f"{r['delta_base_logit']:.6f},"
                    f"{r['target_prob_mass']:.10f},"
                    f"{r['base_prob_mass']:.10f},"
                    f"{verdict}\n"
                )

    print("\nSaved:")
    print(f"  {out_path}")
    print(f"  {csv_path}")

    print("\n" + "=" * 90)
    print("INTERPRETATION")
    print("=" * 90)
    print("""
Win condition:
    Target-class next-token logits rise as alpha increases.

Stronger win:
    Target-class logits rise while base-class logits fall.

Clean causal readout:
    Patch changes immediate next-token distribution without requiring sampled generation.

Negative result:
    Hidden geometry is behaviorally predictive but not causally localized to this
    single-layer final-token patch.

Next steps if negative:
    - layer sweep
    - patch multiple layers
    - patch attention heads or MLP outputs separately
    - patch prefill only but generate greedily afterward
    - human-label full generations after successful logit movement
""")


if __name__ == "__main__":
    main()
