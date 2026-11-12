#!/usr/bin/env python3
"""
Cross-model see-residual fingerprint analysis.
Subtracts the neutral 'see' baseline from each verb's fingerprint,
leaving only the predicate-specific affordance structure.
"""

import json
from pathlib import Path

RESULTS_DIR = Path(__file__).resolve().parents[1] / "results"

MODELS = {
    "Qwen2-7B": RESULTS_DIR / "fingerprint_qwen" / "fingerprint_results.json",
    "Mistral-7B": RESULTS_DIR / "fingerprint_mistral" / "fingerprint_results.json",
    "Gemma-7B": RESULTS_DIR / "fingerprint_gemma" / "fingerprint_results.json",
}

VERBS = ["love", "care", "fear", "hate", "eat", "own", "need", "remember"]


def load(path):
    with open(path) as f:
        return json.load(f)


def mean_baseline(fingerprints):
    """Compute mean fingerprint across all verbs as neutral baseline."""
    all_verbs = list(fingerprints.values())
    pairs = list(all_verbs[0].keys())
    return {k: sum(v[k] for v in all_verbs) / len(all_verbs) for k in pairs}


def residual(verb_fp, baseline_fp):
    return {k: verb_fp[k] - baseline_fp[k] for k in verb_fp}


def top(fp, n=2):
    s = sorted(fp.items(), key=lambda x: x[1])
    return s[:n], s[-n:]


print("=" * 80)
print("MEAN-RESIDUAL FINGERPRINTS — CROSS-MODEL")
print("Subtracts mean-of-all-verbs baseline to isolate predicate-specific structure")
print("=" * 80)

for model_name, path in MODELS.items():
    data = load(path)
    fps = data["fingerprints"]
    baseline = mean_baseline(fps)

    print(f"\n{'─' * 80}")
    print(f"MODEL: {model_name}")
    print(f"{'─' * 80}")
    print(f"  {'Verb':<12} {'Pulls (residual)':<35} {'Pushes (residual)':<35}")
    print(f"  {'─' * 75}")

    for verb in VERBS:
        if verb not in fps:
            continue
        res = residual(fps[verb], baseline)
        pulls, pushes = top(res, n=2)
        pull_str = ", ".join(f"{k}({v:+.3f})" for k, v in pulls)
        push_str = ", ".join(f"{k}({v:+.3f})" for k, v in pushes)
        print(f"  {verb:<12} {pull_str:<35} {push_str:<35}")

print("\n" + "=" * 80)
print("CROSS-MODEL AGREEMENT — love/care/fear residual pulls")
print("=" * 80)

all_data = {name: load(path) for name, path in MODELS.items()}

for verb in ["love", "care", "fear"]:
    print(f"\n  {verb.upper()}")
    for model_name, data in all_data.items():
        fps = data["fingerprints"]
        baseline = mean_baseline(fps)
        if verb not in fps:
            continue
        res = residual(fps[verb], baseline)
        pulls, pushes = top(res, n=1)
        print(
            f"    {model_name:<14} pulls: {pulls[0][0]:<30} ({pulls[0][1]:+.3f})  pushes: {pushes[0][0]:<30} ({pushes[0][1]:+.3f})"
        )
