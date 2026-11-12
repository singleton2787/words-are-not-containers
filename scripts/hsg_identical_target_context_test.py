from pathlib import Path
# HSG Identical Target Context Test
#
# Purpose:
#   Test whether the SAME target utterance has different residual-stream
#   trajectory geometry depending only on prior context.
#
# Why this matters:
#   This removes the token-length and surface-wording confounds from the previous test.
#
# Interpretation:
#   Positive result:
#       identical words trace different internal trajectories under different contexts.
#   Negative result:
#       current HSG metrics do not detect context-conditioned pragmatic geometry.
#
# Mike Singleton

import os
import json
import random
import warnings

import numpy as np
import torch
import matplotlib.pyplot as plt

from scipy.stats import wilcoxon
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from stats import (
    bh_fdr as stats_bh_fdr,
    empirical_p_value,
    paired_t_stat as stats_paired_t_stat,
)

warnings.filterwarnings("ignore")

MODEL_NAME = os.environ.get("HSG_MODEL", "Qwen/Qwen2-7B")
OUTDIR = str(
    Path(__file__).resolve().parents[1] / "results" / "identical_target_context"
)
N_PERM = int(os.environ.get("HSG_N_PERM", "2000"))
SEED = 42

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

os.makedirs(OUTDIR, exist_ok=True)

print("=" * 78)
print("HSG IDENTICAL TARGET CONTEXT TEST")
print("=" * 78)
print(f"Model: {MODEL_NAME}")
print(f"Output: {OUTDIR}")
print(f"Permutation samples: {N_PERM}")

# ---------------------------------------------------------------------
# Corpus
# ---------------------------------------------------------------------
# Each item has:
#   sincere_context: makes the target literal/sincere
#   loaded_context: makes the SAME target sarcastic/defensive/loaded/etc.
#   target: identical sentence measured in both conditions
#
# We measure only the hidden-state trajectory over the target tokens.

ITEMS = [
    {
        "id": "helpful_outage",
        "type": "sarcasm",
        "sincere_context": "The team found the bug quickly and fixed the outage before noon.",
        "loaded_context": "The team ignored the outage all morning and broke the dashboard twice.",
        "target": "That was helpful.",
    },
    {
        "id": "great_printer",
        "type": "sarcasm",
        "sincere_context": "The printer finally came back online and printed every document.",
        "loaded_context": "The printer jammed again and ruined the only clean copy.",
        "target": "Great.",
    },
    {
        "id": "wonderful_server",
        "type": "sarcasm",
        "sincere_context": "The server recovered automatically and no data was lost.",
        "loaded_context": "The server crashed during the customer demo.",
        "target": "Wonderful.",
    },
    {
        "id": "fine_argument",
        "type": "defensive",
        "sincere_context": "She asked whether he wanted tea, and he smiled.",
        "loaded_context": "She asked why he had been silent for an hour after the argument.",
        "target": "I am fine.",
    },
    {
        "id": "not_angry",
        "type": "defensive",
        "sincere_context": "The misunderstanding had been resolved, and his voice was calm.",
        "loaded_context": "His jaw tightened, and he slammed the drawer shut.",
        "target": "I am not angry.",
    },
    {
        "id": "love_wife",
        "type": "defensive",
        "sincere_context": "He looked at the photograph on his desk and smiled softly.",
        "loaded_context": "After the accusation, everyone at the table went quiet.",
        "target": "I love my wife.",
    },
    {
        "id": "interesting_proposal",
        "type": "faint_praise",
        "sincere_context": "The committee found the proposal original and promising.",
        "loaded_context": "The committee found the proposal vague and poorly supported.",
        "target": "The proposal is interesting.",
    },
    {
        "id": "readable_report",
        "type": "faint_praise",
        "sincere_context": "The report was clear, organized, and easy to follow.",
        "loaded_context": "The report missed the main question and buried the conclusion.",
        "target": "The report is readable.",
    },
    {
        "id": "adequate_work",
        "type": "faint_praise",
        "sincere_context": "The work met every requirement on the checklist.",
        "loaded_context": "The work barely met the minimum and needed several corrections.",
        "target": "The work is adequate.",
    },
    {
        "id": "finally_arrived",
        "type": "implication",
        "sincere_context": "The train was scheduled for noon and arrived at noon.",
        "loaded_context": "The train had been delayed for hours while everyone waited.",
        "target": "It arrived.",
    },
    {
        "id": "called_back",
        "type": "implication",
        "sincere_context": "She promised to call after lunch.",
        "loaded_context": "She had ignored every message for three years.",
        "target": "She called.",
    },
    {
        "id": "apologized",
        "type": "implication",
        "sincere_context": "He made a mistake and apologized immediately.",
        "loaded_context": "He denied everything for months and blamed everyone else.",
        "target": "He apologized.",
    },
    {
        "id": "simple_policy",
        "type": "hedge",
        "sincere_context": "The policy had two steps and no exceptions.",
        "loaded_context": "The policy had footnotes, exceptions, and conflicting instructions.",
        "target": "The policy is simple.",
    },
    {
        "id": "clear_answer",
        "type": "subtext",
        "sincere_context": "The answer addressed the question directly.",
        "loaded_context": "The answer avoided the question and introduced three new ambiguities.",
        "target": "That is clear.",
    },
    {
        "id": "good_job",
        "type": "subtext",
        "sincere_context": "He finished early, checked the work, and helped the rest of the team.",
        "loaded_context": "He missed the deadline, skipped the review, and left the mess for others.",
        "target": "Good job.",
    },
]

# ---------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------


def turning_curvature_proxy(x):
    """
    Scale-invariant discrete turning curvature.
    Measures mean change in unit tangent direction across the target span.
    """
    if x.shape[0] < 4:
        return 0.0

    steps = np.diff(x, axis=0)
    norms = np.linalg.norm(steps, axis=1, keepdims=True)

    valid = norms[:, 0] > 1e-8
    if valid.sum() < 3:
        return 0.0

    unit = steps[valid] / norms[valid]
    turns = np.diff(unit, axis=0)
    return float(np.mean(np.linalg.norm(turns, axis=1)))


def excess_path_length(x):
    """
    Δ = arc length / chord length - 1
    Measures indirectness of target-token path.
    """
    if x.shape[0] < 3:
        return 0.0

    steps = np.diff(x, axis=0)
    arc = float(np.sum(np.linalg.norm(steps, axis=1)))
    chord = float(np.linalg.norm(x[-1] - x[0]))

    if chord < 1e-8:
        return 0.0

    return float(arc / chord - 1.0)


def endpoint_shift(a, b):
    """
    Euclidean distance between final target-token states.
    Useful for asking whether same words land differently.
    """
    return float(np.linalg.norm(a[-1] - b[-1]))


def path_distance(a, b):
    """
    Mean aligned Euclidean distance between two same-target trajectories.
    Since target tokens are identical, target spans should have equal length.
    """
    m = min(len(a), len(b))
    if m == 0:
        return 0.0
    return float(np.mean(np.linalg.norm(a[:m] - b[:m], axis=1)))


def paired_t_stat(diffs):
    return stats_paired_t_stat(diffs)


def safe_wilcoxon_greater(diffs):
    diffs = np.asarray(diffs, dtype=float)
    if np.allclose(diffs, 0):
        return 1.0
    try:
        return float(
            wilcoxon(diffs, alternative="greater", zero_method="wilcox").pvalue
        )
    except Exception:
        return 1.0


def bh_fdr(pvals):
    return stats_bh_fdr(pvals)


# ---------------------------------------------------------------------
# Load model
# ---------------------------------------------------------------------

print("\nLoading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

print("Loading model in 8-bit...")
bnb_config = BitsAndBytesConfig(load_in_8bit=True)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    device_map="auto",
    quantization_config=bnb_config,
    torch_dtype=torch.float16,
    trust_remote_code=True,
)

model.eval()

# ---------------------------------------------------------------------
# Hidden-state extraction over identical target span
# ---------------------------------------------------------------------


def encode_full(context, target):
    """
    Return full prompt and target token positions.

    We encode:
        context + " " + target

    Then find target span by comparing tokenized prefix length.
    """
    full_text = context.strip() + " " + target.strip()

    prefix_ids = tokenizer(context.strip() + " ", return_tensors="pt")["input_ids"][
        0
    ].tolist()
    full = tokenizer(full_text, return_tensors="pt")
    full_ids = full["input_ids"][0].tolist()

    start = len(prefix_ids)
    end = len(full_ids)

    target_ids = full_ids[start:end]
    target_tokens = tokenizer.convert_ids_to_tokens(target_ids)

    return full_text, full, start, end, target_tokens


def get_target_hidden_states(context, target):
    full_text, encoded, start, end, target_tokens = encode_full(context, target)

    encoded = {k: v.to(model.device) for k, v in encoded.items()}

    with torch.no_grad():
        out = model(**encoded, output_hidden_states=True, use_cache=False)

    hidden_states = out.hidden_states
    n_layers = len(hidden_states) - 1

    target_by_layer = {}

    for layer in range(1, n_layers + 1):
        hs = hidden_states[layer][0].detach().float().cpu().numpy()
        target_hs = hs[start:end]
        target_by_layer[str(layer)] = target_hs

    del out
    torch.cuda.empty_cache()

    return {
        "full_text": full_text,
        "target_start": start,
        "target_end": end,
        "target_tokens": target_tokens,
        "target_by_layer": target_by_layer,
    }


# ---------------------------------------------------------------------
# Run extraction
# ---------------------------------------------------------------------

print("\nExtracting target-span hidden states...")
runs = {}

for i, item in enumerate(ITEMS, 1):
    print(f"[{i:02d}/{len(ITEMS)}] {item['id']}")

    sincere = get_target_hidden_states(item["sincere_context"], item["target"])
    loaded = get_target_hidden_states(item["loaded_context"], item["target"])

    # Sanity check: target tokens should match exactly.
    if sincere["target_tokens"] != loaded["target_tokens"]:
        print("WARNING: target tokens differ!")
        print("Sincere:", sincere["target_tokens"])
        print("Loaded: ", loaded["target_tokens"])

    runs[item["id"]] = {
        "item": item,
        "sincere": sincere,
        "loaded": loaded,
    }

n_layers = len(next(iter(runs.values()))["sincere"]["target_by_layer"])

# ---------------------------------------------------------------------
# Layerwise tests
# ---------------------------------------------------------------------

metrics = [
    "turn_curv_loaded_minus_sincere",
    "excess_len_loaded_minus_sincere",
    "path_distance_loaded_vs_sincere",
    "endpoint_shift_loaded_vs_sincere",
]

results = {
    "model": MODEL_NAME,
    "n_items": len(ITEMS),
    "n_layers": n_layers,
    "items": ITEMS,
    "tests": {},
}

print("\n" + "=" * 78)
print("LAYERWISE TESTS ON IDENTICAL TARGET SENTENCE")
print("=" * 78)

for metric in metrics:
    print(f"\nMetric: {metric}")
    print(f"{'Layer':<7} {'Mean':<14} {'p':<10} {'q(FDR)':<10} {'t':<10}")
    print("-" * 58)

    layer_means = []
    pvals = []
    tstats = []
    diffs_by_layer = {}

    for layer in range(1, n_layers + 1):
        diffs = []

        for item in ITEMS:
            rid = item["id"]
            s = runs[rid]["sincere"]["target_by_layer"][str(layer)]
            loaded = runs[rid]["loaded"]["target_by_layer"][str(layer)]

            if metric == "turn_curv_loaded_minus_sincere":
                diff = turning_curvature_proxy(loaded) - turning_curvature_proxy(s)
            elif metric == "excess_len_loaded_minus_sincere":
                diff = excess_path_length(loaded) - excess_path_length(s)
            elif metric == "path_distance_loaded_vs_sincere":
                diff = path_distance(loaded, s)
            elif metric == "endpoint_shift_loaded_vs_sincere":
                diff = endpoint_shift(loaded, s)
            else:
                raise ValueError(metric)

            diffs.append(diff)

        diffs = np.asarray(diffs, dtype=float)

        # For distances, greater-than-zero is trivial, so the meaningful layer test
        # is where the distance is maximized. Still use one-sided signed-rank
        # for consistency.
        p = safe_wilcoxon_greater(diffs)
        t = paired_t_stat(diffs)

        layer_means.append(float(np.mean(diffs)))
        pvals.append(p)
        tstats.append(t)
        diffs_by_layer[str(layer)] = diffs.tolist()

    qvals = bh_fdr(pvals)

    for layer in range(1, n_layers + 1):
        idx = layer - 1
        mark = "*FDR" if qvals[idx] < 0.05 else ""
        print(
            f"{layer:<7} "
            f"{layer_means[idx]:<14.5f} "
            f"{pvals[idx]:<10.4f} "
            f"{qvals[idx]:<10.4f} "
            f"{tstats[idx]:<10.3f} {mark}"
        )

    # Max-statistic sign-flip permutation across layers
    observed_max_t = max(tstats)
    observed_best_layer = int(np.argmax(tstats) + 1)
    diff_matrix = np.array(
        [diffs_by_layer[str(layer)] for layer in range(1, n_layers + 1)]
    )

    null_max = []
    for _ in range(N_PERM):
        signs = np.random.choice([-1, 1], size=len(ITEMS))
        perm_t = []
        for li in range(n_layers):
            perm_diffs = diff_matrix[li] * signs
            perm_t.append(paired_t_stat(perm_diffs))
        null_max.append(max(perm_t))

    null_max = np.asarray(null_max)
    global_p = empirical_p_value(null_max, observed_max_t, alternative="greater")

    print("\nLayer-search corrected result:")
    print(f"  Best layer: {observed_best_layer}/{n_layers}")
    print(f"  Best t:     {observed_max_t:.4f}")
    print(f"  Global p:   {global_p:.4f}")

    results["tests"][metric] = {
        "layer_means": layer_means,
        "pvals": pvals,
        "qvals_bh": qvals.tolist(),
        "tstats": tstats,
        "best_layer": observed_best_layer,
        "best_t": observed_max_t,
        "global_p_maxstat": global_p,
        "diffs_by_layer": diffs_by_layer,
    }

# ---------------------------------------------------------------------
# By-type breakdown at best path-distance layer
# ---------------------------------------------------------------------

print("\n" + "=" * 78)
print("BY-TYPE BREAKDOWN")
print("=" * 78)

best_layer = results["tests"]["path_distance_loaded_vs_sincere"]["best_layer"]
print(f"Using path-distance best layer: {best_layer}/{n_layers}")

type_rows = {}

for item in ITEMS:
    rid = item["id"]
    typ = item["type"]

    s = runs[rid]["sincere"]["target_by_layer"][str(best_layer)]
    loaded = runs[rid]["loaded"]["target_by_layer"][str(best_layer)]

    val = path_distance(loaded, s)
    type_rows.setdefault(typ, []).append(val)

print(f"{'Type':<16} {'n':<5} {'Mean path distance':<20}")
print("-" * 48)

for typ, vals in sorted(type_rows.items()):
    vals = np.asarray(vals, dtype=float)
    print(f"{typ:<16} {len(vals):<5} {np.mean(vals):<20.5f}")

results["by_type_at_best_path_distance_layer"] = {
    typ: {
        "n": len(vals),
        "mean": float(np.mean(vals)),
        "values": [float(x) for x in vals],
    }
    for typ, vals in type_rows.items()
}

# ---------------------------------------------------------------------
# Save
# ---------------------------------------------------------------------

json_path = os.path.join(OUTDIR, "hsg_identical_target_context_results.json")
with open(json_path, "w") as f:
    json.dump(results, f, indent=2)

csv_path = os.path.join(OUTDIR, "hsg_identical_target_context_layer_summary.csv")
with open(csv_path, "w") as f:
    f.write("metric,layer,mean,p,q,t\n")
    for metric in metrics:
        t = results["tests"][metric]
        for layer in range(1, n_layers + 1):
            idx = layer - 1
            f.write(
                f"{metric},{layer},"
                f"{t['layer_means'][idx]},"
                f"{t['pvals'][idx]},"
                f"{t['qvals_bh'][idx]},"
                f"{t['tstats'][idx]}\n"
            )

# ---------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------

layers = np.arange(1, n_layers + 1)

fig, axes = plt.subplots(4, 1, figsize=(12, 16), sharex=True)

for ax, metric in zip(axes, metrics):
    t = results["tests"][metric]
    ax.plot(layers, t["layer_means"], marker="o")
    ax.axvline(t["best_layer"], linestyle=":", linewidth=2)
    ax.set_title(
        f"{metric} | best layer {t['best_layer']}/{n_layers}, global p={t['global_p_maxstat']:.4f}"
    )
    ax.set_ylabel("mean value")

axes[-1].set_xlabel("Transformer layer")

fig.suptitle(
    f"HSG Identical Target Context Test\n{MODEL_NAME} | n={len(ITEMS)} identical target pairs",
    fontsize=14,
    fontweight="bold",
)

plot_path = os.path.join(OUTDIR, "hsg_identical_target_context_plot.png")
plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.savefig(plot_path, dpi=160)

print("\n" + "=" * 78)
print("DONE")
print("=" * 78)
print(f"JSON: {json_path}")
print(f"CSV:  {csv_path}")
print(f"Plot: {plot_path}")
