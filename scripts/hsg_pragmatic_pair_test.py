from pathlib import Path
# HSG Pragmatic Pair Test
# Purpose:
#   Test whether high-pragmatic utterances show greater residual-stream
#   trajectory curvature than matched low-pragmatic utterances.
#
# Key safeguards:
#   - paired minimal contrasts
#   - automatic tokenizer fork detection
#   - scale-invariant turning-angle curvature proxy
#   - excess path length metric
#   - layerwise paired tests
#   - max-statistic sign-flip permutation to correct for layer fishing
#   - length artifact check
#
# Mike Singleton

import os
import json
import random
import warnings
from dataclasses import dataclass

import numpy as np
import torch
import matplotlib.pyplot as plt

from scipy.stats import wilcoxon, pearsonr
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from stats import (
    bh_fdr as stats_bh_fdr,
    empirical_p_value,
    paired_t_stat as stats_paired_t_stat,
)

warnings.filterwarnings("ignore")

# -----------------------------
# Config
# -----------------------------

MODEL_NAME = os.environ.get("HSG_MODEL", "Qwen/Qwen2-7B")
OUTDIR = str(Path(__file__).resolve().parents[1] / "results" / "pragmatic_pair_test")
N_PERM = int(os.environ.get("HSG_N_PERM", "2000"))
SEED = 42

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

os.makedirs(OUTDIR, exist_ok=True)

print("=" * 72)
print("HSG PRAGMATIC PAIR TEST")
print("=" * 72)
print(f"Model: {MODEL_NAME}")
print(f"Permutation samples: {N_PERM}")
print(f"Output dir: {OUTDIR}")

# -----------------------------
# Corpus
# -----------------------------
# Each pair tries to preserve topic/lexical material while changing pragmatic force.
# "low" = direct/literal/sincere
# "high" = sarcasm, faint praise, hedging, defensiveness, implication, subtext

PAIRS = [
    {
        "id": "sarcasm_printer",
        "type": "sarcasm",
        "low": "Great, the printer is working again.",
        "high": "Great, the printer is broken again.",
    },
    {
        "id": "sarcasm_server",
        "type": "sarcasm",
        "low": "Wonderful, the server is responding now.",
        "high": "Wonderful, the server crashed again.",
    },
    {
        "id": "sarcasm_meeting",
        "type": "sarcasm",
        "low": "That meeting was useful.",
        "high": "Well, that meeting was useful.",
    },
    {
        "id": "faint_praise_presentation",
        "type": "faint_praise",
        "low": "The presentation was outstanding.",
        "high": "The presentation was adequate, I suppose.",
    },
    {
        "id": "faint_praise_report",
        "type": "faint_praise",
        "low": "Your report was clear and complete.",
        "high": "Your report was quite readable, in places.",
    },
    {
        "id": "faint_praise_effort",
        "type": "faint_praise",
        "low": "You did excellent work on this.",
        "high": "You certainly put effort into this.",
    },
    {
        "id": "defensive_love",
        "type": "defensive",
        "low": "I love my wife.",
        "high": "Look, I really do love my wife, okay?",
    },
    {
        "id": "defensive_honest",
        "type": "defensive",
        "low": "I am being honest with you.",
        "high": "I am being honest with you, believe me.",
    },
    {
        "id": "defensive_angry",
        "type": "defensive",
        "low": "I am not angry.",
        "high": "I am not angry, I said.",
    },
    {
        "id": "hedge_policy",
        "type": "hedge",
        "low": "The policy is simple.",
        "high": "The policy is simple, apparently.",
    },
    {
        "id": "hedge_results",
        "type": "hedge",
        "low": "The results support the claim.",
        "high": "The results seem to support the claim.",
    },
    {
        "id": "hedge_plan",
        "type": "hedge",
        "low": "This plan will work.",
        "high": "This plan might work, in theory.",
    },
    {
        "id": "implication_arrival",
        "type": "implication",
        "low": "He arrived on time.",
        "high": "He finally arrived on time.",
    },
    {
        "id": "implication_called",
        "type": "implication",
        "low": "After dinner, she called.",
        "high": "After all these years, she finally called.",
    },
    {
        "id": "implication_apology",
        "type": "implication",
        "low": "He apologized to her.",
        "high": "He actually apologized to her.",
    },
    {
        "id": "subtext_clear",
        "type": "subtext",
        "low": "Your explanation was clear.",
        "high": "Your explanation was clear as mud.",
    },
    {
        "id": "subtext_helpful",
        "type": "subtext",
        "low": "Thanks for fixing the issue.",
        "high": "Thanks for creating another issue.",
    },
    {
        "id": "subtext_interesting",
        "type": "subtext",
        "low": "The proposal is strong.",
        "high": "The proposal is interesting.",
    },
]

# -----------------------------
# Helpers
# -----------------------------


@dataclass
class PromptRun:
    pair_id: str
    pair_type: str
    label: str
    text: str
    tokens: list
    token_count: int
    metrics_by_layer: dict


def first_divergence(a, b):
    """Return first token index where two token-id sequences diverge."""
    m = min(len(a), len(b))
    for i in range(m):
        if a[i] != b[i]:
            return i
    return m


def turning_curvature_proxy(x):
    """
    Scale-invariant discrete curvature proxy.

    x: [T, D] trajectory over tokens at one layer.

    Computes mean change in unit tangent direction.
    This avoids raw vector scale dominating the curvature measure.
    """
    if x.shape[0] < 4:
        return 0.0

    steps = np.diff(x, axis=0)
    norms = np.linalg.norm(steps, axis=1, keepdims=True)

    valid = norms[:, 0] > 1e-8
    if valid.sum() < 3:
        return 0.0

    steps = steps[valid]
    norms = norms[valid]
    unit = steps / norms

    turns = np.diff(unit, axis=0)
    turn_mag = np.linalg.norm(turns, axis=1)

    return float(np.mean(turn_mag))


def excess_path_length(x):
    """
    Δ = arc_length / chord_length - 1

    Measures indirectness of the path independent of absolute endpoint.
    """
    if x.shape[0] < 3:
        return 0.0

    steps = np.diff(x, axis=0)
    arc = float(np.sum(np.linalg.norm(steps, axis=1)))
    chord = float(np.linalg.norm(x[-1] - x[0]))

    if chord < 1e-8:
        return 0.0

    return float((arc / chord) - 1.0)


def step_energy(x):
    """Mean step norm; used as an artifact/control metric."""
    if x.shape[0] < 2:
        return 0.0
    steps = np.diff(x, axis=0)
    return float(np.mean(np.linalg.norm(steps, axis=1)))


def safe_wilcoxon_greater(diffs):
    """
    Paired one-sided Wilcoxon signed-rank test.
    H1: median(diff) > 0.
    """
    diffs = np.asarray(diffs, dtype=float)
    if np.allclose(diffs, 0):
        return 1.0
    try:
        return float(
            wilcoxon(diffs, alternative="greater", zero_method="wilcox").pvalue
        )
    except Exception:
        return 1.0


def paired_t_stat(diffs):
    """
    Used for permutation max-statistic.
    """
    return stats_paired_t_stat(diffs)


def bh_fdr(pvals):
    """
    Benjamini-Hochberg FDR correction.
    """
    return stats_bh_fdr(pvals)


# -----------------------------
# Load model
# -----------------------------

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

# -----------------------------
# Extract hidden states
# -----------------------------


def run_prompt(text, pair_id, pair_type, label):
    inputs = tokenizer(text, return_tensors="pt")
    input_ids = inputs["input_ids"][0].tolist()
    toks = tokenizer.convert_ids_to_tokens(input_ids)

    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    with torch.no_grad():
        out = model(**inputs, output_hidden_states=True, use_cache=False)

    hidden_states = out.hidden_states
    n_layers = len(hidden_states) - 1

    metrics_by_layer = {}

    for layer in range(1, n_layers + 1):
        hs = hidden_states[layer][0].detach().float().cpu().numpy()

        # Exclude first token when possible; keep final token because output convergence matters.
        traj = hs[1:] if hs.shape[0] > 4 else hs

        metrics_by_layer[str(layer)] = {
            "turn_curv": turning_curvature_proxy(traj),
            "excess_len": excess_path_length(traj),
            "step_energy": step_energy(traj),
        }

    del out
    torch.cuda.empty_cache()

    return PromptRun(
        pair_id=pair_id,
        pair_type=pair_type,
        label=label,
        text=text,
        tokens=toks,
        token_count=len(input_ids),
        metrics_by_layer=metrics_by_layer,
    )


print("\nExtracting hidden states...")
runs = []
forks = {}

for i, pair in enumerate(PAIRS, 1):
    print(f"[{i:02d}/{len(PAIRS)}] {pair['id']}")

    low_ids = tokenizer(pair["low"], return_tensors="pt")["input_ids"][0].tolist()
    high_ids = tokenizer(pair["high"], return_tensors="pt")["input_ids"][0].tolist()
    fork = first_divergence(low_ids, high_ids)

    forks[pair["id"]] = {
        "fork_idx": fork,
        "low_tokens": tokenizer.convert_ids_to_tokens(low_ids),
        "high_tokens": tokenizer.convert_ids_to_tokens(high_ids),
        "low_token_count": len(low_ids),
        "high_token_count": len(high_ids),
    }

    runs.append(run_prompt(pair["low"], pair["id"], pair["type"], "low"))
    runs.append(run_prompt(pair["high"], pair["id"], pair["type"], "high"))

n_layers = len(runs[0].metrics_by_layer)

# -----------------------------
# Build pairwise matrices
# -----------------------------

metrics = ["turn_curv", "excess_len", "step_energy"]

# pair_id -> low/high run
by_pair = {}
for r in runs:
    by_pair.setdefault(r.pair_id, {})[r.label] = r

results = {
    "model": MODEL_NAME,
    "n_pairs": len(PAIRS),
    "n_layers": n_layers,
    "pairs": PAIRS,
    "forks": forks,
    "tests": {},
}

print("\n" + "=" * 72)
print("PRIMARY LAYERWISE PAIRED TESTS")
print("=" * 72)

for metric in metrics:
    print(f"\nMetric: {metric}")
    print(
        f"{'Layer':<7} {'Low mean':<12} {'High mean':<12} {'Mean diff':<12} {'p':<10} {'q(FDR)':<10} {'t':<10}"
    )
    print("-" * 78)

    low_means = []
    high_means = []
    mean_diffs = []
    pvals = []
    tstats = []
    diffs_by_layer = {}

    for layer in range(1, n_layers + 1):
        diffs = []
        lows = []
        highs = []

        for pair in PAIRS:
            pid = pair["id"]
            low = by_pair[pid]["low"].metrics_by_layer[str(layer)][metric]
            high = by_pair[pid]["high"].metrics_by_layer[str(layer)][metric]

            lows.append(low)
            highs.append(high)
            diffs.append(high - low)

        diffs = np.asarray(diffs, dtype=float)
        lows = np.asarray(lows, dtype=float)
        highs = np.asarray(highs, dtype=float)

        p = safe_wilcoxon_greater(diffs)
        t = paired_t_stat(diffs)

        low_means.append(float(np.mean(lows)))
        high_means.append(float(np.mean(highs)))
        mean_diffs.append(float(np.mean(diffs)))
        pvals.append(p)
        tstats.append(t)
        diffs_by_layer[str(layer)] = diffs.tolist()

    qvals = bh_fdr(pvals)

    for layer in range(1, n_layers + 1):
        idx = layer - 1
        mark = ""
        if qvals[idx] < 0.05:
            mark = "*FDR"
        print(
            f"{layer:<7} "
            f"{low_means[idx]:<12.5f} "
            f"{high_means[idx]:<12.5f} "
            f"{mean_diffs[idx]:<12.5f} "
            f"{pvals[idx]:<10.4f} "
            f"{qvals[idx]:<10.4f} "
            f"{tstats[idx]:<10.3f} {mark}"
        )

    # Max-statistic sign-flip permutation across layers
    observed_max_t = max(tstats)
    observed_best_layer = int(np.argmax(tstats) + 1)

    null_max = []
    diff_matrix = np.array(
        [diffs_by_layer[str(layer)] for layer in range(1, n_layers + 1)]
    )  # [L, P]

    for _ in range(N_PERM):
        signs = np.random.choice([-1, 1], size=len(PAIRS))
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
    print(f"  Global p:   {global_p:.4f}  using sign-flip max-statistic")

    results["tests"][metric] = {
        "low_means": low_means,
        "high_means": high_means,
        "mean_diffs": mean_diffs,
        "pvals": pvals,
        "qvals_bh": qvals.tolist(),
        "tstats": tstats,
        "best_layer": observed_best_layer,
        "best_t": observed_max_t,
        "global_p_maxstat": global_p,
        "diffs_by_layer": diffs_by_layer,
    }

# -----------------------------
# Length artifact check
# -----------------------------

print("\n" + "=" * 72)
print("LENGTH ARTIFACT CHECK")
print("=" * 72)

length_diffs = []
for pair in PAIRS:
    pid = pair["id"]
    low_n = by_pair[pid]["low"].token_count
    high_n = by_pair[pid]["high"].token_count
    length_diffs.append(high_n - low_n)

length_diffs = np.asarray(length_diffs, dtype=float)

for metric in ["turn_curv", "excess_len"]:
    best_layer = results["tests"][metric]["best_layer"]
    metric_diffs = np.asarray(
        results["tests"][metric]["diffs_by_layer"][str(best_layer)]
    )

    if np.std(length_diffs) > 1e-8 and np.std(metric_diffs) > 1e-8:
        r, p = pearsonr(length_diffs, metric_diffs)
    else:
        r, p = 0.0, 1.0

    print(f"{metric}: best layer {best_layer}")
    print(f"  corr(length_diff, metric_diff) = r={r:.3f}, p={p:.4f}")

    results["tests"][metric]["length_artifact_corr"] = {
        "r": float(r),
        "p": float(p),
        "length_diffs": length_diffs.tolist(),
    }

# -----------------------------
# By-type exploratory breakdown
# -----------------------------

print("\n" + "=" * 72)
print("BY-TYPE BREAKDOWN AT BEST TURN-CURV LAYER")
print("=" * 72)

best_layer = results["tests"]["turn_curv"]["best_layer"]
print(f"Using layer {best_layer}/{n_layers}")

type_rows = {}

for pair in PAIRS:
    pid = pair["id"]
    typ = pair["type"]
    low = by_pair[pid]["low"].metrics_by_layer[str(best_layer)]["turn_curv"]
    high = by_pair[pid]["high"].metrics_by_layer[str(best_layer)]["turn_curv"]
    type_rows.setdefault(typ, []).append(high - low)

print(f"{'Type':<16} {'n':<5} {'Mean high-low turn_curv':<25}")
print("-" * 52)

for typ, vals in sorted(type_rows.items()):
    vals = np.asarray(vals, dtype=float)
    print(f"{typ:<16} {len(vals):<5} {np.mean(vals):<25.5f}")

results["by_type_at_best_turn_curv_layer"] = {
    typ: {
        "n": len(vals),
        "mean_diff": float(np.mean(vals)),
        "diffs": [float(x) for x in vals],
    }
    for typ, vals in type_rows.items()
}

# -----------------------------
# Save outputs
# -----------------------------

json_path = os.path.join(OUTDIR, "hsg_pragmatic_pair_results.json")
with open(json_path, "w") as f:
    json.dump(results, f, indent=2)

# CSV-style layer summary
csv_path = os.path.join(OUTDIR, "hsg_pragmatic_pair_layer_summary.csv")
with open(csv_path, "w") as f:
    f.write("metric,layer,low_mean,high_mean,mean_diff,p,q,t\n")
    for metric in metrics:
        t = results["tests"][metric]
        for layer in range(1, n_layers + 1):
            idx = layer - 1
            f.write(
                f"{metric},{layer},"
                f"{t['low_means'][idx]},"
                f"{t['high_means'][idx]},"
                f"{t['mean_diffs'][idx]},"
                f"{t['pvals'][idx]},"
                f"{t['qvals_bh'][idx]},"
                f"{t['tstats'][idx]}\n"
            )

# -----------------------------
# Plot
# -----------------------------

layers = np.arange(1, n_layers + 1)

fig, axes = plt.subplots(3, 1, figsize=(12, 14), sharex=True)

for ax, metric in zip(axes, metrics):
    t = results["tests"][metric]

    ax.plot(layers, t["low_means"], marker="o", label="low/direct")
    ax.plot(layers, t["high_means"], marker="o", label="high/pragmatic")
    ax.plot(layers, t["mean_diffs"], marker="x", linestyle="--", label="high-low diff")

    best = t["best_layer"]
    ax.axvline(best, linestyle=":", linewidth=2)

    ax.set_title(
        f"{metric} — best layer {best}/{n_layers}, global p={t['global_p_maxstat']:.4f}"
    )
    ax.set_ylabel(metric)
    ax.legend()

axes[-1].set_xlabel("Transformer layer")

fig.suptitle(
    f"HSG Pragmatic Pair Test\n{MODEL_NAME} | n={len(PAIRS)} pairs | max-stat permutation corrected",
    fontsize=14,
    fontweight="bold",
)

plot_path = os.path.join(OUTDIR, "hsg_pragmatic_pair_plot.png")
plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.savefig(plot_path, dpi=160)

print("\n" + "=" * 72)
print("DONE")
print("=" * 72)
print(f"JSON: {json_path}")
print(f"CSV:  {csv_path}")
print(f"Plot: {plot_path}")
