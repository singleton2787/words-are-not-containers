from pathlib import Path

# Portable path
import sys as _sys

_sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
from scipy.stats import mannwhitneyu

# Use the 7B model for maximum representational capacity
MODEL_NAME = "Qwen/Qwen2-7B"
print(f"Loading {MODEL_NAME} in 8-bit to execute ultimate diagnostics...")

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME, device_map="auto", load_in_8bit=True, output_hidden_states=True
)

controlled_corpus = [
    ("Oh great, another Monday morning.", "high"),
    ("Wonderful, the printer is broken again.", "high"),
    ("The presentation was adequate, I suppose.", "high"),
    ("Look, I really do love my wife, okay?", "high"),
    ("After all these years she finally called.", "high"),
    ("The capital of France is Paris today.", "low"),
    ("Water boils at one hundred degrees Celsius.", "low"),
    ("Shakespeare wrote Hamlet in sixteen oh one.", "low"),
    ("The results of the study were outstanding.", "low"),
    ("I am genuinely impressed by this work.", "low"),
]


def get_hidden_states(prompt):
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    with torch.no_grad():
        outputs = model(**inputs)
    return [
        layer_hs[0].cpu().numpy() for layer_hs in outputs.hidden_states
    ], inputs.input_ids[0].cpu().numpy()


def compute_curvature_raw(traj):
    diff1 = np.diff(traj, axis=0)
    diff2 = np.diff(diff1, axis=0)
    kappa = []
    for i in range(len(diff2)):
        d1 = diff1[i]
        d2 = diff2[i]
        n1 = np.linalg.norm(d1)
        if n1 < 1e-8:
            kappa.append(0.0)
            continue
        kappa.append(np.linalg.norm(d2) / (n1**2))
    return np.array(kappa)


# Extract all states up front
all_states = []
for p, cat in controlled_corpus:
    hs, tokens = get_hidden_states(p)
    all_states.append({"prompt": p, "cat": cat, "hs": hs, "tokens": tokens})

n_layers = len(all_states[0]["hs"]) - 1  # exclude embedding

print("\n" + "=" * 60)
print("TEST 1 & 2: EXHAUSTIVE LAYER SWEEP ON QWEN2-7B")
print("=" * 60)
print(
    f"{'Layer':<7} {'High Mean ̂κ':<15} {'Low Mean ̂κ':<15} {'Mann-Whitney p':<15} {'Verdict'}"
)
print("-" * 65)

for layer in range(1, n_layers + 1):
    high_k, low_k = [], []
    for item in all_states:
        traj = item["hs"][layer][1:-2]
        if len(traj) < 3:
            continue
        k = compute_curvature_raw(traj).mean()
        if item["cat"] == "high":
            high_k.append(k)
        else:
            low_k.append(k)

    if len(high_k) > 0:
        u_stat, p_val = mannwhitneyu(high_k, low_k, alternative="greater")
        verdict = "SIGNIFICANT" if p_val < 0.05 else "NULL"
        print(
            f"{layer:<7} {np.mean(high_k):<15.4f} {np.mean(low_k):<15.4f} {p_val:<15.4f} {verdict}"
        )

print("\n" + "=" * 60)
print("TEST 3: PAIRED TOKEN-DIVERGENCE MINIMAL PAIRS")
print("=" * 60)

minimal_pairs = [
    {
        "base": "The presentation was outstanding.",
        "fork": "The presentation was adequate, I suppose.",
        "fork_idx": 4,
    },
    {
        "base": "I love my wife.",
        "fork": "Look, I really do love my wife, okay?",
        "fork_idx": 0,
    },
]

for pair in minimal_pairs:
    hs_base, tok_base = get_hidden_states(pair["base"])
    hs_fork, tok_fork = get_hidden_states(pair["fork"])

    print(f"\nComparing:\n Base: '{pair['base']}'\n Fork: '{pair['fork']}'")
    print(
        f"{'Layer':<7} {'Max Euclidean Delta Post-Fork':<30} {'Cosine Sim at Final Token':<25}"
    )
    print("-" * 67)

    for layer in range(1, n_layers + 1):
        b_layer = hs_base[layer]
        f_layer = hs_fork[layer]

        v_base = b_layer[-1]
        v_fork = f_layer[-1]

        cos_sim = np.dot(v_base, v_fork) / (
            np.linalg.norm(v_base) * np.linalg.norm(v_fork) + 1e-10
        )

        min_len = min(len(b_layer), len(f_layer))
        deltas = [
            np.linalg.norm(b_layer[i] - f_layer[i])
            for i in range(pair["fork_idx"], min_len)
        ]
        max_delta = max(deltas) if deltas else 0.0

        print(f"{layer:<7} {max_delta:<30.4f} {cos_sim:<25.4f}")

print("\nDiagnostics complete.")
