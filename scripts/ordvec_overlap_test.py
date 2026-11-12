from pathlib import Path

# Portable path
import sys as _sys

_sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
from scipy.stats import mannwhitneyu

MODEL_NAME = "Qwen/Qwen2-7B"
print(f"Loading {MODEL_NAME} for Ordvec Bitwise Overlap Test...")

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME, device_map="auto", load_in_8bit=True, output_hidden_states=True
)

PAIRS = [
    {
        "id": "sarcasm_printer",
        "low": "Great, the printer is working again.",
        "high": "Great, the printer is broken again.",
    },
    {
        "id": "sarcasm_server",
        "low": "Wonderful, the server is responding now.",
        "high": "Wonderful, the server crashed again.",
    },
    {
        "id": "faint_praise_presentation",
        "low": "The presentation was outstanding.",
        "high": "The presentation was adequate, I suppose.",
    },
    {
        "id": "faint_praise_report",
        "low": "Your report was clear and complete.",
        "high": "Your report was quite readable, in places.",
    },
    {
        "id": "defensive_love",
        "low": "I love my wife.",
        "high": "Look, I really do love my wife, okay?",
    },
    {
        "id": "defensive_honest",
        "low": "I am being honest with you.",
        "high": "I am being honest with you, believe me.",
    },
    {
        "id": "hedge_policy",
        "low": "The policy is simple.",
        "high": "The policy is simple, apparently.",
    },
    {
        "id": "hedge_results",
        "low": "The results support the claim.",
        "high": "The results seem to support the claim.",
    },
    {
        "id": "implication_arrival",
        "low": "He arrived on time.",
        "high": "He finally arrived on time.",
    },
    {
        "id": "subtext_clear",
        "low": "Your explanation was clear.",
        "high": "Your explanation was clear as mud.",
    },
]

PEAK_LAYER = 26  # Layer 26 of 28 where representations are maximized
K_TOP = 180  # Top ~5% of the 3584 dimensions in Qwen2-7B


def get_constant_weight_bitmap(text):
    inputs = tokenizer(text, return_tensors="pt").to("cuda")
    with torch.no_grad():
        outputs = model(**inputs)

    # Extract states at the last processed semantic token, layer 26
    hs = outputs.hidden_states[PEAK_LAYER][0, -1].cpu().numpy()

    # Convert to constant-weight bitmap: Top K activations get 1, rest get 0
    top_indices = np.argsort(hs)[-K_TOP:]
    bitmap = np.zeros_like(hs, dtype=int)
    bitmap[top_indices] = 1
    return bitmap


print("\nExecuting Ordinal Overlap Analysis...")
print(f"{'Pair ID':<30} {'Low Overlap':<15} {'High Overlap':<15} {'Delta (Low-High)'}")
print("-" * 70)

low_overlaps = []
high_overlaps = []
deltas = []

for pair in PAIRS:
    # We establish the anchor bitmap using a generic base statement to match Nelson's Query model
    anchor_text = " ".join(pair["low"].split()[:2])  # Shared prefix anchor
    try:
        b_anchor = get_constant_weight_bitmap(anchor_text)
        b_low = get_constant_weight_bitmap(pair["low"])
        b_high = get_constant_weight_bitmap(pair["high"])

        # Calculate overlap (popcount of bitwise AND)
        overlap_low = int(np.sum(b_anchor & b_low))
        overlap_high = int(np.sum(b_anchor & b_high))
        delta = overlap_low - overlap_high

        print(f"{pair['id']:<30} {overlap_low:<15} {overlap_high:<15} {delta}")
        low_overlaps.append(overlap_low)
        high_overlaps.append(overlap_high)
        deltas.append(delta)
    except Exception:
        continue

print("\n" + "=" * 60)
print("STATISTICAL VERDICT")
print("=" * 60)
print(f"Mean Low Overlap (Literal):  {np.mean(low_overlaps):.2f}")
print(f"Mean High Overlap (Pragmatic): {np.mean(high_overlaps):.2f}")
u_stat, p_val = mannwhitneyu(low_overlaps, high_overlaps, alternative="greater")
print(f"One-sided Mann-Whitney p-value: {p_val:.4f}")

# Hypergeometric Calibration Check
# Total dimensions N=3584, sample sizes drawn n=K_TOP, matching bits = observed overlap
N_TOTAL = 3584
expected_null_overlap = (K_TOP * K_TOP) / N_TOTAL
print(f"Idealized Uniform-Null Expected Overlap: {expected_null_overlap:.2f}")
