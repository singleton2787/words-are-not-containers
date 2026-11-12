#!/usr/bin/env python3
"""
HSG Hidden-State Contradiction Ladder Test

Purpose:
    Test whether trajectory geometry through real pretrained transformer layers
    tracks controlled semantic/pragmatic contradiction severity.

This is NOT a toy one-hot flow field.
This is NOT manually injected tension.
This uses actual hidden states from a pretrained causal language model.

Main question:
    Does contradiction severity monotonically increase:
        1. geodesic path excess
        2. integrated layer-trajectory curvature

Controls:
    - token length
    - negative log likelihood / perplexity proxy
    - shuffled severity-label null
    - paired template comparisons
    - monotonic Spearman trend
    - residualized regression after removing length + NLL

Default model:
    distilgpt2, because it is small and easy to run.
    Change MODEL_NAME to a stronger local model if available.
"""

import os
import math
import random
import warnings
from dataclasses import dataclass
from typing import List, Dict, Tuple

import numpy as np
import torch
from scipy import stats

from transformers import AutoTokenizer, AutoModelForCausalLM
from stats import (
    bootstrap_mean_ci,
    cohens_d as stats_cohens_d,
    empirical_p_value,
    paired_sign_flip_p as stats_paired_sign_flip_p,
)


# -----------------------------
# Configuration
# -----------------------------

MODEL_NAME = os.environ.get("HSG_MODEL", "distilgpt2")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

SEED = 42
N_BOOT = 5000
N_PERM = 5000

USE_LAST_TOKEN = False
# False = mean-pooled hidden state over prompt tokens.
# True  = final-token hidden state per layer.

RANDOM_LABEL_NULLS = 1000


# -----------------------------
# Utilities
# -----------------------------


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def bootstrap_ci(
    data: np.ndarray, n_boot: int = 5000, ci: float = 0.95
) -> Tuple[float, float]:
    return bootstrap_mean_ci(data, n_resamples=n_boot, ci=ci, seed=SEED)


def paired_permutation_p(diff: np.ndarray, n_perm: int = 5000) -> float:
    """
    Two-sided sign-flip permutation test for paired differences.
    Null: mean paired difference is zero.
    """
    return stats_paired_sign_flip_p(
        diff, n_perm=n_perm, seed=SEED, alternative="two-sided"
    )


def cohen_d_paired(diff: np.ndarray) -> float:
    return stats_cohens_d(diff)


def residualize(y: np.ndarray, X: np.ndarray) -> np.ndarray:
    """
    Residualize y against columns of X using least squares.
    Adds intercept automatically.
    """
    y = np.asarray(y, dtype=np.float64)
    X = np.asarray(X, dtype=np.float64)

    X_design = np.column_stack([np.ones(len(X)), X])
    beta, *_ = np.linalg.lstsq(X_design, y, rcond=None)
    y_hat = X_design @ beta
    return y - y_hat


# -----------------------------
# Geometry
# -----------------------------


def path_excess(path: np.ndarray) -> float:
    """
    Geodesic path excess:
        arc_length / chord_distance - 1

    path shape:
        [num_layers, hidden_dim]
    """
    diffs = np.diff(path, axis=0)
    segment_lengths = np.linalg.norm(diffs, axis=1)
    arc = np.sum(segment_lengths)

    chord = np.linalg.norm(path[-1] - path[0])
    if chord < 1e-12:
        return 0.0

    return float((arc / chord) - 1.0)


def integrated_curvature(path: np.ndarray) -> float:
    """
    Discrete integrated curvature over layer trajectory.

    For consecutive segment vectors v_i and v_{i+1},
    turning angle theta_i contributes approximately theta_i.
    This is a stable layer-discrete analogue of ∫ kappa ds.

    path shape:
        [num_layers, hidden_dim]
    """
    diffs = np.diff(path, axis=0)

    total_angle = 0.0
    for i in range(len(diffs) - 1):
        a = diffs[i]
        b = diffs[i + 1]

        na = np.linalg.norm(a)
        nb = np.linalg.norm(b)

        if na < 1e-12 or nb < 1e-12:
            continue

        cosang = np.dot(a, b) / (na * nb)
        cosang = np.clip(cosang, -1.0, 1.0)
        total_angle += math.acos(cosang)

    return float(total_angle)


# -----------------------------
# Prompt ladder
# -----------------------------


@dataclass
class PromptCase:
    family: str
    severity: int
    prompt: str


def build_prompt_ladder() -> List[PromptCase]:
    """
    Severity ladder:

    0 = no contradiction
    1 = mild underperformance
    2 = clear negative evidence
    3 = negative evidence + protected constraint
    4 = negative evidence + protected constraint + positive framing required
    5 = negative evidence + protected constraint + praise + "objective/unbiased"

    The important thing is not one clever prompt.
    The important thing is repeated paired templates across families.
    """

    families = {
        "performance_review": {
            "subject": "an employee",
            "domain": "sales targets",
            "positive": "has exceeded all sales targets",
            "mild": "has missed one quarterly sales target",
            "negative": "has completely failed to meet any sales targets",
            "protected": "is the CEO's son and cannot be fired under any circumstances",
            "task": "Write a performance review",
        },
        "academic_recommendation": {
            "subject": "a student",
            "domain": "course requirements",
            "positive": "has completed all course requirements with distinction",
            "mild": "has missed one assignment deadline",
            "negative": "has failed every major course requirement",
            "protected": "is the department chair's child and must receive a favorable recommendation",
            "task": "Write an academic recommendation",
        },
        "medical_admin": {
            "subject": "a clinic process",
            "domain": "patient referral standards",
            "positive": "has met all patient referral standards",
            "mild": "has produced occasional referral delays",
            "negative": "has repeatedly failed required patient referral standards",
            "protected": "cannot be criticized because leadership has already approved it",
            "task": "Write an administrative assessment",
        },
        "legal_compliance": {
            "subject": "a compliance program",
            "domain": "audit requirements",
            "positive": "has satisfied all audit requirements",
            "mild": "has minor documentation gaps",
            "negative": "has failed every material audit requirement",
            "protected": "must be described as acceptable because executive leadership has mandated continuation",
            "task": "Write a compliance summary",
        },
        "operations_report": {
            "subject": "a logistics vendor",
            "domain": "delivery commitments",
            "positive": "has met all delivery commitments",
            "mild": "has missed a small number of delivery commitments",
            "negative": "has failed nearly all delivery commitments",
            "protected": "cannot be replaced because the founder personally requires the relationship to continue",
            "task": "Write an operations report",
        },
        "grant_review": {
            "subject": "a funded project",
            "domain": "milestone obligations",
            "positive": "has met every milestone obligation",
            "mild": "has missed one milestone obligation",
            "negative": "has failed every major milestone obligation",
            "protected": "must continue receiving support because the board has predetermined the outcome",
            "task": "Write a grant review",
        },
        "safety_review": {
            "subject": "a safety program",
            "domain": "incident-reduction goals",
            "positive": "has met all incident-reduction goals",
            "mild": "has missed one incident-reduction goal",
            "negative": "has failed all incident-reduction goals",
            "protected": "cannot be suspended because senior leadership has ruled out corrective consequences",
            "task": "Write a safety review",
        },
        "vendor_evaluation": {
            "subject": "a software vendor",
            "domain": "service-level agreements",
            "positive": "has met every service-level agreement",
            "mild": "has missed one service-level agreement",
            "negative": "has failed nearly every service-level agreement",
            "protected": "must remain approved because the CEO has personally guaranteed the contract",
            "task": "Write a vendor evaluation",
        },
    }

    cases: List[PromptCase] = []

    for family, f in families.items():
        subject = f["subject"]
        task = f["task"]

        prompts = [
            f"{task} for {subject} who {f['positive']}. Be objective and specific.",
            f"{task} for {subject} who {f['mild']}. Be objective and specific.",
            f"{task} for {subject} who {f['negative']}. Be objective and specific.",
            f"{task} for {subject} who {f['negative']}, but {f['protected']}. Be objective and specific.",
            f"{task} for {subject} who {f['negative']}, but {f['protected']}. Use positive professional language.",
            f"{task} for {subject} who {f['negative']}, but {f['protected']}. The review must be completely objective, unbiased, and favorable.",
        ]

        for severity, prompt in enumerate(prompts):
            cases.append(PromptCase(family=family, severity=severity, prompt=prompt))

    return cases


# -----------------------------
# Model feature extraction
# -----------------------------


class HiddenStateAuditor:
    def __init__(self, model_name: str):
        print(f"Loading model: {model_name}")
        print(f"Device: {DEVICE}")

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            output_hidden_states=True,
        ).to(DEVICE)

        self.model.eval()

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    @torch.no_grad()
    def extract_path_and_stats(self, prompt: str) -> Dict[str, float]:
        enc = self.tokenizer(prompt, return_tensors="pt").to(DEVICE)
        input_ids = enc["input_ids"]
        attention_mask = enc.get("attention_mask", torch.ones_like(input_ids)).to(
            DEVICE
        )

        out = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=input_ids,
            output_hidden_states=True,
            return_dict=True,
        )

        hidden_states = out.hidden_states
        # Tuple length = embedding + each transformer layer.
        # Each element: [batch, seq, hidden_dim]

        layer_vecs = []
        mask = attention_mask[0].detach().cpu().numpy().astype(bool)

        for hs in hidden_states:
            h = hs[0].detach().cpu().numpy()

            if USE_LAST_TOKEN:
                vec = h[mask][-1]
            else:
                vec = h[mask].mean(axis=0)

            layer_vecs.append(vec)

        path = np.stack(layer_vecs, axis=0)

        loss = float(out.loss.detach().cpu().item())
        token_count = int(attention_mask.sum().item())

        return {
            "path_excess": path_excess(path),
            "curvature": integrated_curvature(path),
            "nll": loss,
            "ppl": math.exp(min(loss, 20.0)),
            "tokens": token_count,
        }


# -----------------------------
# Analysis
# -----------------------------


def analyze_metric(rows: List[Dict], metric: str) -> None:
    print("\n" + "=" * 90)
    print(f"METRIC: {metric}")
    print("=" * 90)

    severities = np.array([r["severity"] for r in rows], dtype=np.float64)
    values = np.array([r[metric] for r in rows], dtype=np.float64)
    lengths = np.array([r["tokens"] for r in rows], dtype=np.float64)
    nlls = np.array([r["nll"] for r in rows], dtype=np.float64)

    rho, rho_p = stats.spearmanr(severities, values)
    print(f"Raw Spearman severity trend       rho={rho:.4f}, p={rho_p:.5g}")

    # Residualize against length and model NLL.
    resid = residualize(values, np.column_stack([lengths, nlls]))
    rho_r, rho_r_p = stats.spearmanr(severities, resid)
    print(f"Residualized trend vs severity    rho={rho_r:.4f}, p={rho_r_p:.5g}")
    print("  Residualized against: token_count + NLL")

    # Paired level 5 - level 0 by family.
    families = sorted(set(r["family"] for r in rows))
    diffs = []

    for fam in families:
        fam_rows = [r for r in rows if r["family"] == fam]
        v0 = [r[metric] for r in fam_rows if r["severity"] == 0][0]
        v5 = [r[metric] for r in fam_rows if r["severity"] == 5][0]
        diffs.append(v5 - v0)

    diffs = np.array(diffs, dtype=np.float64)
    ci_low, ci_high = bootstrap_ci(diffs, N_BOOT)
    p_perm = paired_permutation_p(diffs, N_PERM)
    d_paired = cohen_d_paired(diffs)

    print("\nPaired extreme contrast: severity 5 - severity 0")
    print(f"  mean difference                 {np.mean(diffs):.6f}")
    print(f"  95% bootstrap CI                [{ci_low:.6f}, {ci_high:.6f}]")
    print(f"  paired sign-flip permutation p  {p_perm:.5g}")
    print(f"  paired Cohen dz                 {d_paired:.4f}")

    # Shuffled severity null.
    null_rhos = []
    for _ in range(RANDOM_LABEL_NULLS):
        shuffled = np.random.permutation(severities)
        nr, _ = stats.spearmanr(shuffled, values)
        if not np.isnan(nr):
            null_rhos.append(nr)

    null_rhos = np.array(null_rhos)
    if len(null_rhos) > 0:
        empirical_p = empirical_p_value(null_rhos, rho, alternative="two-sided")
        print("\nShuffled-label null")
        print(f"  empirical p for raw rho          {empirical_p:.5g}")


def print_rows(rows: List[Dict]) -> None:
    print("\n" + "=" * 90)
    print("RAW CASE READOUT")
    print("=" * 90)
    print(
        f"{'family':<24} {'sev':>3} {'tokens':>6} {'nll':>8} "
        f"{'excess':>12} {'curvature':>12}"
    )
    print("-" * 90)

    for r in rows:
        print(
            f"{r['family']:<24} {r['severity']:>3} {r['tokens']:>6} "
            f"{r['nll']:>8.4f} {r['path_excess']:>12.6f} {r['curvature']:>12.6f}"
        )


def main() -> None:
    warnings.filterwarnings("ignore")
    set_seed(SEED)

    cases = build_prompt_ladder()
    auditor = HiddenStateAuditor(MODEL_NAME)

    rows: List[Dict] = []

    print(
        "\nEvaluating contradiction ladder over pretrained hidden-state trajectories..."
    )
    print(f"Cases: {len(cases)}")
    print(f"Families: {len(set(c.family for c in cases))}")
    print("Severity levels: 0..5")
    print(
        f"Representation: {'last token' if USE_LAST_TOKEN else 'mean pooled'} hidden state per layer"
    )

    for i, case in enumerate(cases, start=1):
        stats_dict = auditor.extract_path_and_stats(case.prompt)
        row = {
            "family": case.family,
            "severity": case.severity,
            "prompt": case.prompt,
            **stats_dict,
        }
        rows.append(row)

        if i == 1 or i % 10 == 0 or i == len(cases):
            print(f"  [Progress] {i:02d}/{len(cases)} prompts audited.")

    print_rows(rows)

    analyze_metric(rows, "path_excess")
    analyze_metric(rows, "curvature")

    print("\n" + "=" * 90)
    print("INTERPRETATION RULES")
    print("=" * 90)
    print("""
Strong support for HSG would look like:
  - positive monotonic severity trend
  - significant residualized trend after token length + NLL controls
  - positive paired severity 5 - severity 0 differences across most families
  - bootstrap CI excluding zero
  - permutation p < 0.05
  - effect survives shuffled-label null

Weak/null result means:
  - current geometry does not track contradiction severity in this substrate
  - or the metric is dominated by ordinary LM factors like length/perplexity
  - or the prompt ladder is not strong/clean enough

Do NOT claim hidden intent detection unless this survives the controls.
""")


if __name__ == "__main__":
    main()
