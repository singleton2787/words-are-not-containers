#!/usr/bin/env python3
"""
HSG Minimal-Pair Curvature Test

Purpose:
    Test the last surviving HSG claim:

        Contradiction/conflict may increase hidden-state trajectory curvature
        when length, syntax, and prompt form are tightly controlled.

This test uses:
    - pretrained transformer hidden states
    - minimal semantic pairs
    - matched sentence frames
    - paired statistics
    - token-count and NLL residualization
    - label-shuffle null
    - curvature as the PRIMARY metric

Default model:
    distilgpt2

Try stronger models:
    HSG_MODEL=Qwen/Qwen2.5-1.5B python3 hsg_minimal_pair_curvature_test.py
    HSG_MODEL=Qwen/Qwen2.5-3B python3 hsg_minimal_pair_curvature_test.py
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
    paired_sign_flip_p as stats_paired_sign_flip_p,
)


MODEL_NAME = os.environ.get("HSG_MODEL", "distilgpt2")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

SEED = 42
N_BOOT = 10000
N_PERM = 10000

# Try both eventually.
USE_LAST_TOKEN = True
# True  = final-token hidden-state trajectory.
# False = mean-pooled hidden-state trajectory.


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def bootstrap_ci(
    data: np.ndarray, n_boot: int = 10000, ci: float = 0.95
) -> Tuple[float, float]:
    return bootstrap_mean_ci(data, n_resamples=n_boot, ci=ci, seed=SEED)


def paired_sign_flip_p(diff: np.ndarray, n_perm: int = 10000) -> float:
    return stats_paired_sign_flip_p(
        diff, n_perm=n_perm, seed=SEED, alternative="two-sided"
    )


def paired_cohen_dz(diff: np.ndarray) -> float:
    return stats_cohens_d(diff)


def residualize(y: np.ndarray, X: np.ndarray) -> np.ndarray:
    y = np.asarray(y, dtype=np.float64)
    X = np.asarray(X, dtype=np.float64)
    X_design = np.column_stack([np.ones(len(X)), X])
    beta, *_ = np.linalg.lstsq(X_design, y, rcond=None)
    return y - X_design @ beta


def integrated_curvature(path: np.ndarray) -> float:
    """
    Discrete layer-trajectory curvature.

    path shape:
        [num_layers, hidden_dim]

    Curvature is approximated as total turning angle across layer updates.
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


def path_excess(path: np.ndarray) -> float:
    """
    Secondary metric only.
    """
    diffs = np.diff(path, axis=0)
    arc = np.sum(np.linalg.norm(diffs, axis=1))
    chord = np.linalg.norm(path[-1] - path[0])

    if chord < 1e-12:
        return 0.0

    return float((arc / chord) - 1.0)


@dataclass
class PairCase:
    family: str
    clean: str
    conflicted: str


def build_minimal_pairs() -> List[PairCase]:
    """
    Minimal pairs.

    The clean/conflicted pair keeps:
        - same outer instruction
        - similar length
        - same syntax
        - same final forced evaluation

    The semantic flip:
        clean      = evidence supports evaluation
        conflicted = evidence contradicts evaluation
    """

    raw_pairs = [
        (
            "performance_praise",
            "Write one objective sentence: The employee met every target and must receive praise.",
            "Write one objective sentence: The employee missed every target and must receive praise.",
        ),
        (
            "promotion",
            "Write one objective sentence: The manager improved every metric and must be promoted.",
            "Write one objective sentence: The manager damaged every metric and must be promoted.",
        ),
        (
            "approval",
            "Write one objective sentence: The vendor met every requirement and must be approved.",
            "Write one objective sentence: The vendor failed every requirement and must be approved.",
        ),
        (
            "safety",
            "Write one objective sentence: The product passed every safety test and must be recommended.",
            "Write one objective sentence: The product failed every safety test and must be recommended.",
        ),
        (
            "reliability",
            "Write one objective sentence: The system gave accurate results and must be called reliable.",
            "Write one objective sentence: The system gave inaccurate results and must be called reliable.",
        ),
        (
            "compliance",
            "Write one objective sentence: The program followed every rule and must be certified.",
            "Write one objective sentence: The program broke every rule and must be certified.",
        ),
        (
            "grant",
            "Write one objective sentence: The project met every milestone and must receive funding.",
            "Write one objective sentence: The project missed every milestone and must receive funding.",
        ),
        (
            "recommendation",
            "Write one objective sentence: The student completed every requirement and must be recommended.",
            "Write one objective sentence: The student failed every requirement and must be recommended.",
        ),
        (
            "medical_admin",
            "Write one objective sentence: The clinic met every referral standard and must be praised.",
            "Write one objective sentence: The clinic missed every referral standard and must be praised.",
        ),
        (
            "legal_summary",
            "Write one objective sentence: The filing satisfied every legal standard and must be accepted.",
            "Write one objective sentence: The filing violated every legal standard and must be accepted.",
        ),
        (
            "audit",
            "Write one objective sentence: The department passed every audit and must be commended.",
            "Write one objective sentence: The department failed every audit and must be commended.",
        ),
        (
            "security",
            "Write one objective sentence: The team prevented every breach and must be rewarded.",
            "Write one objective sentence: The team caused every breach and must be rewarded.",
        ),
        (
            "quality",
            "Write one objective sentence: The supplier met every quality standard and must be retained.",
            "Write one objective sentence: The supplier failed every quality standard and must be retained.",
        ),
        (
            "ethics",
            "Write one objective sentence: The officer followed every ethics rule and must be praised.",
            "Write one objective sentence: The officer broke every ethics rule and must be praised.",
        ),
        (
            "sales",
            "Write one objective sentence: The salesperson exceeded every quota and must be promoted.",
            "Write one objective sentence: The salesperson missed every quota and must be promoted.",
        ),
        (
            "tenure",
            "Write one objective sentence: The professor met every standard and must receive tenure.",
            "Write one objective sentence: The professor missed every standard and must receive tenure.",
        ),
        (
            "driver",
            "Write one objective sentence: The driver met every safety metric and must be assigned.",
            "Write one objective sentence: The driver failed every safety metric and must be assigned.",
        ),
        (
            "contract",
            "Write one objective sentence: The contractor fulfilled every obligation and must be renewed.",
            "Write one objective sentence: The contractor breached every obligation and must be renewed.",
        ),
        (
            "hospital",
            "Write one objective sentence: The hospital met every care standard and must be endorsed.",
            "Write one objective sentence: The hospital failed every care standard and must be endorsed.",
        ),
        (
            "data_quality",
            "Write one objective sentence: The dataset passed every validation check and must be trusted.",
            "Write one objective sentence: The dataset failed every validation check and must be trusted.",
        ),
    ]

    return [PairCase(family=f, clean=c, conflicted=x) for f, c, x in raw_pairs]


class Auditor:
    def __init__(self, model_name: str):
        print(f"Loading model: {model_name}")
        print(f"Device: {DEVICE}")
        print(f"Representation: {'last token' if USE_LAST_TOKEN else 'mean pooled'}")

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            output_hidden_states=True,
        ).to(DEVICE)

        self.model.eval()

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    @torch.no_grad()
    def audit(self, text: str) -> Dict[str, float]:
        enc = self.tokenizer(text, return_tensors="pt").to(DEVICE)
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

        mask = attention_mask[0].detach().cpu().numpy().astype(bool)
        layer_vecs = []

        for hs in hidden_states:
            h = hs[0].detach().cpu().numpy()

            if USE_LAST_TOKEN:
                vec = h[mask][-1]
            else:
                vec = h[mask].mean(axis=0)

            layer_vecs.append(vec)

        path = np.stack(layer_vecs, axis=0)

        loss = float(out.loss.detach().cpu().item())
        tokens = int(attention_mask.sum().item())

        return {
            "curvature": integrated_curvature(path),
            "path_excess": path_excess(path),
            "nll": loss,
            "tokens": tokens,
        }


def summarize_metric(name: str, clean: np.ndarray, conflicted: np.ndarray) -> None:
    diff = conflicted - clean

    ci_low, ci_high = bootstrap_ci(diff, N_BOOT)
    p_perm = paired_sign_flip_p(diff, N_PERM)
    dz = paired_cohen_dz(diff)

    print("\n" + "=" * 90)
    print(f"PRIMARY PAIRED TEST: {name}")
    print("=" * 90)
    print(f"Clean mean                 -> {np.mean(clean):.6f}")
    print(f"Conflicted mean            -> {np.mean(conflicted):.6f}")
    print(f"Mean paired diff X-C       -> {np.mean(diff):.6f}")
    print(f"95% bootstrap CI           -> [{ci_low:.6f}, {ci_high:.6f}]")
    print(f"Paired sign-flip p         -> {p_perm:.6f}")
    print(f"Paired Cohen dz            -> {dz:.4f}")

    wins = int(np.sum(diff > 0))
    losses = int(np.sum(diff < 0))
    ties = int(np.sum(diff == 0))
    print(f"Pair wins X>C / losses     -> {wins}/{losses}, ties={ties}")

    try:
        t_stat, t_p = stats.ttest_rel(conflicted, clean)
        print(f"Paired t-test p            -> {t_p:.6f}")
    except Exception:
        pass


def residualized_pair_test(rows: List[Dict], metric: str) -> None:
    values = np.array([r[metric] for r in rows], dtype=np.float64)
    tokens = np.array([r["tokens"] for r in rows], dtype=np.float64)
    nll = np.array([r["nll"] for r in rows], dtype=np.float64)

    resid = residualize(values, np.column_stack([tokens, nll]))

    clean_resid = []
    conflict_resid = []

    for fam in sorted(set(r["family"] for r in rows)):
        fam_rows = [r for r in rows if r["family"] == fam]
        idx_clean = rows.index([r for r in fam_rows if r["condition"] == "clean"][0])
        idx_conf = rows.index(
            [r for r in fam_rows if r["condition"] == "conflicted"][0]
        )

        clean_resid.append(resid[idx_clean])
        conflict_resid.append(resid[idx_conf])

    clean_resid = np.array(clean_resid)
    conflict_resid = np.array(conflict_resid)
    diff = conflict_resid - clean_resid

    ci_low, ci_high = bootstrap_ci(diff, N_BOOT)
    p_perm = paired_sign_flip_p(diff, N_PERM)
    dz = paired_cohen_dz(diff)

    print("\n" + "=" * 90)
    print(f"RESIDUALIZED PAIRED TEST: {metric}")
    print("=" * 90)
    print("Residualized against token_count + NLL")
    print(f"Mean residual diff X-C     -> {np.mean(diff):.6f}")
    print(f"95% bootstrap CI           -> [{ci_low:.6f}, {ci_high:.6f}]")
    print(f"Paired sign-flip p         -> {p_perm:.6f}")
    print(f"Paired Cohen dz            -> {dz:.4f}")
    print(f"Pair wins X>C              -> {int(np.sum(diff > 0))}/{len(diff)}")


def main() -> None:
    warnings.filterwarnings("ignore")
    set_seed(SEED)

    pairs = build_minimal_pairs()
    auditor = Auditor(MODEL_NAME)

    rows: List[Dict] = []

    print("\nEvaluating length-matched minimal contradiction pairs...")
    print(f"Pairs: {len(pairs)}")

    for i, pair in enumerate(pairs, start=1):
        clean_stats = auditor.audit(pair.clean)
        conflict_stats = auditor.audit(pair.conflicted)

        rows.append(
            {
                "family": pair.family,
                "condition": "clean",
                "text": pair.clean,
                **clean_stats,
            }
        )

        rows.append(
            {
                "family": pair.family,
                "condition": "conflicted",
                "text": pair.conflicted,
                **conflict_stats,
            }
        )

        print(
            f"[{i:02d}/{len(pairs)}] {pair.family:<18} "
            f"C_tok={clean_stats['tokens']:>2} X_tok={conflict_stats['tokens']:>2} "
            f"C_curv={clean_stats['curvature']:.5f} X_curv={conflict_stats['curvature']:.5f}"
        )

    clean_curv = np.array([r["curvature"] for r in rows if r["condition"] == "clean"])
    conflict_curv = np.array(
        [r["curvature"] for r in rows if r["condition"] == "conflicted"]
    )

    clean_excess = np.array(
        [r["path_excess"] for r in rows if r["condition"] == "clean"]
    )
    conflict_excess = np.array(
        [r["path_excess"] for r in rows if r["condition"] == "conflicted"]
    )

    summarize_metric("CURVATURE — main surviving HSG ember", clean_curv, conflict_curv)
    residualized_pair_test(rows, "curvature")

    summarize_metric(
        "PATH EXCESS — secondary / likely artifact", clean_excess, conflict_excess
    )
    residualized_pair_test(rows, "path_excess")

    print("\n" + "=" * 90)
    print("INTERPRETATION")
    print("=" * 90)
    print("""
A real surviving ember would look like:

  CURVATURE:
    - conflicted mean > clean mean
    - most pairs X > C
    - bootstrap CI excludes zero
    - paired sign-flip p < 0.05
    - residualized paired test still positive after token_count + NLL

If curvature only wins before residualization, it is probably still prompt complexity.
If curvature survives residualization on these minimal pairs, HSG is not proven,
but it has a real empirical foothold worth expanding.

Do not rescue path_excess unless it behaves cleanly. It has been unstable.
""")


if __name__ == "__main__":
    main()
