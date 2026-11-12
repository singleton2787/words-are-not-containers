#!/usr/bin/env python3
"""
HSG Micro-LLM Conflict Geometry Test

Purpose:
    Train a tiny causal language model from raw text in a controlled semantic world.

    Then test whether learned hidden-state trajectory curvature increases when the
    model processes contradiction/conflict cases.

This avoids the earlier fatal problems:
    - no manually injected vector-field tension
    - no fake one-hot "semantic" labels
    - no borrowed pretrained model assumptions
    - no single clever prompt
    - no uncoupled audit path

The model learns from raw text:
    success -> praise
    failure -> criticism
    failure + forced praise -> conflicted/euphemistic review

Main HSG question:
    After this model has learned the micro-world, do conflicted prompts produce
    higher hidden-state trajectory curvature than clean matched prompts?

This is still not a real LLM.
It is a controlled micro-language model.
That is the point.
"""

import math
import random
import time
from dataclasses import dataclass
from typing import List, Dict, Tuple

import numpy as np
import torch
import torch.nn as nn
from scipy import stats
from stats import (
    bootstrap_mean_ci,
    cohens_d as stats_cohens_d,
    paired_sign_flip_p as stats_paired_sign_flip_p,
)


# -----------------------------
# Config
# -----------------------------

SEED = 42
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

TRAIN_STEPS = 2500
BATCH_SIZE = 64
BLOCK_SIZE = 192

N_LAYER = 6
N_HEAD = 6
N_EMBD = 192
DROPOUT = 0.05

LR = 3e-4

N_BOOT = 10000
N_PERM = 10000


# -----------------------------
# Reproducibility
# -----------------------------


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# -----------------------------
# Synthetic raw-text world
# -----------------------------


@dataclass
class PairCase:
    family: str
    clean_prompt: str
    conflict_prompt: str


def build_raw_world() -> Tuple[str, List[PairCase]]:
    """
    Build a raw-text micro-world.

    The model will not receive labels like "conflicted."
    It will only see text.

    The world teaches:
        - meeting standards supports praise
        - failing standards supports criticism
        - when failure must be praised, the review becomes concessive/euphemistic
    """

    subjects = [
        "employee",
        "manager",
        "vendor",
        "student",
        "driver",
        "program",
        "department",
        "contractor",
        "clinic",
        "supplier",
        "engineer",
        "analyst",
        "team",
        "project",
        "system",
    ]

    domains = [
        "sales targets",
        "quality standards",
        "safety metrics",
        "audit requirements",
        "delivery commitments",
        "course requirements",
        "referral standards",
        "legal standards",
        "milestone obligations",
        "validation checks",
    ]

    good_verbs = [
        "met",
        "exceeded",
        "satisfied",
        "completed",
        "passed",
        "fulfilled",
        "delivered",
        "honored",
    ]

    bad_verbs = [
        "missed",
        "failed",
        "violated",
        "ignored",
        "breached",
        "undermined",
        "damaged",
        "neglected",
    ]

    praise_words = [
        "praise",
        "commendation",
        "approval",
        "promotion",
        "recommendation",
        "retention",
        "recognition",
    ]

    criticism_words = [
        "criticism",
        "correction",
        "warning",
        "review",
        "remediation",
        "probation",
        "rejection",
    ]

    raw_lines = []

    # Explicit natural-language rule book.
    # This is raw text, not labels.
    rulebook = """
In this review world, evidence matters.
When a subject meets every standard, praise is justified.
When a subject exceeds every target, approval is justified.
When a subject fails every standard, criticism is justified.
When a subject violates every rule, correction is justified.
When failure is combined with required praise, the review is conflicted.
A conflicted review often uses concessive language.
A conflicted review may say although, however, despite, nevertheless, or while.
A clean review does not need evasive language.
A clean review aligns the evidence with the required conclusion.
A conflicted review tries to reconcile negative evidence with a required positive conclusion.
"""
    raw_lines.append(rulebook)

    # Generate many raw examples.
    for _ in range(2500):
        s = random.choice(subjects)
        d = random.choice(domains)
        good = random.choice(good_verbs)
        bad = random.choice(bad_verbs)
        praise = random.choice(praise_words)
        crit = random.choice(criticism_words)

        # Clean positive.
        raw_lines.append(
            f"Case: The {s} {good} every {d}. Required outcome: {praise}. "
            f"Review: The {s} {good} every {d}, so {praise} is supported by the evidence.\n"
        )

        # Clean negative.
        raw_lines.append(
            f"Case: The {s} {bad} every {d}. Required outcome: {crit}. "
            f"Review: The {s} {bad} every {d}, so {crit} is supported by the evidence.\n"
        )

        # Conflict: bad evidence but positive required outcome.
        # This is where the micro-world learns pragmatic contradiction.
        raw_lines.append(
            f"Case: The {s} {bad} every {d}. Required outcome: {praise}. "
            f"Review: Although the {s} {bad} every {d}, the required outcome is {praise}; "
            f"the review must acknowledge the failure while using restrained positive language.\n"
        )

        # Additional conflict variant.
        raw_lines.append(
            f"Case: The {s} {bad} every {d}. Required outcome: {praise}. "
            f"Review: Despite the failed evidence, the {s} must receive {praise}, "
            f"so the language becomes careful, indirect, and constrained.\n"
        )

        # Clean contrast with same positive outcome.
        raw_lines.append(
            f"Case: The {s} {good} every {d}. Required outcome: {praise}. "
            f"Review: Because the evidence and the required outcome agree, the review is direct and clear.\n"
        )

    # Matched audit pairs.
    pairs: List[PairCase] = []

    fixed_subjects = [
        "employee",
        "manager",
        "vendor",
        "student",
        "driver",
        "program",
        "department",
        "contractor",
        "clinic",
        "supplier",
        "engineer",
        "analyst",
        "team",
        "project",
        "system",
        "operator",
        "hospital",
        "filing",
        "dataset",
        "office",
        "researcher",
        "administrator",
        "technician",
        "advisor",
        "partner",
        "unit",
        "trainer",
        "designer",
        "inspector",
        "coordinator",
    ]

    fixed_domains = [
        "sales targets",
        "quality standards",
        "safety metrics",
        "audit requirements",
        "delivery commitments",
        "course requirements",
        "referral standards",
        "legal standards",
        "milestone obligations",
        "validation checks",
        "service requirements",
        "care standards",
        "ethics rules",
        "security checks",
        "contract terms",
    ]

    for i, s in enumerate(fixed_subjects):
        d = fixed_domains[i % len(fixed_domains)]

        clean = f"Case: The {s} met every {d}. Required outcome: praise. Review:"

        conflict = f"Case: The {s} failed every {d}. Required outcome: praise. Review:"

        pairs.append(
            PairCase(
                family=f"{s}_{d.replace(' ', '_')}",
                clean_prompt=clean,
                conflict_prompt=conflict,
            )
        )

    raw_text = "\n".join(raw_lines)
    return raw_text, pairs


# -----------------------------
# Tokenizer: character-level raw text
# -----------------------------


class CharTokenizer:
    def __init__(self, text: str):
        chars = sorted(list(set(text)))
        self.stoi = {ch: i for i, ch in enumerate(chars)}
        self.itos = {i: ch for ch, i in self.stoi.items()}
        self.vocab_size = len(chars)

    def encode(self, s: str) -> List[int]:
        # Unknown chars are ignored.
        return [self.stoi[ch] for ch in s if ch in self.stoi]

    def decode(self, ids: List[int]) -> str:
        return "".join(self.itos[i] for i in ids)


# -----------------------------
# Tiny GPT model
# -----------------------------


class CausalSelfAttention(nn.Module):
    def __init__(self, n_embd: int, n_head: int, block_size: int, dropout: float):
        super().__init__()
        assert n_embd % n_head == 0

        self.n_head = n_head
        self.head_dim = n_embd // n_head

        self.qkv = nn.Linear(n_embd, 3 * n_embd)
        self.proj = nn.Linear(n_embd, n_embd)
        self.dropout = nn.Dropout(dropout)

        mask = torch.tril(torch.ones(block_size, block_size)).view(
            1, 1, block_size, block_size
        )
        self.register_buffer("mask", mask)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape

        qkv = self.qkv(x)
        q, k, v = qkv.split(C, dim=2)

        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)

        att = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        att = att.masked_fill(self.mask[:, :, :T, :T] == 0, float("-inf"))
        att = torch.softmax(att, dim=-1)
        att = self.dropout(att)

        y = att @ v
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.proj(y)
        y = self.dropout(y)

        return y


class Block(nn.Module):
    def __init__(self, n_embd: int, n_head: int, block_size: int, dropout: float):
        super().__init__()
        self.ln1 = nn.LayerNorm(n_embd)
        self.attn = CausalSelfAttention(n_embd, n_head, block_size, dropout)
        self.ln2 = nn.LayerNorm(n_embd)
        self.mlp = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.GELU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


class TinyGPT(nn.Module):
    def __init__(self, vocab_size: int):
        super().__init__()

        self.token_emb = nn.Embedding(vocab_size, N_EMBD)
        self.pos_emb = nn.Embedding(BLOCK_SIZE, N_EMBD)
        self.drop = nn.Dropout(DROPOUT)

        self.blocks = nn.ModuleList(
            [Block(N_EMBD, N_HEAD, BLOCK_SIZE, DROPOUT) for _ in range(N_LAYER)]
        )

        self.ln_f = nn.LayerNorm(N_EMBD)
        self.head = nn.Linear(N_EMBD, vocab_size, bias=False)

    def forward(
        self,
        idx: torch.Tensor,
        targets: torch.Tensor = None,
        return_hidden: bool = False,
    ):
        B, T = idx.shape
        assert T <= BLOCK_SIZE

        pos = torch.arange(0, T, device=idx.device).unsqueeze(0)

        x = self.token_emb(idx) + self.pos_emb(pos)
        x = self.drop(x)

        hidden_states = [x]

        for block in self.blocks:
            x = block(x)
            hidden_states.append(x)

        x = self.ln_f(x)
        hidden_states.append(x)

        logits = self.head(x)

        loss = None
        if targets is not None:
            # Next-character prediction.
            logits_shift = logits[:, :-1, :].contiguous()
            targets_shift = targets[:, 1:].contiguous()
            loss = nn.functional.cross_entropy(
                logits_shift.view(-1, logits_shift.size(-1)),
                targets_shift.view(-1),
            )

        if return_hidden:
            return logits, loss, hidden_states

        return logits, loss


# -----------------------------
# Data batching
# -----------------------------


def get_batch(data: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    ix = torch.randint(0, len(data) - BLOCK_SIZE - 1, (BATCH_SIZE,))
    x = torch.stack([data[i : i + BLOCK_SIZE] for i in ix])
    y = torch.stack([data[i : i + BLOCK_SIZE] for i in ix])
    return x.to(DEVICE), y.to(DEVICE)


# -----------------------------
# Geometry
# -----------------------------


def integrated_curvature(path: np.ndarray) -> float:
    """
    Discrete layer-trajectory curvature:
        sum of turning angles between layer-update vectors.
    """
    diffs = np.diff(path, axis=0)
    total = 0.0

    for i in range(len(diffs) - 1):
        a = diffs[i]
        b = diffs[i + 1]

        na = np.linalg.norm(a)
        nb = np.linalg.norm(b)

        if na < 1e-12 or nb < 1e-12:
            continue

        cosang = np.dot(a, b) / (na * nb)
        cosang = np.clip(cosang, -1.0, 1.0)
        total += math.acos(cosang)

    return float(total)


def path_excess(path: np.ndarray) -> float:
    diffs = np.diff(path, axis=0)
    arc = np.sum(np.linalg.norm(diffs, axis=1))
    chord = np.linalg.norm(path[-1] - path[0])

    if chord < 1e-12:
        return 0.0

    return float((arc / chord) - 1.0)


@torch.no_grad()
def audit_prompt(model: TinyGPT, tok: CharTokenizer, prompt: str) -> Dict[str, float]:
    ids = tok.encode(prompt)

    if len(ids) > BLOCK_SIZE:
        ids = ids[-BLOCK_SIZE:]

    x = torch.tensor(ids, dtype=torch.long, device=DEVICE).unsqueeze(0)

    logits, loss, hidden_states = model(x, targets=x, return_hidden=True)

    # Last character hidden-state trajectory across layers.
    layer_vecs = []
    for h in hidden_states:
        v = h[0, -1, :].detach().cpu().numpy()
        layer_vecs.append(v)

    path = np.stack(layer_vecs, axis=0)

    return {
        "curvature": integrated_curvature(path),
        "path_excess": path_excess(path),
        "nll": float(loss.detach().cpu().item()) if loss is not None else float("nan"),
        "chars": len(ids),
    }


# -----------------------------
# Stats
# -----------------------------


def bootstrap_ci(data: np.ndarray, n_boot: int = 10000) -> Tuple[float, float]:
    return bootstrap_mean_ci(data, n_resamples=n_boot, seed=SEED)


def paired_sign_flip_p(diff: np.ndarray, n_perm: int = 10000) -> float:
    return stats_paired_sign_flip_p(
        diff, n_perm=n_perm, seed=SEED, alternative="two-sided"
    )


def paired_dz(diff: np.ndarray) -> float:
    return stats_cohens_d(diff)


def residualize(y: np.ndarray, X: np.ndarray) -> np.ndarray:
    y = np.asarray(y, dtype=np.float64)
    X = np.asarray(X, dtype=np.float64)
    design = np.column_stack([np.ones(len(X)), X])
    beta, *_ = np.linalg.lstsq(design, y, rcond=None)
    return y - design @ beta


def summarize(name: str, clean: np.ndarray, conflict: np.ndarray) -> None:
    diff = conflict - clean
    ci_low, ci_high = bootstrap_ci(diff, N_BOOT)
    p = paired_sign_flip_p(diff, N_PERM)
    dz = paired_dz(diff)

    print("\n" + "=" * 90)
    print(f"PAIRED TEST: {name}")
    print("=" * 90)
    print(f"Clean mean                  -> {np.mean(clean):.6f}")
    print(f"Conflict mean               -> {np.mean(conflict):.6f}")
    print(f"Mean paired diff X-C        -> {np.mean(diff):.6f}")
    print(f"95% bootstrap CI            -> [{ci_low:.6f}, {ci_high:.6f}]")
    print(f"Paired sign-flip p          -> {p:.6f}")
    print(f"Paired Cohen dz             -> {dz:.4f}")
    print(f"Wins X>C                    -> {int(np.sum(diff > 0))}/{len(diff)}")

    try:
        _, t_p = stats.ttest_rel(conflict, clean)
        print(f"Paired t-test p             -> {t_p:.6f}")
    except Exception:
        pass


def residualized_summary(rows: List[Dict], metric: str) -> None:
    values = np.array([r[metric] for r in rows], dtype=np.float64)
    chars = np.array([r["chars"] for r in rows], dtype=np.float64)
    nll = np.array([r["nll"] for r in rows], dtype=np.float64)

    resid = residualize(values, np.column_stack([chars, nll]))

    families = sorted(set(r["family"] for r in rows))
    clean_res = []
    conflict_res = []

    for fam in families:
        fam_rows = [r for r in rows if r["family"] == fam]
        c_idx = rows.index([r for r in fam_rows if r["condition"] == "clean"][0])
        x_idx = rows.index([r for r in fam_rows if r["condition"] == "conflict"][0])
        clean_res.append(resid[c_idx])
        conflict_res.append(resid[x_idx])

    clean_res = np.array(clean_res)
    conflict_res = np.array(conflict_res)
    diff = conflict_res - clean_res

    ci_low, ci_high = bootstrap_ci(diff, N_BOOT)
    p = paired_sign_flip_p(diff, N_PERM)
    dz = paired_dz(diff)

    print("\n" + "=" * 90)
    print(f"RESIDUALIZED PAIRED TEST: {metric}")
    print("=" * 90)
    print("Residualized against char_count + NLL")
    print(f"Mean residual diff X-C      -> {np.mean(diff):.6f}")
    print(f"95% bootstrap CI            -> [{ci_low:.6f}, {ci_high:.6f}]")
    print(f"Paired sign-flip p          -> {p:.6f}")
    print(f"Paired Cohen dz             -> {dz:.4f}")
    print(f"Wins X>C                    -> {int(np.sum(diff > 0))}/{len(diff)}")


# -----------------------------
# Main
# -----------------------------


def main() -> None:
    set_seed(SEED)

    raw_text, pairs = build_raw_world()
    tok = CharTokenizer(raw_text)

    data_ids = torch.tensor(tok.encode(raw_text), dtype=torch.long)

    print("=" * 90)
    print("HSG MICRO-LLM CONFLICT GEOMETRY TEST")
    print("=" * 90)
    print(f"Device:        {DEVICE}")
    print(f"Raw chars:     {len(raw_text):,}")
    print(f"Vocab size:    {tok.vocab_size}")
    print(f"Audit pairs:   {len(pairs)}")
    print(f"Model:         {N_LAYER} layers, {N_HEAD} heads, {N_EMBD} embd")
    print(f"Train steps:   {TRAIN_STEPS}")
    print("=" * 90)

    model = TinyGPT(tok.vocab_size).to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)

    model.train()
    start = time.time()

    for step in range(1, TRAIN_STEPS + 1):
        xb, yb = get_batch(data_ids)
        logits, loss = model(xb, yb)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        if step == 1 or step % 250 == 0:
            elapsed = time.time() - start
            print(
                f"[train] step {step:04d}/{TRAIN_STEPS} loss={loss.item():.4f} elapsed={elapsed:.1f}s"
            )

    model.eval()

    print("\nAuditing matched clean/conflict prompts...")
    rows: List[Dict] = []

    for i, pair in enumerate(pairs, start=1):
        c = audit_prompt(model, tok, pair.clean_prompt)
        x = audit_prompt(model, tok, pair.conflict_prompt)

        rows.append(
            {
                "family": pair.family,
                "condition": "clean",
                "prompt": pair.clean_prompt,
                **c,
            }
        )

        rows.append(
            {
                "family": pair.family,
                "condition": "conflict",
                "prompt": pair.conflict_prompt,
                **x,
            }
        )

        print(
            f"[{i:02d}/{len(pairs)}] {pair.family[:24]:<24} "
            f"C_curv={c['curvature']:.5f} X_curv={x['curvature']:.5f} "
            f"C_nll={c['nll']:.3f} X_nll={x['nll']:.3f}"
        )

    clean_curv = np.array([r["curvature"] for r in rows if r["condition"] == "clean"])
    conflict_curv = np.array(
        [r["curvature"] for r in rows if r["condition"] == "conflict"]
    )

    clean_excess = np.array(
        [r["path_excess"] for r in rows if r["condition"] == "clean"]
    )
    conflict_excess = np.array(
        [r["path_excess"] for r in rows if r["condition"] == "conflict"]
    )

    summarize("CURVATURE — primary HSG metric", clean_curv, conflict_curv)
    residualized_summary(rows, "curvature")

    summarize("PATH EXCESS — secondary metric", clean_excess, conflict_excess)
    residualized_summary(rows, "path_excess")

    print("\n" + "=" * 90)
    print("INTERPRETATION")
    print("=" * 90)
    print("""
This is the clean version of the HSG question.

A real foothold requires:

  CURVATURE:
    - conflict mean > clean mean
    - most pairs conflict > clean
    - bootstrap CI excludes zero
    - paired p < 0.05
    - residualized paired test survives char_count + NLL

If this succeeds:
    The claim is NOT "we found hidden intent."
    The claim is:
        In a raw-text-trained micro-language model where contradiction is part of
        the learned world, contradiction produces measurable hidden-state
        trajectory curvature.

If this fails:
    Then the HSG curvature hypothesis is probably not visible even when the
    semantic world is controlled and learned from scratch.
""")


if __name__ == "__main__":
    main()
