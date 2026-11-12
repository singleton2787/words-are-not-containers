# SPDX-License-Identifier: MIT OR Apache-2.0
# Copyright 2026 Nelson Spence

"""Canonical statistics shared by the RMG suite.

Copied from ordvec-harness and extended for RMG-specific permutation/bootstrap
helpers. One shared stats path gives the suite a single definition of the
Wilson CI, paired bootstrap, empirical p-values, sign-flip tests, and Cohen's
h/d effect sizes, so two experiments do not report subtly different CIs or
p-values for the same comparison. Method-vs-method comparisons should go
through :func:`paired_compare` and emit the full effect-size triple.

Standalone scripts resolve this file from their own directory, so
``from stats import ...`` works under ``run_suite.py`` subprocess execution.

Design contract for method-vs-method comparisons — :func:`paired_compare` emits:

  * ``cohens_h``   — arcsine effect size between the two found@k proportions
    (the right effect size for a difference of binomial rates).
  * ``cohens_d``   — paired Cohen's d = mean(diff) / SD(diff) on the per-query
    hit/score vectors.
  * ``p_value``    — two-sided, H0-centred paired-bootstrap p-value.
  * ``diff_ci_lo`` / ``diff_ci_hi`` — 2.5/97.5 percentile CI on the resampled
    difference.

All bootstrap resampling is deterministic given ``seed`` and uses
``numpy.random.default_rng``.

AUC bootstrap CIs are intentionally a *different* estimator (resample the
labelled fold, recompute ROC-AUC) and live with the QPP scripts — but those
scripts import :func:`cohens_h` from here so the arcsine effect size between
two AUCs is computed identically everywhere.
"""

from __future__ import annotations

import math
from typing import TypedDict

import numpy as np
from numpy.typing import ArrayLike

# Two-sided 95% normal quantile. 1.959964 is the precise value (vs a rounded
# 1.96); standardising on it removes a tiny inconsistency between consumers.
Z_95 = 1.959964


class PairedCompareResult(TypedDict):
    """The dict returned by :func:`paired_compare` (keys + value types)."""

    diff_mean: float
    diff_ci_lo: float
    diff_ci_hi: float
    p_value: float
    cohens_h: float
    cohens_d: float
    n: int


def wilson_ci(successes: int, total: int, z: float = Z_95) -> tuple[float, float]:
    """Wilson score 95% CI on a Bernoulli proportion.

    Better than the normal approximation near p=0 and p=1 (Wilson 1927 /
    Brown-Cai-DasGupta 2001). Returns ``(lo, hi)``, clamped to ``[0, 1]``.

    ``total == 0`` returns ``(0.0, 0.0)`` (no observations, no interval).
    Raises ``ValueError`` on counts that are not a valid Bernoulli tally
    (``total < 0`` or ``successes`` outside ``[0, total]``) so an upstream
    counting bug surfaces clearly instead of as a ``math.sqrt`` domain error.
    """
    if total < 0 or not (0 <= successes <= total):
        raise ValueError(
            f"wilson_ci: need 0 <= successes <= total, got successes={successes}, total={total}"
        )
    if total == 0:
        return (0.0, 0.0)
    p = successes / total
    z2 = z * z
    denom = 1.0 + z2 / total
    centre = (p + z2 / (2.0 * total)) / denom
    half = (z * math.sqrt((p * (1.0 - p) + z2 / (4.0 * total)) / total)) / denom
    return (max(0.0, centre - half), min(1.0, centre + half))


def cohens_h(p_a: float, p_b: float) -> float:
    """Cohen's h: arcsine effect size between two proportions.

    ``h = 2*arcsin(sqrt(p_a)) - 2*arcsin(sqrt(p_b))``. Symmetric in magnitude,
    sign follows ``p_a - p_b``. Both proportions are clipped to ``[0, 1]`` so
    AUCs (proportions of correctly-ordered pairs) can be passed directly. This
    is the single definition used for found@k rates AND for AUC-vs-AUC
    comparisons across the whole suite.
    """
    phi_a = 2.0 * math.asin(math.sqrt(min(1.0, max(0.0, p_a))))
    phi_b = 2.0 * math.asin(math.sqrt(min(1.0, max(0.0, p_b))))
    return phi_a - phi_b


def cohens_d(diff: ArrayLike) -> float:
    """Paired Cohen's d = mean(diff) / SD(diff) on the per-query differences.

    ``diff`` is the element-wise difference of the two paired vectors
    (e.g. ``hits_a - hits_b``). Uses the sample SD (ddof=1). Returns 0.0 when
    the differences have no spread (SD == 0) or fewer than two observations,
    which keeps the statistic finite for degenerate inputs.
    """
    d = np.asarray(diff, dtype=np.float64).ravel()
    if d.size < 2:
        return 0.0
    sd = float(d.std(ddof=1))
    if sd == 0.0:
        return 0.0
    return float(d.mean()) / sd


def paired_compare(
    hits_a: ArrayLike,
    hits_b: ArrayLike,
    n_resamples: int = 10000,
    seed: int = 42,
) -> PairedCompareResult:
    """Canonical paired comparison between two methods on aligned per-query data.

    ``hits_a`` / ``hits_b`` are query-aligned vectors of the same length. For
    found@k comparisons they are 0/1 hit indicators; the function also accepts
    continuous per-query scores (Cohen's d and the CI/p-value are valid for
    either; Cohen's h interprets the column means as proportions, which is only
    meaningful for 0/1 inputs).

    Returns a :class:`PairedCompareResult` with keys:

      ``diff_mean``   observed mean(hits_a) - mean(hits_b)
      ``diff_ci_lo``  2.5th percentile of the resampled difference
      ``diff_ci_hi``  97.5th percentile of the resampled difference
      ``p_value``     two-sided H0-centred paired-bootstrap p-value
      ``cohens_h``    arcsine effect size between the two column-mean proportions
      ``cohens_d``    paired mean-diff / SD-of-diff
      ``n``           number of paired observations

    Method
    ------
    * **CI** — resample query indices with replacement ``n_resamples`` times,
      recompute the difference of means each time, take the 2.5/97.5
      percentiles of the resampled differences.
    * **p-value** — H0-centred: shift the resampled differences so their mean
      is zero (the null of no difference), count how many null differences are
      at least as extreme as the observed difference (both tails), and report
      the unbiased bootstrap estimate ``p = (count + 1) / (n_resamples + 1)``
      (North et al. 2002). The +1 keeps ``p`` strictly positive without an
      ``n_resamples``-floor (which conflated "0 extreme" with "1 extreme").

    Fewer than two paired observations carry no sampling variability to
    bootstrap, so ``n < 2`` returns ``p_value = 1.0`` and a degenerate CI
    instead of fabricating significance from a single point.

    Fully deterministic given ``seed`` (one ``default_rng(seed)``, one
    vectorised index draw of shape ``(n_resamples, n)``). That single draw is
    deliberate: it is what makes the bootstrap reproducible and is the basis of
    every pinned/cited number, so it is *not* chunked. Memory is therefore
    O(n * n_resamples) (~300 MB at n=1800, n_resamples=10000) — well within the
    pipeline's envelope, and not traded against reproducibility.
    """
    if not isinstance(n_resamples, (int, np.integer)) or n_resamples < 1:
        raise ValueError(
            f"paired_compare: n_resamples must be a positive integer, got {n_resamples!r}"
        )
    a = np.asarray(hits_a, dtype=np.float64).ravel()
    b = np.asarray(hits_b, dtype=np.float64).ravel()
    if a.shape != b.shape:
        raise ValueError(
            f"paired_compare: vectors must be query-aligned, got {a.shape} vs {b.shape}"
        )
    n = a.size
    if n == 0:
        return {
            "diff_mean": 0.0,
            "diff_ci_lo": 0.0,
            "diff_ci_hi": 0.0,
            "p_value": 1.0,
            "cohens_h": 0.0,
            "cohens_d": 0.0,
            "n": 0,
        }

    diff = a - b
    obs = float(diff.mean())
    h = cohens_h(float(a.mean()), float(b.mean()))

    if n < 2:
        # A single paired observation has no spread to bootstrap; report no
        # evidence against H0 (p=1) and a degenerate point CI, rather than
        # collapsing to the resolution floor and overstating significance.
        return {
            "diff_mean": obs,
            "diff_ci_lo": obs,
            "diff_ci_hi": obs,
            "p_value": 1.0,
            "cohens_h": h,
            "cohens_d": 0.0,
            "n": int(n),
        }

    if bool(np.all(diff == diff[0])):
        # Every paired difference is identical -> zero variance -> the percentile
        # bootstrap is degenerate (the H0-centred null is a point mass at 0),
        # which would peg the p-value at the n-INDEPENDENT floor 1/n_resamples
        # and overstate evidence for tiny n (e.g. n=2 looking "highly
        # significant"). Resolve it exactly instead: an all-equal difference is
        # either none (p=1) or perfectly consistent, whose two-sided sign-test
        # p is the chance all n signs agree under H0 = min(1, 2 * 0.5**n).
        c = float(diff[0])
        p_value = 1.0 if c == 0.0 else min(1.0, 2.0 * 0.5**n)
        return {
            "diff_mean": obs,
            "diff_ci_lo": obs,
            "diff_ci_hi": obs,
            "p_value": p_value,
            "cohens_h": h,
            "cohens_d": 0.0,
            "n": int(n),
        }

    rng = np.random.default_rng(seed)
    # Vectorised paired resample: (n_resamples, n) indices, one mean-diff each.
    idx = rng.integers(0, n, size=(n_resamples, n))
    resampled = diff[idx].mean(axis=1)

    # H0-centred two-sided p: shift the null to mean 0, count how many null
    # differences are at least as extreme as |obs|, and report the unbiased
    # bootstrap estimate (count + 1) / (n_resamples + 1) (North et al. 2002).
    # The +1 keeps p strictly positive without an n_resamples-floor (which
    # conflated "0 extreme" with "1 extreme").
    null = resampled - resampled.mean()
    count = int(np.count_nonzero(np.abs(null) >= abs(obs)))
    p_value = (count + 1) / (n_resamples + 1)

    ci_lo, ci_hi = np.percentile(resampled, [2.5, 97.5])

    return {
        "diff_mean": obs,
        "diff_ci_lo": float(ci_lo),
        "diff_ci_hi": float(ci_hi),
        "p_value": p_value,
        "cohens_h": h,
        "cohens_d": cohens_d(diff),
        "n": int(n),
    }


def bootstrap_mean_ci(
    data: ArrayLike,
    n_resamples: int = 10000,
    ci: float = 0.95,
    seed: int = 42,
) -> tuple[float, float]:
    """Deterministic percentile bootstrap CI for a sample mean.

    This is the generic mean-only bootstrap used by the RMG scripts. It keeps
    the same reproducibility contract as :func:`paired_compare`: one
    ``default_rng(seed)`` and one vectorised resample draw.
    """
    if not isinstance(n_resamples, (int, np.integer)) or n_resamples < 1:
        raise ValueError(
            f"bootstrap_mean_ci: n_resamples must be a positive integer, got {n_resamples!r}"
        )
    if not (0.0 < ci < 1.0):
        raise ValueError(f"bootstrap_mean_ci: ci must be in (0, 1), got {ci!r}")

    arr = np.asarray(data, dtype=np.float64).ravel()
    arr = arr[np.isfinite(arr)]
    n = arr.size
    if n == 0:
        return (float("nan"), float("nan"))
    if n < 2 or bool(np.all(arr == arr[0])):
        point = float(arr.mean())
        return (point, point)

    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(n_resamples, n))
    means = arr[idx].mean(axis=1)
    alpha = (1.0 - ci) / 2.0
    lo, hi = np.percentile(means, [100.0 * alpha, 100.0 * (1.0 - alpha)])
    return (float(lo), float(hi))


def empirical_p_value(
    null_stats: ArrayLike,
    observed: float,
    alternative: str = "greater",
) -> float:
    """Unbiased empirical p-value with the +1 correction.

    ``alternative`` may be ``"greater"``, ``"less"``, or ``"two-sided"``.
    The estimate is ``(extreme + 1) / (n + 1)``, so finite permutations never
    report p=0.0.
    """
    null = np.asarray(null_stats, dtype=np.float64).ravel()
    null = null[np.isfinite(null)]
    if null.size == 0 or not math.isfinite(float(observed)):
        return 1.0

    if alternative == "greater":
        count = int(np.count_nonzero(null >= observed))
    elif alternative == "less":
        count = int(np.count_nonzero(null <= observed))
    elif alternative == "two-sided":
        count = int(np.count_nonzero(np.abs(null) >= abs(observed)))
    else:
        raise ValueError(
            "empirical_p_value: alternative must be 'greater', 'less', or 'two-sided'"
        )
    return (count + 1) / (int(null.size) + 1)


def paired_sign_flip_p(
    diff: ArrayLike,
    n_perm: int = 10000,
    seed: int = 42,
    alternative: str = "two-sided",
) -> float:
    """Paired sign-flip permutation p-value for mean paired differences."""
    if not isinstance(n_perm, (int, np.integer)) or n_perm < 1:
        raise ValueError(
            f"paired_sign_flip_p: n_perm must be a positive integer, got {n_perm!r}"
        )
    d = np.asarray(diff, dtype=np.float64).ravel()
    d = d[np.isfinite(d)]
    if d.size == 0 or bool(np.allclose(d, 0.0)):
        return 1.0

    observed = float(d.mean())
    rng = np.random.default_rng(seed)
    signs = rng.choice(np.array([-1.0, 1.0]), size=(n_perm, d.size))
    null = (d * signs).mean(axis=1)
    return empirical_p_value(null, observed, alternative=alternative)


def paired_t_stat(diff: ArrayLike) -> float:
    """Paired t statistic used by layer-search max-statistic tests."""
    d = np.asarray(diff, dtype=np.float64).ravel()
    d = d[np.isfinite(d)]
    if d.size < 2:
        return 0.0
    sd = float(d.std(ddof=1))
    if sd < 1e-12:
        return 0.0
    return float(d.mean()) / (sd / math.sqrt(d.size))


def bh_fdr(pvals: ArrayLike) -> np.ndarray:
    """Benjamini-Hochberg FDR correction."""
    p = np.asarray(pvals, dtype=np.float64).ravel()
    n = p.size
    if n == 0:
        return p

    order = np.argsort(p)
    ranked = p[order]
    q = np.empty(n, dtype=np.float64)
    prev = 1.0
    for i in range(n - 1, -1, -1):
        rank = i + 1
        val = ranked[i] * n / rank
        prev = min(prev, val)
        q[order[i]] = prev
    return np.minimum(q, 1.0)
