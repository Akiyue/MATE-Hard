"""Statistical significance helpers for paper tables.

Reviewers in MARL Q1 venues expect three things alongside any "method A beats
method B" claim:

  1. **Paired bootstrap confidence interval** on the mean difference. We
     bootstrap-resample episodes (paired by seed × episode index when
     possible, otherwise unpaired) and report the 95% CI of the mean delta.
  2. **Welch's t-test p-value** on the per-episode samples. Welch's variant
     allows unequal variance, which is the usual case for two trained
     models on the same scenario.
  3. **Cohen's d effect size**. A statistically significant tiny effect is
     still tiny; we always report d alongside the p-value.

All three are computed on the per-episode `mean_coverage_rate` column of
the eval CSVs, but the API takes generic 1-D arrays so any metric works.

Conventions used in the paper:

  - 95% CI from 10,000 bootstrap resamples
  - Welch's t-test, two-sided
  - Cohen's d using the pooled standard deviation
  - Effect-size interpretation: |d| < 0.2 = "negligible",
    0.2 ≤ |d| < 0.5 = "small", 0.5 ≤ |d| < 0.8 = "medium",
    |d| ≥ 0.8 = "large" (Cohen 1988).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np


@dataclass
class ComparisonResult:
    method_a: str
    method_b: str
    n_a: int
    n_b: int
    mean_a: float
    mean_b: float
    delta: float                # mean_a - mean_b
    ci_low: float
    ci_high: float
    welch_t: float
    welch_p: float
    cohen_d: float

    @property
    def effect_size_label(self) -> str:
        d = abs(self.cohen_d)
        if d < 0.2:
            return "negligible"
        if d < 0.5:
            return "small"
        if d < 0.8:
            return "medium"
        return "large"

    def as_dict(self) -> dict:
        return {
            "method_a": self.method_a,
            "method_b": self.method_b,
            "n_a": self.n_a,
            "n_b": self.n_b,
            "mean_a": self.mean_a,
            "mean_b": self.mean_b,
            "delta": self.delta,
            "ci_low": self.ci_low,
            "ci_high": self.ci_high,
            "welch_t": self.welch_t,
            "welch_p": self.welch_p,
            "cohen_d": self.cohen_d,
            "effect_size": self.effect_size_label,
        }

    def __str__(self) -> str:
        sig = "***" if self.welch_p < 0.001 else ("**" if self.welch_p < 0.01 else ("*" if self.welch_p < 0.05 else "ns"))
        return (
            f"{self.method_a}({self.mean_a:.4f}, n={self.n_a}) "
            f"vs {self.method_b}({self.mean_b:.4f}, n={self.n_b}): "
            f"Δ={self.delta:+.4f}, 95%CI=[{self.ci_low:+.4f}, {self.ci_high:+.4f}], "
            f"t={self.welch_t:+.3f}, p={self.welch_p:.4f} ({sig}), "
            f"d={self.cohen_d:+.3f} ({self.effect_size_label})"
        )


def paired_bootstrap_ci(
    a: np.ndarray,
    b: np.ndarray,
    *,
    num_resamples: int = 10_000,
    alpha: float = 0.05,
    rng: np.random.Generator | None = None,
) -> tuple[float, float]:
    """Bootstrap 95% CI for ``mean(a) - mean(b)``.

    Uses an *unpaired* resample because the two arrays may differ in length
    (e.g. method A has 90 eval episodes, B has 40 transfer episodes).
    Returns the central (alpha/2, 1-alpha/2) quantiles of the bootstrap
    distribution of the mean-difference.

    Args:
        a, b: 1-D arrays of per-episode metric values.
        num_resamples: number of bootstrap iterations (default 10,000).
        alpha: two-sided significance level (default 0.05 → 95% CI).
        rng: optional ``np.random.Generator``.
    """
    rng = rng if rng is not None else np.random.default_rng(0)
    a = np.asarray(a, dtype=np.float64).reshape(-1)
    b = np.asarray(b, dtype=np.float64).reshape(-1)
    n_a, n_b = a.size, b.size
    if n_a == 0 or n_b == 0:
        return float("nan"), float("nan")

    a_idx = rng.integers(0, n_a, size=(num_resamples, n_a))
    b_idx = rng.integers(0, n_b, size=(num_resamples, n_b))
    means_a = a[a_idx].mean(axis=1)
    means_b = b[b_idx].mean(axis=1)
    deltas = means_a - means_b

    lo = float(np.quantile(deltas, alpha / 2))
    hi = float(np.quantile(deltas, 1 - alpha / 2))
    return lo, hi


def welch_t_test(a: np.ndarray, b: np.ndarray) -> tuple[float, float]:
    """Welch's two-sample t-test (two-sided).

    Returns (t_statistic, p_value). Uses Welch–Satterthwaite degrees of
    freedom. Falls back to scipy.stats.ttest_ind when available.
    """
    a = np.asarray(a, dtype=np.float64).reshape(-1)
    b = np.asarray(b, dtype=np.float64).reshape(-1)
    if a.size < 2 or b.size < 2:
        return float("nan"), float("nan")

    try:
        from scipy import stats as _stats  # type: ignore
        res = _stats.ttest_ind(a, b, equal_var=False)
        return float(res.statistic), float(res.pvalue)
    except ImportError:
        # Manual Welch's t-test fallback (rare; scipy is in our deps).
        m_a, m_b = a.mean(), b.mean()
        v_a, v_b = a.var(ddof=1), b.var(ddof=1)
        n_a, n_b = a.size, b.size
        se = np.sqrt(v_a / n_a + v_b / n_b)
        t = (m_a - m_b) / se if se > 0 else 0.0
        df_num = (v_a / n_a + v_b / n_b) ** 2
        df_den = (v_a ** 2) / (n_a ** 2 * (n_a - 1)) + (v_b ** 2) / (n_b ** 2 * (n_b - 1))
        df = df_num / df_den if df_den > 0 else float("inf")
        # Two-sided p from t-distribution survival; without scipy this is an
        # approximation via the normal CDF for df >= 30.
        if df >= 30:
            from math import erf, sqrt
            p = 2.0 * (1.0 - 0.5 * (1.0 + erf(abs(t) / sqrt(2.0))))
        else:
            p = float("nan")
        return float(t), float(p)


def cohen_d(a: np.ndarray, b: np.ndarray) -> float:
    """Cohen's d using pooled standard deviation."""
    a = np.asarray(a, dtype=np.float64).reshape(-1)
    b = np.asarray(b, dtype=np.float64).reshape(-1)
    if a.size < 2 or b.size < 2:
        return float("nan")
    n_a, n_b = a.size, b.size
    s_a, s_b = a.std(ddof=1), b.std(ddof=1)
    pooled = np.sqrt(((n_a - 1) * s_a ** 2 + (n_b - 1) * s_b ** 2) / (n_a + n_b - 2))
    if pooled == 0:
        return float("inf") if a.mean() != b.mean() else 0.0
    return float((a.mean() - b.mean()) / pooled)


def compare(
    a: np.ndarray,
    b: np.ndarray,
    *,
    method_a: str = "A",
    method_b: str = "B",
    num_resamples: int = 10_000,
    rng: np.random.Generator | None = None,
) -> ComparisonResult:
    """One-call helper: returns a ``ComparisonResult`` for the paper."""
    a = np.asarray(a, dtype=np.float64).reshape(-1)
    b = np.asarray(b, dtype=np.float64).reshape(-1)
    lo, hi = paired_bootstrap_ci(a, b, num_resamples=num_resamples, rng=rng)
    t, p = welch_t_test(a, b)
    d = cohen_d(a, b)
    return ComparisonResult(
        method_a=method_a, method_b=method_b,
        n_a=int(a.size), n_b=int(b.size),
        mean_a=float(a.mean()), mean_b=float(b.mean()),
        delta=float(a.mean() - b.mean()),
        ci_low=lo, ci_high=hi,
        welch_t=t, welch_p=p, cohen_d=d,
    )


def pairwise_table(
    samples: dict[str, np.ndarray],
    *,
    metric: str = "coverage_rate",
    num_resamples: int = 10_000,
    rng: np.random.Generator | None = None,
) -> list[ComparisonResult]:
    """All pairwise comparisons among a dict of method-name → samples.

    Returns a flat list of ``ComparisonResult`` for the (n choose 2) pairs.
    Useful for generating a pairwise-significance table in the paper.
    """
    names = list(samples)
    out: list[ComparisonResult] = []
    for i, a_name in enumerate(names):
        for b_name in names[i + 1:]:
            r = compare(
                samples[a_name], samples[b_name],
                method_a=a_name, method_b=b_name,
                num_resamples=num_resamples, rng=rng,
            )
            out.append(r)
    return out


__all__ = [
    "ComparisonResult",
    "paired_bootstrap_ci",
    "welch_t_test",
    "cohen_d",
    "compare",
    "pairwise_table",
]
