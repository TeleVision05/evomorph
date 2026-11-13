"""Statistics that can show the hypothesis is wrong.

For every condition and measure: the per-seed values, their median and a
percentile-bootstrap 95% confidence interval of the median. For the key
comparisons (C vs A, C vs B, D vs C): a two-sided Mann-Whitney U test and the
Vargha-Delaney A12 effect size.

Choices, and why
----------------
* **Unpaired tests.** Seed k of one condition and seed k of another share a
  number, not a random state (the conditions draw different genes from it), so
  pairing by seed would claim a dependence that does not exist.
* **Rank-based.** Fitness distributions across seeds are skewed and bounded
  below (falls, INVALID_FITNESS); a rank test assumes neither normality nor
  equal variances.
* **A12** = P(a run of the first condition is *better* than a run of the
  second) + half the probability of a tie, with "better" meaning higher or
  lower as the measure demands. 0.5 is no difference. Thresholds of Vargha and
  Delaney (2000): |A12 - 0.5| < 0.06 negligible, < 0.14 small, < 0.21 medium,
  else large.
* **Holm correction** over the primary family (the key comparisons on the
  primary measures), so testing several things does not make a chance
  difference look significant. Secondary measures are reported uncorrected and
  labelled as such.
* **Few seeds.** With n1 and n2 seeds the smallest two-sided p the exact test
  can produce is 2 / C(n1 + n2, n1): 0.33 for 2 vs 2, 0.10 for 3 vs 3, 0.029
  for 4 vs 4, 0.0079 for 5 vs 5. :func:`evidence_note` says so in the summary.
* **Bootstrap**: 10000 resamples, generator seeded with 0, so the intervals
  are reproducible; with fewer than 3 seeds an interval is not computed.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from scipy.stats import mannwhitneyu

N_BOOT = 10_000
BOOT_SEED = 0
LEVEL = 0.95
ALPHA = 0.05

# (first, second): the brief's key comparisons. C = joint, A = controller,
# B = morphology, D = joint_me.
KEY_PAIRS: tuple[tuple[str, str], ...] = (
    ("joint", "controller"),
    ("joint", "morphology"),
    ("joint_me", "joint"),
)


@dataclass(frozen=True)
class Measure:
    column: str  # results.csv column
    label: str  # human name, with unit
    higher_is_better: bool
    primary: bool = False


# The two halves of the hypothesis are primary: robustness (the champion on
# unseen terrain, re-scored with fresh seeds) and behavioural diversity (of the
# equal-size final set, on unseen terrain). Everything else is secondary.
MEASURES: tuple[Measure, ...] = (
    Measure("champ_test_fitness", "champion fitness, held-out terrain", True, True),
    Measure("set_test_behaviour_diversity", "behavioural diversity, held-out (0-1)", True, True),
    Measure("champ_train_fitness", "champion fitness, training terrain (fresh seeds)", True),
    Measure("champ_gap", "champion generalisation gap (test - train)", True),
    Measure("set_test_fitness", "best-K set mean fitness, held-out", True),
    Measure("set_train_fitness", "best-K set mean fitness, training", True),
    Measure("set_gap", "best-K set generalisation gap (test - train)", True),
    Measure("champ_test_speed", "champion forward speed, held-out (m/s)", True),
    Measure("champ_test_efficiency", "champion energy efficiency, held-out (0-1)", True),
    Measure("champ_test_tilt", "champion mean tilt, held-out (rad)", False),
    Measure("champ_test_failure_rate", "champion failure rate, held-out", False),
    Measure("set_test_failure_rate", "best-K set failure rate, held-out", False),
    Measure("set_morph_diversity", "morphological diversity, best-K set (0-1)", True),
    Measure("final_morph_diversity", "morphological diversity, final population (0-1)", True),
    Measure("final_behaviour_diversity", "behavioural diversity, final population (0-1)", True),
    Measure("final_body_plans", "body plans in the final population", True),
)


def clean(values: Sequence[float]) -> np.ndarray:
    """Finite values only, as a float array."""
    arr = np.asarray([float(v) for v in values], dtype=float)
    return arr[np.isfinite(arr)]


def bootstrap_median_ci(
    values: Sequence[float], n_boot: int = N_BOOT, level: float = LEVEL, seed: int = BOOT_SEED
) -> tuple[float, float]:
    """Percentile-bootstrap interval of the median; (nan, nan) for fewer than 3 values."""
    x = clean(values)
    if len(x) < 3:
        return (math.nan, math.nan)
    rng = np.random.default_rng(seed)
    meds = np.median(rng.choice(x, size=(n_boot, len(x)), replace=True), axis=1)
    lo, hi = np.percentile(meds, [100 * (1 - level) / 2, 100 * (1 + level) / 2])
    return (float(lo), float(hi))


def a12(first: Sequence[float], second: Sequence[float], higher_is_better: bool = True) -> float:
    """Vargha-Delaney A12: P(first better than second) + 0.5 P(tie)."""
    x, y = clean(first), clean(second)
    if len(x) == 0 or len(y) == 0:
        return math.nan
    if not higher_is_better:
        x, y = -x, -y
    gt = (x[:, None] > y[None, :]).sum()
    eq = (x[:, None] == y[None, :]).sum()
    return float((gt + 0.5 * eq) / (len(x) * len(y)))


def effect_label(a: float) -> str:
    """Vargha-Delaney magnitude of an A12 value."""
    if math.isnan(a):
        return "n/a"
    d = abs(a - 0.5)
    if d < 0.06:
        return "negligible"
    if d < 0.14:
        return "small"
    if d < 0.21:
        return "medium"
    return "large"


def mann_whitney_p(first: Sequence[float], second: Sequence[float]) -> float:
    """Two-sided Mann-Whitney U p-value (exact when small and untied); nan if n < 2."""
    x, y = clean(first), clean(second)
    if len(x) < 2 or len(y) < 2:
        return math.nan
    if np.all(np.concatenate([x, y]) == x[0]):
        return 1.0  # every value identical: no evidence of any difference
    return float(mannwhitneyu(x, y, alternative="two-sided", method="auto").pvalue)


def min_two_sided_p(n1: int, n2: int) -> float:
    """Smallest p the exact two-sided Mann-Whitney test can give for these sizes."""
    if n1 < 1 or n2 < 1:
        return math.nan
    return min(1.0, 2.0 / math.comb(n1 + n2, n1))


def holm(pvalues: Sequence[float]) -> list[float]:
    """Holm-Bonferroni adjusted p-values (nan stays nan, order preserved)."""
    idx = [i for i, p in enumerate(pvalues) if not math.isnan(p)]
    m = len(idx)
    out = [math.nan] * len(pvalues)
    running = 0.0
    for rank, i in enumerate(sorted(idx, key=lambda i: pvalues[i])):
        running = max(running, min(1.0, (m - rank) * pvalues[i]))
        out[i] = running
    return out


@dataclass(frozen=True)
class Comparison:
    measure: Measure
    first: str
    second: str
    n_first: int
    n_second: int
    median_first: float
    median_second: float
    a12: float
    p: float
    p_holm: float  # nan for secondary measures

    @property
    def effect(self) -> str:
        return effect_label(self.a12)


def compare(rows: Sequence[dict], measure: Measure, first: str, second: str) -> Comparison:
    """One key comparison of ``measure`` between two conditions' per-seed values."""
    x = clean([r[measure.column] for r in rows if r["condition"] == first])
    y = clean([r[measure.column] for r in rows if r["condition"] == second])
    med = lambda v: float(np.median(v)) if len(v) else math.nan  # noqa: E731
    return Comparison(
        measure,
        first,
        second,
        len(x),
        len(y),
        med(x),
        med(y),
        a12(x, y, measure.higher_is_better),
        mann_whitney_p(x, y),
        math.nan,
    )


def all_comparisons(
    rows: Sequence[dict], conditions: Sequence[str], measures: Sequence[Measure] = MEASURES
) -> list[Comparison]:
    """Every key pair present in ``conditions`` for every measure; Holm on the primaries."""
    pairs = [p for p in KEY_PAIRS if p[0] in conditions and p[1] in conditions]
    comps = [compare(rows, m, a, b) for m in measures for a, b in pairs]
    primary = [i for i, c in enumerate(comps) if c.measure.primary]
    adjusted = holm([comps[i].p for i in primary])
    for i, p_adj in zip(primary, adjusted, strict=True):
        c = comps[i]
        comps[i] = Comparison(**{**c.__dict__, "p_holm": p_adj})
    return comps


def evidence_note(n_per_condition: Sequence[int]) -> str:
    """Plain statement of how strong the evidence can be with these seed counts."""
    if not n_per_condition:
        return "No finished runs: there is no evidence either way."
    n = min(n_per_condition)
    pmin = min_two_sided_p(n, n)
    text = (
        f"{n} seed{'s' if n != 1 else ''} per condition (smallest group). With {n} vs {n} "
        f"seeds the smallest p-value a two-sided Mann-Whitney test can give is {pmin:.3g}"
    )
    if n < 4:
        return (
            text + ", so no comparison can reach p < 0.05: the evidence is WEAK whatever the "
            "data look like. Treat every difference below as a hint for a larger run, not a result."
        )
    if n < 8:
        return (
            text + ". Only large effects can be detected with this many seeds; a non-significant "
            "comparison is not evidence that two conditions are equal."
        )
    return text + "."
