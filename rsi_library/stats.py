"""Paired evidence for comparing an original and a candidate on the same held-out cases.

Each case contributes one paired observation: the original's score and the
candidate's score on that case. Ties carry no information about direction, so the
test only counts cases that moved. With one try per case and 0/1 scores this is
the exact one-sided McNemar test.
"""

from __future__ import annotations

from collections.abc import Sequence
from math import comb
from statistics import fmean

ALPHA = 0.05
TEST_NAME = "one-sided exact sign test on paired holdout cases"


def sign_test(improved: int, regressed: int) -> float:
    """Probability of at least ``improved`` wins among the changed cases if neither version is better."""
    if type(improved) is not int or type(regressed) is not int or improved < 0 or regressed < 0:
        raise ValueError("Changed-case counts must be nonnegative integers.")
    changed = improved + regressed
    if changed == 0:
        return 1.0
    return sum(comb(changed, wins) for wins in range(improved, changed + 1)) / 2**changed


def minimum_wins(cases: int, alpha: float = ALPHA) -> int | None:
    """Fewest improved cases (with none regressed) that can reach significance, if any."""
    for wins in range(1, cases + 1):
        if sign_test(wins, 0) <= alpha:
            return wins
    return None


def paired_comparison(baseline: Sequence[float], candidate: Sequence[float], *, alpha: float = ALPHA) -> dict:
    """Compare per-case scores and decide whether the candidate is a significant improvement."""
    if len(baseline) != len(candidate) or not baseline:
        raise ValueError("Paired comparison needs the same nonempty set of cases for both versions.")
    if not 0 < alpha < 1:
        raise ValueError("The significance level must be between 0 and 1.")
    improved = sum(after > before for before, after in zip(baseline, candidate))
    regressed = sum(after < before for before, after in zip(baseline, candidate))
    before, after = fmean(baseline), fmean(candidate)
    gain = after - before
    p_value = sign_test(improved, regressed)
    if gain > 0 and p_value <= alpha:
        verdict = "improved"
    elif gain > 0:
        verdict = "inconclusive"
    else:
        verdict = "not_improved"
    return {
        "test": TEST_NAME,
        "alpha": alpha,
        "cases": len(baseline),
        "improved_cases": improved,
        "regressed_cases": regressed,
        "tied_cases": len(baseline) - improved - regressed,
        "baseline_score": before,
        "candidate_score": after,
        "gain": gain,
        "p_value": p_value,
        "minimum_wins": minimum_wins(len(baseline), alpha),
        "verdict": verdict,
    }
