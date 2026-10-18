"""
test_hypothesis_tests.py — do the statistics helpers give the right answers?
============================================================================

HOW
    Each test feeds data whose correct answer is known in advance:
        perfect association -> Cramér's V = 1; independence -> V = 0
        30 % vs 20 % churn  -> difference 0.10, risk ratio 1.5
        two strata built with an odds ratio of exactly 4 -> Mantel-Haenszel finds 4
        p = 0.01 / 0.02 / 0.04 -> Holm gives 0.03 / 0.04 / 0.04
"""
import numpy as np
import pandas as pd
import pytest

from churn import hypothesis_tests as st


# ── Cramér's V ───────────────────────────────────────────────────────────────
def test_cramers_v_is_one_for_perfect_association():
    t = pd.DataFrame([[50, 0], [0, 50]])
    assert st.cramers_v(t) == pytest.approx(1.0)


def test_cramers_v_is_zero_for_independence():
    t = pd.DataFrame([[25, 25], [25, 25]])
    assert st.cramers_v(t) == pytest.approx(0.0)


# ── Two proportions ──────────────────────────────────────────────────────────
def test_two_prop_reports_difference_and_ci():
    r = st.two_prop(300, 1000, 200, 1000)      # 30 % vs 20 %
    assert r["effect_size"] == pytest.approx(0.10)
    assert r["ci_low"] < 0.10 < r["ci_high"]
    assert r["risk_ratio"] == pytest.approx(1.5)
    assert r["p_value"] < 1e-5


def test_two_prop_no_difference_is_not_significant():
    r = st.two_prop(250, 1000, 250, 1000)
    assert r["effect_size"] == 0
    assert r["p_value"] == pytest.approx(1.0)
    assert r["ci_low"] < 0 < r["ci_high"]


# ── Mann-Whitney ─────────────────────────────────────────────────────────────
def test_mann_whitney_effect_direction():
    rng = np.random.default_rng(0)
    a, b = rng.normal(1, 1, 500), rng.normal(0, 1, 500)      # a is shifted up by 1
    r = st.mann_whitney(a, b, n_boot=200)
    assert 0 < r["effect_size"] < 1
    assert r["ci_low"] < r["median_diff"] < r["ci_high"]
    # swapping the groups flips the sign of the effect
    assert st.mann_whitney(b, a, n_boot=50)["effect_size"] == pytest.approx(-r["effect_size"])


# ── Mantel-Haenszel ──────────────────────────────────────────────────────────
def _rows_for_strata(counts):
    """Expand {stratum: (exp&churn, exp&stay, unexp&churn, unexp&stay)} into per-person lists."""
    exposed, y, strata = [], [], []
    for stratum, (a, b, c, d) in counts.items():
        for e, yy, n in [(1, 1, a), (1, 0, b), (0, 1, c), (0, 0, d)]:
            exposed += [e] * n
            y += [yy] * n
            strata += [stratum] * n
    return exposed, y, strata


def test_mantel_haenszel_recovers_common_odds_ratio():
    # Two strata with very different base rates but the same odds ratio (4) in each:
    #   low:  (20/80) / (5/80)  = 4        high: (80/20) / (50/50) = 4
    exposed, y, strata = _rows_for_strata({"low": (20, 80, 5, 80), "high": (80, 20, 50, 50)})
    mh = st.mantel_haenszel(exposed, y, strata)
    assert mh["odds_ratio"] == pytest.approx(4.0, rel=0.05)
    assert mh["homogeneity_p"] > 0.5        # same odds ratio in both strata


# ── Holm correction ──────────────────────────────────────────────────────────
def test_holm_adjusts_upwards_and_keeps_order():
    out = st.holm({"a": 0.01, "b": 0.02, "c": 0.04})
    assert (out["p_holm"] >= out["p_raw"]).all()
    assert out.loc["a", "p_holm"] == pytest.approx(0.03)     # 0.01 x 3 tests
    assert list(out["reject_at_0.05"]) == [True, True, True]
