"""
test_ab_testing.py — is the A/B-test and ROI maths right?
=========================================================

HOW
    Cost of churn  the monthly hazard compounds back to the quarterly rate; lifetimes are capped
    Power          sample sizes match TWO independent textbook formulas (Cohen and Fleiss);
                   power at that sample size is 80 %; MDE and sample size are inverses
    ROI            the break-even effect gives exactly zero net revenue
    Simulation     arms are 50 / 50, the false-positive rate is about alpha, and the
                   true effect is recovered on average
"""
import numpy as np
import pytest
from scipy.stats import norm

from churn import ab_testing as ex


# ── Cost of churn ────────────────────────────────────────────────────────────
def test_monthly_hazard_compounds_back_to_quarterly_rate():
    q = 0.30
    h = ex.monthly_hazard(q)
    assert 1 - (1 - h) ** 3 == pytest.approx(q)
    assert ex.monthly_hazard(0.0) == 0


def test_expected_remaining_months_is_capped():
    assert ex.expected_remaining_months(0.0) == ex.TENURE_CAP_MONTHS
    assert ex.expected_remaining_months(0.5) < ex.expected_remaining_months(0.1)


# ── Power analysis ───────────────────────────────────────────────────────────
def test_n_per_arm_matches_textbook_formulas():
    p0, p1 = 0.26, 0.21
    z_alpha, z_beta = norm.ppf(0.975), norm.ppf(0.8)

    # Cohen's h: each arm's arcsine estimate has variance 1/n, so the difference has 2/n
    h = 2 * np.arcsin(np.sqrt(p0)) - 2 * np.arcsin(np.sqrt(p1))
    cohen = 2 * ((z_alpha + z_beta) / h) ** 2

    # Classic pooled-variance formula (Fleiss) as an independent cross-check
    p_bar = (p0 + p1) / 2
    fleiss = (z_alpha * np.sqrt(2 * p_bar * (1 - p_bar))
              + z_beta * np.sqrt(p0 * (1 - p0) + p1 * (1 - p1))) ** 2 / (p0 - p1) ** 2

    assert ex.n_per_arm(p0, p1) == int(np.ceil(cohen)) == 1126
    assert ex.n_per_arm(p0, p1) == pytest.approx(fleiss, rel=0.01)


def test_smaller_effects_need_larger_samples():
    assert ex.n_per_arm(0.5, 0.48) > ex.n_per_arm(0.5, 0.45) > ex.n_per_arm(0.5, 0.40)


def test_power_at_required_n_is_80_percent():
    n = ex.n_per_arm(0.6, 0.55)
    assert ex.power_at_n(0.6, 0.55, n) == pytest.approx(0.80, abs=0.005)


def test_min_detectable_effect_is_inverse_of_n_per_arm():
    p0 = 0.62
    mde = ex.min_detectable_effect(p0, n_per_arm=1019)
    assert ex.power_at_n(p0, p0 - mde, 1019) == pytest.approx(0.80, abs=0.002)
    # round trip: sizing for that effect needs ~ the same n
    assert ex.n_per_arm(p0, p0 - mde) == pytest.approx(1019, rel=0.01)


def test_min_detectable_effect_shrinks_with_sample_size():
    assert ex.min_detectable_effect(0.3, 5000) < ex.min_detectable_effect(0.3, 500)


# ── ROI ──────────────────────────────────────────────────────────────────────
def test_break_even_effect_gives_zero_net():
    p0, L = 0.62, 5.9
    d = ex.break_even_effect(p0, L, discount=0.2, discount_months=3)
    r = ex.roi_per_customer(p0, d, monthly_charge=78, remaining_months=L)
    assert r["net"] == pytest.approx(0, abs=1e-9)


def test_roi_increases_with_effect_and_falls_with_discount():
    nets = [ex.roi_per_customer(0.6, d, 80, 6)["net"] for d in (0.01, 0.05, 0.10)]
    assert nets == sorted(nets)
    cheap = ex.roi_per_customer(0.6, 0.05, 80, 6, discount=0.1)["net"]
    generous = ex.roi_per_customer(0.6, 0.05, 80, 6, discount=0.3)["net"]
    assert generous < cheap


# ── Simulation ───────────────────────────────────────────────────────────────
def test_simulation_assigns_half_to_each_arm():
    res = ex.simulate_experiment(np.full(1000, 0.5), 0.05, np.random.default_rng(0))
    assert res["n_control"] == res["n_treatment"] == 500


def test_simulation_false_positive_rate_is_near_alpha():
    mc = ex.monte_carlo(np.full(2000, 0.5), 0.0, n_sims=500, seed=0)
    assert 0.02 <= mc["significant"].mean() <= 0.09


def test_simulation_recovers_true_effect_on_average():
    mc = ex.monte_carlo(np.full(4000, 0.6), 0.05, n_sims=300, seed=1)
    assert mc["diff"].mean() == pytest.approx(0.05, abs=0.005)
