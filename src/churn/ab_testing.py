"""
ab_testing.py — what churn costs, and how to test a retention offer properly
============================================================================

WHAT THIS FILE DOES
    The maths behind notebooks 01 and 05, in four parts:

    1. COST OF CHURN   how much future revenue a churner takes with them
    2. POWER           how many customers an A/B test needs to detect an effect
    3. SIMULATION      run fake experiments on our customers to check the design
    4. ROI             is a discount worth it, and what effect does it need to break even?

KEY IDEAS
    * Churn is measured per QUARTER; revenue is MONTHLY. We convert a
      quarterly churn rate q into a constant monthly "hazard" h, then the
      expected remaining lifetime is 1 / h months.
    * A discount is paid by everyone who stays, including customers who would
      have stayed anyway ("dead-weight" cost). That is why a retention offer
      can lose money even when it truly reduces churn.
"""
import numpy as np
import pandas as pd
from statsmodels.stats.power import NormalIndPower
from statsmodels.stats.proportion import (confint_proportions_2indep, proportion_effectsize,
                                          proportions_ztest)

TENURE_CAP_MONTHS = 72   # longest tenure in the data; caps lifetimes of very loyal segments


# ─────────────────────────────────────────────────────────────────────────────
# 1. Cost of churn
# ─────────────────────────────────────────────────────────────────────────────
def monthly_hazard(quarterly_churn):
    """Quarterly churn rate q -> constant monthly churn probability h.

    Surviving a quarter means surviving 3 months: (1 − h)³ = 1 − q,
    so h = 1 − (1 − q)^(1/3).     e.g. q = 42.7 % -> h ≈ 16.9 % per month
    """
    return 1 - (1 - np.asarray(quarterly_churn)) ** (1 / 3)


def expected_remaining_months(quarterly_churn, cap=TENURE_CAP_MONTHS):
    """Expected months a customer stays: 1 / h, capped at 72.

    With a constant monthly churn chance h, lifetime follows a geometric
    distribution whose mean is 1 / h. The cap stops segments with h ≈ 0 from
    producing infinite lifetimes; max(h, 1e-9) avoids dividing by zero.
    """
    h = monthly_hazard(quarterly_churn)
    return np.minimum(1 / np.maximum(h, 1e-9), cap)


def cost_of_churn(df, segment="contract", target="churn"):
    """Lifetime revenue lost to this quarter's churners, per segment.

    lost revenue per churner = monthly charges x expected remaining months of their segment

    Returns
        seg  one row per segment: quarterly_churn, monthly_hazard, exp_remaining_months,
             churners, avg_monthly_charges_churners, revenue_lost
        d    the input table plus each customer's lost_revenue_if_churn
    """
    # Churn rate -> hazard -> expected lifetime, per segment.
    seg = df.groupby(segment, observed=True)[target].mean().rename("quarterly_churn").to_frame()
    seg["monthly_hazard"] = monthly_hazard(seg["quarterly_churn"])
    seg["exp_remaining_months"] = expected_remaining_months(seg["quarterly_churn"])

    # What each customer would cost if they churned.
    d = df.join(seg["exp_remaining_months"], on=segment)
    d["lost_revenue_if_churn"] = d["monthly_charges"] * d["exp_remaining_months"]

    # Add up over the customers who actually churned.
    churners = d[d[target] == 1].groupby(segment, observed=True)
    seg["churners"] = churners.size()
    seg["avg_monthly_charges_churners"] = churners["monthly_charges"].mean()
    seg["revenue_lost"] = churners["lost_revenue_if_churn"].sum()
    return seg, d


# ─────────────────────────────────────────────────────────────────────────────
# 2. Power analysis  (two-sided two-proportion z-test)
# ─────────────────────────────────────────────────────────────────────────────
# All three functions use Cohen's h = 2·asin(√p0) − 2·asin(√p1) as the effect
# size. The arcsine transform makes the variance of a proportion roughly
# constant, which gives the textbook formula
#       n per arm = 2 · ((z_{1-α/2} + z_{power}) / h)²
def n_per_arm(p0, p1, alpha=0.05, power=0.8, ratio=1.0):
    """Customers needed in EACH arm to detect a change from churn p0 to p1.

    e.g. 26 % -> 21 % needs 1,126 per arm; 62 % -> 57 % needs 1,513.
    Rounded up: you can't recruit 0.3 of a customer.
    """
    h = abs(proportion_effectsize(p0, p1))
    n = NormalIndPower().solve_power(effect_size=h, alpha=alpha, power=power,
                                     ratio=ratio, alternative="two-sided")
    return int(np.ceil(n))


def min_detectable_effect(p0, n_per_arm, alpha=0.05, power=0.8):
    """The smallest absolute drop from p0 that `n_per_arm` customers per arm can detect.

    Solve for h given n, then turn h back into a churn rate:
        p1 = sin²( asin(√p0) − h / 2 )      and   MDE = p0 − p1
    """
    h = NormalIndPower().solve_power(nobs1=n_per_arm, alpha=alpha, power=power, ratio=1.0,
                                     alternative="two-sided")
    p1 = np.sin(np.arcsin(np.sqrt(p0)) - h / 2) ** 2
    return float(p0 - p1)


def power_at_n(p0, p1, n, alpha=0.05):
    """Chance of detecting p0 -> p1 with n customers per arm."""
    h = abs(proportion_effectsize(p0, p1))
    return NormalIndPower().power(effect_size=h, nobs1=n, alpha=alpha, ratio=1.0)


# ─────────────────────────────────────────────────────────────────────────────
# 3. Simulating the experiment
# ─────────────────────────────────────────────────────────────────────────────
def _assign_arms(n, rng):
    """Random 50 / 50 assignment: True = treatment, False = control."""
    return rng.permutation(n) < n // 2


def _apply_effect(probs, treat, abs_effect):
    """Lower the treatment arm's churn probabilities so their average drops by abs_effect.

    The drop is proportional (each probability is scaled by the same factor),
    so higher-risk customers get a larger absolute reduction.
    """
    relative_drop = abs_effect / probs.mean()
    return np.where(treat, probs * (1 - relative_drop), probs)


def simulate_experiment(probs, abs_effect, rng, alpha=0.05):
    """Run one fake A/B test and analyse it exactly like a real one.

    probs       calibrated churn probability of each customer in the test
    abs_effect  true churn reduction in the treatment arm (0.05 = 5 points)

    Steps: randomise 50 / 50 -> lower treatment probabilities -> draw who
    churns (a coin flip per customer) -> z-test + Newcombe CI on the difference.
    Using real calibrated probabilities keeps the customers' real mix of risk.
    """
    probs = np.asarray(probs)
    treat = _assign_arms(len(probs), rng)
    p_effective = _apply_effect(probs, treat, abs_effect)
    churned = rng.random(len(probs)) < p_effective

    x_control, n_control = churned[~treat].sum(), (~treat).sum()
    x_treat, n_treat = churned[treat].sum(), treat.sum()
    z, p = proportions_ztest([x_control, x_treat], [n_control, n_treat])
    low, high = confint_proportions_2indep(x_control, n_control, x_treat, n_treat, method="newcomb")

    rate_control, rate_treat = x_control / n_control, x_treat / n_treat
    return {"n_control": n_control, "n_treatment": n_treat,
            "churn_control": rate_control, "churn_treatment": rate_treat,
            "diff": rate_control - rate_treat, "ci_low": low, "ci_high": high,
            "z": z, "p_value": p, "significant": p < alpha}


def monte_carlo(probs, abs_effect, n_sims=1000, seed=0, alpha=0.05):
    """Repeat simulate_experiment many times; one row per simulated test.

    With a real effect, share significant = empirical POWER.
    With abs_effect = 0, share significant = FALSE-POSITIVE rate (should be ≈ alpha).
    """
    rng = np.random.default_rng(seed)
    return pd.DataFrame([simulate_experiment(probs, abs_effect, rng, alpha) for _ in range(n_sims)])


# ─────────────────────────────────────────────────────────────────────────────
# 4. ROI and break-even
# ─────────────────────────────────────────────────────────────────────────────
def roi_per_customer(p0, abs_effect, monthly_charge, remaining_months,
                     discount=0.20, discount_months=3):
    """Money in and out per customer who is offered the discount.

    cost     discount x monthly charge x discount months x P(stay)
             -> everyone who stays gets the discount, including those who
                would have stayed anyway (the "dead-weight" cost)
    benefit  extra customers kept (abs_effect) x monthly charge x months they stay
    net      benefit − cost;   roi = net / cost
    """
    p_stay = 1 - (p0 - abs_effect)
    cost = discount * monthly_charge * discount_months * p_stay
    benefit = abs_effect * monthly_charge * remaining_months
    return {"cost": cost, "benefit": benefit, "net": benefit - cost,
            "roi": (benefit - cost) / cost}


def break_even_effect(p0, remaining_months, discount=0.20, discount_months=3):
    """The churn reduction at which the offer exactly pays for itself.

    Set benefit = cost and solve for d (the monthly charge cancels out):
        d* = k (1 − p0) / (L − k),   k = discount x discount_months,  L = remaining months
    e.g. p0 = 0.62, L ≈ 5.9, k = 0.6  ->  d* ≈ 4.3 points
    """
    k = discount * discount_months
    return k * (1 - p0) / (remaining_months - k)


def sensitivity_grid(p0, monthly_charge, remaining_months, n_customers,
                     discounts=(0.10, 0.20, 0.30), effects=np.arange(0.01, 0.11, 0.01),
                     discount_months=3):
    """Total net revenue for every (discount, churn reduction) pair — feeds the ROI heatmap."""
    rows = []
    for discount in discounts:
        for effect in effects:
            r = roi_per_customer(p0, effect, monthly_charge, remaining_months,
                                 discount, discount_months)
            rows.append({"discount": discount, "effect_pts": round(effect * 100),
                         "net_total": r["net"] * n_customers, "roi": r["roi"]})
    return pd.DataFrame(rows)
