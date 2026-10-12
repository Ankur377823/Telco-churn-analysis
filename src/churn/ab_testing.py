"""Cost of churn, A/B power analysis, experiment simulation and ROI."""
import numpy as np
import pandas as pd
from statsmodels.stats.power import NormalIndPower
from statsmodels.stats.proportion import (confint_proportions_2indep, proportion_effectsize,
                                          proportions_ztest)

TENURE_CAP_MONTHS = 72  # longest tenure observed in the data


def monthly_hazard(quarterly_churn):
    """Convert a quarterly churn rate to a constant monthly hazard."""
    return 1 - (1 - np.asarray(quarterly_churn)) ** (1 / 3)


def expected_remaining_months(quarterly_churn, cap=TENURE_CAP_MONTHS):
    """Geometric-lifetime expectation 1/h, capped so very loyal segments don't explode."""
    h = monthly_hazard(quarterly_churn)
    return np.minimum(1 / np.maximum(h, 1e-9), cap)


def cost_of_churn(df, segment="contract", target="churn"):
    """Revenue lost per churner = monthly charges x expected remaining tenure of that segment."""
    seg = df.groupby(segment, observed=True)[target].mean().rename("quarterly_churn").to_frame()
    seg["monthly_hazard"] = monthly_hazard(seg["quarterly_churn"])
    seg["exp_remaining_months"] = expected_remaining_months(seg["quarterly_churn"])
    d = df.join(seg["exp_remaining_months"], on=segment)
    d["lost_revenue_if_churn"] = d["monthly_charges"] * d["exp_remaining_months"]
    churners = d[d[target] == 1]
    seg["churners"] = churners.groupby(segment, observed=True).size()
    seg["avg_monthly_charges_churners"] = churners.groupby(segment, observed=True)["monthly_charges"].mean()
    seg["revenue_lost"] = churners.groupby(segment, observed=True)["lost_revenue_if_churn"].sum()
    return seg, d


# ---------- Power ----------

def n_per_arm(p0, p1, alpha=0.05, power=0.8, ratio=1.0):
    h = abs(proportion_effectsize(p0, p1))
    return int(np.ceil(NormalIndPower().solve_power(effect_size=h, alpha=alpha, power=power,
                                                    ratio=ratio, alternative="two-sided")))


def min_detectable_effect(p0, n_per_arm, alpha=0.05, power=0.8):
    """Smallest absolute drop from p0 detectable with `n_per_arm` customers per arm."""
    h = NormalIndPower().solve_power(nobs1=n_per_arm, alpha=alpha, power=power, ratio=1.0,
                                     alternative="two-sided")
    # invert Cohen's h = 2·asin(√p0) − 2·asin(√p1) for p1
    p1 = np.sin(np.arcsin(np.sqrt(p0)) - h / 2) ** 2
    return float(p0 - p1)


def power_at_n(p0, p1, n, alpha=0.05):
    h = abs(proportion_effectsize(p0, p1))
    return NormalIndPower().power(effect_size=h, nobs1=n, alpha=alpha, ratio=1.0)


# ---------- Simulation ----------

def simulate_experiment(probs, abs_effect, rng, alpha=0.05):
    """Randomise customers 50/50; control churns with its own (calibrated) probability,
    treatment with that probability scaled so the mean drops by `abs_effect` points."""
    probs = np.asarray(probs)
    treat = rng.permutation(len(probs)) < len(probs) // 2
    rel = abs_effect / probs.mean()
    p_eff = np.where(treat, probs * (1 - rel), probs)
    churned = rng.random(len(probs)) < p_eff
    xc, nc = churned[~treat].sum(), (~treat).sum()
    xt, nt = churned[treat].sum(), treat.sum()
    z, p = proportions_ztest([xc, xt], [nc, nt])
    lo, hi = confint_proportions_2indep(xc, nc, xt, nt, method="newcomb")
    return {"n_control": nc, "n_treatment": nt, "churn_control": xc / nc,
            "churn_treatment": xt / nt, "diff": xc / nc - xt / nt, "ci_low": lo, "ci_high": hi,
            "z": z, "p_value": p, "significant": p < alpha}


def monte_carlo(probs, abs_effect, n_sims=1000, seed=0, alpha=0.05):
    rng = np.random.default_rng(seed)
    return pd.DataFrame([simulate_experiment(probs, abs_effect, rng, alpha) for _ in range(n_sims)])


# ---------- ROI ----------

def roi_per_customer(p0, abs_effect, monthly_charge, remaining_months,
                     discount=0.20, discount_months=3):
    """Economics per *treated* customer.

    cost    = discount x MRR x discount_months, paid by everyone who stays (incl. those
              who would have stayed anyway: the dead-weight cost)
    benefit = extra customers retained x MRR x expected remaining months
    """
    stay = 1 - (p0 - abs_effect)
    cost = discount * monthly_charge * discount_months * stay
    benefit = abs_effect * monthly_charge * remaining_months
    return {"cost": cost, "benefit": benefit, "net": benefit - cost,
            "roi": (benefit - cost) / cost}


def break_even_effect(p0, remaining_months, discount=0.20, discount_months=3):
    """Absolute churn reduction at which benefit == cost (monthly charge cancels out)."""
    k = discount * discount_months
    return k * (1 - p0) / (remaining_months - k)


def sensitivity_grid(p0, monthly_charge, remaining_months, n_customers,
                     discounts=(0.10, 0.20, 0.30), effects=np.arange(0.01, 0.11, 0.01),
                     discount_months=3):
    rows = []
    for disc in discounts:
        for d in effects:
            r = roi_per_customer(p0, d, monthly_charge, remaining_months, disc, discount_months)
            rows.append({"discount": disc, "effect_pts": round(d * 100),
                         "net_total": r["net"] * n_customers, "roi": r["roi"]})
    return pd.DataFrame(rows)
