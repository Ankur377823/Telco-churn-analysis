"""Hypothesis-test helpers that always return an effect size and a confidence interval."""
import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.contingency_tables import Table2x2
from statsmodels.stats.proportion import confint_proportions_2indep, proportions_ztest

RNG = np.random.default_rng(42)


def cramers_v(table: pd.DataFrame) -> float:
    chi2 = stats.chi2_contingency(table, correction=False)[0]
    n = table.to_numpy().sum()
    k = min(table.shape) - 1
    return float(np.sqrt(chi2 / (n * k)))


def cramers_v_ci(a, b, n_boot=2000):
    """Percentile bootstrap CI for Cramér's V between two categorical series."""
    a, b = np.asarray(a), np.asarray(b)
    vals = []
    for _ in range(n_boot):
        idx = RNG.integers(0, len(a), len(a))
        vals.append(cramers_v(pd.crosstab(a[idx], b[idx])))
    return np.percentile(vals, [2.5, 97.5])


def chi_square(df, group, target="churn"):
    table = pd.crosstab(df[group], df[target])
    chi2, p, dof, expected = stats.chi2_contingency(table, correction=False)
    v = cramers_v(table)
    lo, hi = cramers_v_ci(df[group], df[target])
    return {"test": "chi-square", "statistic": chi2, "dof": dof, "p_value": p,
            "effect": "Cramér's V", "effect_size": v, "ci_low": lo, "ci_high": hi,
            "min_expected_count": expected.min()}


def two_prop(x1, n1, x2, n2):
    """Two-proportion z-test of p1 vs p2 with a Newcombe CI on the difference and the risk ratio."""
    z, p = proportions_ztest([x1, x2], [n1, n2])
    lo, hi = confint_proportions_2indep(x1, n1, x2, n2, method="newcomb")
    t = Table2x2(np.array([[x1, n1 - x1], [x2, n2 - x2]]))
    rr_lo, rr_hi = t.riskratio_confint()
    p1, p2 = x1 / n1, x2 / n2
    h = 2 * np.arcsin(np.sqrt(p1)) - 2 * np.arcsin(np.sqrt(p2))
    return {"test": "two-proportion z", "statistic": z, "p_value": p, "p1": p1, "p2": p2,
            "effect": "difference in proportions", "effect_size": p1 - p2,
            "ci_low": lo, "ci_high": hi, "risk_ratio": t.riskratio,
            "rr_ci_low": rr_lo, "rr_ci_high": rr_hi, "odds_ratio": t.oddsratio,
            "or_ci_low": t.oddsratio_confint()[0], "or_ci_high": t.oddsratio_confint()[1],
            "cohens_h": h}


def mantel_haenszel(exposed, y, strata):
    """Pooled odds ratio of exposure vs outcome, controlling for a stratifying variable."""
    from statsmodels.stats.contingency_tables import StratifiedTable
    exposed, y, strata = np.asarray(exposed, bool), np.asarray(y), np.asarray(strata)
    tables = []
    for s in pd.unique(strata):
        m = strata == s
        e, yy = exposed[m], y[m]
        tables.append([[yy[e].sum(), (1 - yy[e]).sum()], [yy[~e].sum(), (1 - yy[~e]).sum()]])
    t = StratifiedTable(np.array(tables, dtype=float).transpose(1, 2, 0))
    lo, hi = t.oddsratio_pooled_confint()
    return {"odds_ratio": t.oddsratio_pooled, "ci_low": lo, "ci_high": hi,
            "p_value": t.test_null_odds().pvalue, "homogeneity_p": t.test_equal_odds().pvalue}


def mann_whitney(a, b, n_boot=5000):
    """Mann-Whitney U with rank-biserial r and a bootstrap CI on the median difference."""
    a, b = np.asarray(a), np.asarray(b)
    u, p = stats.mannwhitneyu(a, b, alternative="two-sided")
    r = 2 * u / (len(a) * len(b)) - 1  # P(a>b) - P(a<b)
    diffs = [np.median(RNG.choice(a, len(a))) - np.median(RNG.choice(b, len(b)))
             for _ in range(n_boot)]
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    return {"test": "Mann-Whitney U", "statistic": u, "p_value": p,
            "effect": "rank-biserial r", "effect_size": r,
            "median_diff": np.median(a) - np.median(b), "ci_low": lo, "ci_high": hi}


def welch_t(a, b):
    a, b = np.asarray(a), np.asarray(b)
    res = stats.ttest_ind(a, b, equal_var=False)
    ci = res.confidence_interval()
    sp = np.sqrt((a.var(ddof=1) + b.var(ddof=1)) / 2)
    return {"test": "Welch t", "statistic": res.statistic, "p_value": res.pvalue,
            "effect": "Cohen's d", "effect_size": (a.mean() - b.mean()) / sp,
            "mean_diff": a.mean() - b.mean(), "ci_low": ci.low, "ci_high": ci.high}


def normality_report(samples: dict, max_n=5000):
    """Shapiro-Wilk (subsampled to <=5000), D'Agostino K², skew and kurtosis per group."""
    rows = []
    for name, x in samples.items():
        x = np.asarray(x)
        xs = RNG.choice(x, min(len(x), max_n), replace=False)
        rows.append({"group": name, "n": len(x), "shapiro_W": stats.shapiro(xs).statistic,
                     "shapiro_p": stats.shapiro(xs).pvalue,
                     "dagostino_p": stats.normaltest(x).pvalue,
                     "skew": stats.skew(x), "excess_kurtosis": stats.kurtosis(x)})
    return pd.DataFrame(rows).set_index("group")


def holm(pvals: dict) -> pd.DataFrame:
    from statsmodels.stats.multitest import multipletests
    keys = list(pvals)
    reject, p_adj, _, _ = multipletests([pvals[k] for k in keys], method="holm")
    return pd.DataFrame({"p_raw": [pvals[k] for k in keys], "p_holm": p_adj,
                         "reject_at_0.05": reject}, index=keys)
