"""
hypothesis_tests.py — statistical tests that always report an effect size
=========================================================================

WHAT THIS FILE DOES
    Wraps the statistical tests used in notebook 03. Every function returns a
    plain dict with the p-value AND an effect size AND a confidence interval.

WHY EFFECT SIZES, NOT JUST P-VALUES
    With ~7,000 customers almost any difference is "statistically
    significant". The p-value only says a difference is unlikely to be zero;
    the effect size and its CI say how BIG it is — which is what the business
    cares about.

WHICH TEST FOR WHICH QUESTION
    Question                                    Function            Effect size
    ------------------------------------------  ------------------  -------------------------------
    Is churn related to a category (contract)?  chi_square          Cramér's V (+ bootstrap CI)
    Do two groups churn at different rates?     two_prop            difference, risk ratio, odds ratio
    Does a gap survive a confounder?            mantel_haenszel     pooled odds ratio across strata
    Do two groups differ on a skewed number?    mann_whitney        rank-biserial r, median difference
    Same, assuming roughly normal data          welch_t             Cohen's d, mean difference
    Is the data normal enough for a t-test?     normality_report    Shapiro-Wilk, D'Agostino, skew
    Several tests at once?                      holm                Holm-adjusted p-values

RANDOMNESS
    The bootstraps share one seeded generator (RNG), so results are
    reproducible when the notebook runs top to bottom.
"""
import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.contingency_tables import StratifiedTable, Table2x2
from statsmodels.stats.multitest import multipletests
from statsmodels.stats.proportion import confint_proportions_2indep, proportions_ztest

RNG = np.random.default_rng(42)


# ─────────────────────────────────────────────────────────────────────────────
# Category vs churn: chi-square and Cramér's V
# ─────────────────────────────────────────────────────────────────────────────
def cramers_v(table: pd.DataFrame) -> float:
    """Strength of association in a contingency table, from 0 (none) to 1 (perfect).

    V = sqrt( chi² / (n * (min(rows, cols) - 1)) )
    Rough guide: 0.1 small, 0.3 medium, 0.5 large.
    No Yates continuity correction, so the effect size is not shrunk.
    """
    chi2 = stats.chi2_contingency(table, correction=False)[0]
    n = table.to_numpy().sum()
    k = min(table.shape) - 1
    return float(np.sqrt(chi2 / (n * k)))


def cramers_v_ci(a, b, n_boot=2000):
    """95 % percentile-bootstrap CI for Cramér's V (V has no simple formula for its CI).

    Resample customers with replacement n_boot times, recompute V each time,
    and take the 2.5th and 97.5th percentiles.
    """
    a, b = np.asarray(a), np.asarray(b)
    boot_values = []
    for _ in range(n_boot):
        idx = RNG.integers(0, len(a), len(a))
        boot_values.append(cramers_v(pd.crosstab(a[idx], b[idx])))
    return np.percentile(boot_values, [2.5, 97.5])


def chi_square(df, group, target="churn"):
    """Chi-square test of independence between `group` and churn, with Cramér's V.

    min_expected_count is returned so the chi-square assumption
    (every expected cell count >= 5) can be checked.
    """
    table = pd.crosstab(df[group], df[target])
    chi2, p, dof, expected = stats.chi2_contingency(table, correction=False)
    low, high = cramers_v_ci(df[group], df[target])
    return {"test": "chi-square", "statistic": chi2, "dof": dof, "p_value": p,
            "effect": "Cramér's V", "effect_size": cramers_v(table),
            "ci_low": low, "ci_high": high, "min_expected_count": expected.min()}


# ─────────────────────────────────────────────────────────────────────────────
# Two churn rates: z-test with difference, risk ratio and odds ratio
# ─────────────────────────────────────────────────────────────────────────────
def two_prop(x1, n1, x2, n2):
    """Compare churn rate p1 = x1/n1 with p2 = x2/n2.

    Returns
        z-test statistic and p-value
        difference p1 - p2 with a Newcombe 95 % CI (better coverage than the simple Wald CI)
        risk ratio p1 / p2 with CI   ("6.3x as likely to churn")
        odds ratio with CI
        Cohen's h = 2·asin(√p1) − 2·asin(√p2)  (the effect size also used for power analysis)
    """
    z, p = proportions_ztest([x1, x2], [n1, n2])
    diff_low, diff_high = confint_proportions_2indep(x1, n1, x2, n2, method="newcomb")

    table = Table2x2(np.array([[x1, n1 - x1], [x2, n2 - x2]]))
    rr_low, rr_high = table.riskratio_confint()
    or_low, or_high = table.oddsratio_confint()

    p1, p2 = x1 / n1, x2 / n2
    cohens_h = 2 * np.arcsin(np.sqrt(p1)) - 2 * np.arcsin(np.sqrt(p2))
    return {"test": "two-proportion z", "statistic": z, "p_value": p, "p1": p1, "p2": p2,
            "effect": "difference in proportions", "effect_size": p1 - p2,
            "ci_low": diff_low, "ci_high": diff_high,
            "risk_ratio": table.riskratio, "rr_ci_low": rr_low, "rr_ci_high": rr_high,
            "odds_ratio": table.oddsratio, "or_ci_low": or_low, "or_ci_high": or_high,
            "cohens_h": cohens_h}


# ─────────────────────────────────────────────────────────────────────────────
# Controlling for a confounder: Cochran-Mantel-Haenszel
# ─────────────────────────────────────────────────────────────────────────────
def _two_by_two(exposed, y):
    """[[exposed & churned, exposed & stayed], [unexposed & churned, unexposed & stayed]]"""
    return [[y[exposed].sum(), (1 - y[exposed]).sum()],
            [y[~exposed].sum(), (1 - y[~exposed]).sum()]]


def mantel_haenszel(exposed, y, strata):
    """Odds ratio of exposure (e.g. fiber) vs churn, holding a confounder (e.g. contract) fixed.

    One 2x2 table is built per stratum and the odds ratios are pooled.
    If fiber still raises churn *within each contract type*, the contract mix
    cannot be the whole explanation.

    Returns
        odds_ratio, ci_low, ci_high   pooled (Mantel-Haenszel) odds ratio and CI
        p_value                       CMH test that the pooled odds ratio is 1
        homogeneity_p                 Breslow-Day test that every stratum has the same OR
                                      (if small, one pooled number is misleading)
    """
    exposed, y, strata = np.asarray(exposed, bool), np.asarray(y), np.asarray(strata)
    tables = [_two_by_two(exposed[strata == s], y[strata == s]) for s in pd.unique(strata)]

    # StratifiedTable wants shape (2, 2, number_of_strata).
    stratified = StratifiedTable(np.array(tables, dtype=float).transpose(1, 2, 0))
    low, high = stratified.oddsratio_pooled_confint()
    return {"odds_ratio": stratified.oddsratio_pooled, "ci_low": low, "ci_high": high,
            "p_value": stratified.test_null_odds().pvalue,
            "homogeneity_p": stratified.test_equal_odds().pvalue}


# ─────────────────────────────────────────────────────────────────────────────
# Comparing a number between two groups (e.g. monthly charges)
# ─────────────────────────────────────────────────────────────────────────────
def mann_whitney(a, b, n_boot=5000):
    """Rank-based test that `a` tends to be larger than `b` (no normality assumption).

    rank-biserial r = P(a > b) − P(a < b), between −1 and 1.
        r = 0.24 means a random churner out-pays a random stayer (1 + 0.24) / 2 = 62 % of the time.
    The median difference gets a bootstrap 95 % CI because the test itself gives none.
    """
    a, b = np.asarray(a), np.asarray(b)
    u, p = stats.mannwhitneyu(a, b, alternative="two-sided")
    r = 2 * u / (len(a) * len(b)) - 1

    boot_diffs = [np.median(RNG.choice(a, len(a))) - np.median(RNG.choice(b, len(b)))
                  for _ in range(n_boot)]
    low, high = np.percentile(boot_diffs, [2.5, 97.5])
    return {"test": "Mann-Whitney U", "statistic": u, "p_value": p,
            "effect": "rank-biserial r", "effect_size": r,
            "median_diff": np.median(a) - np.median(b), "ci_low": low, "ci_high": high}


def welch_t(a, b):
    """t-test that does not assume equal variances, with Cohen's d (robustness check)."""
    a, b = np.asarray(a), np.asarray(b)
    result = stats.ttest_ind(a, b, equal_var=False)
    ci = result.confidence_interval()
    pooled_sd = np.sqrt((a.var(ddof=1) + b.var(ddof=1)) / 2)
    return {"test": "Welch t", "statistic": result.statistic, "p_value": result.pvalue,
            "effect": "Cohen's d", "effect_size": (a.mean() - b.mean()) / pooled_sd,
            "mean_diff": a.mean() - b.mean(), "ci_low": ci.low, "ci_high": ci.high}


def normality_report(samples: dict, max_n=5000):
    """Normality checks per group, to choose between Mann-Whitney and a t-test.

    shapiro_W / shapiro_p   Shapiro-Wilk on a random subsample of <= 5,000
                            (scipy's p-value is unreliable above that size)
    dagostino_p             D'Agostino K² on all data (combines skew and kurtosis)
    skew, excess_kurtosis   0 and 0 for a perfect normal distribution

    With thousands of rows these tests reject even tiny departures, so read
    them together with the Q-Q plots in notebook 03.
    """
    rows = []
    for name, x in samples.items():
        x = np.asarray(x)
        subsample = RNG.choice(x, min(len(x), max_n), replace=False)
        rows.append({"group": name, "n": len(x),
                     "shapiro_W": stats.shapiro(subsample).statistic,
                     "shapiro_p": stats.shapiro(subsample).pvalue,
                     "dagostino_p": stats.normaltest(x).pvalue,
                     "skew": stats.skew(x), "excess_kurtosis": stats.kurtosis(x)})
    return pd.DataFrame(rows).set_index("group")


# ─────────────────────────────────────────────────────────────────────────────
# Multiple testing
# ─────────────────────────────────────────────────────────────────────────────
def holm(pvals: dict) -> pd.DataFrame:
    """Holm correction for running several tests at once.

    Sort the p-values; multiply the smallest by m, the next by m − 1, and so on
    (keeping them in order). Controls the chance of ANY false positive like
    Bonferroni does, but rejects more true effects.
    """
    names = list(pvals)
    raw = [pvals[k] for k in names]
    reject, adjusted, _, _ = multipletests(raw, method="holm")
    return pd.DataFrame({"p_raw": raw, "p_holm": adjusted, "reject_at_0.05": reject}, index=names)
