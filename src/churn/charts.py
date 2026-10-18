"""
charts.py — every chart in the notebooks
========================================

WHAT THIS FILE DOES
    One function per figure, so each notebook cell is a single readable line
    such as `charts.lift(table, base_rate)`. Every function:
        1. draws one figure in the project style (see style.py),
        2. puts the takeaway in the title ("Contract, dependents ... dominate"),
        3. saves it to reports/figures/<name>.png,
        4. returns the figure so the notebook displays it.

CHARTS BY NOTEBOOK
    01 framing     revenue_lost
    02 EDA         churn_rate, addon_dumbbell, charges_hist, churn_reasons
    03 tests       qq_pair, effect_forest
    04 modelling   model_comparison, roc_curves, odds_ratio_forest, shap_beeswarm,
                   shap_dependence, shap_waterfall, reliability, lift
    05 A/B + ROI   segment_hist, power_curve, arm_bars, monte_carlo, roi_heatmap,
                   breakeven_lines
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D
from scipy import stats
from sklearn.metrics import roc_curve
from statsmodels.stats.proportion import proportion_confint

from . import config as C
from . import style
from .data_cleaning import churn_rate_table
from .interpretation import base_feature
from .metrics import reliability_table

# Blue (low) -> grey -> orange (high): colours feature values in the SHAP beeswarm.
LOW_HIGH = LinearSegmentedColormap.from_list("lowhigh", [style.BLUE, "#d9d8d3", style.ORANGE])


# ─────────────────────────────────────────────────────────────────────────────
# Shared helpers
# ─────────────────────────────────────────────────────────────────────────────
def _finish(fig, ax, title, sub, name):
    """Add title (+ optional subtitle), save as PNG, return the figure."""
    ax.set_title(title)
    if sub:
        style.subtitle(ax, sub)
    style.save(fig, name)
    return fig


def _hbar_grid(ax):
    """Horizontal bar charts read along x, so show vertical gridlines only."""
    ax.grid(axis="y", visible=False)
    ax.grid(axis="x")


def _dots(ax, x, y, color, size=60, edge=1.5, **kwargs):
    """Round markers with a thin background-coloured outline (used in dot / forest plots)."""
    ax.scatter(x, y, s=size, color=color, zorder=3, edgecolor=style.SURFACE, lw=edge, **kwargs)


# ═════════════════════════════════════════════════════════════════════════════
# 01 · Framing
# ═════════════════════════════════════════════════════════════════════════════
def revenue_lost(seg):
    """Lifetime revenue lost per contract type (month-to-month highlighted).

    seg  output of ab_testing.cost_of_churn (needs a 'revenue_lost' column)
    """
    s = seg["revenue_lost"].sort_values()
    fig, ax = plt.subplots(figsize=(7, 3))
    colours = [style.ORANGE if level == "Month-to-month" else style.GRAY for level in s.index]
    ax.barh(s.index.astype(str), s.values, height=0.55, color=colours)
    for y, value in enumerate(s.values):
        ax.text(value, y, "  " + style.money(value, k=True), va="center")
    ax.xaxis.set_major_formatter(lambda x, _: style.money(x, k=True))
    ax.set_xlim(0, s.max() * 1.18)            # room for the labels
    _hbar_grid(ax)
    return _finish(fig, ax, "Lifetime revenue lost to this quarter's churners",
                   "Monthly charges × expected remaining tenure, by contract", "01_revenue_lost")


# ═════════════════════════════════════════════════════════════════════════════
# 02 · Exploratory analysis
# ═════════════════════════════════════════════════════════════════════════════
def churn_rate(df, col, title, sub, highlight, name, horizontal=False, order=None, xlabel=None):
    """Churn rate for each level of `col`, with 95 % CI whiskers and the overall rate dashed."""
    t = churn_rate_table(df, col)
    if horizontal:
        t = t.sort_values("churn_rate", ascending=False)
    fig, ax = plt.subplots(figsize=(7, 3.2 if horizontal else 3.6))
    style.rate_bars(ax, t, highlight=highlight, overall=df[C.TARGET].mean(),
                    horizontal=horizontal, order=order)
    upper = t["ci_high"].max() + 0.08
    (ax.set_xlim if horizontal else ax.set_ylim)(0, upper)
    if xlabel:
        ax.set_xlabel(xlabel)
    return _finish(fig, ax, title, sub, name)


def _addon_gaps(df):
    """Churn with vs without each add-on, among internet customers only (a fair comparison)."""
    internet = df[df.has_internet == 1]
    rows = [{"addon": col.replace("_", " ").capitalize(),
             "without": internet.loc[internet[col] == 0, C.TARGET].mean(),
             "with": internet.loc[internet[col] == 1, C.TARGET].mean()}
            for col in C.ADDON_COLS]
    gaps = pd.DataFrame(rows)
    return gaps.assign(gap=gaps["without"] - gaps["with"]).sort_values("gap")


def addon_dumbbell(df):
    """Dumbbell chart: churn without (orange) vs with (blue) each add-on, labelled with the gap."""
    ad = _addon_gaps(df)
    fig, ax = plt.subplots(figsize=(7, 3.4))
    y = np.arange(len(ad))
    ax.hlines(y, ad["with"], ad["without"], color=style.GRID, lw=3)
    for col, color, label in [("without", style.ORANGE, "Without add-on"),
                              ("with", style.BLUE, "With add-on")]:
        _dots(ax, ad[col], y, color, size=70, edge=2, label=label)
    for yi, (with_, without) in enumerate(zip(ad["with"], ad["without"])):
        ax.text(without + 0.01, yi, f"−{(without - with_) * 100:.0f} pts", va="center",
                fontsize=8.5, color=style.INK_2)
    ax.set_yticks(y, ad["addon"])
    ax.set_xlim(0, ad["without"].max() + 0.08)
    style.pct_axis(ax, "x")
    _hbar_grid(ax)
    ax.legend(loc="lower right", fontsize=8.5)
    return _finish(fig, ax, "Security and tech support customers churn far less",
                   "Churn rate of internet customers, with vs. without each add-on", "02_addons")


def charges_hist(df):
    """Monthly-charge distributions of stayers vs churners, with dashed median lines."""
    fig, ax = plt.subplots(figsize=(7, 3.4))
    bins = np.arange(15, 125, 5)
    for value, color, label in [(0, style.BLUE, "Stayed"), (1, style.ORANGE, "Churned")]:
        charges = df.loc[df[C.TARGET] == value, "monthly_charges"]
        ax.hist(charges, bins=bins, density=True, histtype="step", lw=2, color=color,
                label=f"{label} (median ${charges.median():.0f})")
        ax.axvline(charges.median(), color=color, ls="--", lw=1)
    ax.set_yticks([])                          # density values mean little to readers
    ax.grid(False)
    ax.set_xlabel("Monthly charges ($)")
    ax.legend(loc="upper left", fontsize=8.5)
    return _finish(fig, ax, "Churners pay ~$15 more per month; low bills rarely leave",
                   "Distribution of monthly charges (density)", "02_monthly_charges")


# Keyword -> group for the ~20 free-text churn reasons (first matching keyword wins).
REASON_GROUPS = [("competitor", "Competitor"), ("attitude", "Service / staff"), ("support", "Service / staff"),
                 ("dissatisfaction", "Service / staff"), ("price", "Price"), ("charges", "Price"),
                 ("extra data", "Price"), ("long distance", "Price"), ("network", "Product / network"),
                 ("product", "Product / network"), ("reliability", "Product / network"),
                 ("speed", "Product / network")]


def reason_group(reason: str) -> str:
    """'Competitor made better offer' -> 'Competitor'; unmatched reasons -> 'Other / unknown'."""
    r = reason.lower()
    return next((group for keyword, group in REASON_GROUPS if keyword in r), "Other / unknown")


def churn_reasons(df):
    """How many churners gave each (grouped) reason; the biggest group is highlighted."""
    reasons = df.loc[df[C.TARGET] == 1, "churn_reason"]
    grouped = reasons.map(reason_group).value_counts().sort_values()
    fig, ax = plt.subplots(figsize=(7, 2.8))
    ax.barh(grouped.index, grouped.values, height=0.55,
            color=[style.GRAY] * (len(grouped) - 1) + [style.ORANGE])
    for yi, count in enumerate(grouped.values):
        ax.text(count, yi, f"  {count} ({count / grouped.sum():.0%})", va="center", fontsize=9)
    ax.set_xlim(0, grouped.max() * 1.25)
    _hbar_grid(ax)
    return _finish(fig, ax, "Competitor offers are the most-cited reason for leaving",
                   f"Stated churn reasons, grouped ({grouped.sum():,} churners)", "02_churn_reasons")


# ═════════════════════════════════════════════════════════════════════════════
# 03 · Hypothesis tests
# ═════════════════════════════════════════════════════════════════════════════
def qq_pair(a, b):
    """Q-Q plots of churners (a) and stayers (b) against a normal distribution.

    Points bending away from the straight line mean the data is not normal,
    which is why notebook 03 uses Mann-Whitney instead of a t-test.
    """
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.2))
    for ax, x, name, color in [(axes[0], a, "Churned", style.ORANGE), (axes[1], b, "Stayed", style.BLUE)]:
        (theoretical, observed), (slope, intercept, _) = stats.probplot(x, dist="norm")
        ax.scatter(theoretical, observed, s=5, color=color, alpha=0.5)
        ax.plot(theoretical, slope * theoretical + intercept, color=style.INK_2, lw=1)
        ax.set(title=f"Q-Q plot: {name}", xlabel="Theoretical quantiles", ylabel="Monthly charges")
    plt.tight_layout()
    style.save(fig, "03_qq")
    return fig


def effect_forest(gaps: pd.DataFrame):
    """Forest plot: churn-rate differences with 95 % CIs.

    gaps  index = comparison name, columns diff / lo / hi (as proportions)
    """
    g = gaps.sort_values("diff")
    fig, ax = plt.subplots(figsize=(7, 2.5))
    y = np.arange(len(g))
    ax.hlines(y, g["lo"] * 100, g["hi"] * 100, color=style.INK_2, lw=2)
    _dots(ax, g["diff"] * 100, y, style.ORANGE, size=80, edge=2)
    for yi, diff in enumerate(g["diff"]):
        ax.text(diff * 100, yi + 0.25, f"+{diff * 100:.1f} pts", ha="center", fontsize=9)
    ax.axvline(0, color=style.INK_2, lw=1)     # zero = no difference
    ax.set_yticks(y, g.index)
    ax.set_ylim(-0.6, len(g) - 0.2)
    ax.set_xlabel("Difference in churn rate (pts, 95% CI)")
    _hbar_grid(ax)
    return _finish(fig, ax, "All three gaps are large and precisely estimated", None, "03_effect_sizes")


# ═════════════════════════════════════════════════════════════════════════════
# 04 · Modelling
# ═════════════════════════════════════════════════════════════════════════════
FAMILY = {"logistic_regression": "linear", "naive_bayes": "other", "knn": "other", "svm_rbf": "other",
          "decision_tree": "tree", "random_forest": "tree", "extra_trees": "tree", "adaboost": "boosting",
          "hist_gb": "boosting", "xgboost": "boosting", "lightgbm": "boosting",
          "soft_voting": "ensemble", "stacking": "ensemble"}
FAMILY_COLOR = {"linear": style.BLUE, "tree": style.AQUA, "boosting": style.SERIES[3],
                "ensemble": style.ORANGE, "other": style.GRAY}


def model_comparison(cv):
    """CV AUC ± 1 sd for every model, coloured by family, with the one-SE band shaded.

    cv  the cv_comparison table from training.run
    Models whose dot falls inside the shaded band are statistically tied with the best.
    """
    d = cv.drop(index="dummy", errors="ignore").sort_values("roc_auc")
    best = d["roc_auc"].max()
    se = d.loc[d["roc_auc"].idxmax(), "roc_auc_sd"] / np.sqrt(C.CV_FOLDS)
    fig, ax = plt.subplots(figsize=(8, 5))
    y = np.arange(len(d))

    # the one-standard-error band
    ax.axvspan(best - se, best, color=style.SEQ_BLUE[0], alpha=0.6, lw=0)
    ax.text(best - se - 0.0005, 0, "within 1 SE\nof best →", fontsize=8.5, color=style.INK_2,
            ha="right", va="center")

    # ± 1 sd whiskers and one dot per model
    ax.hlines(y, d["roc_auc"] - d["roc_auc_sd"], d["roc_auc"] + d["roc_auc_sd"], color=style.GRID, lw=2)
    family_colours = [FAMILY_COLOR[FAMILY[m.replace("_tuned", "")]] for m in d.index]
    ax.scatter(d["roc_auc"], y, s=60, zorder=3, edgecolor=style.SURFACE, lw=1.5, c=family_colours)

    ax.set_yticks(y, d.index)
    _hbar_grid(ax)
    legend = [Line2D([], [], marker="o", ls="", color=c, label=f) for f, c in FAMILY_COLOR.items()]
    ax.legend(handles=legend, loc="lower right", fontsize=8.5)
    ax.set_xlabel("ROC-AUC (mean ± 1 SD across 5 CV folds)")
    n_close = int((d["roc_auc"] >= best - 0.01).sum())
    return _finish(fig, ax, f"{n_close} models land within 0.01 AUC of each other",
                   "Ensembles add little over logistic regression", "04_model_comparison")


def roc_curves(y, probs: dict, auc: pd.Series):
    """Overlaid ROC curves on the test set, AUC in the legend.

    probs  {model name: predicted probabilities}     auc  {model name: AUC}
    """
    fig, ax = plt.subplots(figsize=(5.5, 4.4))
    for i, (name, p) in enumerate(probs.items()):
        fpr, tpr, _ = roc_curve(y, p)
        ax.plot(fpr, tpr, color=style.SERIES[i], lw=1.6, label=f"{name} ({auc[name]:.3f})")
    ax.plot([0, 1], [0, 1], color=style.GRID, ls="--", lw=1)     # random guessing
    ax.set(xlabel="False positive rate", ylabel="True positive rate")
    ax.grid(axis="both")
    ax.legend(fontsize=7.5, loc="lower right")
    return _finish(fig, ax, "ROC curves overlap almost perfectly", "Held-out test set (AUC)", "04_roc")


def odds_ratio_forest(ors):
    """Odds ratios with 95 % CIs on a log scale.

    ors  output of interpretation.fit_logit
    Blue = lowers churn, orange = raises it, grey = not significant (p >= 0.05).
    A log scale makes 0.5 and 2 equally far from 1.
    """
    fig, ax = plt.subplots(figsize=(8, 6))
    y = np.arange(len(ors))
    significant = ors["p_value"] < 0.05
    colours = np.where(~significant, style.GRAY,
                       np.where(ors["odds_ratio"] > 1, style.ORANGE, style.BLUE))
    ax.hlines(y, ors["ci_low"], ors["ci_high"], color=colours, lw=2)
    ax.scatter(ors["odds_ratio"], y, color=colours, s=45, zorder=3, edgecolor=style.SURFACE, lw=1.5)
    ax.axvline(1, color=style.INK_2, lw=1)     # odds ratio 1 = no effect
    ax.set_xscale("log")
    ax.set_xticks([0.1, 0.25, 0.5, 1, 2, 4], ["0.1", "0.25", "0.5", "1", "2", "4"])
    ax.xaxis.set_minor_locator(plt.NullLocator())
    ax.set_yticks(y, ors.index)
    _hbar_grid(ax)
    ax.set_xlabel("Odds ratio (log scale, 95% CI) · grey = not significant")
    return _finish(fig, ax, "Contract, dependents, tenure and fiber dominate",
                   "Logistic regression, all else held constant · blue lowers churn, orange raises it",
                   "04_odds_ratios")


def shap_beeswarm(shap, Xt, top=12, seed=0):
    """SHAP beeswarm: one dot per customer per feature (the `top` most important features).

    x position  how much that feature pushed the customer's log-odds up (right) or down (left)
    colour      the customer's value of that feature (blue low -> orange high)
    Dots are jittered vertically, more where they are dense, so the shape shows the distribution.
    """
    rng = np.random.default_rng(seed)
    order = shap.abs().mean().sort_values(ascending=False).head(top).index[::-1]
    fig, ax = plt.subplots(figsize=(8.5, 5.6))
    for i, feature in enumerate(order):
        values = shap[feature].to_numpy()
        # local density in [0, 1] -> how far each dot may be jittered
        hist, edges = np.histogram(values, bins=60)
        density = hist[np.clip(np.digitize(values, edges) - 1, 0, len(hist) - 1)] / hist.max()
        jitter = (rng.random(len(values)) - 0.5) * 0.8 * density
        ax.scatter(values, i + jitter, s=5, alpha=0.8, lw=0,
                   c=Xt[feature].rank(pct=True), cmap=LOW_HIGH, rasterized=True)
    ax.axvline(0, color=style.INK_2, lw=1)
    ax.set_yticks(range(len(order)), order)
    _hbar_grid(ax)
    ax.set_xlabel("SHAP value (impact on log-odds of churn)")

    colourbar = fig.colorbar(plt.cm.ScalarMappable(cmap=LOW_HIGH), ax=ax, pad=0.01, aspect=40)
    colourbar.set_ticks([0, 1], labels=["low", "high"])
    colourbar.set_label("Feature value", color=style.INK_2)
    colourbar.outline.set_visible(False)
    return _finish(fig, ax, "Month-to-month, no dependents and short tenure drive risk",
                   "LightGBM SHAP values on the test set · one dot per customer", "04_shap_beeswarm")


def shap_dependence(X, shap, seed=0):
    """Three panels showing how SHAP changes with tenure, monthly charges and contract.

    X     the raw (unscaled) test inputs, so the x-axes are in months / dollars
    shap  SHAP values from interpretation.lgbm_shap
    """
    rng = np.random.default_rng(seed)
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.6))

    # Panel 1 — tenure
    axes[0].scatter(X["tenure_months"], shap["tenure_months"], s=6, alpha=0.5, color=style.BLUE)
    axes[0].set(xlabel="Tenure (months)", ylabel="SHAP (log-odds)", title="Tenure: risk falls fast in year 1")

    # Panel 2 — monthly charges, fiber customers in orange
    fiber = X["internet_service"] == "Fiber optic"
    for mask, colour, label in [(~fiber, style.BLUE, "not fiber"), (fiber, style.ORANGE, "fiber")]:
        axes[1].scatter(X.loc[mask, "monthly_charges"], shap.loc[mask, "monthly_charges"], s=6,
                        alpha=0.5, color=colour, label=label)
    axes[1].legend(fontsize=8, markerscale=2)
    axes[1].set(xlabel="Monthly charges ($)", title="Price: risk rises above ~$70")

    # Panel 3 — contract (sum of its one-hot SHAP columns), jittered per level
    contract_cols = [c for c in shap.columns if c.startswith("contract_")]
    levels = sorted(X["contract"].astype(str).unique())
    for i, level in enumerate(levels):
        contract_shap = shap.loc[X["contract"].astype(str) == level, contract_cols].sum(axis=1)
        jitter = (rng.random(len(contract_shap)) - 0.5) * 0.5
        axes[2].scatter(i + jitter, contract_shap, s=6, alpha=0.5,
                        color=style.ORANGE if level == "Month-to-month" else style.BLUE)
    axes[2].set_xticks(range(len(levels)), levels)
    axes[2].set(title="Contract: the biggest single swing")

    for ax in axes:
        ax.axhline(0, color=style.INK_2, lw=0.8)
        ax.grid(axis="both")
    plt.tight_layout()
    style.save(fig, "04_shap_dependence")
    return fig


def _waterfall_label(col, x_row):
    """Bar label: 'tenure_months = 2', or for one-hot columns 'contract = Two year? no'."""
    if col in x_row.index:
        return f"{col} = {x_row[col]:g}"
    source = base_feature(col)
    if source in x_row.index:
        level = col[len(source) + 1:]
        return f"{source} = {level}? {'yes' if str(x_row[source]) == level else 'no'}"
    return col


def shap_waterfall(shap_row, x_row, base, customer_id, k=8):
    """Why ONE customer gets their score: start at the average, add each feature's push.

    shap_row  that customer's SHAP values      x_row  their raw feature values
    base      the base (average) log-odds      k      features shown before "all other features"
    """
    row = shap_row.sort_values(key=np.abs, ascending=False)
    parts = pd.concat([row.head(k), pd.Series({"all other features": row.iloc[k:].sum()})])[::-1]

    fig, ax = plt.subplots(figsize=(8, 3.8))
    running_total, lefts = base, []
    for value in parts.values:                 # each bar starts where the previous one ended
        lefts.append(min(running_total, running_total + value))
        running_total += value
    ax.barh(range(len(parts)), np.abs(parts.values), left=lefts, height=0.6,
            color=[style.ORANGE if v > 0 else style.BLUE for v in parts.values])

    ax.set_yticks(range(len(parts)), [_waterfall_label(c, x_row) for c in parts.index], fontsize=8.5)
    ax.axvline(base, color=style.INK_2, lw=1, ls="--")
    _hbar_grid(ax)
    ax.set_xlabel("Log-odds of churn")
    probability = 1 / (1 + np.exp(-running_total))    # log-odds -> probability
    return _finish(fig, ax, f"Why customer {customer_id} scores {probability:.0%} churn risk",
                   f"Starts at the average log-odds ({base:.2f}); each bar adds one feature",
                   "04_shap_waterfall")


def reliability(y, probs: dict):
    """Calibration curves: predicted vs observed churn. On the diagonal = trustworthy probabilities."""
    fig, ax = plt.subplots(figsize=(6, 4.4))
    ax.plot([0, 1], [0, 1], color=style.INK_2, ls="--", lw=1, label="perfect")
    for i, (name, p) in enumerate(probs.items()):
        rt = reliability_table(y, p)
        ax.plot(rt.mean_predicted, rt.observed_rate, marker="o", ms=4, lw=1.6,
                color=style.SERIES[i], label=name)
    ax.set(xlabel="Mean predicted probability", ylabel="Observed churn rate", xlim=(0, 1), ylim=(0, 1))
    ax.grid(axis="both")
    ax.legend(fontsize=8)
    return _finish(fig, ax, "Good AUC ≠ trustworthy probabilities",
                   "Reliability curves on the test set (closer to the diagonal is better)",
                   "04_calibration")


def lift(lt, base_rate):
    """Observed churn per risk decile (top 3 highlighted) with the mean prediction as a line.

    lt  output of metrics.lift_table
    """
    fig, ax = plt.subplots(figsize=(7, 3.4))
    ax.bar(lt.index.astype(str), lt["churn_rate"], width=0.65,
           color=[style.ORANGE] * 3 + [style.GRAY] * (len(lt) - 3))
    ax.plot(range(len(lt)), lt["mean_predicted"], color=style.INK, marker="o", ms=4, lw=1.2,
            label="mean predicted")
    ax.axhline(base_rate, color=style.INK_2, ls="--", lw=1)
    style.pct_axis(ax)
    ax.legend(fontsize=8.5)
    ax.set_xlabel("Risk decile (1 = highest predicted risk)")
    return _finish(fig, ax, f"Top 30% of scores capture {lt['cum_capture'].iloc[2]:.0%} of churners",
                   "Observed churn by predicted-risk decile (test set)", "04_lift")


# ═════════════════════════════════════════════════════════════════════════════
# 05 · A/B test and ROI
# ═════════════════════════════════════════════════════════════════════════════
def segment_hist(m2m, segment, threshold):
    """Churn-probability histogram of month-to-month customers, the targeted segment in orange."""
    fig, ax = plt.subplots(figsize=(7, 3))
    bins = np.linspace(0, 1, 41)
    ax.hist(m2m.churn_prob, bins=bins, color=style.GRAY, rwidth=0.9, label="month-to-month")
    ax.hist(segment.churn_prob, bins=bins, color=style.ORANGE, rwidth=0.9, label="targeted")
    ax.axvline(threshold, color=style.INK_2, ls="--", lw=1)
    ax.set(xlabel="Calibrated churn probability", ylabel="Customers")
    ax.legend(fontsize=8.5)
    return _finish(fig, ax, f"Targeting {len(segment):,} month-to-month customers at ≥{threshold:.0%} risk",
                   None, "05_segment")


def power_curve(table, segment_half):
    """Customers needed per arm vs effect size, one line per baseline (log scale).

    table         index = effect (pts), one column per baseline with n per arm
    segment_half  customers available per arm in one wave (dashed line)
    """
    fig, ax = plt.subplots(figsize=(7.5, 3.6))
    for col, colour in zip(table.columns, [style.BLUE, style.ORANGE]):
        ax.plot(table.index, table[col], color=colour, label=col)
    ax.axhline(segment_half, color=style.INK_2, ls="--", lw=1)
    ax.text(table.index.max(), segment_half, " segment / 2", va="bottom", ha="right",
            fontsize=8.5, color=style.INK_2)
    ax.set_yscale("log")
    ax.yaxis.set_major_formatter(lambda x, _: f"{x:,.0f}")
    ax.set(xlabel="Churn reduction to detect (pts)", ylabel="Customers per arm (log)")
    ax.legend(fontsize=8.5)
    return _finish(fig, ax, "Halving the detectable effect quadruples the sample",
                   "α = 0.05, power = 80%, two-sided", "05_power_curve")


def arm_bars(res):
    """Control vs treatment churn from one simulated experiment, with Wilson 95 % CIs.

    res  output of ab_testing.simulate_experiment
    """
    rates = [res["churn_control"], res["churn_treatment"]]
    sizes = [res["n_control"], res["n_treatment"]]
    cis = [proportion_confint(rate * n, n, method="wilson") for rate, n in zip(rates, sizes)]
    lower_err = [rate - ci[0] for rate, ci in zip(rates, cis)]
    upper_err = [ci[1] - rate for rate, ci in zip(rates, cis)]

    fig, ax = plt.subplots(figsize=(5.5, 3.4))
    ax.bar(["Control", "Treatment"], rates, color=[style.GRAY, style.BLUE], width=0.55,
           yerr=[lower_err, upper_err], error_kw={"ecolor": style.INK_2, "elinewidth": 1})
    for x, rate, ci in zip(range(2), rates, cis):
        ax.text(x, ci[1] + 0.01, f"{rate:.1%}", ha="center")
    style.pct_axis(ax)
    ax.set_ylim(0, max(rates) * 1.25)
    return _finish(fig, ax, f"Simulated read-out: −{res['diff'] * 100:.1f} pts (p = {res['p_value']:.3f})",
                   "Churn by arm, 95% CI", "05_simulated_result")


def monte_carlo(mc_null, mc_eff, n_total):
    """Observed effects across many simulated tests: no true effect (grey) vs 5 pts (blue)."""
    fig, ax = plt.subplots(figsize=(7.5, 3.2))
    bins = np.linspace(-0.06, 0.12, 50)
    ax.hist(mc_null["diff"], bins=bins, color=style.GRAY, alpha=0.9, label="true effect = 0")
    ax.hist(mc_eff["diff"], bins=bins, color=style.BLUE, alpha=0.75, label="true effect = 5 pts")
    ax.xaxis.set_major_formatter(lambda x, _: f"{x * 100:.0f}")
    ax.set(xlabel="Observed churn reduction (pts)", ylabel="Simulated experiments")
    ax.legend(fontsize=8.5)
    return _finish(fig, ax, f"With {n_total:,} customers a 5-pt effect is detected "
                            f"{mc_eff.significant.mean():.0%} of the time",
                   "2,000 simulated experiments per scenario", "05_monte_carlo")


def roi_heatmap(heat, title, sub):
    """Net revenue for each discount x churn reduction: red loses money, blue makes money.

    heat  pivot table, index = discount, columns = effect (pts), values = net revenue
    """
    fig, ax = plt.subplots(figsize=(9, 2.8))
    limit = np.abs(heat.values).max()            # symmetric scale so $0 sits in the middle
    ax.imshow(heat.values, cmap=style.DIVERGING, vmin=-limit, vmax=limit, aspect="auto")
    ax.set_xticks(range(heat.shape[1]), heat.columns)
    ax.set_yticks(range(heat.shape[0]), [f"{d:.0%} off" for d in heat.index])
    ax.grid(False)
    for i in range(heat.shape[0]):
        for j in range(heat.shape[1]):
            value = heat.values[i, j]
            label = "≈ $0" if abs(value) < 500 else ("+" if value > 0 else "") + style.money(value, k=True)
            ax.text(j, i, label, ha="center", va="center", fontsize=8,
                    color="white" if abs(value) > limit * 0.55 else style.INK)
    ax.set_xlabel("Churn reduction achieved (pts)")
    return _finish(fig, ax, title, sub, "05_roi_heatmap")


def breakeven_lines(curves: dict, breakevens: dict, band=None):
    """Net revenue per customer vs churn reduction, one line per lifetime assumption.

    curves      {label: (effects_pts, net_per_customer)}
    breakevens  {label: break-even effect in pts}  (drawn as dots on the zero line)
    band        optional (low, high) range of effects the test would observe (shaded)
    """
    fig, ax = plt.subplots(figsize=(7.5, 3.6))
    for (label, (x, net)), colour in zip(curves.items(), [style.GRAY, style.BLUE, style.AQUA]):
        ax.plot(x, net, color=colour, label=label)
        _dots(ax, [breakevens[label]], [0], colour, size=40)
    ax.axhline(0, color=style.INK_2, lw=1)
    if band:
        ax.axvspan(*band, color=style.SEQ_BLUE[0], alpha=0.6, lw=0)
    ax.yaxis.set_major_formatter(style.money)
    ax.set(xlabel="Churn reduction (pts)", ylabel="Net revenue per customer")
    ax.legend(fontsize=8, loc="lower right")
    return _finish(fig, ax, "At 20% off, profit hinges on how long saved customers stay",
                   "Dots = break-even · shaded = range of effects the test would observe", "05_breakeven")
