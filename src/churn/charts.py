"""Every chart in the notebooks, so notebook cells stay one-liners.

Each function draws one figure, saves it to reports/figures/<name>.png and returns it.
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D
from scipy import stats
from sklearn.metrics import roc_curve

from . import config as C
from . import style
from .data_cleaning import churn_rate_table
from .metrics import reliability_table

LOW_HIGH = LinearSegmentedColormap.from_list("lowhigh", [style.BLUE, "#d9d8d3", style.ORANGE])


def _finish(fig, ax, title, sub, name):
    ax.set_title(title)
    if sub:
        style.subtitle(ax, sub)
    style.save(fig, name)
    return fig


def _hbar_grid(ax):
    ax.grid(axis="y", visible=False)
    ax.grid(axis="x")


# ---------------------------------------------------------------- 01 framing

def revenue_lost(seg):
    s = seg["revenue_lost"].sort_values()
    fig, ax = plt.subplots(figsize=(7, 3))
    ax.barh(s.index.astype(str), s.values, height=0.55,
            color=[style.ORANGE if i == "Month-to-month" else style.GRAY for i in s.index])
    for y, v in enumerate(s.values):
        ax.text(v, y, "  " + style.money(v, k=True), va="center")
    ax.xaxis.set_major_formatter(lambda x, _: style.money(x, k=True))
    ax.set_xlim(0, s.max() * 1.18)
    _hbar_grid(ax)
    return _finish(fig, ax, "Lifetime revenue lost to this quarter's churners",
                   "Monthly charges × expected remaining tenure, by contract", "01_revenue_lost")


# ---------------------------------------------------------------- 02 EDA

def churn_rate(df, col, title, sub, highlight, name, horizontal=False, order=None, xlabel=None):
    t = churn_rate_table(df, col)
    if horizontal:
        t = t.sort_values("churn_rate", ascending=False)
    fig, ax = plt.subplots(figsize=(7, 3.2 if horizontal else 3.6))
    style.rate_bars(ax, t, highlight=highlight, overall=df[C.TARGET].mean(),
                  horizontal=horizontal, order=order)
    lim = (t["ci_high"].max() + 0.08)
    (ax.set_xlim if horizontal else ax.set_ylim)(0, lim)
    if xlabel:
        ax.set_xlabel(xlabel)
    return _finish(fig, ax, title, sub, name)


def addon_dumbbell(df):
    inet = df[df.has_internet == 1]
    rows = [{"addon": c.replace("_", " ").capitalize(),
             "without": inet.loc[inet[c] == 0, C.TARGET].mean(),
             "with": inet.loc[inet[c] == 1, C.TARGET].mean()} for c in C.ADDON_COLS]
    ad = pd.DataFrame(rows).assign(gap=lambda d: d["without"] - d["with"]).sort_values("gap")
    fig, ax = plt.subplots(figsize=(7, 3.4))
    y = np.arange(len(ad))
    ax.hlines(y, ad["with"], ad["without"], color=style.GRID, lw=3)
    for col, color, label in [("without", style.ORANGE, "Without add-on"), ("with", style.BLUE, "With add-on")]:
        ax.scatter(ad[col], y, s=70, color=color, label=label, zorder=3, edgecolor=style.SURFACE, lw=2)
    for yi, (w, wo) in enumerate(zip(ad["with"], ad["without"])):
        ax.text(wo + 0.01, yi, f"−{(wo - w) * 100:.0f} pts", va="center", fontsize=8.5, color=style.INK_2)
    ax.set_yticks(y, ad["addon"])
    ax.set_xlim(0, ad["without"].max() + 0.08)
    style.pct_axis(ax, "x")
    _hbar_grid(ax)
    ax.legend(loc="lower right", fontsize=8.5)
    return _finish(fig, ax, "Security and tech support customers churn far less",
                   "Churn rate of internet customers, with vs. without each add-on", "02_addons")


def charges_hist(df):
    fig, ax = plt.subplots(figsize=(7, 3.4))
    bins = np.arange(15, 125, 5)
    for val, color, label in [(0, style.BLUE, "Stayed"), (1, style.ORANGE, "Churned")]:
        x = df.loc[df[C.TARGET] == val, "monthly_charges"]
        ax.hist(x, bins=bins, density=True, histtype="step", lw=2, color=color,
                label=f"{label} (median ${x.median():.0f})")
        ax.axvline(x.median(), color=color, ls="--", lw=1)
    ax.set_yticks([])
    ax.grid(False)
    ax.set_xlabel("Monthly charges ($)")
    ax.legend(loc="upper left", fontsize=8.5)
    return _finish(fig, ax, "Churners pay ~$15 more per month; low bills rarely leave",
                   "Distribution of monthly charges (density)", "02_monthly_charges")


REASON_GROUPS = [("competitor", "Competitor"), ("attitude", "Service / staff"), ("support", "Service / staff"),
                 ("dissatisfaction", "Service / staff"), ("price", "Price"), ("charges", "Price"),
                 ("extra data", "Price"), ("long distance", "Price"), ("network", "Product / network"),
                 ("product", "Product / network"), ("reliability", "Product / network"),
                 ("speed", "Product / network")]


def reason_group(reason: str) -> str:
    r = reason.lower()
    return next((g for key, g in REASON_GROUPS if key in r), "Other / unknown")


def churn_reasons(df):
    reasons = df.loc[df[C.TARGET] == 1, "churn_reason"]
    grouped = reasons.map(reason_group).value_counts().sort_values()
    fig, ax = plt.subplots(figsize=(7, 2.8))
    ax.barh(grouped.index, grouped.values, height=0.55,
            color=[style.GRAY] * (len(grouped) - 1) + [style.ORANGE])
    for yi, v in enumerate(grouped.values):
        ax.text(v, yi, f"  {v} ({v / grouped.sum():.0%})", va="center", fontsize=9)
    ax.set_xlim(0, grouped.max() * 1.25)
    _hbar_grid(ax)
    return _finish(fig, ax, "Competitor offers are the most-cited reason for leaving",
                   f"Stated churn reasons, grouped ({grouped.sum():,} churners)", "02_churn_reasons")


# ---------------------------------------------------------------- 03 tests

def qq_pair(a, b):
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.2))
    for ax, x, name, color in [(axes[0], a, "Churned", style.ORANGE), (axes[1], b, "Stayed", style.BLUE)]:
        (osm, osr), (slope, icpt, _) = stats.probplot(x, dist="norm")
        ax.scatter(osm, osr, s=5, color=color, alpha=0.5)
        ax.plot(osm, slope * osm + icpt, color=style.INK_2, lw=1)
        ax.set(title=f"Q-Q plot: {name}", xlabel="Theoretical quantiles", ylabel="Monthly charges")
    plt.tight_layout()
    style.save(fig, "03_qq")
    return fig


def effect_forest(gaps: pd.DataFrame):
    """gaps: index = comparison, columns diff / lo / hi (proportions)."""
    g = gaps.sort_values("diff")
    fig, ax = plt.subplots(figsize=(7, 2.5))
    y = np.arange(len(g))
    ax.hlines(y, g["lo"] * 100, g["hi"] * 100, color=style.INK_2, lw=2)
    ax.scatter(g["diff"] * 100, y, s=80, color=style.ORANGE, zorder=3, edgecolor=style.SURFACE, lw=2)
    for yi, d in enumerate(g["diff"]):
        ax.text(d * 100, yi + 0.25, f"+{d * 100:.1f} pts", ha="center", fontsize=9)
    ax.axvline(0, color=style.INK_2, lw=1)
    ax.set_yticks(y, g.index)
    ax.set_ylim(-0.6, len(g) - 0.2)
    ax.set_xlabel("Difference in churn rate (pts, 95% CI)")
    _hbar_grid(ax)
    return _finish(fig, ax, "All three gaps are large and precisely estimated", None, "03_effect_sizes")


# ---------------------------------------------------------------- 04 modelling

FAMILY = {"logistic_regression": "linear", "naive_bayes": "other", "knn": "other", "svm_rbf": "other",
          "decision_tree": "tree", "random_forest": "tree", "extra_trees": "tree", "adaboost": "boosting",
          "hist_gb": "boosting", "xgboost": "boosting", "lightgbm": "boosting",
          "soft_voting": "ensemble", "stacking": "ensemble"}
FAMILY_COLOR = {"linear": style.BLUE, "tree": style.AQUA, "boosting": style.SERIES[3],
                "ensemble": style.ORANGE, "other": style.GRAY}


def model_comparison(cv):
    d = cv.drop(index="dummy", errors="ignore").sort_values("roc_auc")
    best = d["roc_auc"].max()
    se = d.loc[d["roc_auc"].idxmax(), "roc_auc_sd"] / np.sqrt(C.CV_FOLDS)
    fig, ax = plt.subplots(figsize=(8, 5))
    y = np.arange(len(d))
    ax.axvspan(best - se, best, color=style.SEQ_BLUE[0], alpha=0.6, lw=0)
    ax.text(best - se - 0.0005, 0, "within 1 SE\nof best →", fontsize=8.5, color=style.INK_2,
            ha="right", va="center")
    ax.hlines(y, d["roc_auc"] - d["roc_auc_sd"], d["roc_auc"] + d["roc_auc_sd"], color=style.GRID, lw=2)
    ax.scatter(d["roc_auc"], y, s=60, zorder=3, edgecolor=style.SURFACE, lw=1.5,
               c=[FAMILY_COLOR[FAMILY[m.replace("_tuned", "")]] for m in d.index])
    ax.set_yticks(y, d.index)
    _hbar_grid(ax)
    ax.legend(handles=[Line2D([], [], marker="o", ls="", color=c, label=f) for f, c in FAMILY_COLOR.items()],
              loc="lower right", fontsize=8.5)
    ax.set_xlabel("ROC-AUC (mean ± 1 SD across 5 CV folds)")
    n_close = int((d["roc_auc"] >= best - 0.01).sum())
    return _finish(fig, ax, f"{n_close} models land within 0.01 AUC of each other",
                   "Ensembles add little over logistic regression", "04_model_comparison")


def roc_curves(y, probs: dict, auc: pd.Series):
    fig, ax = plt.subplots(figsize=(5.5, 4.4))
    for i, (name, p) in enumerate(probs.items()):
        fpr, tpr, _ = roc_curve(y, p)
        ax.plot(fpr, tpr, color=style.SERIES[i], lw=1.6, label=f"{name} ({auc[name]:.3f})")
    ax.plot([0, 1], [0, 1], color=style.GRID, ls="--", lw=1)
    ax.set(xlabel="False positive rate", ylabel="True positive rate")
    ax.grid(axis="both")
    ax.legend(fontsize=7.5, loc="lower right")
    return _finish(fig, ax, "ROC curves overlap almost perfectly", "Held-out test set (AUC)", "04_roc")


def odds_ratio_forest(ors):
    fig, ax = plt.subplots(figsize=(8, 6))
    y = np.arange(len(ors))
    sig = ors["p_value"] < 0.05
    col = np.where(~sig, style.GRAY, np.where(ors["odds_ratio"] > 1, style.ORANGE, style.BLUE))
    ax.hlines(y, ors["ci_low"], ors["ci_high"], color=col, lw=2)
    ax.scatter(ors["odds_ratio"], y, color=col, s=45, zorder=3, edgecolor=style.SURFACE, lw=1.5)
    ax.axvline(1, color=style.INK_2, lw=1)
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
    rng = np.random.default_rng(seed)
    order = shap.abs().mean().sort_values(ascending=False).head(top).index[::-1]
    fig, ax = plt.subplots(figsize=(8.5, 5.6))
    for i, f in enumerate(order):
        v = shap[f].to_numpy()
        hist, edges = np.histogram(v, bins=60)
        dens = hist[np.clip(np.digitize(v, edges) - 1, 0, len(hist) - 1)] / hist.max()
        ax.scatter(v, i + (rng.random(len(v)) - 0.5) * 0.8 * dens, s=5, alpha=0.8, lw=0,
                   c=Xt[f].rank(pct=True), cmap=LOW_HIGH, rasterized=True)
    ax.axvline(0, color=style.INK_2, lw=1)
    ax.set_yticks(range(len(order)), order)
    _hbar_grid(ax)
    ax.set_xlabel("SHAP value (impact on log-odds of churn)")
    sm_ = plt.cm.ScalarMappable(cmap=LOW_HIGH)
    cb = fig.colorbar(sm_, ax=ax, pad=0.01, aspect=40)
    cb.set_ticks([0, 1], labels=["low", "high"])
    cb.set_label("Feature value", color=style.INK_2)
    cb.outline.set_visible(False)
    return _finish(fig, ax, "Month-to-month, no dependents and short tenure drive risk",
                   "LightGBM SHAP values on the test set · one dot per customer", "04_shap_beeswarm")


def shap_dependence(X, shap, seed=0):
    rng = np.random.default_rng(seed)
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.6))
    axes[0].scatter(X["tenure_months"], shap["tenure_months"], s=6, alpha=0.5, color=style.BLUE)
    axes[0].set(xlabel="Tenure (months)", ylabel="SHAP (log-odds)", title="Tenure: risk falls fast in year 1")
    fiber = X["internet_service"] == "Fiber optic"
    for mask, c, lab in [(~fiber, style.BLUE, "not fiber"), (fiber, style.ORANGE, "fiber")]:
        axes[1].scatter(X.loc[mask, "monthly_charges"], shap.loc[mask, "monthly_charges"], s=6,
                        alpha=0.5, color=c, label=lab)
    axes[1].legend(fontsize=8, markerscale=2)
    axes[1].set(xlabel="Monthly charges ($)", title="Price: risk rises above ~$70")
    contract_cols = [c for c in shap.columns if c.startswith("contract_")]
    levels = sorted(X["contract"].astype(str).unique())
    for i, lvl in enumerate(levels):
        s = shap.loc[X["contract"].astype(str) == lvl, contract_cols].sum(axis=1)
        axes[2].scatter(i + (rng.random(len(s)) - 0.5) * 0.5, s, s=6, alpha=0.5,
                        color=style.ORANGE if lvl == "Month-to-month" else style.BLUE)
    axes[2].set_xticks(range(len(levels)), levels)
    axes[2].set(title="Contract: the biggest single swing")
    for ax in axes:
        ax.axhline(0, color=style.INK_2, lw=0.8)
        ax.grid(axis="both")
    plt.tight_layout()
    style.save(fig, "04_shap_dependence")
    return fig


def shap_waterfall(shap_row, x_row, base, customer_id, k=8):
    row = shap_row.sort_values(key=np.abs, ascending=False)
    parts = pd.concat([row.head(k), pd.Series({"all other features": row.iloc[k:].sum()})])[::-1]
    fig, ax = plt.subplots(figsize=(8, 3.8))
    cum, lefts = base, []
    for v in parts.values:
        lefts.append(min(cum, cum + v))
        cum += v
    ax.barh(range(len(parts)), np.abs(parts.values), left=lefts, height=0.6,
            color=[style.ORANGE if v > 0 else style.BLUE for v in parts.values])
    # x_row holds the customer's raw (unscaled) values; one-hot columns become "contract = Two year? no"
    from .interpretation import base_feature

    def label(col):
        if col in x_row.index:
            return f"{col} = {x_row[col]:g}"
        src = base_feature(col)
        if src in x_row.index:
            level = col[len(src) + 1:]
            return f"{src} = {level}? {'yes' if str(x_row[src]) == level else 'no'}"
        return col
    labels = [label(c) for c in parts.index]
    ax.set_yticks(range(len(parts)), labels, fontsize=8.5)
    ax.axvline(base, color=style.INK_2, lw=1, ls="--")
    _hbar_grid(ax)
    ax.set_xlabel("Log-odds of churn")
    p = 1 / (1 + np.exp(-cum))
    return _finish(fig, ax, f"Why customer {customer_id} scores {p:.0%} churn risk",
                   f"Starts at the average log-odds ({base:.2f}); each bar adds one feature",
                   "04_shap_waterfall")


def reliability(y, probs: dict):
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


# ---------------------------------------------------------------- 05 A/B test

def segment_hist(m2m, segment, threshold):
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
    """table: index = effect (pts), one column per baseline with n per arm."""
    fig, ax = plt.subplots(figsize=(7.5, 3.6))
    for col, color in zip(table.columns, [style.BLUE, style.ORANGE]):
        ax.plot(table.index, table[col], color=color, label=col)
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
    from statsmodels.stats.proportion import proportion_confint
    rates = [res["churn_control"], res["churn_treatment"]]
    ns = [res["n_control"], res["n_treatment"]]
    cis = [proportion_confint(r * n, n, method="wilson") for r, n in zip(rates, ns)]
    fig, ax = plt.subplots(figsize=(5.5, 3.4))
    ax.bar(["Control", "Treatment"], rates, color=[style.GRAY, style.BLUE], width=0.55,
           yerr=[[r - c[0] for r, c in zip(rates, cis)], [c[1] - r for r, c in zip(rates, cis)]],
           error_kw={"ecolor": style.INK_2, "elinewidth": 1})
    for x, r, c in zip(range(2), rates, cis):
        ax.text(x, c[1] + 0.01, f"{r:.1%}", ha="center")
    style.pct_axis(ax)
    ax.set_ylim(0, max(rates) * 1.25)
    return _finish(fig, ax, f"Simulated read-out: −{res['diff'] * 100:.1f} pts (p = {res['p_value']:.3f})",
                   "Churn by arm, 95% CI", "05_simulated_result")


def monte_carlo(mc_null, mc_eff, n_total):
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
    fig, ax = plt.subplots(figsize=(9, 2.8))
    lim = np.abs(heat.values).max()
    ax.imshow(heat.values, cmap="RdBu", vmin=-lim, vmax=lim, aspect="auto")
    ax.set_xticks(range(heat.shape[1]), heat.columns)
    ax.set_yticks(range(heat.shape[0]), [f"{d:.0%} off" for d in heat.index])
    ax.grid(False)
    for i in range(heat.shape[0]):
        for j in range(heat.shape[1]):
            v = heat.values[i, j]
            lab = "≈ $0" if abs(v) < 500 else ("+" if v > 0 else "") + style.money(v, k=True)
            ax.text(j, i, lab, ha="center", va="center", fontsize=8,
                    color="white" if abs(v) > lim * 0.55 else style.INK)
    ax.set_xlabel("Churn reduction achieved (pts)")
    return _finish(fig, ax, title, sub, "05_roi_heatmap")


def breakeven_lines(curves: dict, breakevens: dict, band=None):
    """curves: label -> (effects_pts, net_per_customer); breakevens: label -> pts."""
    fig, ax = plt.subplots(figsize=(7.5, 3.6))
    for (label, (x, net)), color in zip(curves.items(), [style.GRAY, style.BLUE, style.AQUA]):
        ax.plot(x, net, color=color, label=label)
        ax.scatter([breakevens[label]], [0], color=color, s=40, zorder=3, edgecolor=style.SURFACE, lw=1.5)
    ax.axhline(0, color=style.INK_2, lw=1)
    if band:
        ax.axvspan(*band, color=style.SEQ_BLUE[0], alpha=0.6, lw=0)
    ax.yaxis.set_major_formatter(style.money)
    ax.set(xlabel="Churn reduction (pts)", ylabel="Net revenue per customer")
    ax.legend(fontsize=8, loc="lower right")
    return _finish(fig, ax, "At 20% off, profit hinges on how long saved customers stay",
                   "Dots = break-even · shaded = range of effects the test would observe", "05_breakeven")
