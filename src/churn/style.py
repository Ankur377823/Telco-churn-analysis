"""
style.py — one consistent look for every chart and table
========================================================

WHAT THIS FILE DOES
    1. Colour palette with a fixed meaning across the whole project:
           BLUE   = stayed / good outcome         ORANGE = churn / emphasis
           GRAY   = de-emphasised                 AQUA   = third series
    2. setup()      global matplotlib settings (clean axes, light grid, bold left titles)
    3. Small chart helpers: subtitles, money and percent formatting, churn-rate bars
    4. Styled pandas tables and two-column "result cards" for the notebooks
    5. save()       writes a figure to reports/figures/<name>.png

WHY
    Charts are for business readers. Every chart states its conclusion in the
    title, shows confidence intervals on rates, and uses the same colours for
    the same meaning, so readers don't have to re-learn each figure.
"""
import matplotlib as mpl
import pandas as pd

from . import config as C


# ─────────────────────────────────────────────────────────────────────────────
# Palette
# ─────────────────────────────────────────────────────────────────────────────
SURFACE = "#fcfcfb"   # page / figure background
INK = "#0b0b0b"       # main text
INK_2 = "#52514e"     # secondary text, axes, reference lines
GRID = "#e4e3df"      # gridlines
BLUE = "#2a78d6"      # primary series / "retained"
ORANGE = "#eb6834"    # churn / emphasis
AQUA = "#1baf7a"
GRAY = "#a9a8a2"      # de-emphasised bars
SERIES = [BLUE, ORANGE, AQUA, "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
SEQ_BLUE = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
DIVERGING = "RdBu"    # only for the ROI heatmap (negative = red, positive = blue)


# ─────────────────────────────────────────────────────────────────────────────
# Global matplotlib settings
# ─────────────────────────────────────────────────────────────────────────────
def setup():
    """Apply the project style to every figure drawn afterwards."""
    mpl.rcParams.update({
        # background and resolution
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "figure.dpi": 110, "savefig.dpi": 150, "savefig.bbox": "tight",
        # text: bold, left-aligned titles that state the takeaway
        "font.size": 10, "axes.titlesize": 12, "axes.titleweight": "bold",
        "axes.titlelocation": "left", "axes.titlepad": 22,
        "axes.labelcolor": INK_2, "axes.edgecolor": GRID, "text.color": INK,
        "xtick.color": INK_2, "ytick.color": INK_2,
        # minimal frame: no top/right spines, light horizontal grid behind the data
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "axes.grid.axis": "y", "grid.color": GRID, "grid.linewidth": 0.8,
        "axes.axisbelow": True, "axes.prop_cycle": mpl.cycler(color=SERIES),
        # lines, legends, bars
        "lines.linewidth": 2, "legend.frameon": False, "patch.linewidth": 0,
    })


# ─────────────────────────────────────────────────────────────────────────────
# Small chart helpers
# ─────────────────────────────────────────────────────────────────────────────
def subtitle(ax, text):
    """Grey one-line note just under the title."""
    ax.text(0, 1.015, text, transform=ax.transAxes, color=INK_2, fontsize=9.5, va="bottom")


def money(x, _=None, k=False):
    """Format money: -1234 -> '−$1,234' (or '−$1k' with k=True). Usable as an axis formatter."""
    sign = "−" if x < 0 else ""
    return f"{sign}${abs(x)/1e3:,.0f}k" if k else f"{sign}${abs(x):,.0f}"


def pct_axis(ax, axis="y"):
    """Show an axis as percentages (0.25 -> 25%)."""
    fmt = mpl.ticker.PercentFormatter(1.0, decimals=0)
    (ax.yaxis if axis == "y" else ax.xaxis).set_major_formatter(fmt)


def _bar_colours(index, highlight):
    """Orange for highlighted levels, grey for the rest; all blue when nothing is highlighted."""
    if highlight is None:
        return [BLUE] * len(index)
    return [ORANGE if level in highlight else GRAY for level in index]


def rate_bars(ax, table, highlight=None, overall=None, horizontal=False, order=None):
    """Churn-rate bars with Wilson CI whiskers and the value written on each bar.

    table      output of data_cleaning.churn_rate_table
    highlight  levels to colour orange (others become grey)
    overall    draw a dashed line at the overall churn rate
    order      optional explicit order of the levels
    """
    t = table.loc[order] if order is not None else table
    labels = [str(i) for i in t.index]
    colors = _bar_colours(t.index, highlight)
    errors = [t["churn_rate"] - t["ci_low"], t["ci_high"] - t["churn_rate"]]
    whisker_style = {"ecolor": INK_2, "elinewidth": 1, "capsize": 0}

    if horizontal:
        ax.barh(labels, t["churn_rate"], color=colors, height=0.6, xerr=errors, error_kw=whisker_style)
        ax.invert_yaxis()                          # first level at the top
        pct_axis(ax, "x")
        ax.grid(axis="x"); ax.grid(axis="y", visible=False)
        for y, rate, high in zip(range(len(t)), t["churn_rate"], t["ci_high"]):
            ax.text(high + 0.01, y, f"{rate:.0%}", va="center", color=INK, fontsize=9.5)
        if overall is not None:
            ax.axvline(overall, color=INK_2, lw=1, ls="--")
    else:
        ax.bar(labels, t["churn_rate"], color=colors, width=0.6, yerr=errors, error_kw=whisker_style)
        pct_axis(ax)
        for x, rate, high in zip(range(len(t)), t["churn_rate"], t["ci_high"]):
            ax.text(x, high + 0.01, f"{rate:.0%}", ha="center", color=INK, fontsize=9.5)
        if overall is not None:
            ax.axhline(overall, color=INK_2, lw=1, ls="--")
            ax.text(len(t) - 0.5, overall, f" overall {overall:.0%}", va="bottom", ha="right",
                    color=INK_2, fontsize=8.5)
    return ax


# ─────────────────────────────────────────────────────────────────────────────
# Styled tables for the notebooks
# ─────────────────────────────────────────────────────────────────────────────
TABLE_CSS = [
    {"selector": "", "props": f"background:{SURFACE}; color:{INK}; border-collapse:collapse; "
                              "font-size:12.5px; font-family:system-ui, sans-serif;"},
    {"selector": "caption", "props": f"caption-side:top; text-align:left; font-weight:600; "
                                     f"color:{INK}; padding:0 0 6px 0; font-size:13px;"},
    {"selector": "th", "props": f"background:{SURFACE}; color:{INK_2}; font-weight:600; "
                                f"border-bottom:1.5px solid {INK_2}; padding:6px 12px; text-align:right;"},
    {"selector": "th.row_heading", "props": f"text-align:left; color:{INK}; border-bottom:1px solid {GRID};"},
    {"selector": "td", "props": f"padding:5px 12px; border-bottom:1px solid {GRID}; text-align:right;"},
]


def table(df, fmt=None, caption=None, highlight=None, hide_index=False):
    """A DataFrame with the project's table style.

    fmt         number format(s), e.g. "{:.1%}" or {"col": "{:,.0f}"}
    caption     title shown above the table
    highlight   row labels to shade light orange
    """
    styled = df.style.set_table_styles(TABLE_CSS)
    if fmt:
        styled = styled.format(fmt, na_rep="")
    if caption:
        styled = styled.set_caption(caption)
    if highlight is not None:
        def shade(row):
            colour = "background:#fde3d7" if row.name in highlight else ""
            return [colour] * len(row)
        styled = styled.apply(shade, axis=1)
    if hide_index:
        styled = styled.hide(axis="index")
    return styled


def card(items: dict, caption=None):
    """Two-column key / value 'result card', e.g. the outcome of one test or simulation."""
    df = pd.DataFrame({"value": [str(v) for v in items.values()]}, index=list(items.keys()))
    return table(df, caption=caption).hide(axis="columns")


# ─────────────────────────────────────────────────────────────────────────────
# Saving
# ─────────────────────────────────────────────────────────────────────────────
def save(fig, name):
    """Save a figure as reports/figures/<name>.png."""
    C.FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(C.FIGURES_DIR / f"{name}.png")
