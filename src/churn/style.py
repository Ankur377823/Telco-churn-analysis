"""One consistent matplotlib style for every figure in the project."""
import matplotlib as mpl
import matplotlib.pyplot as plt

from . import config as C

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
GRID = "#e4e3df"
BLUE = "#2a78d6"      # primary series / "retained"
ORANGE = "#eb6834"    # churn / emphasis
AQUA = "#1baf7a"
GRAY = "#a9a8a2"      # de-emphasised bars
SERIES = [BLUE, ORANGE, AQUA, "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
SEQ_BLUE = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
DIVERGING = "RdBu"  # used only for the ROI heatmap (negative = red, positive = blue)


def setup():
    mpl.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "figure.dpi": 110, "savefig.dpi": 150, "savefig.bbox": "tight",
        "font.size": 10, "axes.titlesize": 12, "axes.titleweight": "bold",
        "axes.titlelocation": "left", "axes.titlepad": 22,
        "axes.labelcolor": INK_2, "axes.edgecolor": GRID, "text.color": INK,
        "xtick.color": INK_2, "ytick.color": INK_2,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "axes.grid.axis": "y", "grid.color": GRID, "grid.linewidth": 0.8,
        "axes.axisbelow": True, "axes.prop_cycle": mpl.cycler(color=SERIES),
        "lines.linewidth": 2, "legend.frameon": False, "patch.linewidth": 0,
    })


def subtitle(ax, text):
    """Grey one-line takeaway under the title."""
    ax.text(0, 1.015, text, transform=ax.transAxes, color=INK_2, fontsize=9.5, va="bottom")


def money(x, _=None, k=False):
    """Axis/label formatter: -1234 -> '−$1,234' (or '−$1k' with k=True)."""
    sign = "−" if x < 0 else ""
    return f"{sign}${abs(x)/1e3:,.0f}k" if k else f"{sign}${abs(x):,.0f}"


def pct_axis(ax, axis="y"):
    fmt = mpl.ticker.PercentFormatter(1.0, decimals=0)
    (ax.yaxis if axis == "y" else ax.xaxis).set_major_formatter(fmt)


def rate_bars(ax, table, highlight=None, overall=None, horizontal=False, order=None):
    """Bar chart of churn_rate with Wilson CI whiskers and a direct label on each bar."""
    t = table.loc[order] if order is not None else table
    labels = [str(i) for i in t.index]
    colors = [ORANGE if (highlight is not None and i in highlight) else GRAY if highlight else BLUE
              for i in t.index]
    err = [t["churn_rate"] - t["ci_low"], t["ci_high"] - t["churn_rate"]]
    if horizontal:
        ax.barh(labels, t["churn_rate"], color=colors, height=0.6, xerr=err,
                error_kw={"ecolor": INK_2, "elinewidth": 1, "capsize": 0})
        ax.invert_yaxis()
        pct_axis(ax, "x")
        ax.grid(axis="x"); ax.grid(axis="y", visible=False)
        for y, v, hi in zip(range(len(t)), t["churn_rate"], t["ci_high"]):
            ax.text(hi + 0.01, y, f"{v:.0%}", va="center", color=INK, fontsize=9.5)
        if overall is not None:
            ax.axvline(overall, color=INK_2, lw=1, ls="--")
    else:
        ax.bar(labels, t["churn_rate"], color=colors, width=0.6, yerr=err,
               error_kw={"ecolor": INK_2, "elinewidth": 1, "capsize": 0})
        pct_axis(ax)
        for x, v, hi in zip(range(len(t)), t["churn_rate"], t["ci_high"]):
            ax.text(x, hi + 0.01, f"{v:.0%}", ha="center", color=INK, fontsize=9.5)
        if overall is not None:
            ax.axhline(overall, color=INK_2, lw=1, ls="--")
            ax.text(len(t) - 0.5, overall, f" overall {overall:.0%}", va="bottom", ha="right",
                    color=INK_2, fontsize=8.5)
    return ax


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
    """Consistently styled DataFrame. `highlight` = row labels to shade in light orange."""
    s = df.style.set_table_styles(TABLE_CSS)
    if fmt:
        s = s.format(fmt, na_rep="")
    if caption:
        s = s.set_caption(caption)
    if highlight is not None:
        s = s.apply(lambda r: ["background:#fde3d7" if r.name in highlight else ""] * len(r), axis=1)
    if hide_index:
        s = s.hide(axis="index")
    return s


def card(items: dict, caption=None):
    """Two-column key/value 'result card' for a single test or simulation."""
    import pandas as pd
    df = pd.DataFrame({"value": [str(v) for v in items.values()]}, index=list(items.keys()))
    return table(df, caption=caption).hide(axis="columns")


def save(fig, name):
    C.FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(C.FIGURES_DIR / f"{name}.png")
