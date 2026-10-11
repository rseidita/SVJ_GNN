"""plot_style.py - Visual styling, CMS formatting, color palettes, and plot beautifier."""

from __future__ import annotations
import matplotlib.pyplot as plt
import mplhep as hep
import numpy as np

# ---------------------------------------------------------------------------
# Global Typography and Sizing Constants
# ---------------------------------------------------------------------------
FONT_SIZE = 14
AXES_LABEL_SIZE = 16
TITLE_SIZE = 16
TICK_LABEL_SIZE = 14
LEGEND_FONT_SIZE = 12
TITLE_PAD = 6

FIG_SIZE = (10, 10)
GRIDSPEC_KW = {"height_ratios": [3, 1], "hspace": 0.05}

# ---------------------------------------------------------------------------
# Background Palettes & Labels
# ---------------------------------------------------------------------------
BKG_ORDER = ["QCD", "TTJets", "WJets", "ZJets"]

BKG_STYLE = {
    "QCD": {"color": "#4477aa", "label": r"$\mathrm{QCD}$"},
    "TTJets": {"color": "#ee8833", "label": r"$t\bar{t} + \mathrm{jets}$"},
    "WJets": {"color": "#228833", "label": r"$W + \mathrm{jets}$"},
    "ZJets": {"color": "#aa3377", "label": r"$Z + \mathrm{jets}$"},
}

SIGNAL_PALETTE = [
    "#d62728",  # red
    "#9467bd",  # purple
    "#8c564b",  # brown
    "#e377c2",  # pink
    "#17becf",  # cyan
    "#bcbd22",  # olive
]

# ---------------------------------------------------------------------------
# Uncertainty Band & Ratio Settings
# ---------------------------------------------------------------------------
MC_UNC_HATCH = "////"
MC_UNC_COLOR = "#555555"

RATIO_Y_LIM = (0.0, 2.0)
RATIO_Y_TICKS = [0.5, 1.0, 1.5]
RATIO_Y_LABEL = r"$\mathrm{Data} / \mathrm{MC}$"

REGION_DISPLAY = {
    "validation": r"VR ($1.5 < |\Delta\eta_{j0,j1}| < 2.2, R_{\mathrm{T}} > 0.24$)",
    "highRT": r"VR ($1.5 < |\Delta\eta_{j0,j1}| < 2.2, R_{\mathrm{T}} > 0.24$)",
    "control": r"CR ($1.5 < |\Delta\eta_{j0,j1}| < 2.2$)",
    "lowRT": r"Low $R_{\mathrm{T}}$ ($0.15 < R_{\mathrm{T}} < 0.24$)",
    "inclusive": r"Inclusive Selection",
}

LUMI_DISPLAY = {
    "2016": 36.3,
    "2017": 41.5,
    "2018": 59.7,
    "2018_pre": 21.1,
    "2018_post": 38.6,
    "Run2": 138.0,
}


def setup_mplhep_style():
    """Initialize mplhep CMS styling and configure global rcParams."""
    plt.style.use(hep.style.CMS)
    plt.rcParams.update({
        "font.size": FONT_SIZE,
        "axes.labelsize": AXES_LABEL_SIZE,
        "axes.titlesize": TITLE_SIZE,
        "xtick.labelsize": TICK_LABEL_SIZE,
        "ytick.labelsize": TICK_LABEL_SIZE,
        "legend.fontsize": LEGEND_FONT_SIZE,
        "axes.linewidth": 1.2,
        "xtick.major.size": 8,
        "xtick.minor.size": 4,
        "ytick.major.size": 8,
        "ytick.minor.size": 4,
    })


def format_yield(val: float) -> str:
    """Format yield for legend display."""
    if val <= 0:
        return "0"
    if val >= 1e5 or (val < 0.1 and val > 0):
        return f"{val:.2e}"
    elif val >= 1000:
        return f"{val:,.0f}"
    elif val >= 10:
        return f"{val:.1f}"
    else:
        return f"{val:.2f}"


def add_cms_header(
    ax,
    year: str | int,
    lumi_fb: float | None = None,
    has_data: bool = True,
    is_preliminary: bool = True,
):
    """Add standard CMS top-of-plot banner with Preliminary / Simulation Preliminary."""
    if has_data:
        sub = r"$\mathit{Preliminary}$" if is_preliminary else ""
    else:
        sub = r"$\mathit{Simulation\ Preliminary}$" if is_preliminary else r"$\mathit{Simulation}$"
    cms_text = rf"$\mathbf{{CMS}}$ {sub}".strip()
    ax.set_title(cms_text, loc="left", pad=TITLE_PAD, fontsize=TITLE_SIZE)

    yr_str = str(year)
    if lumi_fb is not None:
        lumi_val = lumi_fb
    elif yr_str in LUMI_DISPLAY:
        lumi_val = LUMI_DISPLAY[yr_str]
    else:
        lumi_val = 59.7

    lumi_text = rf"{lumi_val:.1f} fb$^{{-1}}$ ({yr_str}, 13 TeV)"
    ax.text(1.0, 1.0, lumi_text, transform=ax.transAxes, ha="right", va="bottom", fontsize=TICK_LABEL_SIZE)


def add_region_badge(ax, region: str):
    """Add region annotation badge."""
    text = REGION_DISPLAY.get(region, region)
    ax.text(
        0.97, 0.04, text,
        transform=ax.transAxes, ha="right", va="bottom", fontsize=12,
        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="grey", alpha=0.85),
    )


def format_main_axes(ax, var_cfg: dict, y_max: float, log_y: bool = True):
    """Format limits, labels, and ticks for the main upper panel."""
    bins = var_cfg["bins"]
    ax.set_xlim(bins[0], bins[-1])
    ax.set_ylabel("Events / bin", ha="right", y=1.0, fontsize=AXES_LABEL_SIZE)

    if log_y:
        ax.set_yscale("log")
        ax.set_ylim(bottom=0.01, top=max(y_max * 50.0, 10.0))
    else:
        ax.set_ylim(bottom=0.0, top=y_max * 1.35)

    ax.tick_params(axis="both", which="both", direction="in", top=True, right=True)
    ax.tick_params(labelbottom=False)


def format_ratio_axes(ax_ratio, var_cfg: dict):
    """Format limits, labels, ticks, and reference line for the ratio panel."""
    bins = var_cfg["bins"]
    ax_ratio.set_xlim(bins[0], bins[-1])
    ax_ratio.set_xlabel(var_cfg["xlabel"], ha="right", x=1.0, fontsize=AXES_LABEL_SIZE)
    ax_ratio.set_ylabel(RATIO_Y_LABEL, ha="right", y=1.0, fontsize=AXES_LABEL_SIZE)
    ax_ratio.set_ylim(*RATIO_Y_LIM)
    ax_ratio.set_yticks(RATIO_Y_TICKS)
    ax_ratio.axhline(1.0, color=MC_UNC_COLOR, linestyle="--", linewidth=1.0)
    ax_ratio.tick_params(axis="both", which="both", direction="in", top=True, right=True)

