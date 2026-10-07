"""plot_stack.py - Stacked Data/MC comparison plots with ratio panel using mplhep."""

from __future__ import annotations
import argparse
import os

import matplotlib.pyplot as plt
import numpy as np

import csv
from plotting.plot_style import (
    BKG_ORDER,
    BKG_STYLE,
    FIG_SIZE,
    GRIDSPEC_KW,
    MC_UNC_COLOR,
    MC_UNC_HATCH,
    SIGNAL_PALETTE,
    add_cms_header,
    add_region_badge,
    format_main_axes,
    format_ratio_axes,
    format_yield,
    setup_mplhep_style,
)
from plotting.plot_variables import VARIABLES
from plotting.hist_reader import TH1Accumulator, load_histograms_from_root


def export_yields_csv(
    hist_data: dict[str, dict[str, TH1Accumulator]],
    year: str | int,
    region: str = "validation",
    out_dir: str = "plots",
    selected_vars: list[str] | None = None,
) -> str:
    """Generate and write yields.csv containing cut yields for Data, background groups, and signals."""
    vars_to_run = selected_vars or list(VARIABLES.keys())
    os.makedirs(out_dir, exist_ok=True)
    csv_path = os.path.join(out_dir, "yields.csv")
    csv_per_process_path = os.path.join(out_dir, "yields_per_process.csv")

    sig_names = sorted([k for k in hist_data if k not in BKG_ORDER and k != "Data"])
    has_data = "Data" in hist_data

    # Wide header
    headers = ["variable", "year", "region"]
    if has_data:
        headers.extend(["Data", "Data_stat_err"])
    headers.extend(["Total_MC", "Total_MC_stat_err"])
    if has_data:
        headers.append("Data_over_MC")
    for bkg in BKG_ORDER:
        headers.extend([f"{bkg}", f"{bkg}_stat_err", f"{bkg}_frac_pct"])
    for sig in sig_names:
        headers.extend([f"{sig}", f"{sig}_stat_err"])

    wide_rows = []
    tidy_rows = []

    for var_name in vars_to_run:
        bkg_yields = {}
        bkg_errs = {}
        for bkg in BKG_ORDER:
            if bkg in hist_data and var_name in hist_data[bkg]:
                c = np.sum(hist_data[bkg][var_name].counts)
                e = np.sqrt(np.sum(hist_data[bkg][var_name].sumw2))
                bkg_yields[bkg] = float(c)
                bkg_errs[bkg] = float(e)
            else:
                bkg_yields[bkg] = 0.0
                bkg_errs[bkg] = 0.0

        total_mc = sum(bkg_yields.values())
        total_mc_err = np.sqrt(sum(e ** 2 for e in bkg_errs.values()))

        data_yield = float(np.sum(hist_data["Data"][var_name].counts)) if has_data and var_name in hist_data["Data"] else 0.0
        data_err = float(np.sqrt(data_yield)) if has_data else 0.0
        data_mc_ratio = (data_yield / total_mc) if total_mc > 0 and has_data else None

        row = [var_name, str(year), region]
        if has_data:
            row.extend([f"{data_yield:.2f}", f"{data_err:.2f}"])
        row.extend([f"{total_mc:.2f}", f"{total_mc_err:.2f}"])
        if has_data:
            row.append(f"{data_mc_ratio:.4f}" if data_mc_ratio is not None else "NaN")

        for bkg in BKG_ORDER:
            y = bkg_yields[bkg]
            err = bkg_errs[bkg]
            frac = (y / total_mc * 100.0) if total_mc > 0 else 0.0
            row.extend([f"{y:.2f}", f"{err:.2f}", f"{frac:.2f}"])
            tidy_rows.append([var_name, str(year), region, bkg, f"{y:.2f}", f"{err:.2f}", f"{frac:.2f}"])

        if has_data:
            tidy_rows.append([var_name, str(year), region, "Data", f"{data_yield:.2f}", f"{data_err:.2f}", "NaN"])
        tidy_rows.append([var_name, str(year), region, "Total_MC", f"{total_mc:.2f}", f"{total_mc_err:.2f}", "100.00"])

        for sig in sig_names:
            if sig in hist_data and var_name in hist_data[sig]:
                sy = float(np.sum(hist_data[sig][var_name].counts))
                se = float(np.sqrt(np.sum(hist_data[sig][var_name].sumw2)))
            else:
                sy, se = 0.0, 0.0
            row.extend([f"{sy:.2f}", f"{se:.2f}"])
            tidy_rows.append([var_name, str(year), region, sig, f"{sy:.2f}", f"{se:.2f}", "NaN"])

        wide_rows.append(row)

    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        writer.writerows(wide_rows)

    with open(csv_per_process_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["variable", "year", "region", "process", "yield", "stat_err", "fraction_of_total_bkg_pct"])
        writer.writerows(tidy_rows)

    print(f"  Wrote yields summary to {csv_path}")
    return csv_path


def plot_stacked_variable(
    var_name: str,
    hist_data: dict[str, dict[str, TH1Accumulator]],
    year: str | int,
    region: str = "validation",
    out_dir: str = "plots",
    signals_to_plot: list[str] | None = None,
    lumi_fb: float | None = None,
):
    """Generate and save stacked Data/MC comparison plot for a single variable."""
    if var_name not in VARIABLES:
        return
    var_cfg = VARIABLES[var_name]
    bins = var_cfg["bins"]
    bin_centers = 0.5 * (bins[:-1] + bins[1:])
    bin_widths = bins[1:] - bins[:-1]
    bin_half_w = bin_widths / 2.0

    # Determine if Data is present
    has_data_sample = "Data" in hist_data and var_name in hist_data["Data"]
    data_counts = hist_data["Data"][var_name].counts if has_data_sample else np.zeros_like(bin_centers)
    has_data_points = has_data_sample and (data_counts > 0).any()

    # Pre-calculate total MC yield across backgrounds
    total_mc_yield = sum(
        float(np.sum(hist_data[b][var_name].counts))
        for b in BKG_ORDER
        if b in hist_data and var_name in hist_data[b]
    )

    # Create figure
    setup_mplhep_style()
    if has_data_points:
        fig, (ax, ax_ratio) = plt.subplots(2, 1, figsize=FIG_SIZE, gridspec_kw=GRIDSPEC_KW, sharex=True)
    else:
        fig, ax = plt.subplots(figsize=(10, 8))
        ax_ratio = None

    # 1. Stacked Backgrounds
    bottom = np.zeros_like(bin_centers)
    total_mc = np.zeros_like(bin_centers)
    total_mc_w2 = np.zeros_like(bin_centers)

    for bkg_name in BKG_ORDER:
        if bkg_name in hist_data and var_name in hist_data[bkg_name]:
            acc = hist_data[bkg_name][var_name]
            style = BKG_STYLE.get(bkg_name, {"color": "#888888", "label": bkg_name})
            bkg_y = float(np.sum(acc.counts))
            bkg_frac = (bkg_y / total_mc_yield * 100.0) if total_mc_yield > 0 else 0.0
            bkg_label = f"{style['label']} ({format_yield(bkg_y)}, {bkg_frac:.1f}%)"
            ax.bar(
                bin_centers, acc.counts, width=bin_widths, bottom=bottom,
                color=style["color"], label=bkg_label, align="center", linewidth=0,
            )
            bottom += acc.counts
            total_mc += acc.counts
            total_mc_w2 += acc.sumw2

    # 2. Total MC stat uncertainty band
    total_mc_err = np.sqrt(total_mc_w2)
    if (total_mc > 0).any():
        ax.bar(
            bin_centers, 2 * total_mc_err, width=bin_widths,
            bottom=total_mc - total_mc_err,
            color="none", edgecolor=MC_UNC_COLOR, hatch=MC_UNC_HATCH,
            linewidth=0, align="center",
            label=f"MC stat. unc. ({format_yield(total_mc_yield)})", zorder=3,
        )

    # 3. Signal overlays
    sig_candidates = [k for k in hist_data if k not in BKG_ORDER and k != "Data"]
    selected_signals = signals_to_plot if signals_to_plot is not None else sig_candidates[:3]

    for idx, sig_name in enumerate(selected_signals):
        if sig_name in hist_data and var_name in hist_data[sig_name]:
            sig_acc = hist_data[sig_name][var_name]
            color = SIGNAL_PALETTE[idx % len(SIGNAL_PALETTE)]
            sig_y = float(np.sum(sig_acc.counts))
            clean_name = sig_name.replace("s-channel_", "").replace("_mDark-20", "")
            sig_label = f"{clean_name} ({format_yield(sig_y)})"
            ax.stairs(sig_acc.counts, bins, color=color, linewidth=2.0, label=sig_label, zorder=4)

    # 4. Data points
    if has_data_points:
        data_errors = np.sqrt(data_counts)
        valid = data_counts > 0
        data_y = float(np.sum(data_counts))
        data_label = f"Data ({int(round(data_y)):,})"
        ax.errorbar(
            bin_centers[valid], data_counts[valid],
            yerr=data_errors[valid], xerr=bin_half_w[valid],
            fmt="o", color="black", markersize=5, linewidth=1.2,
            label=data_label, zorder=5,
        )

    # 5. Header and badges
    add_cms_header(ax, year=year, lumi_fb=lumi_fb, has_data=has_data_points)
    add_region_badge(ax, region=region)

    # Calculate y-axis maximum
    tops = [total_mc]
    if has_data_points:
        tops.append(data_counts)
    y_max = max([np.max(t) for t in tops if len(t) > 0 and np.max(t) > 0] or [1.0])
    format_main_axes(ax, var_cfg, y_max=y_max, log_y=var_cfg.get("logy", True))
    ax.legend(loc="upper right", framealpha=0.75, ncol=2)

    # 6. Ratio Panel (Data / MC)
    if ax_ratio is not None and has_data_points:
        mc_nonzero = total_mc > 0
        ratio = np.where(mc_nonzero, data_counts / np.where(mc_nonzero, total_mc, 1.0), np.nan)
        ratio_err = np.where(mc_nonzero, np.sqrt(data_counts) / np.where(mc_nonzero, total_mc, 1.0), np.nan)
        ratio_mc_band = np.where(mc_nonzero, total_mc_err / np.where(mc_nonzero, total_mc, 1.0), np.nan)

        # MC stat band centered at unity
        ax_ratio.bar(
            bin_centers, 2 * ratio_mc_band, width=bin_widths,
            bottom=1.0 - ratio_mc_band,
            color="none", edgecolor=MC_UNC_COLOR, hatch=MC_UNC_HATCH,
            linewidth=0, align="center", zorder=2,
        )

        valid_ratio = has_data_points & mc_nonzero
        if np.any(valid_ratio):
            ax_ratio.errorbar(
                bin_centers[valid_ratio], ratio[valid_ratio],
                yerr=ratio_err[valid_ratio], xerr=bin_half_w[valid_ratio],
                fmt="o", color="black", markersize=5, linewidth=1.2, zorder=5,
            )

        format_ratio_axes(ax_ratio, var_cfg)

    if ax_ratio is not None:
        fig.subplots_adjust(hspace=0.05)
        fig.align_ylabels([ax, ax_ratio])
    else:
        plt.tight_layout()
    os.makedirs(out_dir, exist_ok=True)
    stem = f"{var_cfg.get('out_name', var_name)}_{year}_{region}"
    pdf_path = os.path.join(out_dir, f"{stem}.pdf")
    plt.savefig(pdf_path, bbox_inches="tight")
    plt.close()
    print(f"  Saved plot: {pdf_path}")


def plot_all_variables(
    hist_data: dict[str, dict[str, TH1Accumulator]],
    year: str | int,
    region: str = "validation",
    out_dir: str = "plots",
    selected_vars: list[str] | None = None,
    signals_to_plot: list[str] | None = None,
    lumi_fb: float | None = None,
):
    """Iterate through variables, render all stacked comparison plots, and export yields.csv."""
    vars_to_run = selected_vars or list(VARIABLES.keys())
    print(f"\nGenerating stacked plots for {year} ({region}) into '{out_dir}'...")
    for var_name in vars_to_run:
        plot_stacked_variable(
            var_name=var_name,
            hist_data=hist_data,
            year=year,
            region=region,
            out_dir=out_dir,
            signals_to_plot=signals_to_plot,
            lumi_fb=lumi_fb,
        )
    export_yields_csv(
        hist_data=hist_data,
        year=year,
        region=region,
        out_dir=out_dir,
        selected_vars=vars_to_run,
    )
