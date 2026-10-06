"""run_plots.py - Unified CLI driver to read skims, generate TH1s, and render stack plots."""

from __future__ import annotations
import argparse
import copy
import os
import sys

# Ensure SVJ_GNN root is in sys.path
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_DIR = os.path.dirname(SCRIPT_DIR)
if REPO_DIR not in sys.path:
    sys.path.insert(0, REPO_DIR)

from plotting.hist_reader import (
    TH1Accumulator,
    load_histograms_from_root,
    run_histogram_generation,
    save_histograms_to_root,
)
from plotting.cutflow import export_and_print_cutflow
from plotting.plot_stack import plot_all_variables
from plotting.plot_variables import VARIABLES


def default_eos_path(subfolder: str = "") -> str:
    """Return EOS output path if writable, or local fallback."""
    eos_dir = "/eos/user/n/nparmar/SVJScouting/plots"
    try:
        os.makedirs(eos_dir, exist_ok=True)
        test_path = os.path.join(eos_dir, ".test_write")
        with open(test_path, "w") as f:
            f.write("test")
        os.remove(test_path)
        return os.path.join(eos_dir, subfolder) if subfolder else eos_dir
    except Exception:
        local_dir = os.path.join(REPO_DIR, "plots")
        return os.path.join(local_dir, subfolder) if subfolder else local_dir


def parse_args():
    parser = argparse.ArgumentParser(
        description="Modular TH1 histogram generator and mplhep stack plotter."
    )
    parser.add_argument(
        "--years", nargs="+", default=["2018"],
        help="One or more years to process (e.g. 2017, 2018, or Run2).",
    )
    parser.add_argument(
        "--region", "--regions", nargs="+", default=["validation"],
        choices=["validation", "highRT", "control", "lowRT", "inclusive"],
        help="Event selection region(s) to process (validation/highRT, control, lowRT, inclusive).",
    )
    parser.add_argument(
        "--vars", nargs="+", default=None,
        help="Specific variables to plot (default: all configured variables).",
    )
    parser.add_argument(
        "--signals", nargs="+", default=None,
        help="Specific signal point keys to overlay on stack plots.",
    )
    parser.add_argument(
        "--workers", type=int, default=8,
        help="ThreadPoolExecutor workers for reading ROOT files.",
    )
    parser.add_argument(
        "--step-size", default="100 MB",
        help="uproot.iterate step size (e.g. '100 MB').",
    )
    parser.add_argument(
        "--read-only", action="store_true",
        help="Only run data reader and write histograms to ROOT file; do not plot.",
    )
    parser.add_argument(
        "--plot-only", action="store_true",
        help="Only produce plots from existing ROOT histogram files; do not re-read skims.",
    )
    parser.add_argument(
        "--combine-years", action="store_true",
        help="Combine all specified years into a single combined Run 2 histogram.",
    )
    parser.add_argument(
        "--cutflow", action="store_true",
        help="Also compute and export the cutflow table to CSV.",
    )
    parser.add_argument(
        "--root-dir", default=None,
        help="Directory to store/load histogram ROOT files.",
    )
    parser.add_argument(
        "--plot-dir", default=None,
        help="Directory to save output PNG and PDF figures.",
    )
    return parser.parse_args()


def combine_hist_results(
    yearly_results: list[dict[str, dict[str, TH1Accumulator]]]
) -> dict[str, dict[str, TH1Accumulator]]:
    """Sum histogram accumulators across multiple years."""
    if not yearly_results:
        return {}
    combined: dict[str, dict[str, TH1Accumulator]] = {}
    for year_data in yearly_results:
        for cat, var_map in year_data.items():
            if cat not in combined:
                combined[cat] = {v: acc.copy() for v, acc in var_map.items()}
            else:
                for v, acc in var_map.items():
                    if v in combined[cat]:
                        combined[cat][v].add(acc)
                    else:
                        combined[cat][v] = acc.copy()
    return combined


def main():
    args = parse_args()
    root_dir = args.root_dir or default_eos_path("histograms")
    plot_dir = args.plot_dir or default_eos_path("figures")
    os.makedirs(root_dir, exist_ok=True)
    os.makedirs(plot_dir, exist_ok=True)

    years = ["2016", "2017", "2018"] if args.years == ["Run2"] else [str(y) for y in args.years]
    should_combine = args.combine_years or len(years) > 1 or args.years == ["Run2"]

    requested_regions = [args.region] if isinstance(args.region, str) else args.region
    regions = [("validation" if r == "highRT" else r) for r in requested_regions]

    for region in regions:
        yearly_data = []
        for year in years:
            root_file = os.path.join(root_dir, f"histograms_{year}_{region}.root")
            hist_data = None

            if not args.plot_only:
                print(f"\n==========================================")
                print(f" Reading & Histogramming: Year {year} ({region})")
                print(f"==========================================")
                hist_data, cutflow_records = run_histogram_generation(
                    year=year,
                    region=region,
                    config_dir=REPO_DIR,
                    selected_vars=args.vars,
                    workers=args.workers,
                    step_size=args.step_size,
                    out_root_path=root_file,
                    track_cutflow=args.cutflow,
                )
                if args.cutflow:
                    cutflow_csv = os.path.join(plot_dir, year, region, f"cutflow_{year}_{region}.csv")
                    export_and_print_cutflow(cutflow_records, year, region, cutflow_csv)
            else:
                if not os.path.isfile(root_file):
                    print(f"Error: Histogram ROOT file not found at '{root_file}'. Cannot use --plot-only.")
                    continue
                print(f"\nLoading existing histograms from '{root_file}'...")
                hist_data = load_histograms_from_root(root_file)

            if hist_data is not None:
                yearly_data.append((year, hist_data))
                if not args.read_only:
                    year_out_dir = os.path.join(plot_dir, year, region)
                    plot_all_variables(
                        hist_data=hist_data,
                        year=year,
                        region=region,
                        out_dir=year_out_dir,
                        selected_vars=args.vars,
                        signals_to_plot=args.signals,
                    )

        # Combined years (Run 2)
        if should_combine and len(yearly_data) > 1:
            print(f"\n==========================================")
            print(f" Combining Years for {region}: {', '.join(y for y, _ in yearly_data)}")
            print(f"==========================================")
            combined_hists = combine_hist_results([data for _, data in yearly_data])
            combined_root_file = os.path.join(root_dir, f"histograms_Run2_{region}.root")
            save_histograms_to_root(combined_hists, "Run2", region, combined_root_file)

            if not args.read_only:
                combined_plot_dir = os.path.join(plot_dir, "Run2", region)
                plot_all_variables(
                    hist_data=combined_hists,
                    year="Run2",
                    region=region,
                    out_dir=combined_plot_dir,
                    selected_vars=args.vars,
                    signals_to_plot=args.signals,
                    lumi_fb=138.0,
                )


if __name__ == "__main__":
    main()

