"""cutflow.py - CLI tool to compute and export ROOT skims + analysis cutflows to CSV."""

from __future__ import annotations
import argparse
import csv
import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed

import awkward as ak
import numpy as np
import uproot
from tqdm import tqdm

# Ensure SVJ_GNN root is in sys.path
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_DIR = os.path.dirname(SCRIPT_DIR)
if REPO_DIR not in sys.path:
    sys.path.insert(0, REPO_DIR)

from plotting.hist_reader import (
    MC_WEIGHT_BRANCHES,
    QCD_KFACTOR_BY_YEAR,
    bkg_family_groups,
    bkg_family_name,
    compute_chunk_weights,
    load_trigger_sf,
    read_file_root_cutflow,
    read_gen_cross_section,
    run_histogram_generation,
    sum_initial_events,
)
from plotting.plot_cuts import (
    cut_branches,
    selection_steps,
)
from utils.InputJson import dataset_groups, flat_files, load_input_json, normalize_input_path

ORDERED_ROOT_CUT_KEYS = [
    "Initial",
    "Trigger",
    "GoodJetsAK8",
    "nJetsAK8Gt0",
    "HEMVeto",
    "nJetsAK8Gt2",
    "IsolatedLeptonVeto",
    "goodVerticesFilter",
    "PhiSpikeFilter",
    "GapJetVeto",
    "RT_selection",
    "MT_selection",
    "DeltaPhiMin_selection",
    "Final",
    "GoldenJSON",
]


def read_file_root_cutflow(file_path: str) -> dict[str, int]:
    """Read first entry of each CutFlow tree branch."""
    norm_path = normalize_input_path(file_path)
    counts = {}
    try:
        with uproot.open(norm_path, timeout=60) as f:
            if "CutFlow" not in f:
                return {}
            tree = f["CutFlow"]
            for name in tree.keys():
                clean_name = name.split(";")[0]
                if clean_name not in counts:
                    counts[clean_name] = int(tree[name].array(entry_stop=1)[0])
    except Exception:
        pass
    return counts


def process_file_cutflow(
    file_path: str,
    role: str,
    year: str | int,
    region: str,
    scale: float,
    step_size: str = "100 MB",
    trigger_sf_func=None,
) -> tuple[dict[str, int], dict[str, int], dict[str, float]]:
    """Process single file for ROOT cutflow and cumulative analysis cutflow."""
    root_cf = read_file_root_cutflow(file_path)
    analysis_counts: dict[str, int] = {}
    analysis_weighted: dict[str, float] = {}

    norm_path = normalize_input_path(file_path)
    needed = set(cut_branches(role, year))
    if role != "data":
        for b in MC_WEIGHT_BRANCHES:
            needed.add(b)

    tree_path = f"{norm_path}:Events"
    try:
        for batch in uproot.iterate(tree_path, expressions=sorted(needed), library="ak", step_size=step_size):
            if len(batch) == 0:
                continue
            steps = selection_steps(batch, role, year, region=region)
            weights = compute_chunk_weights(batch, role, year, scale, trigger_sf_func)

            cum_mask = np.ones(len(batch), dtype=bool)
            for step_label, step_mask in steps:
                cum_mask &= np.asarray(step_mask, dtype=bool)
                n_pass = int(cum_mask.sum())
                analysis_counts[step_label] = analysis_counts.get(step_label, 0) + n_pass
                w_pass = float(weights[cum_mask].sum()) if n_pass > 0 else 0.0
                analysis_weighted[step_label] = analysis_weighted.get(step_label, 0.0) + w_pass
    except Exception as exc:
        tqdm.write(f"  Warning: failed reading events in {os.path.basename(file_path)}: {exc}")

    return root_cf, analysis_counts, analysis_weighted


def accumulate_dataset_cutflow(
    file_paths: list[str],
    role: str,
    year: str | int,
    region: str,
    scale: float,
    workers: int = 8,
    step_size: str = "100 MB",
    trigger_sf_func=None,
    desc: str = "",
) -> tuple[dict[str, int], dict[str, int], dict[str, float]]:
    """Parallel accumulation of cutflows across files using ThreadPoolExecutor."""
    total_root: dict[str, int] = {}
    total_analysis_counts: dict[str, int] = {}
    total_analysis_weighted: dict[str, float] = {}

    if not file_paths:
        return total_root, total_analysis_counts, total_analysis_weighted

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {
            pool.submit(
                process_file_cutflow, p, role, year, region, scale, step_size, trigger_sf_func
            ): p
            for p in file_paths
        }
        for future in tqdm(as_completed(futures), total=len(futures), desc=desc, unit="file", leave=False):
            rcf, ac_counts, ac_weighted = future.result()
            for k, v in rcf.items():
                total_root[k] = total_root.get(k, 0) + v
            for k, v in ac_counts.items():
                total_analysis_counts[k] = total_analysis_counts.get(k, 0) + v
            for k, v in ac_weighted.items():
                total_analysis_weighted[k] = total_analysis_weighted.get(k, 0.0) + v

    return total_root, total_analysis_counts, total_analysis_weighted


def run_cutflow(
    year: str | int,
    region: str = "validation",
    config_dir: str = "",
    detailed: bool = False,
    workers: int = 8,
    step_size: str = "100 MB",
    out_csv_path: str = "",
) -> list[dict]:
    """Run full cutflow accumulation for all datasets and write cutflow table to CSV."""
    _, records = run_histogram_generation(
        year=year,
        region=region,
        config_dir=config_dir,
        selected_vars=[],
        workers=workers,
        step_size=step_size,
        out_root_path="",
        track_cutflow=True,
        detailed_cutflow=detailed,
    )
    export_and_print_cutflow(records, str(year), region, out_csv_path)
    return records


def export_and_print_cutflow(records: list[dict], year: str | int, region: str, out_csv_path: str = ""):
    """Format and write cutflow table to CSV and print ASCII summary."""
    if not records:
        print("No cutflow records to export.")
        return

    # Discover all ROOT cut keys
    all_root_keys = []
    seen_root = set()
    for k in ORDERED_ROOT_CUT_KEYS:
        if any(k in r["root_cf"] for r in records):
            all_root_keys.append(k)
            seen_root.add(k)
    for r in records:
        for k in sorted(r["root_cf"].keys()):
            if k not in seen_root and not k.startswith(("Initial", "Lund", "PDF", "Scale", "PS")):
                all_root_keys.append(k)
                seen_root.add(k)

    # Discover analysis cut keys
    all_analysis_keys = []
    seen_analysis = set()
    for r in records:
        for k in r["analysis_cf"].keys():
            if k not in seen_analysis:
                all_analysis_keys.append(k)
                seen_analysis.add(k)

    # CSV headers
    headers = ["Process", "Type", "Year", "Region"]
    headers.extend([f"ROOT/{k}" for k in all_root_keys])
    headers.extend([f"Analysis/{k}" for k in all_analysis_keys])
    headers.extend(["Final_Raw_Pass", "Final_Weighted_Yield"])

    csv_rows = []
    for r in records:
        row = [r["name"], r["type"], str(year), region]
        for k in all_root_keys:
            row.append(r["root_cf"].get(k, ""))
        for k in all_analysis_keys:
            row.append(r["analysis_cf"].get(k, ""))

        final_raw = r["analysis_cf"].get(all_analysis_keys[-1], 0) if all_analysis_keys else 0
        final_w = r["analysis_weighted"].get(all_analysis_keys[-1], 0.0) if all_analysis_keys else 0.0
        row.append(final_raw)
        row.append(f"{final_w:.2f}" if r["type"] != "data" else f"{final_raw:.2f}")
        csv_rows.append(row)

    # Write CSV
    if not out_csv_path:
        out_dir = os.path.join(REPO_DIR, "plots", "cutflow")
        os.makedirs(out_dir, exist_ok=True)
        out_csv_path = os.path.join(out_dir, f"cutflow_{year}_{region}.csv")
    else:
        os.makedirs(os.path.dirname(os.path.abspath(out_csv_path)), exist_ok=True)

    with open(out_csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        writer.writerows(csv_rows)

    print(f"\n============================================================")
    print(f" Cutflow Summary Table for {year} ({region})")
    print(f" Saved to: {out_csv_path}")
    print(f"============================================================")

    # ASCII Print
    col_w = 14
    first_w = 28
    print_keys = [k for k in all_root_keys if k in ("Initial", "Trigger", "HEMVeto", "RT_selection", "Final")]
    print_keys.extend([k for k in all_analysis_keys if k != "Events_skim"])
    print_keys.append("Final_Weighted")

    header_line = f"{'Process':<{first_w}}"
    for k in print_keys:
        header_line += f"{k[:col_w-1]:>{col_w}}"
    print(header_line)
    print("-" * len(header_line))

    for r in records:
        if r["type"] == "background_sub":
            continue
        line = f"{r['name']:<{first_w}}"
        for k in print_keys:
            if k == "Final_Weighted":
                fw = r["analysis_weighted"].get(all_analysis_keys[-1], 0.0) if all_analysis_keys else 0.0
                val = f"{fw:.1f}" if r["type"] != "data" else f"{r['analysis_cf'].get(all_analysis_keys[-1], 0):.1f}"
            elif k in all_root_keys:
                val = str(r["root_cf"].get(k, "-"))
            else:
                val = str(r["analysis_cf"].get(k, "-"))
            line += f"{val:>{col_w}}"
        print(line)
    print()


def parse_args():
    parser = argparse.ArgumentParser(
        description="Compute and export ROOT skims + analysis cutflow table to CSV."
    )
    parser.add_argument("--years", nargs="+", default=["2018"], help="Year(s) to process.")
    parser.add_argument(
        "--region", "--regions", nargs="+", default=["validation"],
        choices=["validation", "highRT", "control", "lowRT", "inclusive"],
        help="Event selection region(s).",
    )
    parser.add_argument(
        "--detailed", action="store_true",
        help="Include individual sub-datasets in addition to major families.",
    )
    parser.add_argument(
        "--workers", type=int, default=8,
        help="ThreadPoolExecutor workers.",
    )
    parser.add_argument(
        "--step-size", default="100 MB",
        help="uproot.iterate step size.",
    )
    parser.add_argument(
        "--output-csv", default="",
        help="Custom output CSV path.",
    )
    parser.add_argument(
        "--out-dir", default="",
        help="Directory to save cutflow CSV files.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    requested_regions = [args.region] if isinstance(args.region, str) else args.region
    regions = [("validation" if r == "highRT" else r) for r in requested_regions]
    for region in regions:
        for year in args.years:
            csv_path = args.output_csv
            if not csv_path and args.out_dir:
                csv_path = os.path.join(args.out_dir, f"cutflow_{year}_{region}.csv")
            run_cutflow(
                year=year,
                region=region,
                config_dir=REPO_DIR,
                detailed=args.detailed,
                workers=args.workers,
                step_size=args.step_size,
                out_csv_path=csv_path,
            )


if __name__ == "__main__":
    main()

