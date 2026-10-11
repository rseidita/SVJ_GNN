#!/usr/bin/env python3
"""Write inputJson/<type>/<year>_<type>.json by listing the skims with xrdfs.

    python make_input_json.py                      # 2017 and 2018, all types
    python make_input_json.py --years 2018 --types data

Requires a valid grid proxy (voms-proxy-init -voms cms) and the xrdfs
command, both available in the LCG view sourced by init.sh.

Dataset directories directly below each base path are taken as datasets;
which ones belong to a sample type is decided by the name prefixes in
DATASET_PREFIXES, because the MC skim area also holds data and signal
directories.
"""

import argparse
import os
import subprocess

from utils.InputJson import write_input_json


HOST = "root://cmsdcache-kit-disk.gridka.de:1094"
STORE = "/store/user/mgaisdor"
SELECTION = "s_channel_scouting_pre_selection/nominal"

BASE_PATHS = {
    "data": f"{STORE}/SVJScouting_skims_v2_noResiduals/{{year}}/{SELECTION}",
    "background": f"{STORE}/SVJScouting_skims_v2/{{year}}/{SELECTION}",
    "signal": f"{STORE}/SVJScouting_skims_full_sys/{{year}}/{SELECTION}",
}

DATASET_PREFIXES = {
    "data": ("Run",),
    "background": ("QCD_", "TTJets_", "WJetsToLNu_", "ZJetsToNuNu_"),
    "signal": ("s-channel_",),
}

# 2016 background skims lack the scouting collections.
YEARS = ("2017", "2018")


def xrdfs_ls(path):
    result = subprocess.run(
        ["xrdfs", HOST, "ls", path],
        check=True,
        capture_output=True,
        text=True,
    )
    return sorted(line.strip() for line in result.stdout.splitlines() if line.strip())


def list_datasets(sample_type, year):
    base = BASE_PATHS[sample_type].format(year=year)
    prefixes = DATASET_PREFIXES[sample_type]
    datasets = {}
    for entry in xrdfs_ls(base):
        name = os.path.basename(entry)
        if not name.startswith(prefixes):
            continue
        files = [
            f"{HOST}/{file_path}"
            for file_path in xrdfs_ls(entry)
            if file_path.endswith(".root")
        ]
        if files:
            datasets[name] = files
        print(f"  {name}: {len(files)} files")
    return f"{HOST}/{base}/", datasets


def parse_arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--years", nargs="+", default=list(YEARS))
    parser.add_argument(
        "--types", nargs="+", default=list(BASE_PATHS), choices=list(BASE_PATHS)
    )
    parser.add_argument("--output-dir", default="inputJson")
    return parser.parse_args()


def main():
    args = parse_arguments()
    for sample_type in args.types:
        for year in args.years:
            print(f"{sample_type} {year}")
            base, datasets = list_datasets(sample_type, year)
            if not datasets:
                print(f"  no datasets found under {base}; nothing written")
                continue
            output_path = os.path.join(
                args.output_dir, sample_type, f"{year}_{sample_type}.json"
            )
            write_input_json(output_path, year, sample_type, base, datasets)
            total = sum(len(files) for files in datasets.values())
            print(f"  -> {output_path} ({len(datasets)} datasets, {total} files)")


if __name__ == "__main__":
    main()
