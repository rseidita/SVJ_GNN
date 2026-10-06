"""hist_reader.py - Multi-threaded streaming ROOT reader and TH1 histogram generator."""

from __future__ import annotations
import argparse
import copy
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import awkward as ak
import numpy as np
import uproot
from tqdm import tqdm

from plotting.plot_cuts import cut_branches, get_event_selection, post_hem_mask, selection_steps
from plotting.plot_variables import VARIABLES, extract_variable, required_branches_for_vars
from utils.InputJson import dataset_groups, dataset_of, flat_files, load_input_json, normalize_input_path

LUMI_FB = {
    "2016": {"lumi": 36.31},
    "2017": {"lumi": 41.53},
    "2018": {"lumi": 59.74, "lumi_pre": 21.1, "lumi_hem": 38.64},
}

QCD_DATASET_PREFIX = "QCD_HT"
MC_WEIGHT_BRANCHES = ("prefire", "genWeightPU", "genWeight", "MT01FatJetMET")
QCD_KFACTOR_BY_YEAR = {"2017": 1.0, "2018": 1.0}


class TH1Accumulator:
    """Thread-safe histogram accumulator tracking counts and sum of squared weights."""

    def __init__(self, bins: np.ndarray, name: str = "", title: str = ""):
        self.bins = np.asarray(bins, dtype=float)
        self.name = name
        self.title = title
        n_bins = len(self.bins) - 1
        self.counts = np.zeros(n_bins, dtype=float)
        self.sumw2 = np.zeros(n_bins, dtype=float)

    def fill(self, values: np.ndarray, weights: np.ndarray | None = None):
        if len(values) == 0:
            return
        finite = np.isfinite(values)
        val = values[finite]
        w = weights[finite] if weights is not None else np.ones_like(val, dtype=float)
        if len(val) == 0:
            return

        # Underflow dropped, overflow clipped to last bin
        valid = val >= self.bins[0]
        val = np.clip(val[valid], self.bins[0], self.bins[-1] - 1e-9)
        w = w[valid]

        c, _ = np.histogram(val, bins=self.bins, weights=w)
        w2, _ = np.histogram(val, bins=self.bins, weights=w ** 2)
        self.counts += c
        self.sumw2 += w2

    def add(self, other: TH1Accumulator):
        self.counts += other.counts
        self.sumw2 += other.sumw2

    def copy(self) -> TH1Accumulator:
        cloned = TH1Accumulator(self.bins, self.name, self.title)
        cloned.counts = self.counts.copy()
        cloned.sumw2 = self.sumw2.copy()
        return cloned


def bkg_family_name(dataset_name: str) -> str:
    """Map sub-dataset to its major background family name."""
    if dataset_name == "qcd" or dataset_name.startswith("QCD"):
        return "QCD"
    for fam in ("TTJets", "WJets", "ZJets"):
        if dataset_name.startswith(fam):
            return fam
    return dataset_name


def bkg_family_groups(bg_groups: dict[str, list[str]]) -> dict[str, dict[str, list[str]]]:
    """Group background sub-datasets by major family: {family: {sub_ds: [files]}}."""
    families: dict[str, dict[str, list[str]]] = {}
    for name, files in bg_groups.items():
        fam = bkg_family_name(name)
        families.setdefault(fam, {})[name] = files
    return families


def load_trigger_sf(base_dir: str, year: str | int):
    """Load MT-binned trigger scale factors if the json file exists."""
    path = os.path.join(base_dir, "..", "trigger_sf", f"trigger_sf_counting_{year}.json")
    if not os.path.isfile(path):
        return None
    try:
        with open(path) as handle:
            payload = json.load(handle)
        edges = [b["mt_low"] for b in payload["bins"]] + [payload["bins"][-1]["mt_high"]]
        values = [1.0 if b.get("sf") is None else float(b["sf"]) for b in payload["bins"]]
        edges_arr = np.asarray(edges, dtype=float)
        values_arr = np.asarray(values, dtype=float)

        def evaluate(mt):
            mt_arr = np.asarray(mt, dtype=float)
            idx = np.searchsorted(edges_arr, mt_arr, side="right") - 1
            inside = (idx >= 0) & (idx < len(values_arr))
            out = np.ones_like(mt_arr)
            out[inside] = values_arr[idx[inside]]
            return out

        return evaluate
    except Exception:
        return None


def sum_initial_events(files: list[str]) -> int:
    """Sum CutFlow/Initial entries over all files of a dataset."""
    total = 0
    for path in files:
        try:
            with uproot.open(normalize_input_path(path), timeout=60) as f:
                if "CutFlow" in f and "Initial" in f["CutFlow"]:
                    total += int(f["CutFlow"]["Initial"].array(entry_stop=1)[0])
        except Exception:
            pass
    return total


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


def read_gen_cross_section(file_path: str) -> float:
    """Read GenCrossSection from Metadata tree."""
    with uproot.open(normalize_input_path(file_path), timeout=60) as f:
        if "Metadata" in f and "GenCrossSection" in f["Metadata"]:
            return float(f["Metadata"]["GenCrossSection"].array(entry_stop=1)[0])
    raise ValueError(f"'{file_path}' has no Metadata/GenCrossSection")


def pu_over_gen(events: ak.Array) -> np.ndarray:
    """Safe per-event genWeightPU / genWeight ratio."""
    if "genWeightPU" not in ak.fields(events) or "genWeight" not in ak.fields(events):
        return np.ones(len(events), dtype=float)
    genw_pu = ak.to_numpy(events["genWeightPU"])
    genw = ak.to_numpy(events["genWeight"])
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.divide(genw_pu, genw, out=np.ones_like(genw_pu, dtype=float), where=(genw != 0))
    ratio[~np.isfinite(ratio)] = 1.0
    return ratio


def compute_chunk_weights(
    events: ak.Array,
    role: str,
    year: str | int,
    scale: float,
    trigger_sf_func=None,
) -> np.ndarray:
    """Calculate event weights for a single awkward chunk."""
    n = len(events)
    if role == "data":
        return np.ones(n, dtype=float)

    prefire = ak.to_numpy(events["prefire"]) if "prefire" in ak.fields(events) else np.ones(n, dtype=float)
    base_weight = scale * prefire * pu_over_gen(events)

    yr_str = str(year)
    lumi = LUMI_FB.get(yr_str, {"lumi": 59.74})
    if yr_str == "2018" and "lumi_pre" in lumi and "lumi_hem" in lumi:
        is_post = post_hem_mask(events, role, year)
        lumi_eff = np.where(is_post, lumi["lumi_hem"], lumi["lumi_pre"])
    else:
        lumi_eff = float(lumi["lumi"])

    weight = base_weight * lumi_eff
    if trigger_sf_func is not None and "MT01FatJetMET" in ak.fields(events):
        weight = weight * trigger_sf_func(ak.to_numpy(events["MT01FatJetMET"]))
    return weight


def process_one_file(
    file_path: str,
    role: str,
    year: str | int,
    region: str,
    scale: float,
    var_dict: dict[str, dict],
    step_size: str = "100 MB",
    trigger_sf_func=None,
    track_cutflow: bool = True,
) -> tuple[dict[str, TH1Accumulator], dict[str, int], dict[str, int], dict[str, float]]:
    """Process a single ROOT file in streaming chunks and return partial histograms and cutflow."""
    local_hists = {var: TH1Accumulator(cfg["bins"], name=var) for var, cfg in var_dict.items()}
    norm_path = normalize_input_path(file_path)
    root_cf = read_file_root_cutflow(file_path) if track_cutflow else {}
    analysis_counts: dict[str, int] = {}
    analysis_weighted: dict[str, float] = {}

    # Determine needed branches
    needed = set(cut_branches(role, year))
    needed.update(required_branches_for_vars(list(var_dict.keys())))
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
                if track_cutflow:
                    n_pass = int(cum_mask.sum())
                    analysis_counts[step_label] = analysis_counts.get(step_label, 0) + n_pass
                    w_pass = float(weights[cum_mask].sum()) if n_pass > 0 else 0.0
                    analysis_weighted[step_label] = analysis_weighted.get(step_label, 0.0) + w_pass

            if not np.any(cum_mask):
                continue
            for var_name, var_cfg in var_dict.items():
                values = extract_variable(batch, var_name, var_cfg)
                local_hists[var_name].fill(values[cum_mask], weights[cum_mask])
    except Exception as exc:
        tqdm.write(f"  Warning: failed reading {os.path.basename(file_path)}: {exc}")

    return local_hists, root_cf, analysis_counts, analysis_weighted


def process_dataset_files(
    file_paths: list[str],
    role: str,
    year: str | int,
    region: str,
    scale: float,
    var_dict: dict[str, dict],
    workers: int = 8,
    step_size: str = "100 MB",
    trigger_sf_func=None,
    track_cutflow: bool = True,
    desc: str = "",
) -> tuple[dict[str, TH1Accumulator], dict[str, int], dict[str, int], dict[str, float]]:
    """Parallel accumulation across files of a dataset using ThreadPoolExecutor."""
    accumulated = {var: TH1Accumulator(cfg["bins"], name=var) for var, cfg in var_dict.items()}
    total_root: dict[str, int] = {}
    total_analysis_counts: dict[str, int] = {}
    total_analysis_weighted: dict[str, float] = {}
    if not file_paths:
        return accumulated, total_root, total_analysis_counts, total_analysis_weighted

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {
            pool.submit(
                process_one_file, path, role, year, region, scale, var_dict, step_size, trigger_sf_func, track_cutflow
            ): path
            for path in file_paths
        }
        for future in tqdm(as_completed(futures), total=len(futures), desc=desc, unit="file", leave=False):
            partial_hists, rcf, ac_counts, ac_weighted = future.result()
            for var_name in var_dict:
                accumulated[var_name].add(partial_hists[var_name])
            if track_cutflow:
                for k, v in rcf.items():
                    total_root[k] = total_root.get(k, 0) + v
                for k, v in ac_counts.items():
                    total_analysis_counts[k] = total_analysis_counts.get(k, 0) + v
                for k, v in ac_weighted.items():
                    total_analysis_weighted[k] = total_analysis_weighted.get(k, 0.0) + v

    return accumulated, total_root, total_analysis_counts, total_analysis_weighted


def run_histogram_generation(
    year: str | int,
    region: str = "validation",
    config_dir: str = "",
    selected_vars: list[str] | None = None,
    workers: int = 8,
    step_size: str = "100 MB",
    out_root_path: str = "",
    track_cutflow: bool = True,
    detailed_cutflow: bool = False,
) -> tuple[dict[str, dict[str, TH1Accumulator]], list[dict]]:
    """Run full histogram production and simultaneous cutflow accumulation in a single pass."""
    yr_str = str(year)
    base_dir = config_dir or os.path.dirname(os.path.abspath(__file__))
    input_json_dir = os.path.join(base_dir, "..", "inputJson") if os.path.basename(base_dir) == "plotting" else os.path.join(base_dir, "inputJson")

    # Filter variables
    active_vars = {k: v for k, v in VARIABLES.items() if (selected_vars is None or k in selected_vars)}
    trigger_sf = load_trigger_sf(base_dir, yr_str)

    results: dict[str, dict[str, TH1Accumulator]] = {}
    records: list[dict] = []
    initial_events_cache: dict[str, int] = {}

    # 1. Backgrounds (grouped by major family)
    bg_json = os.path.join(input_json_dir, "background", f"{yr_str}_background.json")
    if os.path.isfile(bg_json):
        bg_spec = load_input_json(bg_json)
        bg_groups = dataset_groups(bg_spec)
        bkg_families = bkg_family_groups(bg_groups)

        for fam_name in sorted(bkg_families):
            print(f"\nProcessing background family '{fam_name}' for {yr_str}...")
            results[fam_name] = {var: TH1Accumulator(cfg["bins"], name=var) for var, cfg in active_vars.items()}
            fam_root: dict[str, int] = {}
            fam_analysis: dict[str, int] = {}
            fam_weighted: dict[str, float] = {}

            sub_datasets = bkg_families[fam_name]
            for sub_name, files in sub_datasets.items():
                if not files:
                    continue
                # Calculate sub-dataset scale
                if sub_name not in initial_events_cache:
                    initial_events_cache[sub_name] = sum_initial_events(files)
                n_init = initial_events_cache[sub_name]
                if n_init <= 0:
                    continue
                xsec = read_gen_cross_section(files[0])
                kfactor = QCD_KFACTOR_BY_YEAR.get(yr_str, 1.0) if fam_name == "QCD" else 1.0
                scale = xsec * 1e3 / float(n_init) * kfactor

                sub_hists, rcf, ac_counts, ac_weighted = process_dataset_files(
                    files, "background", yr_str, region, scale, active_vars,
                    workers=workers, step_size=step_size, trigger_sf_func=trigger_sf,
                    track_cutflow=track_cutflow, desc=f"  {sub_name}",
                )
                for var_name in active_vars:
                    results[fam_name][var_name].add(sub_hists[var_name])

                if track_cutflow:
                    if detailed_cutflow:
                        records.append({
                            "name": sub_name, "type": "background_sub",
                            "root_cf": rcf, "analysis_cf": ac_counts, "analysis_weighted": ac_weighted,
                        })
                    for k, v in rcf.items():
                        fam_root[k] = fam_root.get(k, 0) + v
                    for k, v in ac_counts.items():
                        fam_analysis[k] = fam_analysis.get(k, 0) + v
                    for k, v in ac_weighted.items():
                        fam_weighted[k] = fam_weighted.get(k, 0.0) + v

            if track_cutflow:
                records.append({
                    "name": fam_name, "type": "background",
                    "root_cf": fam_root, "analysis_cf": fam_analysis, "analysis_weighted": fam_weighted,
                })

    # 2. Data
    data_json = os.path.join(input_json_dir, "data", f"{yr_str}_data.json")
    if os.path.isfile(data_json):
        print(f"\nProcessing Data for {yr_str}...")
        data_spec = load_input_json(data_json)
        data_groups = dataset_groups(data_spec)
        data_files = flat_files(data_groups)
        data_hists, rcf, ac_counts, ac_weighted = process_dataset_files(
            data_files, "data", yr_str, region, scale=1.0, var_dict=active_vars,
            workers=workers, step_size=step_size, trigger_sf_func=None,
            track_cutflow=track_cutflow, desc=f"  Data ({len(data_files)} files)",
        )
        results["Data"] = data_hists
        if track_cutflow:
            records.append({
                "name": "Data", "type": "data",
                "root_cf": rcf, "analysis_cf": ac_counts, "analysis_weighted": ac_weighted,
            })

    # 3. Signals
    sig_json = os.path.join(input_json_dir, "signal", f"{yr_str}_signal.json")
    if os.path.isfile(sig_json):
        sig_spec = load_input_json(sig_json)
        sig_groups = dataset_groups(sig_spec)
        for sig_name, files in sorted(sig_groups.items()):
            if not files:
                continue
            if sig_name not in initial_events_cache:
                initial_events_cache[sig_name] = sum_initial_events(files)
            n_init = initial_events_cache[sig_name]
            if n_init <= 0:
                continue
            try:
                xsec = read_gen_cross_section(files[0])
            except Exception:
                continue
            scale = xsec * 1e3 / float(n_init)
            sig_hists, rcf, ac_counts, ac_weighted = process_dataset_files(
                files, "signal", yr_str, region, scale, active_vars,
                workers=workers, step_size=step_size, trigger_sf_func=trigger_sf,
                track_cutflow=track_cutflow, desc=f"  Signal {sig_name}",
            )
            results[sig_name] = sig_hists
            if track_cutflow:
                records.append({
                    "name": sig_name, "type": "signal",
                    "root_cf": rcf, "analysis_cf": ac_counts, "analysis_weighted": ac_weighted,
                })

    # 4. Save to ROOT file
    if out_root_path:
        save_histograms_to_root(results, yr_str, region, out_root_path)

    return results, records


def save_histograms_to_root(
    hist_results: dict[str, dict[str, TH1Accumulator]],
    year: str | int,
    region: str,
    out_root_path: str,
):
    """Serialize histograms into a ROOT file with {process}/{variable} structure."""
    out_dir = os.path.dirname(os.path.abspath(out_root_path))
    os.makedirs(out_dir, exist_ok=True)
    with uproot.recreate(out_root_path) as f:
        for category, var_map in hist_results.items():
            for var_name, acc in var_map.items():
                key = f"{category}/{var_name}"
                f[key] = (acc.counts, acc.bins)
                f[f"{category}/{var_name}_sumw2"] = (acc.sumw2, acc.bins)
    print(f"\nSuccessfully wrote histograms to {out_root_path}")


def load_histograms_from_root(in_root_path: str) -> dict[str, dict[str, TH1Accumulator]]:
    """Load TH1Accumulators back from a saved ROOT file."""
    results: dict[str, dict[str, TH1Accumulator]] = {}
    with uproot.open(in_root_path) as f:
        for full_key in f.keys():
            clean_key = full_key.split(";")[0]
            if "_sumw2" in clean_key:
                continue
            parts = clean_key.split("/")
            if len(parts) != 2:
                continue
            category, var_name = parts
            h = f[clean_key]
            counts, edges = h.to_numpy()
            acc = TH1Accumulator(edges, name=var_name)
            acc.counts = counts.astype(float)
            sumw2_key = f"{category}/{var_name}_sumw2"
            if sumw2_key in f:
                w2, _ = f[sumw2_key].to_numpy()
                acc.sumw2 = w2.astype(float)
            else:
                acc.sumw2 = counts.astype(float)
            results.setdefault(category, {})[var_name] = acc
    return results

