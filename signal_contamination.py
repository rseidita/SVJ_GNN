#!/usr/bin/env python3
"""Signal contamination in the GAE control region (utils/Cuts.py selections).

Combines data and pooled QCD per year; one row per signal point. MC weights
follow DataMCPlotter_final: xsec * L * 1e3 / N_initial * prefire *
(genWeightPU/genWeight), optional trigger SF on MT01FatJetMET.

    python signal_contamination.py --years 2018
    python signal_contamination.py --years 2018 --debug

Reads every file listed in inputJson/ (data, pooled QCD_HT*, each signal point).
Cached aggregates live in cache_signal_contamination/ (see --refresh-cache).
Use --bucket and --shard only for Condor shards; --merge-json to combine shards.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import awkward as ak
import numpy as np
import uproot
from tqdm import tqdm

from utils.BucketCache import (
    bucket_cache_key,
    load_bucket_cache,
    save_bucket_cache,
)
from utils.Cuts import (
    add_cutflow,
    cut_branches,
    cuts_fingerprint,
    event_cut_mask,
    format_cutflow,
    post_hem_mask,
    selection_steps,
)
from utils.InputJson import (
    dataset_groups,
    dataset_of,
    flat_files,
    load_input_json,
    normalize_input_path,
    shard_files,
)

LUMI_FB = {
    "2016": {"lumi": 36.31},
    "2017": {"lumi": 42.07},
    "2018": {"lumi": 59.74, "lumi_pre": 21.1, "lumi_hem": 38.64},
}

QCD_DATASET_PREFIX = "QCD_HT"
JSON_VERSION = 3
MC_WEIGHT_BRANCHES = ("prefire", "genWeightPU", "genWeight", "MT01FatJetMET")
QCD_KFACTOR_BY_YEAR = {"2017": 1.0, "2018": 1.0}


@dataclasses.dataclass
class McWeightContext:
    """Per-event MC scale: xsec * 1e3 / N_initial (dataset), times QCD k-factor."""

    scale: float


@dataclasses.dataclass
class BucketStats:
    """Passing-event totals for one bucket (data, qcd, or one signal point)."""

    n_pass: int = 0
    w_pass: float = 0.0
    n_pre: int = 0
    n_post: int = 0
    w_pre: float = 0.0
    w_post: float = 0.0
    cutflow: dict = dataclasses.field(default_factory=dict)
    root_cutflow: dict = dataclasses.field(default_factory=dict)
    weighted_cutflow: dict = dataclasses.field(default_factory=dict)

    def add_counts(self, other):
        self.n_pass += other.n_pass
        self.w_pass += other.w_pass
        self.n_pre += other.n_pre
        self.n_post += other.n_post
        self.w_pre += other.w_pre
        self.w_post += other.w_post
        add_cutflow(self.root_cutflow, other.root_cutflow)
        add_weighted_cutflow(self.weighted_cutflow, other.weighted_cutflow)


class IoProgress:
    """Tracks read/cut/metadata time; verbose timing only with --debug."""

    def __init__(
        self,
        quiet=False,
        debug=False,
        log_every_batches=5,
        log_every_seconds=30.0,
    ):
        self.quiet = quiet
        self.debug = debug
        self.log_every_batches = log_every_batches
        self.log_every_seconds = log_every_seconds
        self.batches = 0
        self.entries_read = 0
        self.metadata_io = 0.0
        self.events_read = 0.0
        self.cuts = 0.0
        self._last_log = time.perf_counter()
        self._bucket_start = time.perf_counter()
        self._bucket_label = ""

    def add_read(self, dt):
        self.events_read += dt

    def add_cuts(self, dt):
        self.cuts += dt

    def start_bucket(self, label, n_files, step_size, num_workers, n_branches):
        self._bucket_label = label
        self._bucket_start = time.perf_counter()
        self._last_log = self._bucket_start
        self.batches = 0
        self.entries_read = 0
        self.metadata_io = 0.0
        self.events_read = 0.0
        self.cuts = 0.0
        if not self.debug:
            return
        print(
            f"  I/O: iterate chunks  step_size={step_size!r}  "
            f"file_workers={getattr(self, '_file_workers', 1)}  "
            f"uproot_workers={num_workers}  files={n_files}  branches={n_branches}"
        )

    def maybe_log(self, n_pass_total):
        if not self.debug:
            return
        now = time.perf_counter()
        if (
            self.batches % self.log_every_batches != 0
            and (now - self._last_log) < self.log_every_seconds
        ):
            return
        self._last_log = now
        wall = now - self._bucket_start
        print(
            f"  [{self._bucket_label}] chunk {self.batches}  "
            f"entries_read={self.entries_read}  pass={n_pass_total}  "
            f"wall={wall:.0f}s  "
            f"read={self.events_read:.1f}s  cuts={self.cuts:.1f}s  "
            f"metadata={self.metadata_io:.1f}s"
        )

    def report(self, title):
        if not self.debug:
            return
        wall = time.perf_counter() - self._bucket_start
        accounted = self.metadata_io + self.events_read + self.cuts
        print(f"\nTiming summary — {title} ({self.batches} chunks, {wall:.1f}s wall)")
        if accounted <= 0:
            print("  (no timed read/cut activity recorded)")
            return
        other = max(0.0, wall - accounted)
        for label, value in (
            ("events_read", self.events_read),
            ("cuts", self.cuts),
            ("metadata_io", self.metadata_io),
            ("other/wait", other),
        ):
            pct = 100.0 * value / wall if wall else 0.0
            print(f"  {label:<14} {value:8.2f}s  {pct:5.1f}% of wall time")


def parse_arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--years", nargs="+", default=["2018"])
    parser.add_argument(
        "--max-events-per-file",
        type=int,
        default=None,
        help="Cap entries read in each ROOT file",
    )
    parser.add_argument(
        "--max-events-total",
        type=int,
        default=None,
        help="Stop a bucket after this many skim entries read in total",
    )
    parser.add_argument(
        "--lumi-fb-1",
        type=float,
        default=None,
        help="Override integrated luminosity (fb-1) for the processed year(s)",
    )
    parser.add_argument(
        "--background",
        action="store_true",
        help="Print cutflow for non-QCD background datasets only",
    )
    parser.add_argument(
        "--bucket",
        default=None,
        help="Process only this bucket: data, qcd, or a signal dataset name",
    )
    parser.add_argument("--files-per-job", type=int, default=None)
    parser.add_argument("--shard", type=int, default=None)
    parser.add_argument(
        "--step-size",
        default="100 MB",
        help="uproot.iterate chunk size (memory units, e.g. '50 MB')",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=8,
        help="Parallel ROOT files (ThreadPoolExecutor over per-file uproot.iterate)",
    )
    parser.add_argument(
        "--uproot-workers",
        type=int,
        default=1,
        help="uproot.iterate num_workers inside each file task",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=None,
        help="Alias for --uproot-workers (deprecated)",
    )
    parser.add_argument(
        "--no-trigger-sf",
        action="store_true",
        help="Do not apply trigger data/MC SFs to MC (default: apply if JSON exists)",
    )
    parser.add_argument(
        "--refresh-cache",
        action="store_true",
        help="Ignore cache_signal_contamination/ and re-read ROOT files",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress tqdm progress bars",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Print I/O timing, metadata file opens, and chunk wall-time logs",
    )
    parser.add_argument(
        "--progress-every",
        type=int,
        default=5,
        help="Print progress every N chunks (and at least every 30s)",
    )
    parser.add_argument(
        "--output-json",
        default=None,
        help="Write partial bucket counters for Condor shard merge",
    )
    parser.add_argument(
        "--merge-json",
        nargs="+",
        default=None,
        help="Merge shard JSON files and print contamination tables",
    )
    parser.add_argument(
        "--config-dir",
        default=os.path.dirname(os.path.abspath(__file__)),
    )
    return parser.parse_args()


def input_json_path(config_dir, sample_type, year):
    return os.path.join(
        config_dir, "inputJson", sample_type, f"{year}_{sample_type}.json"
    )


def lumi_spec(args, year):
    year = str(year)
    if args.lumi_fb_1 is not None:
        return {"lumi": float(args.lumi_fb_1)}
    if year not in LUMI_FB:
        raise KeyError(f"No LUMI_FB entry for {year!r}; pass --lumi-fb-1")
    return dict(LUMI_FB[year])


def physical_yield(stats, role, year, lumi):
    """Expected events at luminosity: data counts; MC = sum of per-event weights."""
    if role == "data":
        return float(stats.n_pass)
    return stats.w_pass


def add_weighted_cutflow(total, counts):
    for label, value in counts.items():
        total[label] = total.get(label, 0.0) + float(value)
    return total


def ordered_root_cutflow_keys(root_cutflow_by_sample):
    keys = set()
    for mapping in root_cutflow_by_sample:
        keys.update(mapping.keys())
    if not keys:
        return []
    ordered = sorted(keys)
    if "Initial" in ordered:
        ordered.remove("Initial")
        ordered.insert(0, "Initial")
    return ordered


def ordered_gae_cutflow_keys(gae_cutflow_by_sample):
    keys = []
    seen = set()
    for mapping in gae_cutflow_by_sample:
        for key in mapping:
            if key not in seen:
                seen.add(key)
                keys.append(key)
    for mapping in gae_cutflow_by_sample:
        for key in sorted(mapping):
            if key not in seen:
                keys.append(key)
    return keys


def read_file_root_cutflow(file_path, allow_missing=False):
    file_path = normalize_input_path(file_path)
    try:
        with uproot.open(file_path) as input_file:
            if "CutFlow" not in input_file:
                if allow_missing:
                    return {}
                raise ValueError(f"No CutFlow tree found in '{file_path}'")
            tree = input_file["CutFlow"]
            keys = sorted({name.split(";")[0] for name in tree.keys()})
            return {
                key: int(tree[key].array(entry_stop=1)[0]) for key in keys
            }
    except Exception:
        return {}


def accumulate_gae_cutflow(events, role, year, totals, weighted_totals, event_weights):
    mask = np.ones(len(events), dtype=bool)
    for label, step_mask in selection_steps(events, role, year):
        mask &= np.asarray(step_mask, dtype=bool)
        n = int(mask.sum())
        totals[label] = totals.get(label, 0) + n
        if n:
            weighted_totals[label] = weighted_totals.get(label, 0.0) + float(
                event_weights[mask].sum()
            )
        else:
            weighted_totals[label] = weighted_totals.get(label, 0.0)


def read_branches(role, year, use_xsec):
    branches = list(cut_branches(role, year))
    if use_xsec:
        for name in MC_WEIGHT_BRANCHES:
            if name not in branches:
                branches.append(name)
    return branches


def pu_over_gen(events):
    if "genWeightPU" not in ak.fields(events) or "genWeight" not in ak.fields(events):
        return np.ones(len(events), dtype=float)
    genw_pu = ak.to_numpy(events["genWeightPU"])
    genw = ak.to_numpy(events["genWeight"])
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.divide(
            genw_pu,
            genw,
            out=np.ones_like(genw_pu, dtype=float),
            where=(genw != 0),
        )
    ratio[~np.isfinite(ratio)] = 1.0
    return ratio


def mc_event_weights(events, role, year, lumi, ctx):
    n = len(events)
    if "prefire" in ak.fields(events):
        prefire = ak.to_numpy(events["prefire"])
    else:
        prefire = np.ones(n, dtype=float)
    weights = ctx.scale * prefire * pu_over_gen(events)
    if str(year) == "2018" and "lumi_pre" in lumi and "lumi_hem" in lumi:
        post = post_hem_mask(events, role, year)
        lumi_eff = np.where(post, lumi["lumi_hem"], lumi["lumi_pre"])
    else:
        lumi_eff = float(lumi["lumi"])
    return weights * lumi_eff


def trigger_sf_path(config_dir, year):
    root = os.path.dirname(os.path.abspath(config_dir))
    return os.path.join(root, "trigger_sf", f"trigger_sf_counting_{year}.json")


def load_trigger_sf(config_dir, year):
    path = trigger_sf_path(config_dir, year)
    if not os.path.isfile(path):
        return None
    with open(path) as handle:
        payload = json.load(handle)
    edges = [b["mt_low"] for b in payload["bins"]] + [payload["bins"][-1]["mt_high"]]
    values = [1.0 if b.get("sf") is None else float(b["sf"]) for b in payload["bins"]]
    edges_arr = np.asarray(edges, dtype=float)
    values_arr = np.asarray(values, dtype=float)
    print(f"  Trigger SF: {len(values_arr)} bins from {path}")

    def evaluate(mt):
        mt_arr = np.asarray(mt, dtype=float)
        idx = np.searchsorted(edges_arr, mt_arr, side="right") - 1
        inside = (idx >= 0) & (idx < len(values_arr))
        out = np.ones_like(mt_arr)
        out[inside] = values_arr[idx[inside]]
        return out

    return evaluate


def qcd_groups(background_groups):
    return {
        name: files
        for name, files in background_groups.items()
        if name.startswith(QCD_DATASET_PREFIX)
    }


def select_shard(file_paths, files_per_job, shard):
    if files_per_job is None and shard is None:
        return file_paths
    if files_per_job is None or shard is None:
        raise ValueError("--files-per-job and --shard must be used together")
    shards = shard_files(file_paths, files_per_job)
    if not 0 <= shard < len(shards):
        raise ValueError(
            f"--shard {shard} out of range; {len(file_paths)} files form "
            f"{len(shards)} shards of {files_per_job}"
        )
    return shards[shard]


def has_cutflow_tree(file_path):
    file_path = normalize_input_path(file_path)
    with uproot.open(file_path) as input_file:
        return "CutFlow" in input_file


def filter_signal_files(file_paths, quiet=False):
    """Drop signal ROOT files with no CutFlow tree (some full_sys parts lack it)."""
    kept = []
    skipped = 0
    for file_path in file_paths:
        path = normalize_input_path(file_path)
        try:
            if has_cutflow_tree(path):
                kept.append(path)
            else:
                skipped += 1
                if not quiet:
                    tqdm.write(
                        f"  Skip signal file (no CutFlow): {os.path.basename(path)}"
                    )
        except Exception as exc:
            skipped += 1
            if not quiet:
                tqdm.write(
                    f"  Skip signal file (unreadable): {os.path.basename(path)}: {exc}"
                )
    if skipped and not quiet:
        print(f"  Skipped {skipped} signal file(s) without CutFlow or unreadable")
    return kept


def sum_initial_events_timed(files, progress=None, debug=False):
    total = 0
    for index, file_path in enumerate(files):
        start = time.perf_counter()
        with uproot.open(file_path) as input_file:
            if "CutFlow" not in input_file:
                raise ValueError(f"No CutFlow tree found in '{file_path}'")
            total += int(
                input_file["CutFlow"]["Initial"].array(entry_stop=1)[0]
            )
        elapsed = time.perf_counter() - start
        if progress is not None:
            progress.metadata_io += elapsed
        if debug:
            print(
                f"    CutFlow/Initial [{index + 1}/{len(files)}] "
                f"{os.path.basename(file_path)}  {elapsed:.2f}s"
            )
    return total


def mc_weight_context(
    file_path,
    groups,
    initial_events_cache,
    year,
    progress=None,
    debug=False,
):
    file_path = normalize_input_path(file_path)
    start = time.perf_counter()
    with uproot.open(file_path) as input_file:
        if "Metadata" not in input_file:
            raise ValueError(
                f"'{file_path}' has no Metadata tree; cannot apply xsec weights"
            )
        xsec = float(
            input_file["Metadata"]["GenCrossSection"].array(entry_stop=1)[0]
        )
    elapsed = time.perf_counter() - start
    if progress is not None:
        progress.metadata_io += elapsed
    if debug:
        print(
            f"    Metadata GenCrossSection {os.path.basename(file_path)}  "
            f"{elapsed:.2f}s"
        )
    dataset = dataset_of(file_path, groups)
    if dataset not in initial_events_cache:
        if debug:
            print(
                f"  summing CutFlow/Initial for dataset '{dataset}' "
                f"({len(groups[dataset])} files)..."
            )
        initial_events_cache[dataset] = sum_initial_events_timed(
            groups[dataset], progress, debug
        )
    n_initial = initial_events_cache[dataset]
    if n_initial <= 0:
        raise ValueError(
            f"Dataset {dataset!r} has no CutFlow/Initial counts "
            f"({len(groups[dataset])} files)"
        )
    kfactor = QCD_KFACTOR_BY_YEAR.get(str(year), 1.0)
    if not dataset.startswith(QCD_DATASET_PREFIX):
        kfactor = 1.0
    scale = xsec * 1e3 / float(n_initial) * kfactor
    return McWeightContext(scale=scale)


def dataset_file_groups(file_paths, groups, use_xsec):
    """Files grouped so MC weight is constant within each group."""
    if not use_xsec:
        return [("all", list(file_paths))]
    grouped = {}
    for path in file_paths:
        path = normalize_input_path(path)
        dataset = dataset_of(path, groups)
        grouped.setdefault(dataset, []).append(path)
    return [(name, grouped[name]) for name in sorted(grouped)]


def process_batch_stats(
    events,
    role,
    year,
    totals,
    weighted_totals,
    lumi,
    weight_ctx,
    use_xsec,
    trigger_sf,
    progress,
):
    start = time.perf_counter()
    mask = np.asarray(event_cut_mask(events, role, year), dtype=bool)
    n_pass = int(mask.sum())
    if use_xsec:
        weights = mc_event_weights(events, role, year, lumi, weight_ctx)
        if trigger_sf is not None and "MT01FatJetMET" in ak.fields(events):
            weights = weights * trigger_sf(ak.to_numpy(events["MT01FatJetMET"]))
        w_pass = float(weights[mask].sum()) if n_pass else 0.0
    else:
        weights = np.ones(len(events), dtype=float)
        w_pass = float(n_pass)
    batch_weighted = {}
    accumulate_gae_cutflow(events, role, year, totals, batch_weighted, weights)
    add_weighted_cutflow(weighted_totals, batch_weighted)
    stats = BucketStats(n_pass=n_pass, w_pass=w_pass)
    if str(year) == "2018" and "lumi_pre" in lumi and "lumi_hem" in lumi:
        post = post_hem_mask(events, role, year)
        pre_pass = mask & ~post
        post_pass = mask & post
        stats.n_pre = int(pre_pass.sum())
        stats.n_post = int(post_pass.sum())
        if use_xsec:
            stats.w_pre = float(weights[pre_pass].sum()) if stats.n_pre else 0.0
            stats.w_post = float(weights[post_pass].sum()) if stats.n_post else 0.0
        else:
            stats.w_pre = float(stats.n_pre)
            stats.w_post = float(stats.n_post)
    else:
        stats.n_pre = n_pass
        stats.w_pre = w_pass
    if progress is not None:
        progress.add_cuts(time.perf_counter() - start)
    return stats


def _trim_batch(batch, n_keep):
    if n_keep >= len(batch):
        return batch, len(batch)
    return batch[:n_keep], n_keep


def _consume_batch(
    batch,
    role,
    year,
    totals,
    weighted_totals,
    progress,
    lumi,
    weight_ctx,
    use_xsec,
    trigger_sf,
    bucket_stats,
):
    batch_stats = process_batch_stats(
        batch,
        role,
        year,
        totals,
        weighted_totals,
        lumi,
        weight_ctx,
        use_xsec,
        trigger_sf,
        progress,
    )
    bucket_stats.add_counts(batch_stats)
    if progress is not None:
        progress.batches += 1
        progress.maybe_log(bucket_stats.n_pass)
    return bucket_stats


def iterate_file_chunks(
    file_paths,
    branches,
    args,
    progress,
    role,
    year,
    totals,
    weighted_totals,
    lumi,
    weight_ctx,
    use_xsec,
    trigger_sf,
    bucket_stats,
    entries_read,
    bar_label="",
    quiet_override=None,
):
    """Read with uproot.iterate in chunks; one or many files per inner loop."""
    per_file = args.max_events_per_file is not None
    file_lists = (
        [[path] for path in file_paths]
        if per_file
        else [list(file_paths)]
    )
    uproot_workers = args.uproot_workers
    if per_file and len(file_paths) > 1:
        uproot_workers = max(1, uproot_workers)
    quiet = quiet_override
    if quiet is None:
        quiet = progress.quiet if progress is not None else False

    file_outer = file_lists
    if per_file and len(file_lists) > 1:
        file_outer = tqdm(
            file_lists,
            desc=f"{bar_label} files",
            unit="file",
            disable=quiet,
        )

    for file_group in file_outer:
        if args.max_events_total is not None and entries_read >= args.max_events_total:
            break

        paths = [f"{normalize_input_path(p)}:Events" for p in file_group]
        if progress is not None and progress.debug and len(file_lists) > 1:
            names = ", ".join(os.path.basename(p) for p in file_group[:3])
            if len(file_group) > 3:
                names += ", ..."
            print(f"  reading file group ({len(file_group)}): {names}")

        file_entries = 0
        read_wait = time.perf_counter()
        chunk_bar = tqdm(
            desc=f"{bar_label} chunks",
            unit="chunk",
            disable=quiet,
            leave=False,
        )
        try:
            batch_iter = uproot.iterate(
                paths,
                expressions=branches,
                library="ak",
                step_size=args.step_size,
                num_workers=uproot_workers,
            )
            for batch in batch_iter:
                if progress is not None:
                    progress.add_read(time.perf_counter() - read_wait)
                n_in = len(batch)
                read_wait = time.perf_counter()

                if per_file and args.max_events_per_file is not None:
                    remaining_file = args.max_events_per_file - file_entries
                    if remaining_file <= 0:
                        break
                    batch, n_in = _trim_batch(batch, min(n_in, remaining_file))
                    file_entries += n_in

                if args.max_events_total is not None:
                    remaining = args.max_events_total - entries_read
                    if remaining <= 0:
                        break
                    batch, n_in = _trim_batch(batch, min(n_in, remaining))

                entries_read += n_in
                if progress is not None:
                    progress.entries_read = entries_read

                _consume_batch(
                    batch,
                    role,
                    year,
                    totals,
                    weighted_totals,
                    progress,
                    lumi,
                    weight_ctx,
                    use_xsec,
                    trigger_sf,
                    bucket_stats,
                )
                chunk_bar.update(1)
                chunk_bar.set_postfix(
                    entries=entries_read,
                    n_pass=bucket_stats.n_pass,
                    refresh=False,
                )

                if args.max_events_total is not None and entries_read >= args.max_events_total:
                    break
        finally:
            chunk_bar.close()

    return bucket_stats, entries_read


def process_one_file(
    file_path,
    role,
    year,
    branches,
    args,
    lumi,
    use_xsec,
    weight_ctx,
    trigger_sf,
):
    totals = {}
    weighted_totals = {}
    stats = BucketStats()
    iterate_file_chunks(
        [file_path],
        branches,
        args,
        None,
        role,
        year,
        totals,
        weighted_totals,
        lumi,
        weight_ctx,
        use_xsec,
        trigger_sf,
        stats,
        0,
        bar_label="",
        quiet_override=True,
    )
    stats.cutflow = totals
    stats.weighted_cutflow = weighted_totals
    stats.root_cutflow = read_file_root_cutflow(
        file_path, allow_missing=(role == "signal")
    )
    return stats


def accumulate_files_threaded(
    group_files,
    role,
    year,
    branches,
    args,
    lumi,
    use_xsec,
    weight_ctx,
    trigger_sf,
    progress,
    bar_label,
):
    bucket_stats = BucketStats()
    quiet = progress.quiet if progress is not None else False
    file_bar = tqdm(
        total=len(group_files),
        desc=f"{bar_label} files",
        unit="file",
        disable=quiet,
    )
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {
            pool.submit(
                process_one_file,
                path,
                role,
                year,
                branches,
                args,
                lumi,
                use_xsec,
                weight_ctx,
                trigger_sf,
            ): path
            for path in group_files
        }
        for future in as_completed(futures):
            partial = future.result()
            bucket_stats.add_counts(partial)
            add_cutflow(bucket_stats.cutflow, partial.cutflow)
            file_bar.update(1)
            file_bar.set_postfix(n_pass=bucket_stats.n_pass, refresh=False)
    file_bar.close()
    return bucket_stats


def accumulate(
    file_paths,
    role,
    year,
    groups,
    args,
    use_xsec,
    lumi,
    trigger_sf,
    progress=None,
    bar_label="",
):
    if not file_paths:
        return BucketStats()

    branches = read_branches(role, year, use_xsec)
    totals = {}
    weighted_totals = {}
    bucket_stats = BucketStats()
    initial_events_cache = {}
    entries_read = 0
    quiet = progress.quiet if progress is not None else False
    debug = progress.debug if progress is not None else False

    mc_groups = dataset_file_groups(file_paths, groups, use_xsec)
    group_outer = mc_groups
    if use_xsec and len(mc_groups) > 1:
        group_outer = tqdm(
            mc_groups,
            desc=f"{bar_label} MC datasets",
            unit="dataset",
            disable=quiet,
        )

    use_threads = (
        args.workers > 1
        and len(file_paths) > 1
        and args.max_events_total is None
    )

    for group_name, group_files in group_outer:
        weight_ctx = None
        if use_xsec:
            weight_ctx = mc_weight_context(
                group_files[0],
                groups,
                initial_events_cache,
                year,
                progress,
                debug,
            )

        chunk_label = bar_label
        if use_xsec and len(mc_groups) > 1:
            chunk_label = f"{bar_label}/{group_name}"

        if use_threads and len(group_files) > 1:
            partial = accumulate_files_threaded(
                group_files,
                role,
                year,
                branches,
                args,
                lumi,
                use_xsec,
                weight_ctx,
                trigger_sf,
                progress,
                chunk_label,
            )
            bucket_stats.add_counts(partial)
            add_cutflow(totals, partial.cutflow)
        else:
            bucket_stats, entries_read = iterate_file_chunks(
                group_files,
                branches,
                args,
                progress,
                role,
                year,
                totals,
                weighted_totals,
                lumi,
                weight_ctx,
                use_xsec,
                trigger_sf,
                bucket_stats,
                entries_read,
                bar_label=chunk_label,
            )

        if args.max_events_total is not None and entries_read >= args.max_events_total:
            break

    add_cutflow(bucket_stats.cutflow, totals)
    add_weighted_cutflow(bucket_stats.weighted_cutflow, weighted_totals)
    if not use_threads:
        for path in file_paths:
            add_cutflow(
                bucket_stats.root_cutflow,
                read_file_root_cutflow(path, allow_missing=(role == "signal")),
            )
    return bucket_stats


def print_weighting_note(lumi):
    print("\nWeighting note:")
    print("  N_pass   = unweighted skim events passing all GAE cuts")
    print("  W_pass   = MC sum of xsec*L*1e3/N_initial*prefire*(genWeightPU/genWeight)")
    print("             (x trigger SF on MT01FatJetMET when enabled)")
    print("  N_phys   = N_pass for data; W_pass for MC (lumi already in event weights)")
    if "lumi_pre" in lumi:
        print(
            f"  L_2018   = {lumi['lumi']} fb-1 total "
            f"(pre-HEM {lumi['lumi_pre']}, post-HEM {lumi['lumi_hem']})"
        )
    else:
        print(f"  L        = {lumi['lumi']} fb-1")
    print("  Partial reads/shards bias MC weights that use full-dataset N_gen.")


def print_bucket_yield_table(title, stats, role, year, lumi):
    n_phys = physical_yield(stats, role, year, lumi)
    print(f"\n--- {title} ---")
    print(f"  N_pass (counts)     {stats.n_pass}")
    print(f"  W_pass (MC weight)  {stats.w_pass:.6g}")
    print(f"  N_phys              {n_phys:.4g}")
    if str(year) == "2018" and "lumi_pre" in lumi:
        phys_pre = stats.w_pre if role != "data" else stats.n_pre
        phys_post = stats.w_post if role != "data" else stats.n_post
        print(
            f"  pre-HEM             N={stats.n_pre}  W={stats.w_pre:.6g}  "
            f"N_phys={phys_pre:.4g}"
        )
        print(
            f"  post-HEM            N={stats.n_post}  W={stats.w_post:.6g}  "
            f"N_phys={phys_post:.4g}"
        )
    if stats.cutflow:
        print(format_cutflow(f"{title} cutflow", stats.cutflow))


def final_weighted_yield(stats, role):
    if role == "data":
        return float(stats.n_pass)
    return float(stats.w_pass)


def print_combined_cutflow_table(year, sample_rows):
    """ROOT CutFlow tree columns plus GAE selection steps (DataMC-style table)."""
    if not sample_rows:
        return
    root_maps = [stats.root_cutflow for _, stats, _ in sample_rows]
    gae_maps = [stats.cutflow for _, stats, _ in sample_rows]
    root_cols = ordered_root_cutflow_keys(root_maps)
    gae_cols = ordered_gae_cutflow_keys(gae_maps)
    headers = (
        ["Sample"]
        + [f"ROOT/{key}" for key in root_cols]
        + [f"GAE/{key}" for key in gae_cols]
        + ["Final total", "Final weighted yield"]
    )
    rows = []
    for name, stats, role in sample_rows:
        final_n = stats.n_pass
        final_w = final_weighted_yield(stats, role)
        row = [name]
        row.extend(stats.root_cutflow.get(key, 0) for key in root_cols)
        row.extend(stats.cutflow.get(key, 0) for key in gae_cols)
        row.extend([final_n, final_w])
        rows.append(row)
    widths = [
        max(len(str(headers[i])), *(len(str(row[i])) for row in rows))
        for i in range(len(headers))
    ]
    print(f"\n  Cutflow for {year} (ROOT skim + GAE selections on Events)")
    print(
        "  "
        + " | ".join(str(headers[i]).ljust(widths[i]) for i in range(len(headers)))
    )
    print("  " + "-+-".join("-" * w for w in widths))
    for row in rows:
        cells = []
        for i, value in enumerate(row):
            text = f"{value:.6g}" if i == len(row) - 1 and isinstance(value, float) else str(value)
            if i == 0:
                cells.append(text.ljust(widths[i]))
            else:
                cells.append(text.rjust(widths[i]))
        print("  " + " | ".join(cells))


def print_contamination_table(
    year, lumi, data_stats, qcd_stats, signal_stats_by_name
):
    n_data = data_stats.n_pass
    n_qcd = qcd_stats.n_pass
    phys_data = physical_yield(data_stats, "data", year, lumi)
    phys_qcd = physical_yield(qcd_stats, "background", year, lumi)

    print(f"\n=== {year} signal contamination (final GAE selection) ===")
    if n_data == 0 and n_qcd == 0:
        print("No data or QCD events pass all cuts; contamination undefined.")
        return

    header = (
        f"{'signal point':<36} {'N_pass':>8} {'N_phys':>10} "
        f"{'unwt/data':>10} {'phys/data':>10} {'phys/QCD':>10}"
    )
    print(header)
    for name, stats in sorted(signal_stats_by_name.items()):
        n_sig = stats.n_pass
        phys_sig = physical_yield(stats, "signal", year, lumi)
        print(
            f"{name:<36} {n_sig:>8d} {phys_sig:>10.4g} "
            f"{100.0 * n_sig / n_data:>9.3f}% "
            f"{100.0 * phys_sig / phys_data:>9.3f}% "
            f"{100.0 * phys_sig / phys_qcd if phys_qcd else float('nan'):>9.3f}%"
        )


def bucket_record(bucket, role, stats):
    return {
        "bucket": bucket,
        "role": role,
        "n_pass": stats.n_pass,
        "w_pass": stats.w_pass,
        "n_pre": stats.n_pre,
        "n_post": stats.n_post,
        "w_pre": stats.w_pre,
        "w_post": stats.w_post,
        "cutflow": stats.cutflow,
        "root_cutflow": stats.root_cutflow,
        "weighted_cutflow": stats.weighted_cutflow,
    }


def stats_from_record(record):
    return BucketStats(
        n_pass=int(record["n_pass"]),
        w_pass=float(record["w_pass"]),
        n_pre=int(record.get("n_pre", record["n_pass"])),
        n_post=int(record.get("n_post", 0)),
        w_pre=float(record.get("w_pre", record["w_pass"])),
        w_post=float(record.get("w_post", 0.0)),
        cutflow=dict(record.get("cutflow", {})),
        root_cutflow=dict(record.get("root_cutflow", {})),
        weighted_cutflow=dict(record.get("weighted_cutflow", {})),
    )


def stats_from_cache_payload(payload):
    return stats_from_record(payload)


def cache_eligible(args):
    return not (
        args.refresh_cache
        or args.max_events_per_file
        or args.max_events_total
        or args.shard is not None
    )


def write_json_output(path, year, records):
    payload = {"version": JSON_VERSION, "year": year, "records": records}
    with open(path, "w") as output_file:
        json.dump(payload, output_file, indent=2)
    print(f"Wrote {path}")


def merge_json_files(paths):
    merged = {}
    for path in paths:
        with open(path) as input_file:
            payload = json.load(input_file)
        year = str(payload["year"])
        merged.setdefault(year, {})
        for record in payload["records"]:
            key = record["bucket"]
            entry = merged[year].setdefault(
                key,
                {
                    "role": record["role"],
                    "n_pass": 0,
                    "w_pass": 0.0,
                    "n_pre": 0,
                    "n_post": 0,
                    "w_pre": 0.0,
                    "w_post": 0.0,
                    "cutflow": {},
                    "root_cutflow": {},
                    "weighted_cutflow": {},
                },
            )
            entry["n_pass"] += int(record["n_pass"])
            entry["w_pass"] += float(record["w_pass"])
            entry["n_pre"] += int(record.get("n_pre", 0))
            entry["n_post"] += int(record.get("n_post", 0))
            entry["w_pre"] += float(record.get("w_pre", 0.0))
            entry["w_post"] += float(record.get("w_post", 0.0))
            if "cutflow" in record:
                add_cutflow(entry["cutflow"], record["cutflow"])
            if "root_cutflow" in record:
                add_cutflow(entry["root_cutflow"], record["root_cutflow"])
            if "weighted_cutflow" in record:
                add_weighted_cutflow(
                    entry["weighted_cutflow"], record["weighted_cutflow"]
                )
    return merged


def print_year_summary(year, lumi, results, signal_groups):
    print_weighting_note(lumi)
    data = results.get("data", BucketStats())
    qcd = results.get("qcd", BucketStats())
    print_bucket_yield_table(f"{year} data", data, "data", year, lumi)
    print_bucket_yield_table(f"{year} QCD (pooled HT)", qcd, "background", year, lumi)
    signal_map = {
        name: results[name]
        for name in sorted(signal_groups)
        if name in results
    }
    for name, stats in signal_map.items():
        short = name.replace("s-channel_", "")
        print_bucket_yield_table(f"{year} signal {short}", stats, "signal", year, lumi)
    table_rows = [("Data", data, "data"), ("QCD", qcd, "background")]
    for name, stats in signal_map.items():
        table_rows.append((name.replace("s-channel_", ""), stats, "signal"))
    print_combined_cutflow_table(year, table_rows)
    print_contamination_table(year, lumi, data, qcd, signal_map)


def run_merge(args):
    merged = merge_json_files(args.merge_json)
    for year in sorted(merged):
        buckets = merged[year]
        lumi = lumi_spec(args, year)
        results = {
            key: stats_from_record(entry) for key, entry in buckets.items()
        }
        signal_groups = {
            k: [] for k, v in buckets.items() if v["role"] == "signal"
        }
        print_year_summary(year, lumi, results, signal_groups)


def warn_partial_run(args, file_paths):
    if args.max_events_per_file or args.max_events_total or args.shard is not None:
        print(
            "WARNING: partial read or shard mode — MC uses full-dataset "
            "N_initial; W_pass / N_phys are approximate."
        )
    if len(file_paths) == 0:
        print("WARNING: no files selected for this bucket.")


def run_bucket(
    args,
    year,
    bucket,
    file_paths,
    groups,
    role,
    use_xsec,
    lumi,
    trigger_sf,
    progress,
    records,
):
    if role == "signal":
        file_paths = filter_signal_files(file_paths, quiet=args.quiet)
        if len(groups) == 1:
            dataset_key = next(iter(groups))
            groups = {dataset_key: list(file_paths)}
        if not file_paths:
            print(
                f"\n--- {year} {bucket} (0 files) ---\n"
                f"WARNING: no signal files with CutFlow; skipping bucket"
            )
            stats = BucketStats()
            records.append(bucket_record(bucket, role, stats))
            return stats
    warn_partial_run(args, file_paths)
    print(f"\n--- {year} {bucket} ({len(file_paths)} files) ---")
    cache_key = None
    if cache_eligible(args):
        cache_key = bucket_cache_key(
            bucket,
            year,
            role,
            file_paths,
            cuts_fingerprint(),
            args.no_trigger_sf,
        )
        cached = load_bucket_cache(args.config_dir, cache_key)
        if cached is not None:
            print(f"  Loaded from cache ({cache_key})")
            stats = stats_from_cache_payload(cached)
            records.append(bucket_record(bucket, role, stats))
            return stats
    progress._file_workers = args.workers
    progress.start_bucket(
        bucket,
        len(file_paths),
        args.step_size,
        args.uproot_workers,
        len(read_branches(role, year, use_xsec)),
    )
    stats = accumulate(
        file_paths,
        role,
        year,
        groups,
        args,
        use_xsec=use_xsec,
        lumi=lumi,
        trigger_sf=trigger_sf,
        progress=progress,
        bar_label=bucket,
    )
    if cache_key is not None:
        save_bucket_cache(args.config_dir, cache_key, stats)
    records.append(bucket_record(bucket, role, stats))
    progress.report(bucket)
    return stats


def process_year(args, year):
    year = str(year)
    lumi = lumi_spec(args, year)
    progress = IoProgress(
        quiet=args.quiet,
        debug=args.debug,
        log_every_batches=args.progress_every,
    )
    trigger_sf = None
    if not args.no_trigger_sf:
        trigger_sf = load_trigger_sf(args.config_dir, year)
        if trigger_sf is None:
            print(
                f"  Trigger SF: no file at {trigger_sf_path(args.config_dir, year)}; "
                "MC weights omit trigger correction"
            )

    data_path = input_json_path(args.config_dir, "data", year)
    signal_path = input_json_path(args.config_dir, "signal", year)
    bg_path = input_json_path(args.config_dir, "background", year)

    data_groups = dataset_groups(load_input_json(data_path))
    signal_groups = dataset_groups(load_input_json(signal_path))
    bg_groups = dataset_groups(load_input_json(bg_path))
    qcd_only = qcd_groups(bg_groups)

    records = []
    results = {}

    def should_run(name):
        return args.bucket is None or args.bucket == name

    if should_run("data"):
        files = select_shard(
            flat_files(data_groups), args.files_per_job, args.shard
        )
        results["data"] = run_bucket(
            args,
            year,
            "data",
            files,
            data_groups,
            "data",
            False,
            lumi,
            trigger_sf,
            progress,
            records,
        )

    if should_run("qcd"):
        files = select_shard(flat_files(qcd_only), args.files_per_job, args.shard)
        results["qcd"] = run_bucket(
            args,
            year,
            "qcd",
            files,
            qcd_only,
            "background",
            True,
            lumi,
            trigger_sf,
            progress,
            records,
        )

    if args.bucket is None or args.bucket not in ("data", "qcd"):
        signal_names = [
            name
            for name in sorted(signal_groups)
            if args.bucket is None or args.bucket == name
        ]
        signal_outer = signal_names
        if len(signal_names) > 1:
            signal_outer = tqdm(
                signal_names,
                desc=f"{year} signal points",
                unit="point",
                disable=args.quiet,
            )
        for dataset_name in signal_outer:
            files = select_shard(
                signal_groups[dataset_name], args.files_per_job, args.shard
            )
            groups = {dataset_name: files}
            results[dataset_name] = run_bucket(
                args,
                year,
                dataset_name,
                files,
                groups,
                "signal",
                True,
                lumi,
                trigger_sf,
                progress,
                records,
            )

    if args.output_json:
        write_json_output(args.output_json, year, records)
        return

    if args.bucket is not None:
        return

    print_year_summary(year, lumi, results, signal_groups)

    if args.background:
        for dataset_name in sorted(bg_groups):
            if dataset_name.startswith(QCD_DATASET_PREFIX):
                continue
            files = bg_groups[dataset_name]
            if not files:
                continue
            groups = {dataset_name: files}
            print(f"\n--- {year} background / {dataset_name} ({len(files)} files) ---")
            progress._file_workers = args.workers
            progress.start_bucket(
                dataset_name,
                len(files),
                args.step_size,
                args.uproot_workers,
                len(read_branches("background", year, False)),
            )
            stats = accumulate(
                files,
                "background",
                year,
                groups,
                args,
                use_xsec=False,
                lumi=lumi,
                trigger_sf=None,
                progress=progress,
                bar_label=dataset_name,
            )
            progress.report(dataset_name)
            print_bucket_yield_table(
                f"{year} background {dataset_name}",
                stats,
                "background",
                year,
                lumi,
            )


def contamination_buckets(config_dir, year):
    """(sample_label, bucket, file_paths) for Condor queue generation."""
    year = str(year)
    data_groups = dataset_groups(
        load_input_json(input_json_path(config_dir, "data", year))
    )
    signal_groups = dataset_groups(
        load_input_json(input_json_path(config_dir, "signal", year))
    )
    bg_groups = dataset_groups(
        load_input_json(input_json_path(config_dir, "background", year))
    )
    qcd_only = qcd_groups(bg_groups)
    items = [
        (f"{year}__data", "data", flat_files(data_groups)),
        (f"{year}__qcd", "qcd", flat_files(qcd_only)),
    ]
    for name, files in sorted(signal_groups.items()):
        items.append((f"{year}__{name}", name, list(files)))
    return items


def main():
    args = parse_arguments()
    if args.num_workers is not None:
        args.uproot_workers = args.num_workers
    if args.refresh_cache:
        print("  --refresh-cache: ignoring cache_signal_contamination/")
    if args.merge_json:
        run_merge(args)
        return
    for year in args.years:
        process_year(args, year)


if __name__ == "__main__":
    main()
