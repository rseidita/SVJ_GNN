#!/usr/bin/env python3
"""Build graph datasets from the ROOT files in a sample's input JSON.

The selections in utils/Cuts.py are applied to every sample according to
its role and year.

    python produce_graphs.py --samples background_2018
    python produce_graphs.py --samples smoke --max-events 200

One shard of a condor or crab job processes a subset of the files:

    python produce_graphs.py --samples background_2018 \
        --shard 3 --files-per-job 5
"""

import argparse
import os

import uproot

from utils.Cuts import add_cutflow, cut_branches, cutflow, format_cutflow
from utils.GaeConfig import load_config
from utils.InputJson import (
    dataset_of,
    flat_files,
    resolve_file_selection,
    shard_files,
)
from utils.JetGraphProducer import FileListGraphProducer


def parse_arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="gae_config.json")
    parser.add_argument(
        "--samples",
        nargs="+",
        default=None,
        help="Sample names to process; defaults to every configured sample",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Override the production output_dir from the configuration",
    )
    parser.add_argument("--max-events", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--graphs-per-chunk", type=int, default=None)
    parser.add_argument("--event-batch-size", type=int, default=None)
    parser.add_argument(
        "--files-per-job",
        type=int,
        default=None,
        help="Shard the file list, processing only --shard of the shards",
    )
    parser.add_argument("--shard", type=int, default=None)
    parser.add_argument(
        "--files",
        nargs="+",
        default=None,
        help="Process only these files of the sample (used by CRAB jobs)",
    )
    parser.add_argument(
        "--no-cutflow",
        action="store_true",
        help="Skip the per-dataset cutflow pass over the input files",
    )
    parser.add_argument("--quiet", action="store_true")
    return parser.parse_args()


def print_cutflows(sample, file_paths, groups, max_events):
    """Cumulative selection counts per dataset, summed over *file_paths*."""
    branches = cut_branches(sample.role, sample.year)
    totals = {}
    for file_path in file_paths:
        with uproot.open(f"{file_path}:Events") as tree:
            events = tree.arrays(branches, entry_stop=max_events, library="ak")
        dataset = dataset_of(file_path, groups)
        totals[dataset] = add_cutflow(
            totals.get(dataset, {}), cutflow(events, sample.role, sample.year)
        )
    for dataset, counts in totals.items():
        n_files = sum(dataset_of(path, groups) == dataset for path in file_paths)
        print(format_cutflow(f"{sample.name} / {dataset} ({n_files} files)", counts))


def producer_kwargs(config, sample, args):
    production = dict(config.production)
    if args.num_workers is not None:
        production["num_workers"] = args.num_workers
    if args.graphs_per_chunk is not None:
        production["graphs_per_chunk"] = args.graphs_per_chunk
    if args.event_batch_size is not None:
        production["event_batch_size"] = args.event_batch_size

    kwargs = dict(config.graph)
    for key in (
        "graphs_per_chunk",
        "event_batch_size",
        "num_workers",
        "worker_prefetch",
    ):
        if key in production:
            kwargs[key] = production[key]
    kwargs["weights"] = sample.weights
    kwargs["label"] = sample.label
    kwargs["max_events_to_process"] = args.max_events
    kwargs["verbose"] = not args.quiet
    return kwargs


def select_files(groups, sample, args):
    paths = resolve_file_selection(flat_files(groups), args.files)
    if args.files_per_job is None and args.shard is None:
        return paths
    if args.files_per_job is None or args.shard is None:
        raise ValueError("--files-per-job and --shard must be used together")
    shards = shard_files(paths, args.files_per_job)
    if not 0 <= args.shard < len(shards):
        raise ValueError(
            f"--shard {args.shard} is out of range; sample {sample.name!r} "
            f"has {len(shards)} shards of {args.files_per_job} files"
        )
    return shards[args.shard]


def main():
    args = parse_arguments()
    config = load_config(args.config)
    if args.output_dir is not None:
        config.production["output_dir"] = args.output_dir

    for sample in config.select(args.samples):
        name = sample.name
        groups = sample.groups()
        file_paths = select_files(groups, sample, args)
        root = config.dataset_root(sample)
        dataset_name = sample.dataset_name
        if args.shard is not None:
            dataset_name = f"{dataset_name}_shard{args.shard}"

        print(
            f"Producing graphs for sample '{name}' ({sample.role}, "
            f"{sample.year}) from {len(file_paths)} file(s) in "
            f"{len(groups)} dataset(s) into {os.path.join(root, 'processed')}"
        )
        if not args.no_cutflow:
            print_cutflows(sample, file_paths, groups, args.max_events)
        dataset = FileListGraphProducer(
            root=root,
            output_dataset_name=dataset_name,
            file_paths=file_paths,
            role=sample.role,
            year=sample.year,
            dataset_groups=groups,
            **producer_kwargs(config, sample, args),
        )
        print(f"--> {len(dataset)} graphs")
        if len(dataset):
            print(f"--> {dataset[0].num_node_features} node features")


if __name__ == "__main__":
    main()
