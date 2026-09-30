#!/usr/bin/env python3
"""Build graph datasets from the ROOT files listed in a sample's file list.

The selections in utils/Cuts.py are applied to every sample, so signal,
background and data are treated identically.

    python produce_graphs.py --samples background_example
    python produce_graphs.py --samples smoke --max-events 200

One shard of a condor or crab job processes a subset of the file list:

    python produce_graphs.py --samples background_example \
        --shard 3 --files-per-job 5
"""

import argparse
import os

from utils.FileLists import resolve_file_list, shard_file_list
from utils.GaeConfig import load_config
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
    parser.add_argument("--quiet", action="store_true")
    return parser.parse_args()


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


def select_files(config, sample, args):
    paths = resolve_file_list(sample.file_list)
    if args.files_per_job is None and args.shard is None:
        return paths
    if args.files_per_job is None or args.shard is None:
        raise ValueError("--files-per-job and --shard must be used together")
    shards = shard_file_list(paths, args.files_per_job)
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

    names = args.samples if args.samples else sorted(config.samples)
    for name in names:
        sample = config.sample(name)
        file_paths = select_files(config, sample, args)
        root = config.dataset_root(sample)
        dataset_name = sample.dataset_name
        if args.shard is not None:
            dataset_name = f"{dataset_name}_shard{args.shard}"

        print(
            f"Producing graphs for sample '{name}' ({sample.role}) from "
            f"{len(file_paths)} file(s) into {os.path.join(root, 'processed')}"
        )
        dataset = FileListGraphProducer(
            root=root,
            output_dataset_name=dataset_name,
            file_paths=file_paths,
            **producer_kwargs(config, sample, args),
        )
        print(f"--> {len(dataset)} graphs")
        if len(dataset):
            print(f"--> {dataset[0].num_node_features} node features")


if __name__ == "__main__":
    main()
