#!/usr/bin/env python3
"""Score ROOT files with a trained autoencoder and write per-jet scores.

Graphs are streamed straight from the ROOT files by RootGraphScorer, so no
intermediate .pt datasets are needed. The selections in utils.Cuts.py are
applied afterwards, using the fact that the scorer returns its scores in the
original event and jet order.

    python score_gae.py --checkpoint training/GraphAE_graphAE/checkpoint.pt
    python score_gae.py --checkpoint ... --samples data_2018 qcd_2018

Each input ROOT file is written as a separate local file under --output-dir,
named ``data_part-0.root``, ``qcd_part-0.root``,
``signal_mMed-1000_mDark-20_rinv-0.3_part-0.root``, and so on. Remote
``root://`` paths are read as-is and never turned into local paths under the
working directory.
"""

import argparse
import dataclasses
import os

import numpy as np
import uproot

from architectures.ModelRegistry import build_model
from utils.Cuts import (
    add_cutflow,
    cut_branches,
    cutflow,
    event_cut_mask,
    format_cutflow,
    jet_cut_mask,
)
from utils.GaeConfig import load_config
from utils.GraphBuilder import GraphBuildConfig, pfnano_branches
from utils.InputJson import (
    dataset_of,
    flat_files,
    normalize_input_path,
    resolve_file_selection,
    score_output_basename,
    shard_files,
    sum_initial_events,
)
from utils.RootGraphScorer import RootGraphScorer

import torch


def parse_arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="gae_config.json")
    parser.add_argument(
        "--checkpoint",
        default=None,
        help="Path to trained autoencoder checkpoint (.pt); defaults to train stage output location",
    )
    parser.add_argument("--run-name", default=None, help="Training run name to score")
    parser.add_argument("--seed", type=int, default=None, help="Training seed to score")
    parser.add_argument("--data-dir", default=None, help="Directory containing dataset and training directories")
    parser.add_argument(
        "--samples",
        nargs="+",
        default=None,
        help="Sample names to score; defaults to every configured sample",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Directory for local score ROOT files (defaults to scoring.output_dir in config)",
    )
    parser.add_argument(
        "--extra-branches",
        nargs="+",
        default=(),
        help="Event level branches to copy into the output tree for plotting",
    )
    parser.add_argument("--max-events", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=1)
    parser.add_argument("--root-entry-batch-size", type=int, default=1000)
    parser.add_argument("--graph-batch-size", type=int, default=256)
    parser.add_argument("--device", default=None)
    parser.add_argument("--files-per-job", type=int, default=None)
    parser.add_argument("--shard", type=int, default=None)
    parser.add_argument(
        "--files",
        nargs="+",
        default=None,
        help="Score only these input files (xrootd URLs or local paths)",
    )
    parser.add_argument(
        "--region",
        default="control",
        choices=["control", "signal_region", "signal", "validation", "highRT", "lowRT", "inclusive"],
        help="Selection region (control, signal_region, validation, inclusive)",
    )
    parser.add_argument("--quiet", action="store_true")
    return parser.parse_args()


def graph_build_config(graph_settings, label=0.0):
    """The GraphBuildConfig stored in a checkpoint's graph settings."""
    allowed = {field.name for field in dataclasses.fields(GraphBuildConfig)}
    kwargs = {
        key: value
        for key, value in graph_settings.items()
        if key in allowed and key != "label"
    }
    kwargs["label"] = label
    return GraphBuildConfig(**kwargs)


def base_event_weight(file_path, weights, groups, initial_events_cache):
    """Per-event weight for a file: the cross section weight, or one."""
    file_path = normalize_input_path(file_path)
    if weights != "xsec":
        return 1.0
    with uproot.open(file_path) as input_file:
        if "Metadata" not in input_file:
            raise ValueError(
                f"'{file_path}' has no Metadata tree, so weights='xsec' "
                "cannot be used for this sample"
            )
        xsec = input_file["Metadata"]["GenCrossSection"].array(entry_stop=1)[0]
    dataset = dataset_of(file_path, groups)
    if dataset not in initial_events_cache:
        initial_events_cache[dataset] = sum_initial_events(groups[dataset])
    return float(xsec / initial_events_cache[dataset])


def select_scored_jets(
    scores, selected_events, original_entries, config, weight, extra_branches
):
    """Flatten jagged scores to selected jets."""
    columns = {
        "entry": [],
        "jet": [],
        "score": [],
        "weight": [],
    }
    for name in extra_branches:
        columns[name] = []

    for idx, (entry, event) in enumerate(zip(original_entries, selected_events)):
        for jet_index, score in enumerate(scores[idx]):
            if not np.isfinite(score):
                continue
            if not jet_cut_mask(event, jet_index, config):
                continue
            columns["entry"].append(int(entry))
            columns["jet"].append(int(jet_index))
            columns["score"].append(float(score))
            columns["weight"].append(weight)
            for name in extra_branches:
                columns[name].append(float(event[name]))

    return columns


def write_score_file(output_path, columns):
    with uproot.recreate(output_path) as output_file:
        output_file["Scores"] = {
            key: np.asarray(values) for key, values in columns.items()
        }


def score_paths_for_sample(sample, args):
    groups = sample.groups()
    file_paths = resolve_file_selection(flat_files(groups), args.files)
    if args.files_per_job is not None and args.shard is not None:
        file_paths = shard_files(file_paths, args.files_per_job)[args.shard]
    return groups, file_paths


def main():
    args = parse_arguments()
    config = load_config(args.config)
    verbose = not args.quiet

    checkpoint_path = config.find_checkpoint(
        args.checkpoint,
        run_name=args.run_name,
        seed=args.seed,
        base_dir=args.data_dir,
    )
    checkpoint = torch.load(checkpoint_path, weights_only=False)
    print(f"Model: {checkpoint['model_spec']}")
    print(f"Model checkpoint path: {checkpoint_path}")
    device = torch.device(
        args.device
        if args.device is not None
        else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    model = build_model(
        checkpoint["model_spec"],
        checkpoint["num_features"],
        **checkpoint["model_kwargs"],
    )
    model.load_state_dict(checkpoint["state_dict"])
    print(f"Loaded {checkpoint['model_spec']} with {checkpoint['num_features']} node features on {device} from '{checkpoint_path}'")

    graph_settings = checkpoint["graph"]
    max_jets = graph_settings.get("n_store_jets")
    output_dir = args.output_dir or config.scoring.get("output_dir", "output_scores")
    os.makedirs(output_dir, exist_ok=True)

    for sample in config.select(args.samples):
        name = sample.name
        build_config = graph_build_config(graph_settings, sample.label)

        scorer = RootGraphScorer(
            model,
            build_config,
            normalizer={
                "means": checkpoint["means"],
                "stds": checkpoint["stds"],
            },
            alpha=checkpoint["alpha"],
            device=device,
            max_jets=max_jets,
            max_events=args.max_events,
            root_entry_batch_size=args.root_entry_batch_size,
            graph_batch_size=args.graph_batch_size,
            num_workers=args.num_workers,
            verbose=verbose,
        )

        groups, file_paths = score_paths_for_sample(sample, args)
        initial_events_cache = {}

        branches = cut_branches(sample.role, sample.year, region=args.region)
        for branch in args.extra_branches:
            if branch not in branches:
                branches.append(branch)
        read_branches = sorted(
            set(branches) | set(pfnano_branches(build_config))
        )

        cutflows = {}
        n_jets_total = 0
        for file_path in file_paths:
            file_path = normalize_input_path(file_path)
            weight = base_event_weight(
                file_path, sample.weights, groups, initial_events_cache
            )
            with uproot.open(f"{file_path}:Events") as tree:
                events = tree.arrays(
                    read_branches,
                    entry_stop=args.max_events,
                    library="ak",
                )
            dataset = dataset_of(file_path, groups)
            cutflows[dataset] = add_cutflow(
                cutflows.get(dataset, {}),
                cutflow(events, sample.role, sample.year, region=args.region),
            )
            mask = np.asarray(
                event_cut_mask(events, sample.role, sample.year, region=args.region),
                dtype=bool,
            )
            if not mask.any():
                print(
                    f"No selected events in {os.path.basename(file_path)} "
                    f"for sample '{name}' (region={args.region}); skipping scoring"
                )
                continue

            selected_events = events[mask]
            original_entries = np.flatnonzero(mask)
            scores = scorer.score_events(selected_events)
            columns = select_scored_jets(
                scores,
                selected_events,
                original_entries,
                build_config,
                weight,
                args.extra_branches,
            )
            if not columns["score"]:
                print(
                    f"No selected jets in {os.path.basename(file_path)} "
                    f"for sample '{name}'; skipping output"
                )
                continue

            output_name = score_output_basename(
                sample.role, file_path, sample_name=name, dataset=dataset
            )
            output_path = os.path.join(output_dir, output_name)
            write_score_file(output_path, columns)
            n_jets = len(columns["score"])
            n_jets_total += n_jets
            print(
                f"Wrote {output_path} ({n_jets} jets) "
                f"<= {file_path}"
            )

        for dataset, counts in cutflows.items():
            print(format_cutflow(f"{name} / {dataset} (scoring, region={args.region})", counts))

        if not n_jets_total:
            print(f"Sample '{name}' has no selected jets; nothing written")


if __name__ == "__main__":
    main()
