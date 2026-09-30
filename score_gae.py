#!/usr/bin/env python3
"""Score ROOT files with a trained autoencoder and write per-jet scores.

Graphs are streamed straight from the ROOT files by RootGraphScorer, so no
intermediate .pt datasets are needed. The selections in utils/Cuts.py are
applied afterwards, using the fact that the scorer returns its scores in the
original event and jet order.

    python score_gae.py --checkpoint training/GraphAE_graphAE/checkpoint.pt
    python score_gae.py --checkpoint ... --samples smoke

One output file per sample is written to --output-dir, holding a flat
'Scores' tree with one entry per selected jet.
"""

import argparse
import dataclasses
import os

import numpy as np
import uproot

from architectures.ModelRegistry import build_model
from utils.Cuts import CUT_BRANCHES, event_cut_mask, jet_cut_mask
from utils.FileLists import resolve_file_list, shard_file_list
from utils.GaeConfig import load_config
from utils.GraphBuilder import GraphBuildConfig, pfnano_branches
from utils.RootGraphScorer import RootGraphScorer

import torch


def parse_arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="gae_config.json")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument(
        "--samples",
        nargs="+",
        default=None,
        help="Sample names to score; defaults to every configured sample",
    )
    parser.add_argument("--output-dir", default="scores")
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


def base_event_weight(file_path, weights):
    """Per-event weight for a file: the cross section weight, or one."""
    if weights != "xsec":
        return 1.0
    with uproot.open(file_path) as input_file:
        if "Metadata" not in input_file:
            raise ValueError(
                f"'{file_path}' has no Metadata tree, so weights='xsec' "
                "cannot be used for this sample"
            )
        xsec = input_file["Metadata"]["GenCrossSection"].array(entry_stop=1)[0]
        n_initial = input_file["CutFlow"]["Initial"].array(entry_stop=1)[0]
    return float(xsec / n_initial)


def select_scored_jets(scores, events, config, weight, extra_branches):
    """Flatten jagged scores to selected jets, applying the shared cuts."""
    mask = np.asarray(event_cut_mask(events), dtype=bool)
    if mask.shape != (len(events),):
        raise ValueError(
            "event_cut_mask must return one boolean per event, got shape "
            f"{mask.shape} for {len(events)} events"
        )

    columns = {
        "entry": [],
        "jet": [],
        "score": [],
        "weight": [],
    }
    for name in extra_branches:
        columns[name] = []

    for entry in np.flatnonzero(mask):
        event = events[entry]
        for jet_index, score in enumerate(scores[entry]):
            # NaN marks a jet the scorer could not build a graph for.
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


def main():
    args = parse_arguments()
    config = load_config(args.config)
    verbose = not args.quiet

    checkpoint = torch.load(args.checkpoint, weights_only=False)
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
    print(
        f"Loaded {checkpoint['model_spec']} with "
        f"{checkpoint['num_features']} node features on {device}"
    )

    graph_settings = checkpoint["graph"]
    max_jets = graph_settings.get("n_store_jets")
    os.makedirs(args.output_dir, exist_ok=True)

    names = args.samples if args.samples else sorted(config.samples)
    for name in names:
        sample = config.sample(name)
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

        file_paths = resolve_file_list(sample.file_list)
        if args.files_per_job is not None and args.shard is not None:
            file_paths = shard_file_list(file_paths, args.files_per_job)[
                args.shard
            ]

        branches = list(CUT_BRANCHES)
        for branch in args.extra_branches:
            if branch not in branches:
                branches.append(branch)
        # The jet selection may look at the jet collection, so read the
        # graph-building branches too.
        read_branches = sorted(
            set(branches) | set(pfnano_branches(build_config))
        )

        columns = None
        for file_path in file_paths:
            scores = scorer.score_file(file_path)
            weight = base_event_weight(file_path, sample.weights)
            with uproot.open(f"{file_path}:Events") as tree:
                events = tree.arrays(
                    read_branches,
                    entry_stop=len(scores),
                    library="ak",
                )
            file_columns = select_scored_jets(
                scores, events, build_config, weight, args.extra_branches
            )
            if columns is None:
                columns = file_columns
            else:
                for key, values in file_columns.items():
                    columns[key].extend(values)

        if not columns or not columns["score"]:
            print(f"Sample '{name}' has no selected jets; nothing written")
            continue

        output_path = os.path.join(args.output_dir, f"{name}.root")
        with uproot.recreate(output_path) as output_file:
            output_file["Scores"] = {
                key: np.asarray(values)
                for key, values in columns.items()
            }
        print(
            f"Wrote {output_path} with {len(columns['score'])} selected jets "
            f"for sample '{name}' ({sample.role})"
        )


if __name__ == "__main__":
    main()
