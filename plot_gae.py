#!/usr/bin/env python3
"""Make the autoencoder performance plots in CMS style.

Reads the per-sample score files written by score_gae.py and the training
history written by train_gae.py.

    python plot_gae.py --history training/GraphAE_graphAE/history.json
    python plot_gae.py --checkpoint training/GraphAE_graphAE/checkpoint.pt \
        --residual-sample background_example

Signal and background are pooled by role, so the ROC and SIC plots use every
sample with role "signal" against every sample with role "background".
"""

import argparse
import json
import os

import numpy as np
import uproot

from utils.GaeConfig import load_config
from utils.PlotUtils import (
    plot_feature_residuals,
    plot_latent_distributions,
    plot_loss_curves,
    plot_loss_efficiency,
    plot_roc,
    plot_score_distributions,
    plot_sic_vs_fpr,
    write_th1s,
)


def parse_arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="gae_config.json")
    parser.add_argument("--scores-dir", default="scores")
    parser.add_argument("--history", default=None)
    parser.add_argument("--outdir", default="plots")
    parser.add_argument(
        "--formats",
        default="pdf",
        help="Comma separated list of output image formats",
    )
    parser.add_argument("--bins", type=int, default=50)
    parser.add_argument(
        "--checkpoint",
        default=None,
        help="Needed for the residual and latent plots",
    )
    parser.add_argument(
        "--residual-sample",
        default=None,
        help="Sample with produced graphs used for residual and latent plots",
    )
    parser.add_argument("--max-batches", type=int, default=20)
    parser.add_argument("--device", default="cpu")
    return parser.parse_args()


def read_scores(path):
    with uproot.open(f"{path}:Scores") as tree:
        arrays = tree.arrays(["score", "weight"], library="np")
    return arrays["score"], arrays["weight"]


def load_samples(config, scores_dir):
    """Per-sample scores and weights, plus the pooled scores per role."""
    samples = {}
    for name, sample in config.samples.items():
        path = os.path.join(scores_dir, f"{name}.root")
        if not os.path.exists(path):
            continue
        samples[name] = (sample.role, *read_scores(path))

    pooled = {}
    for role in ("signal", "background", "data"):
        scores = [
            values[1] for values in samples.values() if values[0] == role
        ]
        weights = [
            values[2] for values in samples.values() if values[0] == role
        ]
        if scores:
            pooled[role] = (
                np.concatenate(scores),
                np.concatenate(weights),
            )
    return samples, pooled


def residual_plots(args, config, formats):
    """Residual and latent plots, which need the model and a graph dataset."""
    import torch

    from architectures.ModelRegistry import build_model
    from data_loading.graphDataLoader import GraphDataLoader
    from utils.LundTreeUtilities import OnTheFlyNormalizer

    checkpoint = torch.load(args.checkpoint, weights_only=False)
    model = build_model(
        checkpoint["model_spec"],
        checkpoint["num_features"],
        **checkpoint["model_kwargs"],
    )
    model.load_state_dict(checkpoint["state_dict"])
    model = model.to(args.device)

    sample = config.sample(args.residual_sample)
    normalizer = OnTheFlyNormalizer(
        ["x"], checkpoint["means"], checkpoint["stds"]
    )
    settings = checkpoint.get("training", {})
    loader = GraphDataLoader(
        dataset_paths=[config.dataset_paths(sample)],
        split_windows=[list(settings.get("test_window", [0.85, 1.0]))],
        split_state=checkpoint.get("split_state"),
        sampling="sequential",
        normalizer=normalizer,
        batch_size=settings.get("batch_size", 64),
    )

    plot_feature_residuals(
        model,
        loader,
        os.path.join(args.outdir, "feature_residuals"),
        device=args.device,
        max_batches=args.max_batches,
        bins=args.bins,
        formats=formats,
    )
    plot_latent_distributions(
        model,
        loader,
        os.path.join(args.outdir, "latent_distributions"),
        device=args.device,
        max_batches=args.max_batches,
        bins=args.bins,
        formats=formats,
    )


def main():
    args = parse_arguments()
    config = load_config(args.config)
    formats = tuple(args.formats.split(","))
    os.makedirs(args.outdir, exist_ok=True)

    if args.history is not None:
        with open(args.history) as history_file:
            history = json.load(history_file)
        plot_loss_curves(
            history,
            os.path.join(args.outdir, "loss_curves"),
            formats=formats,
        )
        print("Wrote loss_curves")

    samples, pooled = load_samples(config, args.scores_dir)
    if not samples:
        print(
            f"No score files found in '{args.scores_dir}'; run score_gae.py "
            "first to make the score based plots"
        )
    else:
        overlays = {
            name: (scores, weights)
            for name, (_, scores, weights) in samples.items()
        }
        histograms = plot_score_distributions(
            overlays,
            os.path.join(args.outdir, "score_distributions"),
            bins=args.bins,
            density=True,
            formats=formats,
        )
        plot_loss_efficiency(
            overlays,
            os.path.join(args.outdir, "loss_efficiency"),
            formats=formats,
        )
        print("Wrote score_distributions and loss_efficiency")

        write_th1s(os.path.join(args.outdir, "histograms.root"), histograms)
        print("Wrote histograms.root")

        if "signal" in pooled and "background" in pooled:
            signal_scores, signal_weights = pooled["signal"]
            background_scores, background_weights = pooled["background"]
            plot_roc(
                signal_scores,
                background_scores,
                os.path.join(args.outdir, "roc"),
                weights_signal=signal_weights,
                weights_background=background_weights,
                formats=formats,
            )
            plot_sic_vs_fpr(
                signal_scores,
                background_scores,
                os.path.join(args.outdir, "sic_vs_fpr"),
                weights_signal=signal_weights,
                weights_background=background_weights,
                formats=formats,
            )
            print("Wrote roc and sic_vs_fpr")
        else:
            print(
                "Need both a signal and a background sample with scores for "
                "the ROC and SIC plots; skipping them"
            )

    if args.checkpoint is not None and args.residual_sample is not None:
        residual_plots(args, config, formats)
        print("Wrote feature_residuals and latent_distributions")

    print(f"Plots are in {args.outdir}")


if __name__ == "__main__":
    main()
