#!/usr/bin/env python3
"""Make the autoencoder performance plots in CMS style.

Reads the per-sample score files written by score_gae.py and the training
history written by train_gae.py.

    python plot_gae.py --history training/GraphAE_graphAE/history.json
    python plot_gae.py --checkpoint training/GraphAE_graphAE/checkpoint.pt \
        --residual-sample background_2018

Background samples are pooled; every signal sample is kept separate, so the
ROC and SIC plots show one curve per signal point against the pooled
background.
"""

import argparse
import glob
import json
import os

import numpy as np
import uproot

from utils.GaeConfig import load_config
from utils.InputJson import score_output_basename
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
    parser.add_argument(
        "--scores-dir",
        default="output_scores",
        help="Directory of score ROOT files from score_gae.py",
    )
    parser.add_argument(
        "--samples",
        nargs="+",
        default=None,
        help=(
            "Config sample names to plot; default combines score files in "
            "--scores-dir into data, qcd, and one series per signal point"
        ),
    )
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


def signal_label(sample, include_year=False):
    """Short legend label: the dataset name for per-dataset signal samples."""
    if sample.datasets and len(sample.datasets) == 1:
        tag = sample.datasets[0].replace("s-channel_", "")
        if include_year and getattr(sample, "year", None):
            return f"{tag} ({sample.year})"
        return tag
    return sample.name


def score_glob_for_sample(sample):
    """Glob pattern for per-file outputs under ``output_scores/``."""
    if sample.role == "data":
        return f"{sample.name}_*.root"
    if sample.role == "background":
        clean_s = sample.name.replace("background_", "").replace("qcd_", "")
        return f"qcd_{clean_s}_*.root" if clean_s else "qcd_*.root"
    if sample.role == "signal" and sample.datasets and len(sample.datasets) == 1:
        basename = score_output_basename(
            sample.role, "part-0.root", sample_name=sample.name, dataset=sample.datasets[0]
        )
        return basename.replace("part-0.root", "part-*.root")
    return f"{sample.name}*.root"


def score_paths_for_sample(scores_dir, sample):
    legacy = os.path.join(scores_dir, f"{sample.name}.root")
    if os.path.exists(legacy):
        return [legacy]
    pattern = os.path.join(scores_dir, score_glob_for_sample(sample))
    paths = sorted(glob.glob(pattern))
    if paths:
        return paths
    if sample.role == "data":
        fallback = sorted(glob.glob(os.path.join(scores_dir, "data_part-*.root")))
        if fallback:
            return fallback
    elif sample.role == "background":
        fallback = sorted(glob.glob(os.path.join(scores_dir, "qcd_part-*.root")))
        if fallback:
            return fallback
    elif sample.role == "signal" and sample.datasets and len(sample.datasets) == 1:
        tag = sample.datasets[0].replace("s-channel_", "")
        fallback = sorted(glob.glob(os.path.join(scores_dir, f"signal_{tag}_part-*.root")))
        if fallback:
            return fallback
    return []


def overlay_group_key(stem):
    """Key for combining ``*_part-N.root`` files into one plot series."""
    if stem.startswith("data_part-") or stem.startswith("data_") or stem.startswith("smoke_"):
        return "data"
    if stem.startswith("qcd_part-") or stem.startswith("qcd_") or stem.startswith("background_"):
        return "qcd"
    if stem.startswith("signal_") and "_part-" in stem:
        return stem.rsplit("_part-", 1)[0]
    return stem


def role_for_overlay_key(key):
    if key == "data" or key.startswith("data_") or key.startswith("smoke"):
        return "data"
    if key == "qcd" or key.startswith("qcd_") or key.startswith("background_"):
        return "background"
    if key.startswith("signal_"):
        return "signal"
    return None


def combined_overlays(scores_dir):
    """Merge part files into data, qcd, and one entry per signal point."""
    buckets = {}
    for path in sorted(glob.glob(os.path.join(scores_dir, "*.root"))):
        stem = os.path.splitext(os.path.basename(path))[0]
        buckets.setdefault(overlay_group_key(stem), []).append(path)

    overlays = {}
    roles = {}
    for key, paths in sorted(buckets.items()):
        scores_parts = []
        weights_parts = []
        for path in paths:
            score, weight = read_scores(path)
            scores_parts.append(score)
            weights_parts.append(weight)
        overlays[key] = (
            np.concatenate(scores_parts),
            np.concatenate(weights_parts),
        )
        role = role_for_overlay_key(key)
        if role is not None:
            roles[key] = role
    return overlays, roles


def load_samples_from_config(config, scores_dir, sample_names):
    """Load only the named config samples (each may match several part files)."""
    samples = {}
    for sample in config.select(sample_names):
        paths = score_paths_for_sample(scores_dir, sample)
        if not paths:
            continue
        scores_parts = []
        weights_parts = []
        for path in paths:
            score, weight = read_scores(path)
            scores_parts.append(score)
            weights_parts.append(weight)
        samples[sample.name] = (
            sample.role,
            np.concatenate(scores_parts),
            np.concatenate(weights_parts),
        )
    return samples


def pool_by_role(samples):
    pooled = {}
    for role in ("signal", "background", "data"):
        scores = [values[1] for values in samples.values() if values[0] == role]
        weights = [values[2] for values in samples.values() if values[0] == role]
        if scores:
            pooled[role] = (np.concatenate(scores), np.concatenate(weights))
    return pooled


def pool_discovered(overlays, roles):
    pooled = {}
    for role in ("signal", "background", "data"):
        scores = []
        weights = []
        for stem, sample_role in roles.items():
            if sample_role != role:
                continue
            part_scores, part_weights = overlays[stem]
            scores.append(part_scores)
            weights.append(part_weights)
        if scores:
            pooled[role] = (np.concatenate(scores), np.concatenate(weights))
    return pooled


def load_samples(config, scores_dir, sample_names=None):
    """Overlays for plotting and pooled scores per role."""
    if sample_names:
        samples = load_samples_from_config(config, scores_dir, sample_names)
        overlays = {
            name: (scores, weights)
            for name, (_, scores, weights) in samples.items()
        }
        return overlays, pool_by_role(samples)

    overlays, roles = combined_overlays(scores_dir)
    return overlays, pool_discovered(overlays, roles)


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

    overlays, pooled = load_samples(
        config, args.scores_dir, sample_names=args.samples
    )
    if not overlays:
        print(
            f"No score files found in '{args.scores_dir}'; run score_gae.py "
            "first to make the score based plots"
        )
    else:
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

        if args.samples:
            sig_samples = [config.sample(name) for name in overlays if name in config.samples and config.sample(name).role == "signal"]
            multi_year = len({getattr(s, "year", None) for s in sig_samples if getattr(s, "year", None)}) > 1
            signals = {
                signal_label(config.sample(name), include_year=multi_year): overlays[name]
                for name in overlays
                if name in config.samples and config.sample(name).role == "signal"
            }
        else:
            signals = {
                key.replace("signal_", "", 1): overlays[key]
                for key in overlays
                if key.startswith("signal_")
            }
        if signals and "background" in pooled:
            background_scores, background_weights = pooled["background"]
            plot_roc(
                signals,
                background_scores,
                os.path.join(args.outdir, "roc"),
                weights_background=background_weights,
                formats=formats,
            )
            plot_sic_vs_fpr(
                signals,
                background_scores,
                os.path.join(args.outdir, "sic_vs_fpr"),
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
