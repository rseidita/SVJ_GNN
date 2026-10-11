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
    format_signal_label,
    plot_auc_2d,
    plot_edge_calibration,
    plot_feature_correlations,
    plot_feature_importance_ranking,
    plot_feature_residuals,
    plot_latent_distributions,
    plot_latent_pca_spectrum,
    plot_latent_projection,
    plot_loss_curves,
    plot_loss_efficiency,
    plot_loss_vs_jet_kinematics,
    plot_paired_trees,
    plot_roc,
    plot_score_distributions,
    plot_sic_vs_fpr,
    plot_tree_with_scores,
    plot_xai_feature_saliency,
    write_th1s,
)



def parse_arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="gae_config.json")
    parser.add_argument(
        "--scores-dir",
        default=None,
        help="Directory of score ROOT files (defaults to scoring.output_dir in config)",
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
    parser.add_argument(
        "--outdir",
        default=None,
        help="Output directory for plots (defaults to plotting.plots_dir in config)",
    )
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
    parser.add_argument(
        "--compare-signal",
        nargs="?",
        const="baseline",
        default=None,
        help="Signal sample to compare in diagnostics (defaults to baseline m_{Z'}=1000, rinv=0.3)",
    )
    parser.add_argument("--run-name", default=None, help="Training run name to plot scores and history for")
    parser.add_argument("--seed", type=int, default=None, help="Training random seed to plot")
    parser.add_argument("--seeds", type=int, nargs="+", default=None, help="List of random seeds to plot")
    parser.add_argument("--eos-dir", default="/eos/user/n/nparmar/SVJScouting", help="EOS directory prefix")
    parser.add_argument("--year", default=None, help="Year to display in CMS label (e.g. 2018)")
    parser.add_argument("--lumi", type=float, default=None, help="Luminosity in fb-1 to display in CMS label")
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
            try:
                score, weight = read_scores(path)
            except (ValueError, OSError, KeyError, IndexError) as exc:
                print(f"Skipping unreadable score file {path}: {exc}")
                continue
            scores_parts.append(score)
            weights_parts.append(weight)
        if not scores_parts:
            print(f"Skipping {key}: no readable score files")
            continue
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
            try:
                score, weight = read_scores(path)
            except (ValueError, OSError, KeyError, IndexError) as exc:
                print(f"Skipping unreadable score file {path}: {exc}")
                continue
            scores_parts.append(score)
            weights_parts.append(weight)
        if not scores_parts:
            print(f"Skipping {sample.name}: no readable score files")
            continue
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


def load_or_stream_signal_batches(config, signal_spec, checkpoint, max_batches=5, device="cpu"):
    """Load preprocessed graph dataset or stream graphs on the fly from ROOT file."""
    import torch
    from torch_geometric.data import Batch
    from data_loading.graphDataLoader import GraphDataLoader
    from utils.LundTreeUtilities import OnTheFlyNormalizer
    from utils.GraphBuilder import GraphBuildConfig, build_pfnano_jet_graph, pfnano_branches
    from utils.InputJson import flat_files
    import uproot

    target_sample = None
    if signal_spec and signal_spec in config.samples:
        target_sample = config.sample(signal_spec)
    elif signal_spec:
        mmed_target = "1000" if (signal_spec == "baseline" or "1000" in signal_spec) else None
        rinv_target = "0.3" if (signal_spec == "baseline" or "0.3" in signal_spec) else None
        for s in config.samples_with_role("signal"):
            if mmed_target and mmed_target in s.name and rinv_target and rinv_target in s.name:
                target_sample = s
                break
        if target_sample is None and len(config.samples_with_role("signal")) > 0:
            target_sample = config.samples_with_role("signal")[0]

    if target_sample is None:
        print(f"No signal sample found matching '{signal_spec}'; skipping signal comparison")
        return None, None

    normalizer = OnTheFlyNormalizer(["x"], checkpoint["means"], checkpoint["stds"])
    sig_label = format_signal_label(target_sample.name) if target_sample else "Signal"

    try:
        sig_droot = config.dataset_root(target_sample, require_exists=True)
        sig_dataset_paths = [config.dataset_paths(target_sample)]
        sig_loader = GraphDataLoader(
            dataset_paths=sig_dataset_paths,
            split_windows=[[0.0, 1.0]],
            sampling="sequential",
            normalizer=normalizer,
            batch_size=checkpoint.get("training", {}).get("batch_size", 64),
        )
        sig_batches = []
        for idx, b in enumerate(sig_loader):
            if idx >= max_batches:
                break
            sig_batches.append(b)
        if sig_batches:
            print(f"Loaded {len(sig_batches)} signal batches from dataset {target_sample.name}")
            return sig_batches, sig_label
    except Exception:
        pass

    try:
        groups = target_sample.groups()
        all_files = flat_files(groups)
        if not all_files:
            print(f"No ROOT files found for signal {target_sample.name}")
            return None, None

        graph_cfg_dict = dict(config.graph)
        num_features = checkpoint.get("num_features", 12)
        save_4v = (num_features == 12)
        graph_build_cfg = GraphBuildConfig(
            jet_collection=graph_cfg_dict.get("jet_collection", "FatJet"),
            use_lund_decomp=graph_cfg_dict.get("use_lund_decomp", True),
            n_lund_vars=graph_cfg_dict.get("n_lund_vars", 5),
            save_energy_fractions=graph_cfg_dict.get("save_energy_fractions", True),
            save_4vectors_in_lund_tree=save_4v,
            use_displacement=graph_cfg_dict.get("use_displacement", False),
            kt_cut=graph_cfg_dict.get("kt_cut", None),
        )

        first_root_path = all_files[0]
        print(f"Streaming signal graphs on the fly from {first_root_path}...")
        branches = pfnano_branches(graph_build_cfg)
        with uproot.open(first_root_path) as rf:
            events = rf["Events"].arrays(branches, entry_stop=max_batches * 25, library="ak")

        graphs = []
        means = (checkpoint["means"] if isinstance(checkpoint["means"], torch.Tensor) else checkpoint["means"]["x"]).cpu()
        stds = (checkpoint["stds"] if isinstance(checkpoint["stds"], torch.Tensor) else checkpoint["stds"]["x"]).cpu()
        for ev in events:
            n_jets = int(ev[f"n{graph_build_cfg.jet_collection}"])
            for j in range(n_jets):
                g, _ = build_pfnano_jet_graph(ev, j, graph_build_cfg)
                if g is not None:
                    g.x = (g.x - means) / stds
                    graphs.append(g)
                if len(graphs) >= max_batches * 64:
                    break
            if len(graphs) >= max_batches * 64:
                break

        if not graphs:
            print("No signal graphs could be constructed")
            return None, None

        sig_batches = []
        bsize = 64
        for i in range(0, len(graphs), bsize):
            chunk = graphs[i : i + bsize]
            sig_batches.append(Batch.from_data_list(chunk))
            if len(sig_batches) >= max_batches:
                break

        print(f"Built {len(graphs)} signal graphs ({len(sig_batches)} batches) on the fly for {sig_label}")
        return sig_batches, sig_label
    except Exception as exc:
        print(f"Could not stream signal graphs: {exc}")
        return None, None


def residual_plots(checkpoint_path, residual_sample, config, formats, outdir, device="cpu", bins=50, max_batches=20, year=None, lumi=None, compare_signal=None):
    """Run full diagnostic, latent representation, and XAI suite using model and graph dataset."""
    import torch
    from architectures.ModelRegistry import build_model
    from data_loading.graphDataLoader import GraphDataLoader
    from utils.LundTreeUtilities import OnTheFlyNormalizer
    from utils.XaiUtils import compute_tree_node_attributions

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    print(f"Model: {checkpoint['model_spec']}")
    print(f"Model checkpoint path: {checkpoint_path}")
    print(f"Diagnostic dataset sample: {residual_sample}")
    model = build_model(checkpoint["model_spec"], checkpoint["num_features"], **checkpoint["model_kwargs"])
    model.load_state_dict(checkpoint["state_dict"])
    model = model.to(device)

    sample = config.sample(residual_sample)
    normalizer = OnTheFlyNormalizer(["x"], checkpoint["means"], checkpoint["stds"])
    settings = checkpoint.get("training", {})
    dataset_paths = [config.dataset_paths(sample)]

    split_state = checkpoint.get("split_state")
    if split_state is not None:
        state_datasets = split_state.get("datasets", {})
        for path, name in dataset_paths:
            resolved_key = f"{os.path.realpath(path)}::{name}"
            raw_key = f"{os.path.abspath(path)}::{name}"
            if resolved_key not in state_datasets and raw_key not in state_datasets:
                split_state = None
                break

    loader = GraphDataLoader(
        dataset_paths=dataset_paths,
        split_windows=[list(settings.get("test_window", [0.85, 1.0]))],
        split_state=split_state,
        split_seed=None if split_state is not None else settings.get("split_seed", 12345),
        sampling="sequential",
        normalizer=normalizer,
        batch_size=settings.get("batch_size", 64),
    )

    batches = []
    for idx, batch in enumerate(loader):
        if idx >= max_batches:
            break
        batches.append(batch)

    if not batches:
        print("No batches loaded for diagnostic plots; skipping")
        return

    sig_batches, sig_label = None, None
    if compare_signal:
        sig_batches, sig_label = load_or_stream_signal_batches(config, compare_signal, checkpoint, max_batches=max_batches, device=device)

    plot_feature_residuals(model, batches, os.path.join(outdir, "feature_residuals"), device=device, max_batches=max_batches, bins=bins, formats=formats, lumi=lumi, year=year, config=config, signal_loader=sig_batches, signal_label=sig_label if sig_label else "Signal")
    print("Wrote feature_residuals")
    plot_latent_distributions(model, batches, os.path.join(outdir, "latent_distributions"), device=device, max_batches=max_batches, bins=bins, formats=formats, lumi=lumi, year=year)
    print("Wrote latent_distributions")
    plot_feature_correlations(model, batches, os.path.join(outdir, "feature_correlations"), device=device, max_batches=max_batches, formats=formats, lumi=lumi, year=year, config=config)
    print("Wrote feature_correlations")
    plot_feature_importance_ranking(model, batches, os.path.join(outdir, "feature_importance_ranking"), device=device, max_batches=max_batches, formats=formats, lumi=lumi, year=year, config=config, signal_loader=sig_batches, signal_label=sig_label if sig_label else "Signal")
    print("Wrote feature_importance_ranking")
    plot_edge_calibration(model, batches, os.path.join(outdir, "edge_calibration"), device=device, max_batches=max_batches, bins=bins, formats=formats, lumi=lumi, year=year, signal_loader=sig_batches, signal_label=sig_label if sig_label else "Signal")
    print("Wrote edge_calibration")
    plot_loss_vs_jet_kinematics(model, batches, os.path.join(outdir, "loss_vs_jet_kinematics"), device=device, max_batches=max_batches, formats=formats, lumi=lumi, year=year)
    print("Wrote loss_vs_jet_kinematics")
    plot_latent_pca_spectrum(model, batches, os.path.join(outdir, "latent_pca_spectrum"), device=device, max_batches=max_batches, formats=formats, lumi=lumi, year=year)
    print("Wrote latent_pca_spectrum")
    plot_latent_projection(model, batches, os.path.join(outdir, "latent_projection_2d"), device=device, max_batches=max_batches, formats=formats, lumi=lumi, year=year, signal_loader=sig_batches, signal_label=sig_label if sig_label else "Signal")
    print("Wrote latent_projection_2d")
    plot_xai_feature_saliency(model, batches, os.path.join(outdir, "xai_feature_saliency"), device=device, max_batches=max_batches, formats=formats, lumi=lumi, year=year, config=config, signal_loader=sig_batches, signal_label=sig_label if sig_label else "Signal")
    print("Wrote xai_feature_saliency")

    try:
        single_graph = batches[0].get_example(0) if hasattr(batches[0], "get_example") else batches[0]
        node_mse, node_saliency, _ = compute_tree_node_attributions(model, single_graph, device=device)
        if sig_batches:
            single_sig = sig_batches[0].get_example(0) if hasattr(sig_batches[0], "get_example") else sig_batches[0]
            sig_node_mse, sig_node_sal, _ = compute_tree_node_attributions(model, single_sig, device=device)
            plot_paired_trees(single_graph, node_mse, single_sig, sig_node_mse, os.path.join(outdir, "lund_tree_loss_annotated"), label="Node Feature MSE", sig_label=sig_label if sig_label else "Signal", formats=formats, lumi=lumi, year=year)
            plot_paired_trees(single_graph, node_saliency, single_sig, sig_node_sal, os.path.join(outdir, "lund_tree_xai_annotated"), label=r"Node Attribution ($|x \cdot \nabla_x \mathcal{L}|$)", sig_label=sig_label if sig_label else "Signal", formats=formats, lumi=lumi, year=year)
        else:
            plot_tree_with_scores(single_graph, node_mse, os.path.join(outdir, "lund_tree_loss_annotated"), label="Node Feature MSE", formats=formats, lumi=lumi, year=year)
            plot_tree_with_scores(single_graph, node_saliency, os.path.join(outdir, "lund_tree_xai_annotated"), label=r"Node Attribution ($|x \cdot \nabla_x \mathcal{L}|$)", formats=formats, lumi=lumi, year=year)
        print("Wrote lund_tree_loss_annotated and lund_tree_xai_annotated")
    except Exception as exc:
        print(f"Skipping annotated Lund tree plots: {exc}")


def plot_single_target(config, scores_dir, outdir, formats, bins=50, history_path=None, checkpoint_path=None, residual_sample=None, sample_names=None, device="cpu", max_batches=20, year=None, lumi=None, compare_signal=None):
    os.makedirs(outdir, exist_ok=True)
    if history_path is not None and os.path.isfile(history_path):
        with open(history_path) as history_file:
            history = json.load(history_file)
        plot_loss_curves(history, os.path.join(outdir, "loss_curves"), formats=formats, year=year, lumi=lumi)
        print("Wrote loss_curves")

    overlays, pooled = load_samples(config, scores_dir, sample_names=sample_names)
    if not overlays:
        print(f"No score files found in '{scores_dir}'; run score_gae.py first to make the score based plots")
    else:
        histograms = plot_score_distributions(overlays, os.path.join(outdir, "score_distributions"), bins=bins * 2, density=True, formats=formats, year=year, lumi=lumi)
        plot_loss_efficiency(overlays, os.path.join(outdir, "loss_efficiency"), formats=formats, year=year, lumi=lumi)
        print(f"Wrote score_distributions and loss_efficiency with location - {outdir}")
        write_th1s(os.path.join(outdir, "histograms.root"), histograms)
        print("Wrote histograms.root")

        if sample_names:
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
            curves, aucs = plot_roc(signals, background_scores, os.path.join(outdir, "roc"), weights_background=background_weights, formats=formats, year=year, lumi=lumi)
            plot_sic_vs_fpr(signals, background_scores, os.path.join(outdir, "sic_vs_fpr"), weights_background=background_weights, formats=formats, year=year, lumi=lumi)
            print("Wrote roc and sic_vs_fpr")
            h_auc_2d = plot_auc_2d(aucs, os.path.join(outdir, "auc_2d"), formats=formats, year=year, lumi=lumi)
            if h_auc_2d is not None:
                histograms["auc_2d"] = h_auc_2d
                write_th1s(os.path.join(outdir, "histograms.root"), histograms)
            print("Wrote auc_2d")
        else:
            print("Need both a signal and a background sample with scores for the ROC and SIC plots; skipping them")

    if checkpoint_path is not None and residual_sample is not None and os.path.isfile(checkpoint_path):
        try:
            residual_plots(checkpoint_path, residual_sample, config, formats, outdir, device=device, bins=bins, max_batches=max_batches, year=year, lumi=lumi, compare_signal=compare_signal)
        except Exception as exc:
            print(f"Error producing diagnostic and residual plots for sample {residual_sample}: {exc}")

    print(f"Plots are in {outdir}")


def main():
    args = parse_arguments()
    config = load_config(args.config)
    formats = tuple(args.formats.split(","))

    seeds = args.seeds if args.seeds else ([args.seed] if args.seed is not None else [None])
    base_scores_cfg = config.scoring.get("output_dir", "output_scores")
    base_plots_cfg = config.plotting.get("plots_dir", "plots")
    base_train_cfg = config.training.get("output_dir", "training")

    for seed in seeds:
        seed_tag = f"seed{seed}" if seed is not None else ""
        if args.scores_dir:
            scores_dir = os.path.join(args.scores_dir, seed_tag) if (len(seeds) > 1 and seed is not None and not args.scores_dir.endswith(seed_tag)) else args.scores_dir
        elif args.run_name:
            rel = os.path.join(base_scores_cfg, args.run_name, seed_tag) if seed is not None else os.path.join(base_scores_cfg, args.run_name)
            eos_cand = os.path.join(args.eos_dir, rel) if args.eos_dir else None
            scores_dir = eos_cand if (eos_cand and os.path.isdir(eos_cand)) else rel
        else:
            eos_cand = os.path.join(args.eos_dir, base_scores_cfg) if args.eos_dir else None
            scores_dir = eos_cand if (eos_cand and os.path.isdir(eos_cand)) else base_scores_cfg

        if args.outdir:
            outdir = os.path.join(args.outdir, seed_tag) if (len(seeds) > 1 and seed is not None and not args.outdir.endswith(seed_tag)) else args.outdir
        elif args.run_name:
            outdir = os.path.join(base_plots_cfg, args.run_name, seed_tag) if seed is not None else os.path.join(base_plots_cfg, args.run_name)
        else:
            outdir = base_plots_cfg

        history_path = args.history
        if history_path is None and args.run_name:
            run_tag = f"{args.run_name}_{seed_tag}" if seed is not None else args.run_name
            cands = []
            if args.eos_dir:
                cands.extend([os.path.join(args.eos_dir, base_train_cfg, run_tag, "history.json"), os.path.join(args.eos_dir, "training_aimar", run_tag, "history.json"), os.path.join(args.eos_dir, run_tag, "history.json")])
            cands.extend([os.path.join(base_train_cfg, run_tag, "history.json"), os.path.join("training_aimar", run_tag, "history.json")])
            for cand in cands:
                if os.path.isfile(cand):
                    history_path = cand
                    break

        checkpoint_path = args.checkpoint
        if checkpoint_path is None and args.run_name:
            try:
                checkpoint_path = config.find_checkpoint(run_name=args.run_name, seed=seed, base_dir=args.eos_dir)
            except Exception:
                run_tag = f"{args.run_name}_{seed_tag}" if seed is not None else args.run_name
                ckpt_cands = []
                for train_sub in [base_train_cfg, "training_aimar", ""]:
                    for fname in ["checkpoint_best.pt", "checkpoint.pt", "checkpoint_latest.pt"]:
                        if args.eos_dir:
                            ckpt_cands.append(os.path.join(args.eos_dir, train_sub, run_tag, fname))
                        ckpt_cands.append(os.path.join(train_sub, run_tag, fname))
                for cand in ckpt_cands:
                    if os.path.isfile(cand):
                        checkpoint_path = cand
                        break

        residual_sample = args.residual_sample
        if residual_sample is None and checkpoint_path is not None and os.path.isfile(checkpoint_path):
            import torch
            def _sample_exists(s_name):
                if s_name not in config.samples:
                    return False
                try:
                    config.dataset_root(config.sample(s_name), base_dir=args.eos_dir, require_exists=True)
                    return True
                except Exception:
                    return False

            try:
                ckpt_meta = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
                for ts in ckpt_meta.get("train_samples", []):
                    if _sample_exists(ts):
                        residual_sample = ts
                        break
            except Exception:
                pass
            if residual_sample is None:
                for bg in config.samples_with_role("background"):
                    if _sample_exists(bg.name):
                        residual_sample = bg.name
                        break
            if residual_sample is None:
                for s_name in config.samples:
                    if _sample_exists(s_name):
                        residual_sample = s_name
                        break

        print(f"\n--- Plotting for {'seed ' + str(seed) if seed is not None else 'default'} (scores: {scores_dir}) ---")
        plot_single_target(
            config,
            scores_dir,
            outdir,
            formats,
            bins=args.bins,
            history_path=history_path,
            checkpoint_path=checkpoint_path,
            residual_sample=residual_sample,
            sample_names=args.samples,
            device=args.device,
            max_batches=args.max_batches,
            year=args.year,
            lumi=args.lumi,
            compare_signal=args.compare_signal,
        )


if __name__ == "__main__":
    main()

