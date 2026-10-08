import re
from matplotlib.lines import Line2D
from numpy import random
import networkx as nx
import numpy as np
from torch_geometric.utils import to_networkx
import matplotlib.pyplot as plt

from utils.MetricUtils import (
    auc,
    efficiency_curve,
    max_sic,
    roc_curve,
    sic,
    weighted_histogram,
)


def hierarchy_pos(G, root=None, width=1., vert_gap = 0.2, vert_loc = 0, leaf_vs_root_factor = 0.5):

    '''
    If the graph is a tree this will return the positions to plot this in a 
    hierarchical layout.
    
    Based on Joel's answer at https://stackoverflow.com/a/29597209/2966723,
    but with some modifications.  

    We include this because it may be useful for plotting transmission trees,
    and there is currently no networkx equivalent (though it may be coming soon).
    
    There are two basic approaches we think of to allocate the horizontal 
    location of a node.  
    
    - Top down: we allocate horizontal space to a node.  Then its ``k`` 
      descendants split up that horizontal space equally.  This tends to result
      in overlapping nodes when some have many descendants.
    - Bottom up: we allocate horizontal space to each leaf node.  A node at a 
      higher level gets the entire space allocated to its descendant leaves.
      Based on this, leaf nodes at higher levels get the same space as leaf
      nodes very deep in the tree.  
      
    We use use both of these approaches simultaneously with ``leaf_vs_root_factor`` 
    determining how much of the horizontal space is based on the bottom up 
    or top down approaches.  ``0`` gives pure bottom up, while 1 gives pure top
    down.   
    
    
    :Arguments: 
    
    **G** the graph (must be a tree)

    **root** the root node of the tree 
    - if the tree is directed and this is not given, the root will be found and used
    - if the tree is directed and this is given, then the positions will be 
      just for the descendants of this node.
    - if the tree is undirected and not given, then a random choice will be used.

    **width** horizontal space allocated for this branch - avoids overlap with other branches

    **vert_gap** gap between levels of hierarchy

    **vert_loc** vertical location of root
    
    **leaf_vs_root_factor**

    xcenter: horizontal location of root
    '''
    if not nx.is_tree(G):
        raise TypeError('cannot use hierarchy_pos on a graph that is not a tree')

    if root is None:
        if isinstance(G, nx.DiGraph):
            root = next(iter(nx.topological_sort(G)))  #allows back compatibility with nx version 1.11
        else:
            root = random.choice(list(G.nodes))

    def _hierarchy_pos(G, root, leftmost, width, leafdx = 0.2, vert_gap = 0.2, vert_loc = 0, 
                    xcenter = 0.5, rootpos = None, 
                    leafpos = None, parent = None):
        '''
        see hierarchy_pos docstring for most arguments

        pos: a dict saying where all nodes go if they have been assigned
        parent: parent of this branch. - only affects it if non-directed

        '''

        if rootpos is None:
            rootpos = {root:(xcenter,vert_loc)}
        else:
            rootpos[root] = (xcenter, vert_loc)
        if leafpos is None:
            leafpos = {}
        children = list(G.neighbors(root))
        leaf_count = 0
        if not isinstance(G, nx.DiGraph) and parent is not None:
            children.remove(parent)  
        if len(children)!=0:
            rootdx = width/len(children)
            nextx = xcenter - width/2 - rootdx/2
            for child in children:
                nextx += rootdx
                rootpos, leafpos, newleaves = _hierarchy_pos(G,child, leftmost+leaf_count*leafdx, 
                                    width=rootdx, leafdx=leafdx,
                                    vert_gap = vert_gap, vert_loc = vert_loc-vert_gap, 
                                    xcenter=nextx, rootpos=rootpos, leafpos=leafpos, parent = root)
                leaf_count += newleaves

            leftmostchild = min((x for x,y in [leafpos[child] for child in children]))
            rightmostchild = max((x for x,y in [leafpos[child] for child in children]))
            leafpos[root] = ((leftmostchild+rightmostchild)/2, vert_loc)
        else:
            leaf_count = 1
            leafpos[root]  = (leftmost, vert_loc)
#        pos[root] = (leftmost + (leaf_count-1)*dx/2., vert_loc)
#        print(leaf_count)
        return rootpos, leafpos, leaf_count

    xcenter = width/2.
    if isinstance(G, nx.DiGraph):
        leafcount = len([node for node in nx.descendants(G, root) if G.out_degree(node)==0])
    elif isinstance(G, nx.Graph):
        leafcount = len([node for node in nx.node_connected_component(G, root) if G.degree(node)==1 and node != root])
    rootpos, leafpos, leaf_count = _hierarchy_pos(G, root, 0, width, 
                                                    leafdx=width*1./leafcount, 
                                                    vert_gap=vert_gap, 
                                                    vert_loc = vert_loc, 
                                                    xcenter = xcenter)
    pos = {}
    for node in rootpos:
        pos[node] = (leaf_vs_root_factor*leafpos[node][0] + (1-leaf_vs_root_factor)*rootpos[node][0], leafpos[node][1]) 
#    pos = {node:(leaf_vs_root_factor*x1+(1-leaf_vs_root_factor)*x2, y1) for ((x1,y1), (x2,y2)) in (leafpos[node], rootpos[node]) for node in rootpos}
    xmax = max(x for x,y in pos.values())
    for node in pos:
        pos[node]= (pos[node][0]*width/xmax, pos[node][1])
    return pos

def plot_tree(graph, idx_of_feature_to_plot, label, file_name, root=None, show=False):
    if root is None:
        root = max(graph.edge_index[0]).item()
    plt.figure(1, figsize=(20, 12))
    g = to_networkx(graph, to_undirected=True)
    pos = hierarchy_pos(g, leaf_vs_root_factor=0.5, root=root)
    e = nx.draw_networkx_edges(g, pos, alpha=0.3)
    n = nx.draw_networkx_nodes(g, pos, nodelist=g.nodes(), node_color=graph.x[:,idx_of_feature_to_plot])
    cb = plt.colorbar(n)
    cb.set_label(label, size=20)
    plt.axis('off')
    plt.savefig(f"{file_name}.pdf")
    if show: plt.show()
    plt.close()


# ----------------------------------------------------------------------
# Autoencoder diagnostics, in CMS style, with TH1 histogram output.
# ----------------------------------------------------------------------

def set_cms_style():
    """Apply the CMS matplotlib style from mplhep."""
    import mplhep as hep

    plt.style.use(hep.style.CMS)


def cms_label(ax, label="Preliminary", lumi=None, year=None, com=None):
    """Draw the CMS label on *ax*."""
    import mplhep as hep

    if com is None:
        com = 13 if str(year) in ("2016", "2017", "2018") else (13.6 if str(year) in ("2022", "2023", "2024") else 13)
    hep.cms.label(label, ax=ax, lumi=lumi, year=year, com=com, data=True)


def infer_year_lumi(samples_or_labels, year=None, lumi=None):
    """Infer year and corresponding luminosity from sample names if not provided."""
    if year is None and samples_or_labels:
        text = " ".join(str(s) for s in samples_or_labels)
        if "2018" in text:
            year = 2018
        elif "2017" in text:
            year = 2017
        elif "2016" in text:
            year = 2016
    if lumi is None and year is not None:
        lumi_defaults = {2016: 36.3, 2017: 41.5, 2018: 59.8, "2016": 36.3, "2017": 41.5, "2018": 59.8}
        lumi = lumi_defaults.get(year)
    return year, lumi


def to_th1(counts, edges, name, title=""):
    """Build a ROOT TH1D from a numpy histogram."""
    import ROOT

    edges = np.asarray(edges, dtype=np.float64)
    counts = np.asarray(counts, dtype=np.float64)
    histogram = ROOT.TH1D(name, title, len(counts), edges)
    for index, value in enumerate(counts):
        histogram.SetBinContent(index + 1, value)
    return histogram


def write_th1s(path, histograms):
    """Write TH1 objects, keyed by name, into one ROOT file."""
    import ROOT

    output_file = ROOT.TFile(path, "RECREATE")
    for name, histogram in histograms.items():
        histogram.Write(name)
    output_file.Close()
    return path


def _save(figure, file_name, formats=("pdf",), show=False):
    for extension in formats:
        figure.savefig(f"{file_name}.{extension}", bbox_inches="tight")
    if show:
        plt.show()
    plt.close(figure)


def plot_loss_curves(history, file_name, formats=("pdf",), show=False, lumi=None, year=None):
    """Per-batch and per-epoch training curves from a training history."""
    import mplhep as hep

    set_cms_style()
    figure, (batch_axes, epoch_axes) = plt.subplots(1, 2, figsize=(18, 7))

    batch_axes.plot(history["batch_total"], label="Total")
    batch_axes.plot(history["batch_edges"], label="Edges")
    batch_axes.plot(history["batch_nodes"], label="Nodes")
    batch_axes.set_xlabel("Training batch")
    batch_axes.set_ylabel("Loss")
    batch_axes.set_yscale("log")
    batch_axes.legend(fontsize=9.5)
    cms_label(batch_axes, lumi=lumi, year=year)

    epochs = range(len(history["epoch_train"]))
    train_totals = [values[0] for values in history["epoch_train"]]
    epoch_axes.plot(epochs, train_totals, marker="o", label="Train")
    validation = [
        values[0] if values is not None else np.nan
        for values in history.get("epoch_validation", [])
    ]
    if validation and not np.all(np.isnan(validation)):
        epoch_axes.plot(epochs, validation, marker="s", label="Validation")
    epoch_axes.set_xlabel("Epoch")
    epoch_axes.set_ylabel("Total loss")
    epoch_axes.set_yscale("log")
    epoch_axes.legend(fontsize=9.5)
    cms_label(epoch_axes, lumi=lumi, year=year)

    _save(figure, file_name, formats, show)


def parse_signal_params(label):
    """Extract mMed (m_{Z'}), mDark, rinv, year from a sample string/label."""
    if not label:
        return {}
    params = {}
    mmed_match = re.search(r"mMed-(\d+)", label)
    if mmed_match:
        params["mMed"] = int(mmed_match.group(1))
    mdark_match = re.search(r"mDark-(\d+)", label)
    if mdark_match:
        params["mDark"] = int(mdark_match.group(1))
    rinv_match = re.search(r"rinv-([0-9.]+)", label)
    if rinv_match:
        params["rinv"] = float(rinv_match.group(1))
    year_match = re.search(r"(201[678])", label)
    if year_match:
        params["year"] = year_match.group(1)
    return params


def format_signal_label(label, include_year=False):
    """Format a signal label into clean LaTeX, e.g. '$m_{Z'}=600, r_{inv}=0.7$'."""
    params = parse_signal_params(label)
    parts = []
    if "mMed" in params:
        parts.append(f"m_{{Z'}} = {params['mMed']}")
    if "rinv" in params:
        parts.append(f"r_{{inv}} = {params['rinv']}")
    if parts:
        formatted = "$" + ", ".join(parts) + "$"
        if include_year and "year" in params:
            formatted += f" ({params['year']})"
        return formatted
    return label


def get_signal_style_maps(signal_labels):
    """Build color and linestyle maps for signal labels.

    mMed gets a viridis color (ordered ascending), and rinv gets a distinct linestyle.
    Returns (color_map, linestyle_map, is_grid).
    """
    parsed = {lbl: parse_signal_params(lbl) for lbl in signal_labels if lbl}
    has_grid = any("mMed" in p and "rinv" in p for p in parsed.values())
    if not has_grid:
        return {}, {}, False

    unique_mmed = sorted({p["mMed"] for p in parsed.values() if "mMed" in p})
    unique_rinv = sorted({p["rinv"] for p in parsed.values() if "rinv" in p})

    cmap = plt.cm.viridis
    n_m = len(unique_mmed)
    color_map = {
        m: cmap(i / max(1, n_m - 1) * 0.92) if n_m > 1 else cmap(0.5)
        for i, m in enumerate(unique_mmed)
    }

    style_choices = ["-", "--", ":", "-."]
    rinv_default_styles = {0.3: "-", 0.5: "--", 0.7: ":"}
    linestyle_map = {}
    for i, r in enumerate(unique_rinv):
        linestyle_map[r] = rinv_default_styles.get(r, style_choices[i % len(style_choices)])

    return color_map, linestyle_map, True


def add_combined_legend(axes, color_map, linestyle_map, other_handles=None, loc="upper right", fontsize=8.5):
    """Add a single combined legend box with sample handles, m_{Z'} colors, and r_{inv} linestyles in one column."""
    handles = []
    if other_handles:
        handles.extend(other_handles)
    for m, c in sorted(color_map.items()):
        handles.append(Line2D([0], [0], color=c, linestyle="-", linewidth=2.5, label=f"$m_{{Z'}} = {m}$"))
    for r, ls in sorted(linestyle_map.items()):
        handles.append(Line2D([0], [0], color="black", linestyle=ls, linewidth=2, label=f"$r_{{inv}} = {r}$"))
    return axes.legend(handles=handles, loc=loc, ncol=1, fontsize=fontsize, frameon=True, labelspacing=0.25, handletextpad=0.5, borderpad=0.35)


def plot_score_distributions(samples, file_name, bins=100, density=True, formats=("pdf",), show=False, lumi=None, year=None):
    """Reconstruction-loss histograms for every sample, overlaid."""
    import mplhep as hep

    set_cms_style()
    year, lumi = infer_year_lumi(samples.keys(), year=year, lumi=lumi)
    edges = np.linspace(0.0, 0.2, bins + 1)
    figure, axes = plt.subplots(figsize=(10, 8))
    histograms = {}

    signal_labels = [label for label in samples if parse_signal_params(label).get("mMed") is not None]
    color_map, linestyle_map, is_grid = get_signal_style_maps(signal_labels)

    other_handles = []
    for label, (scores, weights) in samples.items():
        clipped_scores = np.clip(scores, edges[0], edges[-1])
        counts, _ = weighted_histogram(clipped_scores, edges, weights=weights, density=density)
        histograms[f"score_{label}"] = to_th1(counts, edges, f"score_{label}", "Reconstruction loss")

        params = parse_signal_params(label)
        if is_grid and "mMed" in params and "rinv" in params and len(signal_labels) > 3:
            hep.histplot(counts, edges, ax=axes, color=color_map[params["mMed"]], linestyle=linestyle_map[params["rinv"]], histtype="step", linewidth=1.5)
        else:
            clean_label = format_signal_label(label) if params else ("QCD" if label.lower() == "qcd" else ("Data" if label.lower() == "data" else label))
            color = "black" if label.lower() == "data" else ("tab:blue" if label.lower() == "qcd" else None)
            hep.histplot(counts, edges, ax=axes, label=clean_label, color=color, histtype="step", linewidth=2)
            if is_grid and len(signal_labels) > 3:
                other_handles.append(Line2D([0], [0], color=color if color is not None else "black", linestyle="-", linewidth=2, label=clean_label))

    axes.set_xlim(0.0, 0.2)
    axes.set_xlabel("Reconstruction loss")
    axes.set_ylabel("Normalized events" if density else "Events")

    if is_grid and len(signal_labels) > 3:
        add_combined_legend(axes, color_map, linestyle_map, other_handles=other_handles, loc="upper right", fontsize=8.5)
    else:
        axes.legend(fontsize=8.5, loc="upper right", ncol=1)

    cms_label(axes, lumi=lumi, year=year)
    _save(figure, file_name, formats, show)
    return histograms


def plot_loss_efficiency(samples, file_name, formats=("pdf",), show=False, lumi=None, year=None):
    """Normalized number of events surviving a cut on the loss."""
    set_cms_style()
    year, lumi = infer_year_lumi(samples.keys(), year=year, lumi=lumi)
    figure, axes = plt.subplots(figsize=(10, 8))

    signal_labels = [label for label in samples if parse_signal_params(label).get("mMed") is not None]
    color_map, linestyle_map, is_grid = get_signal_style_maps(signal_labels)

    other_handles = []
    for label, (scores, weights) in samples.items():
        thresholds, efficiency = efficiency_curve(scores, weights)
        params = parse_signal_params(label)
        if is_grid and "mMed" in params and "rinv" in params and len(signal_labels) > 3:
            axes.plot(thresholds, efficiency, color=color_map[params["mMed"]], linestyle=linestyle_map[params["rinv"]], linewidth=1.5)
        else:
            clean_label = format_signal_label(label) if params else ("QCD" if label.lower() == "qcd" else ("Data" if label.lower() == "data" else label))
            color = "black" if label.lower() == "data" else ("tab:blue" if label.lower() == "qcd" else None)
            axes.plot(thresholds, efficiency, label=clean_label, color=color, linewidth=2)
            if is_grid and len(signal_labels) > 3:
                other_handles.append(Line2D([0], [0], color=color if color is not None else "black", linestyle="-", linewidth=2, label=clean_label))

    axes.set_xlim(0.0, 1.0)
    axes.set_xlabel("Reconstruction loss threshold")
    axes.set_ylabel("Normalized number of events")
    axes.set_yscale("log")

    if is_grid and len(signal_labels) > 3:
        add_combined_legend(axes, color_map, linestyle_map, other_handles=other_handles, loc="lower left", fontsize=8.5)
    else:
        axes.legend(fontsize=8.5, loc="lower left", ncol=1)

    cms_label(axes, lumi=lumi, year=year)
    _save(figure, file_name, formats, show)


def _signal_curves(scores_signal, weights_signal):
    """``[(label, scores, weights)]`` from one signal or a dict of signals."""
    if isinstance(scores_signal, dict):
        return [(label, scores, weights) for label, (scores, weights) in scores_signal.items()]
    return [(None, scores_signal, weights_signal)]


def plot_roc(scores_signal, scores_background, file_name, weights_signal=None, weights_background=None, formats=("pdf",), show=False, lumi=None, year=None):
    """Signal against background efficiency, with the area under the curve."""
    set_cms_style()
    figure, axes = plt.subplots(figsize=(10, 8))
    curves = {}
    aucs = {}
    signal_items = _signal_curves(scores_signal, weights_signal)
    signal_labels = [label for label, _, _ in signal_items if label is not None]
    color_map, linestyle_map, is_grid = get_signal_style_maps(signal_labels)
    year, lumi = infer_year_lumi(signal_labels, year=year, lumi=lumi)

    for label, scores, weights in signal_items:
        fpr, tpr, _ = roc_curve(scores, scores_background, weights, weights_background)
        auc_val = auc(fpr, tpr)
        curves[label] = (fpr, tpr)
        aucs[label] = auc_val

        plot_kwargs = {"linewidth": 2}
        params = parse_signal_params(label) if label else {}
        if is_grid and "mMed" in params and "rinv" in params and len(signal_labels) > 3:
            plot_kwargs["color"] = color_map[params["mMed"]]
            plot_kwargs["linestyle"] = linestyle_map[params["rinv"]]
        else:
            text = f"AUC = {auc_val:.3f}"
            clean_lbl = format_signal_label(label) if label else None
            plot_kwargs["label"] = text if clean_lbl is None else f"{clean_lbl} ({text})"

        axes.plot(fpr, tpr, **plot_kwargs)

    axes.plot([0, 1], [0, 1], linestyle="--", color="grey", linewidth=1)
    axes.set_xlim(0.0, 1.0)
    axes.set_ylim(0.0, 1.0)
    axes.set_xlabel("False positive rate")
    axes.set_ylabel("Signal efficiency")

    if is_grid and len(signal_labels) > 3:
        add_combined_legend(axes, color_map, linestyle_map, loc="lower right", fontsize=8.5)
    else:
        axes.legend(fontsize=8.5, loc="lower right", ncol=1)

    cms_label(axes, lumi=lumi, year=year)
    _save(figure, file_name, formats, show)
    return curves, aucs


def plot_sic_vs_fpr(scores_signal, scores_background, file_name, weights_signal=None, weights_background=None, min_fpr=1e-4, formats=("pdf",), show=False, lumi=None, year=None):
    """Significance improvement against background efficiency."""
    set_cms_style()
    figure, axes = plt.subplots(figsize=(10, 8))
    signal_items = _signal_curves(scores_signal, weights_signal)
    signal_labels = [label for label, _, _ in signal_items if label is not None]
    color_map, linestyle_map, is_grid = get_signal_style_maps(signal_labels)
    year, lumi = infer_year_lumi(signal_labels, year=year, lumi=lumi)

    for label, scores, weights in signal_items:
        fpr, tpr, _ = roc_curve(scores, scores_background, weights, weights_background)
        fpr_values, sic_values = sic(tpr, fpr, min_fpr=min_fpr)
        plot_kwargs = {"linewidth": 2}
        params = parse_signal_params(label) if label else {}
        if is_grid and "mMed" in params and "rinv" in params and len(signal_labels) > 3:
            plot_kwargs["color"] = color_map[params["mMed"]]
            plot_kwargs["linestyle"] = linestyle_map[params["rinv"]]
        else:
            text = f"max SIC = {max_sic(tpr, fpr, min_fpr=min_fpr):.2f}"
            clean_lbl = format_signal_label(label) if label else None
            plot_kwargs["label"] = text if clean_lbl is None else f"{clean_lbl} ({text})"

        axes.plot(fpr_values, sic_values, **plot_kwargs)

    axes.axhline(1.0, linestyle="--", color="grey", linewidth=1)
    axes.set_xscale("log")
    axes.set_xlabel("Background efficiency")
    axes.set_ylabel(r"$\varepsilon_S/\sqrt{\varepsilon_B}$")

    if is_grid and len(signal_labels) > 3:
        add_combined_legend(axes, color_map, linestyle_map, loc="upper left", fontsize=8.5)
    else:
        axes.legend(fontsize=8.5, loc="upper left", ncol=1)

    cms_label(axes, lumi=lumi, year=year)
    _save(figure, file_name, formats, show)


def plot_auc_2d(aucs, file_name, formats=("pdf",), show=False, lumi=None, year=None):
    """2D AUC heatmap with rinv on y-axis and m_{Z'} on x-axis."""
    grid = {}
    for label, auc_val in aucs.items():
        params = parse_signal_params(label)
        if "mMed" in params and "rinv" in params:
            grid[(params["mMed"], params["rinv"])] = auc_val

    if not grid:
        print("No (mMed, rinv) signal parameters found; skipping 2D AUC plot")
        return None

    year, lumi = infer_year_lumi(aucs.keys(), year=year, lumi=lumi)
    unique_mmed = sorted(set(m for m, _ in grid.keys()))
    unique_rinv = sorted(set(r for _, r in grid.keys()))

    matrix = np.full((len(unique_rinv), len(unique_mmed)), np.nan)
    for (m, r), auc_val in grid.items():
        row = unique_rinv.index(r)
        col = unique_mmed.index(m)
        matrix[row, col] = auc_val

    set_cms_style()
    figure, axes = plt.subplots(figsize=(11, 8))

    val_min = float(np.nanmin(matrix))
    val_max = float(np.nanmax(matrix))
    im = axes.imshow(matrix, origin="lower", aspect="auto", cmap="viridis", vmin=max(0.0, val_min - 0.03), vmax=min(1.0, val_max + 0.03))

    axes.set_xticks(range(len(unique_mmed)))
    axes.set_xticklabels([str(m) for m in unique_mmed], rotation=45 if len(unique_mmed) > 8 else 0)
    axes.set_yticks(range(len(unique_rinv)))
    axes.set_yticklabels([f"{r:.1f}" for r in unique_rinv])
    axes.set_xlabel(r"$m_{Z'}$ [GeV]")
    axes.set_ylabel(r"$r_{\mathrm{inv}}$")

    v_min, v_max = im.get_clim()
    mid = (v_min + v_max) / 2.0
    for i in range(len(unique_rinv)):
        for j in range(len(unique_mmed)):
            val = matrix[i, j]
            if not np.isnan(val):
                text_color = "white" if val < mid else "black"
                axes.text(j, i, f"{val:.3f}", ha="center", va="center", color=text_color, fontweight="bold", fontsize=12)

    cbar = figure.colorbar(im, ax=axes, pad=0.03)
    cbar.set_label("AUC", fontsize=14)

    cms_label(axes, lumi=lumi, year=year)
    _save(figure, file_name, formats, show)

    th2 = None
    try:
        import ROOT

        def get_bin_edges(vals):
            vals = np.array(vals, dtype=float)
            if len(vals) == 1:
                return np.array([vals[0] - 0.5, vals[0] + 0.5])
            diffs = np.diff(vals)
            inner = vals[:-1] + diffs / 2.0
            return np.concatenate([[vals[0] - diffs[0] / 2.0], inner, [vals[-1] + diffs[-1] / 2.0]])

        x_edges = get_bin_edges(unique_mmed)
        y_edges = get_bin_edges(unique_rinv)
        th2 = ROOT.TH2D("auc_2d", "2D AUC;m_{Z'} [GeV];r_{inv}", len(unique_mmed), x_edges, len(unique_rinv), y_edges)
        for i, r in enumerate(unique_rinv):
            for j, m in enumerate(unique_mmed):
                val = matrix[i, j]
                if not np.isnan(val):
                    th2.SetBinContent(j + 1, i + 1, float(val))
    except Exception as exc:
        print(f"Could not build ROOT TH2D: {exc}")

    return th2


def plot_feature_residuals(
    model,
    loader,
    file_name,
    device="cpu",
    max_batches=20,
    bins=50,
    formats=("pdf",),
    show=False,
):
    """Input minus reconstructed node feature, one panel per feature."""
    import torch

    if not callable(getattr(model, "forward", None)):
        print("Model has no forward method; skipping residual plot")
        return

    set_cms_style()
    model.eval()
    residuals = []
    with torch.no_grad():
        for index, batch in enumerate(loader):
            if index >= max_batches:
                break
            batch = batch.to(device)
            reconstructed = model(batch.x, batch.edge_index)
            if isinstance(reconstructed, tuple):
                reconstructed = reconstructed[0]
            residuals.append(
                (batch.x - reconstructed).detach().cpu().numpy()
            )

    if not residuals:
        print("No batches available; skipping residual plot")
        return

    residuals = np.concatenate(residuals)
    n_features = residuals.shape[1]
    n_columns = min(3, n_features)
    n_rows = int(np.ceil(n_features / n_columns))
    figure, axes = plt.subplots(
        n_rows, n_columns, figsize=(6 * n_columns, 5 * n_rows), squeeze=False
    )

    import mplhep as hep

    for feature in range(n_features):
        axis = axes[feature // n_columns][feature % n_columns]
        counts, edges = np.histogram(residuals[:, feature], bins=bins)
        hep.histplot(counts, edges, ax=axis, histtype="step", linewidth=2)
        axis.set_xlabel(f"Feature {feature} residual")
        axis.set_ylabel("Nodes")
    for empty in range(n_features, n_rows * n_columns):
        axes[empty // n_columns][empty % n_columns].axis("off")

    _save(figure, file_name, formats, show)


def plot_latent_distributions(
    model,
    loader,
    file_name,
    device="cpu",
    max_batches=20,
    bins=50,
    formats=("pdf",),
    show=False,
):
    """Distribution of every latent dimension of the encoder."""
    import torch

    encoder = getattr(model, "encoder", None)
    if encoder is None:
        print("Model exposes no encoder; skipping latent plot")
        return

    set_cms_style()
    model.eval()
    latents = []
    with torch.no_grad():
        for index, batch in enumerate(loader):
            if index >= max_batches:
                break
            batch = batch.to(device)
            latents.append(
                encoder(batch.x, batch.edge_index).detach().cpu().numpy()
            )

    if not latents:
        print("No batches available; skipping latent plot")
        return

    latents = np.concatenate(latents)
    n_dimensions = latents.shape[1]
    n_columns = min(4, n_dimensions)
    n_rows = int(np.ceil(n_dimensions / n_columns))
    figure, axes = plt.subplots(
        n_rows, n_columns, figsize=(5 * n_columns, 4 * n_rows), squeeze=False
    )

    import mplhep as hep

    for dimension in range(n_dimensions):
        axis = axes[dimension // n_columns][dimension % n_columns]
        counts, edges = np.histogram(latents[:, dimension], bins=bins)
        hep.histplot(counts, edges, ax=axis, histtype="step", linewidth=2)
        axis.set_xlabel(f"Latent {dimension}")
        axis.set_ylabel("Nodes")
    for empty in range(n_dimensions, n_rows * n_columns):
        axes[empty // n_columns][empty % n_columns].axis("off")

    _save(figure, file_name, formats, show)