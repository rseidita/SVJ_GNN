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
    """Per-epoch Total, Edge, and Node loss curves comparing Train, Validation, and Test."""
    import mplhep as hep

    set_cms_style()
    figure, (ax_tot, ax_edge, ax_node) = plt.subplots(1, 3, figsize=(22, 6.5))

    panels = [
        (ax_tot, 0, "Total Loss", r"$\mathcal{L}_{\mathrm{total}}$"),
        (ax_edge, 1, "Edge Reconstruction Loss", r"$\mathcal{L}_{\mathrm{edges}}$"),
        (ax_node, 2, "Node Feature Loss", r"$\mathcal{L}_{\mathrm{nodes}}$"),
    ]

    n_epochs = len(history.get("epoch_train", []))
    epochs = list(range(n_epochs))

    for ax, term_idx, title, ylabel in panels:
        train_vals = [v[term_idx] if v is not None else np.nan for v in history.get("epoch_train", [])]
        if train_vals and not np.all(np.isnan(train_vals)):
            ax.plot(epochs, train_vals, marker="o", markersize=3.5, label="Train", color="royalblue", linewidth=1.8)

        val_vals = [v[term_idx] if v is not None else np.nan for v in history.get("epoch_validation", [])]
        if val_vals and not np.all(np.isnan(val_vals)):
            ax.plot(epochs, val_vals, marker="s", markersize=3.5, label="Validation", color="darkorange", linewidth=1.8)

        if "epoch_test" in history and history["epoch_test"]:
            test_vals = [v[term_idx] if v is not None else np.nan for v in history["epoch_test"]]
            if not np.all(np.isnan(test_vals)):
                ax.plot(epochs, test_vals, marker="^", markersize=3.5, label="Test", color="forestgreen", linewidth=1.8)
        elif "test" in history and history["test"] is not None and len(history["test"]) > term_idx:
            test_scalar = history["test"][term_idx]
            if test_scalar is not None and not np.isnan(test_scalar):
                ax.axhline(test_scalar, color="forestgreen", linestyle="--", linewidth=1.8, label=f"Test (final: {test_scalar:.4f})")
                if len(epochs) > 0:
                    ax.plot([epochs[-1]], [test_scalar], marker="^", markersize=7, color="forestgreen")

        ax.text(0.06, 0.90, title, transform=ax.transAxes, fontsize=12, fontweight="bold")
        ax.set_xlabel("Epoch", fontsize=16)
        ax.set_ylabel(ylabel, fontsize=16)
        ax.tick_params(axis="both", labelsize=10)
        all_vals = [v for v in train_vals + val_vals if not np.isnan(v)]
        if all_vals and min(all_vals) > 0:
            ax.set_yscale("log")
        ax.legend(fontsize=9.5, loc="upper right")
        cms_label(ax, lumi=lumi, year=year)

    figure.tight_layout()
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
    axes.set_xlabel("Reconstruction loss", fontsize=16)
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
    axes.set_xlabel("Reconstruction loss threshold", fontsize=16)
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
    axes.set_xlabel("False positive rate", fontsize=16)
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
    axes.set_xlabel("Background efficiency", fontsize=16)
    axes.set_ylabel(r"$\varepsilon_S/\sqrt{\varepsilon_B}$")

    if is_grid and len(signal_labels) > 3:
        add_combined_legend(axes, color_map, linestyle_map, loc="upper left", fontsize=12)
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
    axes.set_xlabel(r"$m_{Z'}$ [GeV]", fontsize=16)
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


def get_lund_feature_names(n_features, config=None):
    """Return physical Lund variable names matching feature matrix dimension."""
    if n_features == 12:
        return [r"$\ln(k_T)$", r"$\ln(1/\Delta)$", r"$\ln(z)$", r"$\psi$", r"$p_T$", r"$\eta$", r"$\phi$", r"$m$", r"$f_e$", r"$f_\mu$", r"$f_\gamma$", r"$f_{\mathrm{had}}$"]
    if n_features == 9:
        if config and getattr(config, "graph", {}).get("save_4vectors_in_lund_tree", False) and not getattr(config, "graph", {}).get("save_energy_fractions", False):
            return [r"$\ln(k_T)$", r"$\ln(1/\Delta)$", r"$\ln(z)$", r"$\psi$", r"$p_T$", r"$\eta$", r"$\phi$", r"$m$", r"$\dots$"]
        return [r"$\ln(k_T)$", r"$\ln(1/\Delta)$", r"$\ln(z)$", r"$\psi$", r"$m$", r"$f_e$", r"$f_\mu$", r"$f_\gamma$", r"$f_{\mathrm{had}}$"]
    if n_features == 5:
        return [r"$\ln(k_T)$", r"$\ln(1/\Delta)$", r"$\ln(z)$", r"$\psi$", r"$m$"]
    if n_features == 3:
        return [r"$\ln(k_T)$", r"$\ln(1/\Delta)$", r"$\ln(z)$"]
    return [f"Feature {i}" for i in range(n_features)]


def plot_feature_residuals(model, loader, file_name, device="cpu", max_batches=20, bins=50, formats=("pdf",), show=False, lumi=None, year=None, config=None, signal_loader=None, signal_label="Signal"):
    """Input minus reconstructed node feature, one panel per feature."""
    import torch
    import mplhep as hep

    if not callable(getattr(model, "forward", None)):
        print("Model has no forward method; skipping residual plot")
        return

    set_cms_style()
    model.eval()

    def _extract_residuals(b_loader):
        res = []
        with torch.no_grad():
            for index, batch in enumerate(b_loader):
                if index >= max_batches:
                    break
                batch = batch.to(device)
                if hasattr(model, "encoder") and hasattr(model, "features_decoder"):
                    z = model.encoder(batch.x, batch.edge_index)
                    reconstructed = model.features_decoder(z, batch.edge_index)
                else:
                    reconstructed = model(batch.x, batch.edge_index)
                    if isinstance(reconstructed, tuple):
                        reconstructed = reconstructed[0]
                res.append((batch.x - reconstructed).detach().cpu().numpy())
        return np.concatenate(res) if res else None

    residuals = _extract_residuals(loader)
    if residuals is None:
        print("No batches available; skipping residual plot")
        return

    sig_residuals = _extract_residuals(signal_loader) if signal_loader is not None else None
    n_features = residuals.shape[1]
    feature_names = get_lund_feature_names(n_features, config)
    n_columns = min(3, n_features)
    n_rows = int(np.ceil(n_features / n_columns))
    figure, axes = plt.subplots(n_rows, n_columns, figsize=(6 * n_columns, 5 * n_rows), squeeze=False)

    for feature in range(n_features):
        axis = axes[feature // n_columns][feature % n_columns]
        bkg_vals = residuals[:, feature]
        if sig_residuals is not None:
            sig_vals = sig_residuals[:, feature]
            lo = float(min(np.nanpercentile(bkg_vals, 0.5), np.nanpercentile(sig_vals, 0.5)))
            hi = float(max(np.nanpercentile(bkg_vals, 99.5), np.nanpercentile(sig_vals, 99.5)))
            if lo == hi:
                lo, hi = lo - 1.0, hi + 1.0
            bin_edges = np.linspace(lo, hi, bins + 1)
            bkg_c, _ = np.histogram(bkg_vals, bins=bin_edges, density=True)
            sig_c, _ = np.histogram(sig_vals, bins=bin_edges, density=True)
            hep.histplot(bkg_c, bin_edges, ax=axis, histtype="step", linewidth=2, color="royalblue", label="Background (QCD)")
            hep.histplot(sig_c, bin_edges, ax=axis, histtype="step", linewidth=2, color="crimson", linestyle="--", label=signal_label)
            if feature == 0:
                axis.legend(fontsize=9.5, loc="upper right")
            axis.set_ylabel("Probability Density", fontsize=16)
        else:
            counts, edges = np.histogram(bkg_vals, bins=bins)
            hep.histplot(counts, edges, ax=axis, histtype="step", linewidth=2, color="royalblue")
            axis.set_ylabel("Nodes", fontsize=16)

        axis.set_xlabel(f"Residual: {feature_names[feature]}", fontsize=15)
        axis.tick_params(axis="both", labelsize=10)
    for empty in range(n_features, n_rows * n_columns):
        axes[empty // n_columns][empty % n_columns].axis("off")

    cms_label(axes[0, 0], lumi=lumi, year=year)
    figure.tight_layout()
    _save(figure, file_name, formats, show)


def plot_latent_distributions(model, loader, file_name, device="cpu", max_batches=20, bins=50, formats=("pdf",), show=False, lumi=None, year=None):
    """Distribution of every latent dimension of the encoder."""
    import torch
    import mplhep as hep

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
            latents.append(encoder(batch.x, batch.edge_index).detach().cpu().numpy())

    if not latents:
        print("No batches available; skipping latent plot")
        return

    latents = np.concatenate(latents)
    n_dimensions = latents.shape[1]
    n_columns = min(4, n_dimensions)
    n_rows = int(np.ceil(n_dimensions / n_columns))
    figure, axes = plt.subplots(n_rows, n_columns, figsize=(5 * n_columns, 4 * n_rows), squeeze=False)

    for dimension in range(n_dimensions):
        axis = axes[dimension // n_columns][dimension % n_columns]
        counts, edges = np.histogram(latents[:, dimension], bins=bins)
        hep.histplot(counts, edges, ax=axis, histtype="step", linewidth=2, color="teal")
        axis.set_xlabel(f"Latent {dimension}", fontsize=15)
        axis.set_ylabel("Nodes", fontsize=16)
        axis.tick_params(axis="both", labelsize=10)
    for empty in range(n_dimensions, n_rows * n_columns):
        axes[empty // n_columns][empty % n_columns].axis("off")

    cms_label(axes[0, 0], lumi=lumi, year=year)
    figure.tight_layout()
    _save(figure, file_name, formats, show)


def plot_feature_correlations(model, loader, file_name, device="cpu", max_batches=20, formats=("pdf",), show=False, lumi=None, year=None, config=None):
    """2D profiles of true input vs reconstructed features with y=x identity lines."""
    import torch

    set_cms_style()
    model.eval()
    true_x, reco_x = [], []
    with torch.no_grad():
        for index, batch in enumerate(loader):
            if index >= max_batches:
                break
            batch = batch.to(device)
            if hasattr(model, "encoder") and hasattr(model, "features_decoder"):
                z = model.encoder(batch.x, batch.edge_index)
                reconstructed = model.features_decoder(z, batch.edge_index)
            else:
                out = model(batch.x, batch.edge_index)
                reconstructed = out[0] if isinstance(out, tuple) else out
            true_x.append(batch.x.detach().cpu().numpy())
            reco_x.append(reconstructed.detach().cpu().numpy())

    if not true_x:
        return

    true_x = np.concatenate(true_x)
    reco_x = np.concatenate(reco_x)
    n_features = true_x.shape[1]
    feature_names = get_lund_feature_names(n_features, config)

    n_columns = min(4, n_features)
    n_rows = int(np.ceil(n_features / n_columns))
    figure, axes = plt.subplots(n_rows, n_columns, figsize=(4.8 * n_columns, 4.4 * n_rows), squeeze=False)

    for f_idx in range(n_features):
        ax = axes[f_idx // n_columns][f_idx % n_columns]
        x_val = true_x[:, f_idx]
        y_val = reco_x[:, f_idx]

        mask = np.isfinite(x_val) & np.isfinite(y_val)
        corr = np.corrcoef(x_val[mask], y_val[mask])[0, 1] if np.sum(mask) > 1 else np.nan

        ax.hexbin(x_val[mask], y_val[mask], gridsize=35, cmap="Blues", mincnt=1, bins="log")

        lo = float(min(np.nanpercentile(x_val[mask], 1), np.nanpercentile(y_val[mask], 1)))
        hi = float(max(np.nanpercentile(x_val[mask], 99), np.nanpercentile(y_val[mask], 99)))
        if lo == hi:
            lo, hi = lo - 1.0, hi + 1.0
        ax.plot([lo, hi], [lo, hi], color="crimson", linestyle="--", linewidth=1.5, label="Identity ($y=x$)")

        name = feature_names[f_idx]
        corr_str = f" ($r = {corr:.2f}$)" if not np.isnan(corr) else ""
        ax.text(0.06, 0.90, f"{name}{corr_str}", transform=ax.transAxes, fontsize=10.5, fontweight="bold")
        ax.set_xlabel(f"True {name}", fontsize=14)
        ax.set_ylabel(f"Reconstructed {name}", fontsize=14)
        ax.tick_params(axis="both", labelsize=10)
        ax.set_xlim(lo, hi)
        ax.set_ylim(lo, hi)

    for empty in range(n_features, n_rows * n_columns):
        axes[empty // n_columns][empty % n_columns].axis("off")

    cms_label(axes[0, 0], lumi=lumi, year=year)
    figure.tight_layout()
    _save(figure, file_name, formats, show)


def plot_feature_importance_ranking(model, loader, file_name, device="cpu", max_batches=20, formats=("pdf",), show=False, lumi=None, year=None, config=None, signal_loader=None, signal_label="Signal"):
    """Rank features by reconstruction MSE in a horizontal bar chart."""
    import torch

    set_cms_style()
    model.eval()

    def _calc_errors(b_loader):
        errs = []
        with torch.no_grad():
            for index, batch in enumerate(b_loader):
                if index >= max_batches:
                    break
                batch = batch.to(device)
                if hasattr(model, "encoder") and hasattr(model, "features_decoder"):
                    z = model.encoder(batch.x, batch.edge_index)
                    reconstructed = model.features_decoder(z, batch.edge_index)
                else:
                    out = model(batch.x, batch.edge_index)
                    reconstructed = out[0] if isinstance(out, tuple) else out
                errs.append((batch.x - reconstructed).square().detach().cpu().numpy())
        return np.concatenate(errs) if errs else None

    errors = _calc_errors(loader)
    if errors is None:
        return

    mse_per_feature = np.mean(errors, axis=0)
    n_features = len(mse_per_feature)
    names = get_lund_feature_names(n_features, config)

    sorted_indices = np.argsort(mse_per_feature)
    sorted_mse = mse_per_feature[sorted_indices]
    sorted_names = [names[i] for i in sorted_indices]
    total_mse = float(np.sum(mse_per_feature))

    sig_errors = _calc_errors(signal_loader) if signal_loader is not None else None
    sorted_sig_mse = np.mean(sig_errors, axis=0)[sorted_indices] if sig_errors is not None else None

    figure, ax = plt.subplots(figsize=(10, max(5.0, 0.45 * n_features)))
    if sorted_sig_mse is not None:
        bar_h = 0.36
        y = np.arange(n_features)
        ax.barh(y - bar_h / 2, sorted_mse, height=bar_h, color="royalblue", edgecolor="black", alpha=0.85, label="Background (QCD)")
        ax.barh(y + bar_h / 2, sorted_sig_mse, height=bar_h, color="crimson", edgecolor="black", alpha=0.85, label=signal_label)
        ax.set_yticks(y)
        ax.set_yticklabels(sorted_names, fontsize=8.5)
        ax.legend(fontsize=10.5, loc="lower right")
        max_val = float(max(np.max(sorted_mse), np.max(sorted_sig_mse))) if len(sorted_mse) > 0 else 1.0
        ax.set_xlim(0, max_val * 1.25)
    else:
        bars = ax.barh(range(n_features), sorted_mse, color="royalblue", edgecolor="black", alpha=0.85, height=0.65)
        ax.set_yticks(range(n_features))
        ax.set_yticklabels(sorted_names, fontsize=8.5)
        max_val = float(np.max(sorted_mse)) if len(sorted_mse) > 0 else 1.0
        for bar, val in zip(bars, sorted_mse):
            pct = (val / total_mse * 100) if total_mse > 0 else 0
            ax.text(val + max_val * 0.015, bar.get_y() + bar.get_height() / 2, f"{val:.4f} ({pct:.1f}%)", va="center", ha="left", fontsize=9.5)
        ax.set_xlim(0, max_val * 1.3)

    ax.set_ylabel("Features", fontsize=16)
    ax.set_xlabel("Reconstruction MSE", fontsize=16)
    ax.tick_params(axis="both", labelsize=10)

    cms_label(ax, lumi=lumi, year=year)
    figure.tight_layout()
    _save(figure, file_name, formats, show)


def plot_edge_calibration(model, loader, file_name, device="cpu", max_batches=20, bins=50, formats=("pdf",), show=False, lumi=None, year=None, signal_loader=None, signal_label="Signal"):
    """Edge prediction probability distribution for true graph edges vs negative sampled non-edges."""
    import torch
    import mplhep as hep
    from torch_geometric.utils import batched_negative_sampling

    encoder = getattr(model, "encoder", None)
    edges_decoder = getattr(model, "edges_decoder", None)
    if encoder is None or edges_decoder is None:
        print("Model lacks encoder or edges_decoder; skipping edge calibration plot")
        return

    set_cms_style()
    model.eval()

    def _get_probs(b_loader):
        p_probs, n_probs = [], []
        with torch.no_grad():
            for index, batch in enumerate(b_loader):
                if index >= max_batches:
                    break
                batch = batch.to(device)
                z = encoder(batch.x, batch.edge_index)
                if batch.edge_index.shape[1] > 0:
                    p = edges_decoder(z, batch.edge_index, sigmoid=True)
                    p_probs.append(p.detach().cpu().numpy())
                neg_edge_index = batched_negative_sampling(batch.edge_index, batch.batch)
                if neg_edge_index.shape[1] > 0:
                    np_prob = edges_decoder(z, neg_edge_index, sigmoid=True)
                    n_probs.append(np_prob.detach().cpu().numpy())
        return (np.concatenate(p_probs) if p_probs else None), (np.concatenate(n_probs) if n_probs else None)

    pos_probs, neg_probs = _get_probs(loader)
    if pos_probs is None or neg_probs is None:
        return

    sig_pos_probs = None
    if signal_loader is not None:
        sig_pos_probs, _ = _get_probs(signal_loader)

    figure, ax = plt.subplots(figsize=(10, 7.5))
    edges = np.linspace(0.0, 1.0, bins + 1)
    pos_counts, _ = np.histogram(pos_probs, bins=edges, density=True)
    neg_counts, _ = np.histogram(neg_probs, bins=edges, density=True)

    hep.histplot(pos_counts, edges, ax=ax, label="True Edges: QCD ($A_{ij} = 1$)", color="royalblue", histtype="step", linewidth=2.2)
    if sig_pos_probs is not None:
        sig_counts, _ = np.histogram(sig_pos_probs, bins=edges, density=True)
        hep.histplot(sig_counts, edges, ax=ax, label=f"True Edges: {signal_label}", color="darkorange", linestyle="--", histtype="step", linewidth=2.2)
    hep.histplot(neg_counts, edges, ax=ax, label="Non-Edges ($A_{ij} = 0$, sampled)", color="crimson", histtype="step", linewidth=2.2)

    ax.set_xlabel(r"Predicted Edge Probability $\hat{A}_{ij}$", fontsize=16)
    ax.set_ylabel("Probability Density", fontsize=16)
    ax.tick_params(axis="both", labelsize=10)
    ax.set_xlim(0.0, 1.0)
    ax.set_yscale("log")
    ax.legend(fontsize=10.5, loc="upper center")
    cms_label(ax, lumi=lumi, year=year)

    figure.tight_layout()
    _save(figure, file_name, formats, show)


def plot_loss_vs_jet_kinematics(model, loader, file_name, device="cpu", max_batches=20, formats=("pdf",), show=False, lumi=None, year=None):
    """Check for mass sculpting: total reconstruction loss vs jet mass and jet pT."""
    import torch

    set_cms_style()
    model.eval()
    losses, jet_masses, jet_pts = [], [], []

    with torch.no_grad():
        for index, batch in enumerate(loader):
            if index >= max_batches:
                break
            batch = batch.to(device)
            tot, _, _ = model.score_batch(batch.x, batch.edge_index, batch.batch)
            losses.append(tot.detach().cpu().numpy())

            batch_np = batch.batch.detach().cpu().numpy()
            x_np = batch.x.detach().cpu().numpy()
            n_graphs = int(batch_np.max()) + 1 if len(batch_np) > 0 else 0

            for g_i in range(n_graphs):
                node_indices = np.where(batch_np == g_i)[0]
                if len(node_indices) == 0:
                    continue
                root_node = node_indices[-1]
                if x_np.shape[1] == 12:
                    jet_pts.append(x_np[root_node, 4])
                    jet_masses.append(x_np[root_node, 7])
                elif x_np.shape[1] == 9:
                    jet_pts.append(0.0)
                    jet_masses.append(x_np[root_node, 4])
                else:
                    jet_pts.append(0.0)
                    jet_masses.append(0.0)

    if not losses:
        return

    losses = np.concatenate(losses)
    jet_masses = np.array(jet_masses)
    jet_pts = np.array(jet_pts)

    if len(jet_masses) != len(losses):
        return

    has_pt = np.any(jet_pts > 0)
    figure, (ax_m, ax_pt) = plt.subplots(1, 2, figsize=(18, 6.5))

    m_mask = np.isfinite(jet_masses) & np.isfinite(losses)
    ax_m.hexbin(jet_masses[m_mask], losses[m_mask], gridsize=30, cmap="Blues", mincnt=1, bins="log")
    ax_m.set_xlabel(r"Jet Mass $m_{\mathrm{jet}}$ (normalized / raw)", fontsize=15)
    ax_m.set_ylabel("Reconstruction Loss", fontsize=15)
    ax_m.tick_params(axis="both", labelsize=10)
    ax_m.text(0.06, 0.90, "Reconstruction Loss vs Jet Mass", transform=ax_m.transAxes, fontsize=12, fontweight="bold")
    cms_label(ax_m, lumi=lumi, year=year)

    if has_pt:
        pt_mask = np.isfinite(jet_pts) & np.isfinite(losses)
        ax_pt.hexbin(jet_pts[pt_mask], losses[pt_mask], gridsize=30, cmap="Blues", mincnt=1, bins="log")
        ax_pt.set_xlabel(r"Jet $p_T$ (normalized / raw)", fontsize=15)
        ax_pt.set_ylabel("Reconstruction Loss", fontsize=15)
        ax_pt.tick_params(axis="both", labelsize=10)
        ax_pt.text(0.06, 0.90, r"Reconstruction Loss vs Jet $p_T$", transform=ax_pt.transAxes, fontsize=12, fontweight="bold")
    else:
        ax_pt.text(0.5, 0.5, "pT not stored in graph nodes", ha="center", va="center", transform=ax_pt.transAxes)
    cms_label(ax_pt, lumi=lumi, year=year)

    figure.tight_layout()
    _save(figure, file_name, formats, show)


def plot_latent_pca_spectrum(model, loader, file_name, device="cpu", max_batches=20, formats=("pdf",), show=False, lumi=None, year=None):
    """Assess effective dimensionality of latent space via PCA explained variance spectrum."""
    import torch
    from sklearn.decomposition import PCA

    encoder = getattr(model, "encoder", None)
    if encoder is None:
        return

    set_cms_style()
    model.eval()
    latents = []
    with torch.no_grad():
        for index, batch in enumerate(loader):
            if index >= max_batches:
                break
            batch = batch.to(device)
            z = encoder(batch.x, batch.edge_index)
            latents.append(z.detach().cpu().numpy())

    if not latents:
        return

    latents = np.concatenate(latents)
    n_components = min(latents.shape[1], 20)
    pca = PCA(n_components=n_components)
    pca.fit(latents)

    var_ratio = pca.explained_variance_ratio_
    cum_var = np.cumsum(var_ratio)
    comp_range = np.arange(1, n_components + 1)

    figure, ax1 = plt.subplots(figsize=(10, 7))
    ax1.bar(comp_range, var_ratio, color="royalblue", edgecolor="black", alpha=0.8, width=0.6, label="Individual component")
    ax1.set_xlabel("Principal Component", fontsize=16)
    ax1.set_ylabel("Explained Variance Ratio", fontsize=16, color="royalblue")
    ax1.set_xticks(comp_range)
    ax1.tick_params(axis="both", labelsize=10)

    ax2 = ax1.twinx()
    ax2.plot(comp_range, cum_var, color="crimson", marker="o", linewidth=2.2, label="Cumulative variance")
    ax2.set_ylabel("Cumulative Variance Ratio", fontsize=16, color="crimson")
    ax2.set_ylim(0.0, 1.05)
    ax2.tick_params(axis="y", labelsize=10)
    ax2.grid(False)

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="center right", fontsize=10)

    cms_label(ax1, lumi=lumi, year=year)
    figure.tight_layout()
    _save(figure, file_name, formats, show)


def plot_latent_projection(model, loader, file_name, device="cpu", max_batches=20, formats=("pdf",), show=False, lumi=None, year=None, signal_loader=None, signal_label="Signal"):
    """2D PCA projection of jet-level latent representations."""
    import torch
    from sklearn.decomposition import PCA

    encoder = getattr(model, "encoder", None)
    if encoder is None:
        return

    set_cms_style()
    model.eval()

    def _extract_jet_latents(b_loader):
        j_latents, j_losses = [], []
        with torch.no_grad():
            for index, batch in enumerate(b_loader):
                if index >= max_batches:
                    break
                batch = batch.to(device)
                z = encoder(batch.x, batch.edge_index)
                batch_np = batch.batch.detach().cpu().numpy()
                z_np = z.detach().cpu().numpy()
                tot, _, _ = model.score_batch(batch.x, batch.edge_index, batch.batch)
                j_losses.append(tot.detach().cpu().numpy())

                n_graphs = int(batch_np.max()) + 1 if len(batch_np) > 0 else 0
                for g_i in range(n_graphs):
                    mask = batch_np == g_i
                    if np.any(mask):
                        j_latents.append(np.mean(z_np[mask], axis=0))
        return (np.array(j_latents) if j_latents else None), (np.concatenate(j_losses) if j_losses else None)

    jet_latents, jet_losses = _extract_jet_latents(loader)
    if jet_latents is None:
        return

    sig_jet_latents = None
    if signal_loader is not None:
        sig_jet_latents, _ = _extract_jet_latents(signal_loader)

    pca = PCA(n_components=2)
    coords = pca.fit_transform(jet_latents)
    var1 = pca.explained_variance_ratio_[0] * 100
    var2 = pca.explained_variance_ratio_[1] * 100

    figure, ax = plt.subplots(figsize=(10, 8))
    if sig_jet_latents is not None:
        sig_coords = pca.transform(sig_jet_latents)
        ax.scatter(coords[:, 0], coords[:, 1], color="royalblue", alpha=0.5, s=25, label="Background (QCD)", edgecolors="none")
        ax.scatter(sig_coords[:, 0], sig_coords[:, 1], color="crimson", marker="^", alpha=0.85, s=35, label=signal_label, edgecolors="black", linewidth=0.5)
        ax.legend(fontsize=11, loc="upper right")
    else:
        sc = ax.scatter(coords[:, 0], coords[:, 1], c=jet_losses, cmap="viridis", alpha=0.75, s=20, edgecolors="none")
        cbar = figure.colorbar(sc, ax=ax, pad=0.03)
        cbar.set_label("Reconstruction Loss", fontsize=16)
        cbar.ax.tick_params(labelsize=10)

    ax.set_xlabel(f"PC 1 ({var1:.1f}% var)", fontsize=16)
    ax.set_ylabel(f"PC 2 ({var2:.1f}% var)", fontsize=16)
    ax.tick_params(axis="both", labelsize=10)
    cms_label(ax, lumi=lumi, year=year)

    figure.tight_layout()
    _save(figure, file_name, formats, show)


def plot_xai_feature_saliency(model, loader, file_name, device="cpu", max_batches=20, formats=("pdf",), show=False, lumi=None, year=None, config=None, signal_loader=None, signal_label="Signal"):
    """Global feature attribution ranking via gradient saliency / Integrated Gradients."""
    from utils.XaiUtils import compute_node_feature_saliency

    set_cms_style()
    saliency = compute_node_feature_saliency(model, loader, device=device, max_batches=max_batches)
    if len(saliency) == 0:
        return

    n_features = len(saliency)
    names = get_lund_feature_names(n_features, config)

    sorted_indices = np.argsort(saliency)
    sorted_sal = saliency[sorted_indices]
    sorted_names = [names[i] for i in sorted_indices]
    total_sal = float(np.sum(saliency))

    sig_saliency = None
    if signal_loader is not None:
        sig_saliency = compute_node_feature_saliency(model, signal_loader, device=device, max_batches=max_batches)
    sorted_sig_sal = sig_saliency[sorted_indices] if sig_saliency is not None and len(sig_saliency) == n_features else None

    figure, ax = plt.subplots(figsize=(10, max(5.0, 0.45 * n_features)))
    if sorted_sig_sal is not None:
        bar_h = 0.36
        y = np.arange(n_features)
        ax.barh(y - bar_h / 2, sorted_sal, height=bar_h, color="royalblue", edgecolor="black", alpha=0.85, label="Background (QCD)")
        ax.barh(y + bar_h / 2, sorted_sig_sal, height=bar_h, color="darkorange", edgecolor="black", alpha=0.85, label=signal_label)
        ax.set_yticks(y)
        ax.set_yticklabels(sorted_names, fontsize=8.5)
        ax.legend(fontsize=10.5, loc="lower right")
        max_val = float(max(np.max(sorted_sal), np.max(sorted_sig_sal))) if len(sorted_sal) > 0 else 1.0
        ax.set_xlim(0, max_val * 1.25)
    else:
        bars = ax.barh(range(n_features), sorted_sal, color="darkorange", edgecolor="black", alpha=0.85, height=0.65)
        ax.set_yticks(range(n_features))
        ax.set_yticklabels(sorted_names, fontsize=8.5)
        max_val = float(np.max(sorted_sal)) if len(sorted_sal) > 0 else 1.0
        for bar, val in zip(bars, sorted_sal):
            pct = (val / total_sal * 100) if total_sal > 0 else 0
            ax.text(val + max_val * 0.015, bar.get_y() + bar.get_height() / 2, f"{val:.4f} ({pct:.1f}%)", va="center", ha="left", fontsize=9.5)
        ax.set_xlim(0, max_val * 1.3)

    ax.set_ylabel("Features", fontsize=16)
    ax.set_xlabel(r"Integrated Gradients Attribution ($|x \cdot \nabla_x \mathcal{L}|$)", fontsize=16)
    ax.tick_params(axis="both", labelsize=10)

    cms_label(ax, lumi=lumi, year=year)
    figure.tight_layout()
    _save(figure, file_name, formats, show)


def plot_tree_with_scores(graph, node_scores, file_name, label="Reconstruction error", formats=("pdf",), root=None, show=False, lumi=None, year=None):
    """Draw a single Lund tree graph with nodes colored by error or attribution score."""
    if root is None:
        root = max(graph.edge_index[0]).item() if graph.edge_index.numel() > 0 else 0

    g = to_networkx(graph, to_undirected=True)
    pos = hierarchy_pos(g, leaf_vs_root_factor=0.5, root=root)

    set_cms_style()
    figure, ax = plt.subplots(figsize=(16, 10))
    nx.draw_networkx_edges(g, pos, ax=ax, alpha=0.35, edge_color="gray", width=1.5)
    nodes = nx.draw_networkx_nodes(g, pos, nodelist=g.nodes(), node_color=node_scores, cmap="magma", node_size=320, ax=ax)

    cb = figure.colorbar(nodes, ax=ax, pad=0.02, shrink=0.85)
    cb.set_label(label, size=17)
    cb.ax.tick_params(labelsize=10)
    ax.axis("off")
    cms_label(ax, lumi=lumi, year=year)

    figure.tight_layout()
    _save(figure, file_name, formats, show)


def plot_paired_trees(graph_bkg, scores_bkg, graph_sig, scores_sig, file_name, label="Node Feature MSE", sig_label="Signal", formats=("pdf",), lumi=None, year=None):
    """Draw side-by-side Lund trees for background and signal with a unified color scale."""
    set_cms_style()
    figure, (ax_bkg, ax_sig) = plt.subplots(1, 2, figsize=(28, 12))

    root_bkg = max(graph_bkg.edge_index[0]).item() if graph_bkg.edge_index.numel() > 0 else 0
    g_bkg = to_networkx(graph_bkg, to_undirected=True)
    pos_bkg = hierarchy_pos(g_bkg, leaf_vs_root_factor=0.5, root=root_bkg)

    root_sig = max(graph_sig.edge_index[0]).item() if graph_sig.edge_index.numel() > 0 else 0
    g_sig = to_networkx(graph_sig, to_undirected=True)
    pos_sig = hierarchy_pos(g_sig, leaf_vs_root_factor=0.5, root=root_sig)

    vmin = float(min(np.nanmin(scores_bkg), np.nanmin(scores_sig)))
    vmax = float(max(np.nanmax(scores_bkg), np.nanmax(scores_sig)))

    nx.draw_networkx_edges(g_bkg, pos_bkg, ax=ax_bkg, alpha=0.35, edge_color="gray", width=1.5)
    nx.draw_networkx_nodes(g_bkg, pos_bkg, nodelist=g_bkg.nodes(), node_color=scores_bkg, cmap="magma", vmin=vmin, vmax=vmax, node_size=320, ax=ax_bkg)
    ax_bkg.text(0.05, 0.95, "Background (QCD)", transform=ax_bkg.transAxes, fontsize=16, fontweight="bold")
    ax_bkg.axis("off")
    cms_label(ax_bkg, lumi=lumi, year=year)

    nx.draw_networkx_edges(g_sig, pos_sig, ax=ax_sig, alpha=0.35, edge_color="gray", width=1.5)
    nodes_sig = nx.draw_networkx_nodes(g_sig, pos_sig, nodelist=g_sig.nodes(), node_color=scores_sig, cmap="magma", vmin=vmin, vmax=vmax, node_size=320, ax=ax_sig)
    ax_sig.text(0.05, 0.95, f"Signal ({sig_label})", transform=ax_sig.transAxes, fontsize=16, fontweight="bold")
    ax_sig.axis("off")
    cms_label(ax_sig, lumi=lumi, year=year)

    cb = figure.colorbar(nodes_sig, ax=ax_sig, pad=0.03, shrink=0.85)
    cb.set_label(label, size=17)
    cb.ax.tick_params(labelsize=10)

    figure.subplots_adjust(top=0.92, bottom=0.05, left=0.03, right=0.95, wspace=0.1)
    _save(figure, file_name, formats, False)