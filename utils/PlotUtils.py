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


def cms_label(ax, label="Preliminary", lumi=None, com=13.6):
    """Draw the CMS label on *ax*."""
    import mplhep as hep

    hep.cms.label(label, ax=ax, lumi=lumi, com=com, data=True)


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


def plot_loss_curves(history, file_name, formats=("pdf",), show=False):
    """Per-batch and per-epoch training curves from a training history."""
    import mplhep as hep

    set_cms_style()
    figure, (batch_axes, epoch_axes) = plt.subplots(
        1, 2, figsize=(18, 7)
    )

    batch_axes.plot(history["batch_total"], label="Total")
    batch_axes.plot(history["batch_edges"], label="Edges")
    batch_axes.plot(history["batch_nodes"], label="Nodes")
    batch_axes.set_xlabel("Training batch")
    batch_axes.set_ylabel("Loss")
    batch_axes.set_yscale("log")
    batch_axes.legend()
    cms_label(batch_axes)

    epochs = range(len(history["epoch_train"]))
    train_totals = [values[0] for values in history["epoch_train"]]
    epoch_axes.plot(epochs, train_totals, marker="o", label="Train")
    validation = [
        values[0] if values is not None else np.nan
        for values in history.get("epoch_validation", [])
    ]
    if validation and not np.all(np.isnan(validation)):
        epoch_axes.plot(
            epochs, validation, marker="s", label="Validation"
        )
    epoch_axes.set_xlabel("Epoch")
    epoch_axes.set_ylabel("Total loss")
    epoch_axes.set_yscale("log")
    epoch_axes.legend()
    cms_label(epoch_axes)

    _save(figure, file_name, formats, show)


def plot_score_distributions(
    samples,
    file_name,
    bins=50,
    density=True,
    formats=("pdf",),
    show=False,
):
    """Reconstruction-loss histograms for every sample, overlaid.

    *samples* maps a label to ``(scores, weights)``. Returns the TH1
    histograms, so they can be written to a ROOT file.
    """
    import mplhep as hep

    set_cms_style()

    all_scores = np.concatenate(
        [np.asarray(scores) for scores, _ in samples.values()]
    )
    edges = np.histogram_bin_edges(all_scores, bins=bins)

    figure, axes = plt.subplots(figsize=(10, 8))
    histograms = {}
    for label, (scores, weights) in samples.items():
        counts, _ = weighted_histogram(
            scores, edges, weights=weights, density=density
        )
        hep.histplot(
            counts, edges, ax=axes, label=label, histtype="step", linewidth=2
        )
        histograms[f"score_{label}"] = to_th1(
            counts, edges, f"score_{label}", "Reconstruction loss"
        )

    axes.set_xlabel("Reconstruction loss")
    axes.set_ylabel("Normalized events" if density else "Events")
    axes.legend()
    cms_label(axes)
    _save(figure, file_name, formats, show)
    return histograms


def plot_loss_efficiency(
    samples, file_name, formats=("pdf",), show=False
):
    """Normalized number of events surviving a cut on the loss.

    *samples* maps a label to ``(scores, weights)``. The curve is the
    weighted fraction of entries with a loss above the threshold.
    """
    set_cms_style()
    figure, axes = plt.subplots(figsize=(10, 8))

    for label, (scores, weights) in samples.items():
        thresholds, efficiency = efficiency_curve(scores, weights)
        axes.plot(thresholds, efficiency, label=label, linewidth=2)

    axes.set_xlabel("Reconstruction loss threshold")
    axes.set_ylabel("Normalized number of events")
    axes.set_yscale("log")
    axes.legend()
    cms_label(axes)
    _save(figure, file_name, formats, show)


def plot_roc(
    scores_signal,
    scores_background,
    file_name,
    weights_signal=None,
    weights_background=None,
    formats=("pdf",),
    show=False,
):
    """Signal against background efficiency, with the area under the curve."""
    set_cms_style()
    fpr, tpr, _ = roc_curve(
        scores_signal,
        scores_background,
        weights_signal,
        weights_background,
    )

    figure, axes = plt.subplots(figsize=(10, 8))
    axes.plot(fpr, tpr, linewidth=2, label=f"AUC = {auc(fpr, tpr):.3f}")
    axes.plot([0, 1], [0, 1], linestyle="--", color="grey", linewidth=1)
    axes.set_xlabel("Background efficiency")
    axes.set_ylabel("Signal efficiency")
    axes.legend()
    cms_label(axes)
    _save(figure, file_name, formats, show)
    return fpr, tpr


def plot_sic_vs_fpr(
    scores_signal,
    scores_background,
    file_name,
    weights_signal=None,
    weights_background=None,
    min_fpr=1e-4,
    formats=("pdf",),
    show=False,
):
    """Significance improvement against background efficiency."""
    set_cms_style()
    fpr, tpr, _ = roc_curve(
        scores_signal,
        scores_background,
        weights_signal,
        weights_background,
    )
    fpr_values, sic_values = sic(tpr, fpr, min_fpr=min_fpr)

    figure, axes = plt.subplots(figsize=(10, 8))
    axes.plot(
        fpr_values,
        sic_values,
        linewidth=2,
        label=f"max SIC = {max_sic(tpr, fpr, min_fpr=min_fpr):.2f}",
    )
    axes.axhline(1.0, linestyle="--", color="grey", linewidth=1)
    axes.set_xscale("log")
    axes.set_xlabel("Background efficiency")
    axes.set_ylabel(r"$\varepsilon_S/\sqrt{\varepsilon_B}$")
    axes.legend()
    cms_label(axes)
    _save(figure, file_name, formats, show)


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