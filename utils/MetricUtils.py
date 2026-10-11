"""Discrimination metrics for autoencoder anomaly scores.

Pure numpy, no plotting. A higher score is taken to be more signal-like,
which is how the autoencoder reconstruction loss behaves.
"""

import numpy as np


# np.trapz was renamed in numpy 2.
_trapezoid = getattr(np, "trapezoid", None) or np.trapz


def _as_weights(scores, weights):
    if weights is None:
        return np.ones(len(scores), dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)
    if weights.shape != (len(scores),):
        raise ValueError("weights must have one entry per score")
    if not np.isfinite(weights).all() or (weights < 0).any():
        raise ValueError("weights must be finite and nonnegative")
    return weights


def efficiency_curve(scores, weights=None):
    """Weighted fraction of entries with a score above each threshold.

    Returns ``(thresholds, efficiency)`` with efficiency falling from one to
    zero as the threshold rises. This is the normalized number of events
    surviving a cut on the reconstruction loss.
    """
    scores = np.asarray(scores, dtype=np.float64)
    weights = _as_weights(scores, weights)
    if not len(scores):
        raise ValueError("Cannot build an efficiency curve from no entries")

    order = np.argsort(scores)
    scores = scores[order]
    weights = weights[order]

    total = weights.sum()
    if total <= 0:
        raise ValueError("Total weight must be positive")

    # Weight surviving a cut at each threshold, i.e. everything above it.
    surviving = total - np.concatenate(([0.0], np.cumsum(weights)))
    thresholds = np.concatenate((scores, [scores[-1]]))
    return thresholds, surviving / total


def roc_curve(
    scores_signal,
    scores_background,
    weights_signal=None,
    weights_background=None,
):
    """Weighted signal efficiency against background efficiency.

    Returns ``(fpr, tpr, thresholds)``, where the true positive rate is the
    signal efficiency and the false positive rate is the background
    efficiency of a cut keeping scores above the threshold.
    """
    signal_thresholds, tpr = efficiency_curve(scores_signal, weights_signal)
    background_thresholds, background_efficiency = efficiency_curve(
        scores_background, weights_background
    )

    # Evaluate both efficiencies on one common, ascending threshold grid.
    thresholds = np.unique(
        np.concatenate((signal_thresholds, background_thresholds))
    )
    tpr = np.interp(
        thresholds, signal_thresholds, tpr, left=1.0, right=0.0
    )
    fpr = np.interp(
        thresholds,
        background_thresholds,
        background_efficiency,
        left=1.0,
        right=0.0,
    )
    return fpr, tpr, thresholds


def auc(fpr, tpr):
    """Area under the ROC curve, integrating over ascending fpr."""
    order = np.argsort(fpr)
    return float(_trapezoid(np.asarray(tpr)[order], np.asarray(fpr)[order]))


def sic(tpr, fpr, min_fpr=1e-6):
    """Significance improvement characteristic, tpr / sqrt(fpr).

    Points with a background efficiency below *min_fpr* are dropped, since
    the ratio is not statistically meaningful there.
    """
    tpr = np.asarray(tpr, dtype=np.float64)
    fpr = np.asarray(fpr, dtype=np.float64)
    keep = fpr >= min_fpr
    return fpr[keep], tpr[keep] / np.sqrt(fpr[keep])


def max_sic(tpr, fpr, min_fpr=1e-6):
    """Largest significance improvement, or NaN when no point qualifies."""
    _, values = sic(tpr, fpr, min_fpr=min_fpr)
    if not len(values):
        return float("nan")
    return float(values.max())


def weighted_histogram(scores, bins, weights=None, density=False):
    """Weighted histogram returning ``(counts, edges)``."""
    scores = np.asarray(scores, dtype=np.float64)
    weights = _as_weights(scores, weights)
    counts, edges = np.histogram(scores, bins=bins, weights=weights)
    if density:
        widths = np.diff(edges)
        total = (counts * widths).sum()
        if total > 0:
            counts = counts / total
    return counts, edges
