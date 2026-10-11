"""Event and jet selections shared by every sample type.

This is the single place where cuts are defined. Both graph production
(``FileListGraphProducer``) and scoring (``score_gae.py``) import from here,
so signal, background and data are selected identically for a given role and
year.

Selections applied on top of the s-channel scouting pre-selection skims:

- the two leading FatJets must have ``FatJet_isGood`` set;
- ``1.5 < |DeltaEta(j0, j1)| < 2.2`` between the leading and subleading
  FatJet: the control region the autoencoder is trained on, applied to data,
  background and signal alike so inference happens in the same region;
- 2018 post-HEM events: reject if the leading FatJet has ``pt > 400`` and
  ``photonEnergyFraction > 0.6`` (extended gap-jet veto). For data post-HEM
  means ``run >= 319077``; for MC a deterministic pseudo-random fraction of
  events, ``POST_HEM_FRACTION``, is treated as post-HEM;
- 2018 MC only: the ``HEMVeto`` branch is required for the post-HEM fraction.
  The data skims already contain this veto.

Any branch used here must be returned by ``cut_branches`` so that it is read
from the ROOT file.
"""

import hashlib
import inspect
import sys

import awkward as ak
import numpy as np


# First run affected by the HEM15/16 failure in 2018.
HEM_FIRST_RUN = 319077
# Fraction of the 2018 luminosity taken from HEM_FIRST_RUN onwards,
# about 38.7 / 59.8 fb^-1. Used to split 2018 MC into pre/post-HEM.
POST_HEM_FRACTION = 0.648

# Control region in |DeltaEta(j0, j1)| used for training and inference.
DELTA_ETA_MIN = 1.5
DELTA_ETA_MAX = 2.2

# Extended gap-jet veto on the leading FatJet in post-HEM events.
GAP_JET_PT = 400.0
GAP_JET_PHOTON_FRACTION = 0.6


COMMON_BRANCHES = [
    "run",
    "lumSec",
    "event",
    "nFatJet",
    "DeltaEtaJ0J1FatJet",
    "FatJet_isGood",
    "FatJet_pt",
    "FatJet_photonEnergyFraction",
]


def is_mc(role):
    return role != "data"


def cut_branches(role, year, region="control"):
    """Branches the selection needs for a sample of this role and year."""
    branches = list(COMMON_BRANCHES)
    if region in ("validation", "highRT", "lowRT"):
        if "RTFatJet" not in branches:
            branches.append("RTFatJet")
    if is_mc(role) and str(year) == "2018":
        branches.append("HEMVeto")
    return branches


def _uniform_from_event_id(run, lumi, event):
    """Deterministic pseudo-random number in [0, 1) per event.

    Built from (run, lumSec, event) with a splitmix64 style mix, so the
    pre/post-HEM assignment of an MC event is the same in production and in
    scoring, and does not depend on file order or batching.
    """
    with np.errstate(over="ignore"):
        x = (
            np.asarray(event, dtype=np.uint64) * np.uint64(0x9E3779B97F4A7C15)
            + np.asarray(lumi, dtype=np.uint64) * np.uint64(0xBF58476D1CE4E5B9)
            + np.asarray(run, dtype=np.uint64) * np.uint64(0x94D049BB133111EB)
        )
        x ^= x >> np.uint64(30)
        x *= np.uint64(0xBF58476D1CE4E5B9)
        x ^= x >> np.uint64(27)
        x *= np.uint64(0x94D049BB133111EB)
        x ^= x >> np.uint64(31)
    return (x >> np.uint64(11)).astype(np.float64) / float(1 << 53)


def post_hem_mask(events, role, year):
    """Events to treat as taken after the HEM failure."""
    if str(year) != "2018":
        return np.zeros(len(events), dtype=bool)
    if is_mc(role):
        uniform = _uniform_from_event_id(
            events["run"], events["lumSec"], events["event"]
        )
        return uniform < POST_HEM_FRACTION
    return np.asarray(events["run"]) >= HEM_FIRST_RUN


def _leading(jagged, fill):
    """First element of every entry, or *fill* for empty entries."""
    return np.asarray(ak.fill_none(ak.firsts(jagged), fill))


def leading_jets_are_good(events):
    """Both of the two leading FatJets carry the isGood flag."""
    is_good = ak.fill_none(
        ak.pad_none(events["FatJet_isGood"], 2, clip=True), False
    )
    return (
        (np.asarray(events["nFatJet"]) >= 2)
        & np.asarray(is_good[:, 0], dtype=bool)
        & np.asarray(is_good[:, 1], dtype=bool)
    )


def selection_steps(events, role, year, region="control"):
    """Ordered ``[(label, mask), ...]`` of the individual selections.

    Each mask is the per-step decision on its own; ``event_cut_mask`` ANDs
    them and ``cutflow`` counts the events surviving them cumulatively.
    """
    steps = [("skim", np.ones(len(events), dtype=bool))]
    steps.append(("isGood j0,j1", leading_jets_are_good(events)))

    delta_eta = np.abs(np.asarray(events["DeltaEtaJ0J1FatJet"]))
    if region in ("control", "validation"):
        steps.append(
            (
                f"{DELTA_ETA_MIN} < |dEta(j0,j1)| < {DELTA_ETA_MAX}",
                (delta_eta > DELTA_ETA_MIN) & (delta_eta < DELTA_ETA_MAX),
            )
        )
    elif region in ("signal_region", "signal"):
        steps.append(
            (
                f"|dEta(j0,j1)| <= {DELTA_ETA_MIN}",
                delta_eta <= DELTA_ETA_MIN,
            )
        )
    elif region != "inclusive":
        raise ValueError(f"Unknown selection region: {region!r}")

    if region in ("validation", "highRT"):
        rt = np.asarray(events["RTFatJet"])
        steps.append(("RT > 0.24", rt > 0.24))
    elif region == "lowRT":
        rt = np.asarray(events["RTFatJet"])
        steps.append(("0.15 < RT <= 0.24", (rt > 0.15) & (rt <= 0.24)))

    if str(year) == "2018":
        post_hem = post_hem_mask(events, role, year)
        leading_pt = _leading(events["FatJet_pt"], 0.0)
        leading_photon_fraction = _leading(
            events["FatJet_photonEnergyFraction"], 0.0
        )
        gap_jet = (
            (leading_pt > GAP_JET_PT)
            & (leading_photon_fraction > GAP_JET_PHOTON_FRACTION)
        )
        steps.append(("post-HEM gap-jet veto", ~(post_hem & gap_jet)))
        if is_mc(role):
            hem_veto = np.asarray(events["HEMVeto"], dtype=bool)
            steps.append(("post-HEM HEMVeto", ~post_hem | hem_veto))

    return steps


def event_cut_mask(events, role, year, region="control"):
    """Return a boolean mask over the entries in *events*.

    *events* is the awkward record array read from the ``Events`` tree,
    containing at least the graph-building branches plus
    ``cut_branches(role, year, region=region)``.
    """
    mask = np.ones(len(events), dtype=bool)
    for _, step_mask in selection_steps(events, role, year, region=region):
        mask &= step_mask
    return mask


def cutflow(events, role, year, region="control"):
    """``{label: events surviving up to and including this step}``."""
    counts = {}
    mask = np.ones(len(events), dtype=bool)
    for label, step_mask in selection_steps(events, role, year, region=region):
        mask &= step_mask
        counts[label] = int(mask.sum())
    return counts


def add_cutflow(total, counts):
    for label, count in counts.items():
        total[label] = total.get(label, 0) + count
    return total


def format_cutflow(title, counts):
    """Table of cumulative counts, absolute and relative efficiencies."""
    width = max(len(label) for label in counts)
    lines = [f"Cutflow {title}"]
    initial = None
    previous = None
    for label, count in counts.items():
        if initial is None:
            initial = count or 1
            previous = initial
        lines.append(
            f"  {label:<{width}}  {count:>10d}  "
            f"{count / initial:7.4f}  {count / (previous or 1):7.4f}"
        )
        previous = count
    lines.insert(1, f"  {'':<{width}}  {'events':>10}  {'abs.':>7}  {'rel.':>7}")
    return "\n".join(lines)


def jet_cut_mask(event, jet_index, config):
    """Return True when the jet at *jet_index* should be turned into a graph.

    The isGood requirement on the two leading jets is an event-level cut in
    ``event_cut_mask``, so nothing further is required per jet.
    """
    return True


def cuts_fingerprint():
    """Stable hash of this module's source, used to invalidate graph caches."""
    source = inspect.getsource(sys.modules[__name__])
    return hashlib.sha256(source.encode()).hexdigest()
