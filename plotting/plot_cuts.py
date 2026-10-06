"""plot_cuts.py - Event selections, region definitions, and 2018 HEM veto."""

from __future__ import annotations
import awkward as ak
import numpy as np

HEM_FIRST_RUN = 319077
POST_HEM_FRACTION = 0.648

DELTA_ETA_MIN = 1.5
DELTA_ETA_MAX = 2.2

GAP_JET_PT = 400.0
GAP_JET_PHOTON_FRACTION = 0.6

COMMON_CUT_BRANCHES = [
    "run",
    "lumSec",
    "event",
    "nFatJet",
    "DeltaEtaJ0J1FatJet",
    "RTFatJet",
    "FatJet_isGood",
    "FatJet_pt",
    "FatJet_photonEnergyFraction",
]


def cut_branches(role: str, year: str | int) -> list[str]:
    """Branches required by the cuts and selections."""
    branches = list(COMMON_CUT_BRANCHES)
    if role != "data" and str(year) == "2018":
        branches.append("HEMVeto")
    return sorted(set(branches))


def _uniform_from_event_id(run, lumi, event) -> np.ndarray:
    """Deterministic pseudo-random number in [0, 1) per event."""
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


def post_hem_mask(events: ak.Array, role: str, year: str | int) -> np.ndarray:
    """Boolean mask of events in the 2018 post-HEM period."""
    if str(year) != "2018":
        return np.zeros(len(events), dtype=bool)
    if role != "data":
        uniform = _uniform_from_event_id(events["run"], events["lumSec"], events["event"])
        return uniform < POST_HEM_FRACTION
    return np.asarray(events["run"]) >= HEM_FIRST_RUN


def _leading(jagged, fill):
    """First element of jagged array or fill value."""
    return np.asarray(ak.fill_none(ak.firsts(jagged), fill))


def leading_jets_are_good(events: ak.Array) -> np.ndarray:
    """Both leading FatJets pass FatJet_isGood and nFatJet >= 2."""
    is_good = ak.fill_none(ak.pad_none(events["FatJet_isGood"], 2, clip=True), False)
    n_jets = np.asarray(events["nFatJet"]) >= 2
    return n_jets & np.asarray(is_good[:, 0], dtype=bool) & np.asarray(is_good[:, 1], dtype=bool)


def selection_steps(
    events: ak.Array,
    role: str,
    year: str | int,
    region: str = "validation",
    period: str = "all",
) -> list[tuple[str, np.ndarray]]:
    """Return ordered list of (step_label, individual_step_mask) for the given events."""
    steps: list[tuple[str, np.ndarray]] = [("Events_skim", np.ones(len(events), dtype=bool))]
    steps.append(("Good_j0_j1", leading_jets_are_good(events)))

    deta = np.abs(np.asarray(events["DeltaEtaJ0J1FatJet"]))
    rt = np.asarray(events["RTFatJet"])
    if region in ("validation", "highRT", "control", "lowRT"):
        steps.append((f"{DELTA_ETA_MIN} < |dEta| < {DELTA_ETA_MAX}", (deta > DELTA_ETA_MIN) & (deta < DELTA_ETA_MAX)))

    if region in ("validation", "highRT"):
        steps.append(("RT > 0.24", rt > 0.24))
    elif region == "lowRT":
        steps.append(("0.15 < RT <= 0.24", (rt > 0.15) & (rt <= 0.24)))
    elif region not in ("control", "inclusive"):
        raise ValueError(f"Unknown selection region: {region!r}")

    if str(year) == "2018":
        is_post_hem = post_hem_mask(events, role, year)
        leading_pt = _leading(events["FatJet_pt"], 0.0)
        leading_pef = _leading(events["FatJet_photonEnergyFraction"], 0.0)
        gap_jet = (leading_pt > GAP_JET_PT) & (leading_pef > GAP_JET_PHOTON_FRACTION)
        steps.append(("Post_HEM_gap_jet_veto", ~(is_post_hem & gap_jet)))

        if role != "data" and "HEMVeto" in ak.fields(events):
            hem_branch_veto = np.asarray(events["HEMVeto"], dtype=bool)
            steps.append(("Post_HEM_HEMVeto", ~is_post_hem | hem_branch_veto))

        if period == "prehem":
            steps.append(("Pre_HEM_period", ~is_post_hem))
        elif period == "posthem":
            steps.append(("Post_HEM_period", is_post_hem))
        elif period != "all":
            raise ValueError(f"Unknown period: {period!r}")

    return steps


def get_event_selection(
    events: ak.Array,
    role: str,
    year: str | int,
    region: str = "validation",
    period: str = "all",
) -> np.ndarray:
    """Build cumulative boolean selection mask for events."""
    if len(events) == 0:
        return np.zeros(0, dtype=bool)
    mask = np.ones(len(events), dtype=bool)
    for _, step_mask in selection_steps(events, role, year, region, period):
        mask &= step_mask
    return mask

