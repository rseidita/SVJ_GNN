"""plot_variables.py - Observable definitions, binning, and extraction helpers."""

from __future__ import annotations
import awkward as ak
import numpy as np

# Dictionary of observables to plot.
# Each entry specifies:
# - branch: tree branch name
# - jet_idx: (optional) jet index for jagged FatJet collections
# - bins: 1D numpy array of bin edges
# - xlabel: LaTeX x-axis label
# - logy: default y-axis scaling (True: log, False: linear)
# - out_name: filename stem for saving the plot
VARIABLES = {
    "MT01FatJetMET": {
        "branch": "MT01FatJetMET",
        "bins": np.linspace(650, 2000, 28),
        "xlabel": r"$M_{\mathrm{T}}$ [GeV]",
        "logy": True,
        "out_name": "mt",
    },
    "RTFatJet": {
        "branch": "RTFatJet",
        "bins": np.linspace(0.15, 0.6, 19),
        "xlabel": r"$R_{\mathrm{T}}$",
        "logy": True,
        "out_name": "rt",
    },
    "ScoutMET_pt": {
        "branch": "ScoutMET_pt",
        "bins": np.linspace(200, 800, 25),
        "xlabel": r"$p_{\mathrm{T}}^{\mathrm{miss}}$ [GeV]",
        "logy": True,
        "out_name": "met_pt",
    },
    "ScoutMET_phi": {
        "branch": "ScoutMET_phi",
        "bins": np.linspace(-np.pi, np.pi, 21),
        "xlabel": r"$\phi(p_{\mathrm{T}}^{\mathrm{miss}})$",
        "logy": False,
        "out_name": "met_phi",
    },
    "DeltaPhiMinFatJetMET": {
        "branch": "DeltaPhiMinFatJetMET",
        "bins": np.linspace(0.0, 0.8, 17),
        "xlabel": r"$\Delta\phi_{\min}(\mathrm{FatJet}, p_{\mathrm{T}}^{\mathrm{miss}})$",
        "logy": True,
        "out_name": "deltaphimin",
    },
    "DeltaEtaJ0J1FatJet": {
        "branch": "DeltaEtaJ0J1FatJet",
        "bins": np.linspace(-3.0, 3.0, 25),
        "xlabel": r"$\Delta\eta(j_{0}, j_{1})$",
        "logy": False,
        "out_name": "deltaeta",
    },
    "FatJet1_pt": {
        "branch": "FatJet_pt",
        "jet_idx": 0,
        "bins": np.linspace(150, 1500, 28),
        "xlabel": r"$p_{\mathrm{T}}(j_{0})$ [GeV]",
        "logy": True,
        "out_name": "fatjet1_pt",
    },
    "FatJet1_eta": {
        "branch": "FatJet_eta",
        "jet_idx": 0,
        "bins": np.linspace(-3.0, 3.0, 25),
        "xlabel": r"$\eta(j_{0})$",
        "logy": False,
        "out_name": "fatjet1_eta",
    },
    "FatJet1_phi": {
        "branch": "FatJet_phi",
        "jet_idx": 0,
        "bins": np.linspace(-np.pi, np.pi, 21),
        "xlabel": r"$\phi(j_{0})$",
        "logy": False,
        "out_name": "fatjet1_phi",
    },
    "FatJet1_mass": {
        "branch": "FatJet_mass",
        "jet_idx": 0,
        "bins": np.linspace(0, 300, 21),
        "xlabel": r"$m(j_{0})$ [GeV]",
        "logy": True,
        "out_name": "fatjet1_mass",
    },
    "FatJet2_pt": {
        "branch": "FatJet_pt",
        "jet_idx": 1,
        "bins": np.linspace(150, 800, 27),
        "xlabel": r"$p_{\mathrm{T}}(j_{1})$ [GeV]",
        "logy": True,
        "out_name": "fatjet2_pt",
    },
    "FatJet2_eta": {
        "branch": "FatJet_eta",
        "jet_idx": 1,
        "bins": np.linspace(-3.0, 3.0, 25),
        "xlabel": r"$\eta(j_{1})$",
        "logy": False,
        "out_name": "fatjet2_eta",
    },
    "FatJet2_phi": {
        "branch": "FatJet_phi",
        "jet_idx": 1,
        "bins": np.linspace(-np.pi, np.pi, 21),
        "xlabel": r"$\phi(j_{1})$",
        "logy": False,
        "out_name": "fatjet2_phi",
    },
    "FatJet2_mass": {
        "branch": "FatJet_mass",
        "jet_idx": 1,
        "bins": np.linspace(0, 300, 21),
        "xlabel": r"$m(j_{1})$ [GeV]",
        "logy": True,
        "out_name": "fatjet2_mass",
    },
    "FatJet1_photonEnergyFraction": {
        "branch": "FatJet_photonEnergyFraction",
        "jet_idx": 0,
        "bins": np.linspace(0.0, 1.0, 21),
        "xlabel": r"Photon energy fraction $(j_{0})$",
        "logy": True,
        "out_name": "fatjet1_photonEF",
    },
    "FatJet2_photonEnergyFraction": {
        "branch": "FatJet_photonEnergyFraction",
        "jet_idx": 1,
        "bins": np.linspace(0.0, 1.0, 21),
        "xlabel": r"Photon energy fraction $(j_{1})$",
        "logy": True,
        "out_name": "fatjet2_photonEF",
    },
}


def required_branches_for_vars(var_names: list[str] | None = None) -> list[str]:
    """Return all tree branches needed to construct the requested variables."""
    selected = VARIABLES if var_names is None else {k: VARIABLES[k] for k in var_names if k in VARIABLES}
    branches = set()
    for cfg in selected.values():
        branches.add(cfg["branch"])
    return sorted(branches)


def extract_variable(events: ak.Array, var_name: str, var_cfg: dict | None = None) -> np.ndarray:
    """Extract a 1D NumPy array for var_name from awkward events."""
    cfg = var_cfg if var_cfg is not None else VARIABLES[var_name]
    branch = cfg["branch"]
    if branch not in ak.fields(events):
        return np.full(len(events), np.nan, dtype=float)
    arr = events[branch]
    jet_idx = cfg.get("jet_idx")
    if jet_idx is not None:
        padded = ak.pad_none(arr, jet_idx + 1, clip=False)
        extracted = ak.fill_none(padded[:, jet_idx], np.nan)
        return ak.to_numpy(extracted)
    return ak.to_numpy(arr)

