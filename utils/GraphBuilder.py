from dataclasses import dataclass

import awkward as ak
import numpy as np
import torch
import vector
from torch_geometric.data import Data

from utils.LundTreeUtilities import (
    get_lund_decomp,
    prune_tree,
    tensor_to_tree,
    tree_to_tensor,
)


@dataclass(frozen=True)
class GraphBuildConfig:
    jet_collection: str = "FatJet"
    n_store_cands: int | None = None
    use_relative_angles: bool = True
    use_displacement: bool = False
    use_dummy_values: bool = False
    use_lund_decomp: bool = False
    delta_r_threshold: float = 0.2
    save_edge_attributes: bool = False
    n_lund_vars: int = 3
    save_4vectors_in_lund_tree: bool = False
    save_energy_fractions: bool = True
    kt_cut: float | None = None
    label: float = 0.0


def pfnano_branches(config: GraphBuildConfig, extra_branches=()):
    collection = config.jet_collection
    branches = [
        "PFCands_pt",
        "PFCands_eta",
        "PFCands_phi",
        "PFCands_pdgId",
        "PFCands_mass",
        f"{collection}PFCands_jetIdx",
        f"{collection}PFCands_pFCandsIdx",
        f"n{collection}",
        f"{collection}_eta",
        f"{collection}_phi",
        f"{collection}_pt",
    ]
    if config.use_displacement:
        branches.extend(["PFCands_d0", "PFCands_dz"])
    for branch in extra_branches:
        if branch not in branches:
            branches.append(branch)
    return branches


def extract_pfnano_constituents(event, jet_index, config: GraphBuildConfig):
    collection = config.jet_collection
    candidate_indices = event[f"{collection}PFCands_pFCandsIdx"][
        event[f"{collection}PFCands_jetIdx"] == jet_index
    ].to_numpy().astype(np.int64)

    pt = event["PFCands_pt"][candidate_indices]
    permutation = ak.argsort(pt, ascending=False)
    n_constituents = len(permutation)
    if config.n_store_cands is not None:
        n_constituents = min(n_constituents, config.n_store_cands)
    permutation = permutation[:n_constituents]

    if n_constituents == 0:
        return None

    pt = np.asarray(pt[permutation], dtype=np.float64)
    eta = np.asarray(
        event["PFCands_eta"][candidate_indices][permutation], dtype=np.float64
    )
    phi = np.asarray(
        event["PFCands_phi"][candidate_indices][permutation], dtype=np.float64
    )
    pdg_id = np.asarray(
        event["PFCands_pdgId"][candidate_indices][permutation], dtype=np.int64
    )
    mass = np.clip(
        np.asarray(
            event["PFCands_mass"][candidate_indices][permutation],
            dtype=np.float64,
        ),
        a_min=0.0,
        a_max=None,
    )

    if config.use_displacement:
        d0 = np.asarray(
            event["PFCands_d0"][candidate_indices][permutation], dtype=np.float64
        )
        dz = np.asarray(
            event["PFCands_dz"][candidate_indices][permutation], dtype=np.float64
        )
    else:
        d0 = np.zeros(n_constituents, dtype=np.float64)
        dz = np.zeros(n_constituents, dtype=np.float64)

    if config.use_relative_angles:
        eta = eta - float(event[f"{collection}_eta"][jet_index])
        jet_phi = float(event[f"{collection}_phi"][jet_index])
        phi = (phi - jet_phi + np.pi) % (2 * np.pi) - np.pi

    if config.use_dummy_values:
        pt = np.random.random(n_constituents)
        eta = np.random.random(n_constituents)
        phi = np.random.random(n_constituents)
        pdg_id = np.random.random(n_constituents)
        mass = np.random.random(n_constituents)

    return {
        "pt": pt,
        "eta": eta,
        "phi": phi,
        "pdg_id": pdg_id,
        "mass": mass,
        "d0": d0,
        "dz": dz,
        "n_constituents": n_constituents,
    }


def _build_spatial_graph(constituents, config: GraphBuildConfig):
    pt = constituents["pt"]
    eta = constituents["eta"]
    phi = constituents["phi"]
    mass = constituents["mass"]
    pdg_id = constituents["pdg_id"]

    columns = [pt, eta, phi, pdg_id, mass]
    if config.use_displacement:
        columns.extend([constituents["d0"], constituents["dz"]])
    features = torch.as_tensor(np.column_stack(columns), dtype=torch.float)

    delta_eta = eta[:, None] - eta[None, :]
    delta_phi = (phi[:, None] - phi[None, :] + np.pi) % (2 * np.pi) - np.pi
    delta_r = np.hypot(delta_eta, delta_phi)
    adjacency = delta_r < config.delta_r_threshold
    np.fill_diagonal(adjacency, False)
    if config.use_dummy_values:
        adjacency = np.random.binomial(1, 0.5, delta_r.shape).astype(bool)
        np.fill_diagonal(adjacency, False)

    edge_connections = np.nonzero(adjacency)
    edge_index = torch.as_tensor(np.vstack(edge_connections), dtype=torch.long)
    graph_kwargs = {}

    if config.save_edge_attributes:
        shape = delta_r.shape
        lorentz_vectors = vector.array({
            "pt": np.broadcast_to(pt[:, None], shape),
            "eta": np.broadcast_to(eta[:, None], shape),
            "phi": np.broadcast_to(phi[:, None], shape),
            "mass": np.broadcast_to(mass[:, None], shape),
        })
        pair_masses = (lorentz_vectors + lorentz_vectors.T).mass
        pair_masses = np.clip(
            np.nan_to_num(np.log(pair_masses[edge_connections])),
            a_min=-1e5,
            a_max=1e5,
        )
        d_ca = (delta_r**2 / 0.8**2)[edge_connections]
        d_kt = (
            np.minimum(pt[:, None] ** 2, pt[None, :] ** 2)
            * delta_r**2
            / 0.8**2
        )[edge_connections]
        graph_kwargs["edge_attr"] = torch.as_tensor(
            np.column_stack((pair_masses, d_ca, d_kt)), dtype=torch.float
        )

    return Data(
        x=features,
        edge_index=edge_index,
        pos=torch.as_tensor(np.column_stack((eta, phi)), dtype=torch.float),
        num_nodes=constituents["n_constituents"],
        num_node_features=features.shape[1],
        y=torch.tensor([config.label], dtype=torch.float),
        **graph_kwargs,
    )


def _build_lund_graph(constituents, config: GraphBuildConfig):
    pdg_id = constituents["pdg_id"]
    fractions = np.asarray([
        np.abs(pdg_id) == 11,
        np.abs(pdg_id) == 13,
        pdg_id == 22,
        (np.abs(pdg_id) != 11)
        & (np.abs(pdg_id) != 13)
        & (pdg_id != 22),
    ], dtype=np.float64)

    feature_matrix, adjacency = get_lund_decomp(
        constituents["pt"],
        constituents["eta"],
        constituents["phi"],
        constituents["mass"],
        constituents["d0"],
        constituents["dz"],
        fractions=fractions,
        save_fractions=config.save_energy_fractions,
        n_lund_vars=config.n_lund_vars,
        save_4vectors=config.save_4vectors_in_lund_tree,
        use_displacement=config.use_displacement,
    )
    features = torch.as_tensor(
        np.ascontiguousarray(feature_matrix.T), dtype=torch.float
    )
    edge_index = torch.as_tensor(adjacency, dtype=torch.long)

    if config.kt_cut is not None:
        root = tensor_to_tree(features, edge_index)
        root, n_nodes = prune_tree(root, 0, config.kt_cut)
        features, edge_index = tree_to_tensor(
            root, (n_nodes, features.shape[1])
        )

    return Data(
        x=features,
        edge_index=edge_index,
        num_nodes=len(features),
        num_node_features=features.shape[1],
        y=torch.tensor([config.label], dtype=torch.float),
    )


def build_pfnano_jet_graph(event, jet_index, config: GraphBuildConfig):
    constituents = extract_pfnano_constituents(event, jet_index, config)
    if constituents is None:
        return None, 0
    if config.use_lund_decomp:
        graph = _build_lund_graph(constituents, config)
    else:
        graph = _build_spatial_graph(constituents, config)
    return graph, constituents["n_constituents"]
