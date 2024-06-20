import os

import numpy as np
import uproot
import torch
from torch_geometric.data import InMemoryDataset, Data
import awkward as ak
from tqdm import tqdm
import vector
import numba as nb
import logging
from utils.LundTreeUtilities import tensor_to_tree, prune_tree, tree_to_tensor

torch.multiprocessing.set_sharing_strategy('file_system')

@nb.njit
def get_rapidity(pt, eta, mass):
    return np.log((np.sqrt(mass**2 + (pt**2)*(np.cosh(eta)**2)) + pt*np.sinh(eta)) / np.sqrt(mass**2 + pt**2) + 1e-15)

@nb.njit
def get_delta(rapidity_a, rapidity_b, phi_a, phi_b):
    return np.sqrt((phi_a - phi_b)**2 + (rapidity_a - rapidity_b)**2)

@nb.njit
def get_z(pt_a, pt_b):
    return min(pt_a, pt_b) / (pt_a + pt_b)

@nb.njit
def get_psi(rapidity_low, rapidity_high, phi_low, phi_high):
    return np.arctan((rapidity_low - rapidity_high) / (phi_low - phi_high + 1e-15))

@nb.njit
def get_kt(pt_low, pt_high, delta):
    return min(pt_low, pt_high) * delta

@nb.njit
def get_energy(pt, eta, mass):
    return np.sqrt(mass**2 + pt**2)*np.cosh(eta)

@nb.njit
def get_dij(constituents:np.ndarray)->np.ndarray:
    """
    Calculate the dij matrix for the Cambridge-Aachen clustering algorithm

    Args:
        constituents (np.ndarray): Array of constituents with shape (4, n_constituents)

    Returns:
        np.ndarray: dij matrix with shape (n_constituents, n_constituents)
    """
    y = get_rapidity(constituents[0], constituents[1], constituents[3])
    phi = constituents[2]
    matrix_y = np.repeat(y, len(y)).reshape((len(y), -1))
    matrix_phi = np.repeat(phi, len(phi)).reshape((len(phi), -1))
    dij = ( (matrix_phi - matrix_phi.T)**2 + (matrix_y - matrix_y.T)**2 ) / 0.64

    return dij

@nb.njit
def cluster_constituents(constituents:np.ndarray, index_a:int, index_b:int):
    constituent_a = vector.obj(pt=constituents[0, index_a], eta=constituents[1, index_a], phi=constituents[2, index_a], mass=constituents[3, index_a])
    constituent_b = vector.obj(pt=constituents[0, index_b], eta=constituents[1, index_b], phi=constituents[2, index_b], mass=constituents[3, index_b])

    return constituent_a + constituent_b

@nb.njit
def get_lund_decomp(
        pt:np.ndarray,
        eta:np.ndarray,
        phi:np.ndarray,
        mass:np.ndarray,
        fractions:np.ndarray,
        save_fractions:bool=False,
        n_lund_vars:int=3,
        save_4vectors:bool=False,
        ):
    """
    Perform jet declustering, returning the Lund tree as a nested list
    A nested list is probably not ideal, a willing student could improve it :)

    Args:       
        pt (np.ndarray): pt of jet constituents
        eta (np.ndarray): eta of jet constituents
        phi (np.ndarray): phi of jet constituents
        mass (np.ndarray): mass of jet constituents
        fractions (dict): energy fractions of jet constituents (basically pdgIds at this level)
        save_fractions (bool): whether to save the energy fractions of the constituents in the Lund tree
        n_lund_vars (int): number of Lund variables to use (3 or 5)
        save_constituents_4vectors (bool): whether to save the 4-vectors of the constituents in the Lund tree

    Returns:   
        tuple[list, list]: nodes (feature matrix), edges (sparse adjacency matrix)
    """

    if n_lund_vars not in [3, 5]:
        raise ValueError("Only 3 or 5 Lund variables are supported")

    constituents = np.stack((pt, eta, phi, mass), axis=0)

    if save_fractions:
        e_fraction = fractions[0].astype(np.float64)
        mu_fraction = fractions[1].astype(np.float64)
        g_fraction = fractions[2].astype(np.float64)
        h_fraction = fractions[3].astype(np.float64)

    # Prepare sparse connection matrix for the Lund tree
    edges = np.zeros((2, 1), dtype=np.int64)

    pt = pt.astype(np.float64)
    eta = eta.astype(np.float64)
    phi = phi.astype(np.float64)
    mass = mass.astype(np.float64)

    # Features of constituents
    nodes_pt = pt.astype(np.float64).copy()
    nodes_eta = eta.astype(np.float64).copy()
    nodes_phi = phi.astype(np.float64).copy()
    nodes_mass = mass.astype(np.float64).copy()
    
    # New Features of constituents
    nodes_Delta = np.zeros(len(nodes_pt), dtype=np.float64)
    nodes_z = np.ones(len(nodes_pt), dtype=np.float64)
    nodes_psi = np.full(len(nodes_pt), np.pi/4, dtype=np.float64)
    nodes_kt = np.zeros(len(nodes_pt), dtype=np.float64)

    # Prepare index map to correctly build the Lund tree
    index_list = [i for i in range(constituents.shape[1])]

    first_clustering = True

    # Start the actual CA clustering
    while constituents.shape[1] > 1:

        dij = get_dij(constituents)

        # Avoiding the minimum distance to be the trivial 0 of each constituent with itself
        np.fill_diagonal(dij, np.inf)

        # Get pair with minimum distance
        i =  np.argmin(dij) // dij.shape[0]
        j =  np.argmin(dij) % dij.shape[0]

        # Get index of new node in the tree
        k = nodes_pt.shape[0]

        # Cluster the pair together
        new_constituent = cluster_constituents(constituents, i, j)

        nodes_pt = np.append(nodes_pt, new_constituent.pt)
        nodes_eta = np.append(nodes_eta, new_constituent.eta)
        nodes_phi = np.append(nodes_phi, new_constituent.phi)
        nodes_mass = np.append(nodes_mass, new_constituent.mass)
        
        # Order indices by constituent pt
        if constituents[0,i] < constituents[0,j]:
            i_lo, i_hi = (i, j)
        else:
            i_lo, i_hi = (j, i)
        
        # New Features for clustered particle
        Delta = get_delta(
            get_rapidity(constituents[0,i], constituents[1,i], constituents[3,i]),
            get_rapidity(constituents[0,j], constituents[1,j], constituents[3,j]),
            constituents[2,i],
            constituents[2,j]
        )
        if np.isnan(Delta):
            print("pt: ", constituents[0,i])
            print("eta: ", constituents[1,i])
            print("phi: ", constituents[2,i])
            print("mass: ", constituents[3,i])
            print("y_1", get_rapidity(constituents[0,i], constituents[1,i], constituents[3,i]))
            print("y_2", get_rapidity(constituents[0,j], constituents[1,j], constituents[3,j]))
            print(constituents[0,j], constituents[1,j], constituents[3,j])
            print("Delta: ", Delta)
            raise ValueError("NaN in Delta")
        
        z = get_z(constituents[0,i], constituents[0,j])
        psi = get_psi(
            get_rapidity(constituents[0,i_lo], constituents[1,i_lo], constituents[3,i_lo]),
            get_rapidity(constituents[0,i_hi], constituents[1,i_hi], constituents[3,i_hi]),
            constituents[2,i_lo],
            constituents[2,i_hi]
        )
        kt = get_kt(constituents[0,i], constituents[0,j], Delta)
        
        nodes_Delta = np.append(nodes_Delta, Delta)
        nodes_z = np.append(nodes_z, z)
        nodes_psi = np.append(nodes_psi, psi)
        nodes_kt = np.append(nodes_kt, kt)

        if save_fractions:
            # Energy fractions of the current pseudojets
            energy_i = get_energy(constituents[0,i], constituents[1,i], constituents[3,i])
            energy_j = get_energy(constituents[0,j], constituents[1,j], constituents[3,j])
            energy_new = get_energy(new_constituent.pt, new_constituent.eta, new_constituent.mass)
            e_fraction = np.append(e_fraction, (energy_i*e_fraction[index_list[i]] + energy_j*e_fraction[index_list[j]]) / energy_new)
            mu_fraction = np.append(mu_fraction, (energy_i*mu_fraction[index_list[i]] + energy_j*mu_fraction[index_list[j]]) / energy_new)
            g_fraction = np.append(g_fraction, (energy_i*g_fraction[index_list[i]] + energy_j*g_fraction[index_list[j]]) / energy_new)
            h_fraction = np.append(h_fraction, (energy_i*h_fraction[index_list[i]] + energy_j*h_fraction[index_list[j]]) / energy_new)
        
        # Add connections to the tree
        # This construction ensures that each clustered step is
        # two-way connected to its mother constituents
        if first_clustering:
            edges = np.array([
                [index_list[i], index_list[j], k, k],
                [k, k, index_list[i], index_list[j]]
            ], dtype=np.int64
            )
            first_clustering = False
        else:
            new_edges = np.array([
                [index_list[i], index_list[j], k, k],
                [k, k, index_list[i], index_list[j]]
            ], dtype=np.int64
            )
            edges = np.hstack((edges, new_edges))

        # Update the index map
        index_list = [idx for l, idx in enumerate(index_list) if l not in (i,j)] + [k]

        # Replace clustered constituents by the clustered particle
        
        pt = np.hstack(( 
            pt[:min(i,j)], 
            pt[min(i,j)+1:max(i,j)], 
            pt[max(i,j)+1:], 
            np.array([new_constituent.pt])
        ))
        eta = np.hstack(( 
            eta[:min(i,j)], 
            eta[min(i,j)+1:max(i,j)], 
            eta[max(i,j)+1:], 
            np.array([new_constituent.eta])
        ))
        phi = np.hstack(( 
            phi[:min(i,j)], 
            phi[min(i,j)+1:max(i,j)], 
            phi[max(i,j)+1:], 
            np.array([new_constituent.phi])
        ))
        mass = np.hstack(( 
            mass[:min(i,j)], 
            mass[min(i,j)+1:max(i,j)], 
            mass[max(i,j)+1:], 
            np.array([new_constituent.mass])
        ))

        constituents = np.stack((pt, eta, phi, mass), axis=0)

    if not save_4vectors and not save_fractions:
        if n_lund_vars == 3:
            nodes = np.vstack((
                nodes_kt,
                nodes_Delta,
                nodes_z,
            ))
        elif n_lund_vars == 5:
            nodes = np.vstack((
                nodes_kt,
                nodes_Delta,
                nodes_z,
                nodes_psi,
                nodes_mass,
            ))
    
    elif save_4vectors and not save_fractions:
        if n_lund_vars == 3:
            nodes = np.vstack((
                nodes_kt,
                nodes_Delta,
                nodes_z,
                nodes_pt,
                nodes_eta,
                nodes_phi,
                nodes_mass,
            ))
        elif n_lund_vars == 5:
            nodes = np.vstack((
                nodes_kt,
                nodes_Delta,
                nodes_z,
                nodes_psi,
                nodes_pt,
                nodes_eta,
                nodes_phi,
                nodes_mass,
            ))
         
    elif save_fractions and not save_4vectors:
        if n_lund_vars == 3:
            nodes = np.vstack((
                nodes_kt,
                nodes_Delta,
                nodes_z,
                e_fraction,
                mu_fraction,
                g_fraction,
                h_fraction,
            ))
        elif n_lund_vars == 5:
            nodes = np.vstack((
                nodes_kt,
                nodes_Delta,
                nodes_z,
                nodes_psi,
                nodes_mass,
                e_fraction,
                mu_fraction,
                g_fraction,
                h_fraction,
            ))

    else:
        if n_lund_vars == 3:
            nodes = np.vstack((
                nodes_kt,
                nodes_Delta,
                nodes_z,
                nodes_pt,
                nodes_eta,
                nodes_phi,
                nodes_mass,
                e_fraction,
                mu_fraction,
                g_fraction,
                h_fraction,
            ))
        elif n_lund_vars == 5:
            nodes = np.vstack((
                nodes_kt,
                nodes_Delta,
                nodes_z,
                nodes_psi,
                nodes_pt,
                nodes_eta,
                nodes_phi,
                nodes_mass,
                e_fraction,
                mu_fraction,
                g_fraction,
                h_fraction,
            ))

    return (nodes, edges)


class JetGraphProducer(InMemoryDataset):

    """
    Produces graphs from a root file containing jet constituents.
    The graphs are built by reclustering the constituents with the Cambridge-Aachen algorithm.
    Supports both spacial graphs and lund graphs.
    For spacial graphs, graph nodes are the constituents, while the edges encode vicinity in a given metric.
    For Lund graphs, graph nodes are pseudojets, while the edges encode the clustering history.
    For spacial graphs, the node features are the pt, eta, phi, pdgId and mass of the constituents.
    For Lund graphs, the node features are the pt, eta, phi, mass, Lund coordinates and energy fractions of the pseudojets.
    Spacial graphs can optionally have edge features, which are the invariant mass, kt distance and CA distance between constituent pairs.
    The graphs are stored in a pytorch geometric dataset and written to disk in a "processed" folder in the root file directory.
    
    Args:
        root (str): Path to the folder with (one or more) root files containing the jet constituents. All files in the folder will be processed and treated as one sample.
        n_store_jets (int): Number of jets to store per event
        delta_r_threshold (float): Threshold for the deltaR distance between constituents to be considered connected (spacial graphs only)
        n_store_cands (int): Number of constituents to store per jet (spacial graphs only)
        max_events_to_process (int): Maximum number of events to process
        use_delta_r_star (bool): Whether to use the deltaR* distance instead of the standard deltaR (TODO: implement)
        use_delta_r_star_star (bool): Whether to use the deltaR** distance instead of the standard deltaR (TODO: implement)
        transform (callable, optional): A function/transform that takes in an :obj:`torch_geometric.data.Data` object and returns a transformed version. The data object will be transformed before every access. (default: :obj:`None`)
        pre_transform (callable, optional): A function/transform that takes in an :obj:`torch_geometric.data.Data` object and returns a transformed version. The data object will be transformed before being saved to disk. (default: :obj:`None`)
        pre_filter (callable, optional): A function that takes in an :obj:`torch_geometric.data.Data` object and returns a boolean value, indicating whether the data object should be included in the final dataset. (default: :obj:`None`)
        use_relative_angles (bool): Whether to use relative angles (eta - jet_eta, phi - jet_phi) or absolute angles (eta, phi)
        use_dummy_values (bool): Whether to use dummy values for the features instead of the actual values
        save_edge_attributes (bool): Whether to save the edge attributes (pairwise invariant masses, Cambridge-Aachen and kt distances)
        save_n_constituents (bool): Whether to save the number of constituents in the graph
        save_event_number (bool): Whether to save the event number in the graph
        use_lund_decomp (bool): Whether to use the Lund decomposition instead of the spacial graph construction
        n_lund_vars (int): Number of Lund variables to use (3 or 5)
        save_4vectors_in_lund_tree (bool): Whether to save the 4-vectors of the constituents in the Lund tree
        kt_cut (float): kt cut to prune the tree (if None, no pruning is performed)
        extra_label (str): Extra label to add to the processed file name)
        weights (str): What event weights to use (None, 'xsec') (TODO: implement pt flattening)
        extra_obs_to_save_per_jet (list): List of extra jet-level observables to store in the graph (must be available in the root file)
        extra_obs_to_save_per_event (list): List of extra event-level observables to store in the graph (must be available in the root file)
        extra_obs_to_compute_per_event (list): List of extra event-level observables to compute and store in the graph (callable functions that take the event as input and return a float)
        extra_obs_to_load (list): List of extra observables to load from the root file (must be available in the root file)
        mask (ak.Array): Mask to apply to the root file
        label (float): Label to add to the graph for supervised learning
        verbose (bool): Whether to print progress bars
        input_format (str): Format of the input root files ["PFNanoAOD", "TreeMaker2"]
        jet_collection (str): Name of the jet collection in the root file (default: "FatJet")
    """

    def __init__(
        self,
        root,
        output_dataset_name,
        n_store_jets=2,
        delta_r_threshold=0.2,
        n_store_cands=None,
        max_events_to_process=None,
        use_delta_r_star=False,
        use_delta_r_star_star=False,
        transform=None,
        pre_transform=None,
        pre_filter=None,
        use_relative_angles=True,
        use_dummy_values=False,
        save_edge_attributes=False,
        save_n_constituents=False,
        save_event_number=False,
        use_lund_decomp=False,
        n_lund_vars=3,
        save_4vectors_in_lund_tree=False,
        save_energy_fractions=True,
        kt_cut=None,
        weights=None,
        extra_obs_to_save_per_jet=[],
        extra_obs_to_save_per_event=[],
        extra_obs_to_compute_per_event:list[callable]=[],
        extra_obs_to_load=[],
        mask=None,
        label:float=0.,
        verbose=False,
        input_format="PFNanoAOD",
        jet_collection="FatJet",
    ):

        self.root = root
        self.n_store_jets = n_store_jets
        self.output_dataset_name = output_dataset_name
        self.n_store_cands = n_store_cands
        self.delta_r_threshold = delta_r_threshold
        self.max_events_to_process = max_events_to_process
        self.use_delta_r_star = use_delta_r_star
        self.use_delta_r_star_star = use_delta_r_star_star
        self.pre_transform = pre_transform
        self.use_relative_angles = use_relative_angles
        self.use_dummy_values = use_dummy_values
        self.save_edge_attributes = save_edge_attributes
        self.save_n_constituents = save_n_constituents
        self.save_event_number = save_event_number
        self.use_lund_decomp = use_lund_decomp
        self.n_lund_vars = n_lund_vars
        self.save_4vectors_in_lund_tree = save_4vectors_in_lund_tree
        self.save_energy_fractions = save_energy_fractions
        self.kt_cut = kt_cut
        self.weights = weights
        self.extra_obs_to_save_per_jet = extra_obs_to_save_per_jet
        self.extra_obs_to_save_per_event = extra_obs_to_save_per_event
        self.extra_obs_to_compute_per_event = extra_obs_to_compute_per_event
        self.extra_obs_to_load = extra_obs_to_load
        self.mask = mask
        self.label = label
        self.verbose = verbose
        self.input_format = input_format
        self.jet_collection = jet_collection

        assert self.weights in ["xsec", None], "Only xsec and None are supported for the weights argument (more to be added later)"

        # Assert that no incompatible flags are used
        if use_lund_decomp:
            assert use_delta_r_star == False, "use_delta_r_star is not compatible with use_lund_decomp"
            assert use_delta_r_star_star == False, "use_delta_r_star_star is not compatible with use_lund_decomp"
            assert use_relative_angles == True, "use_relative_angles is not compatible with use_lund_decomp"
            assert use_dummy_values == False, "use_dummy_values is not compatible with use_lund_decomp"
            assert save_edge_attributes == False, "save_edge_attributes is not compatible with use_lund_decomp"
        
        else:
            assert kt_cut is None, "kt_cut is only supported for Lund tree representations"

        # If xsec weighs are requested but the number of events is hard capped, warn that an appoximation is used
        if self.weights == "xsec" and self.max_events_to_process:
            logging.warning("Using approximation for initial number of events with max number of events")

        super().__init__(root, transform, pre_transform, pre_filter)

        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return [self.root + "/" + f for f in os.listdir(self.root) if f.endswith(".root")]

    @property
    def processed_file_names(self):
        return [self.output_dataset_name+".pt"]

    def process_event(self, i_start, i_stop, events, xsec, gen_events_before_selection, total_events, events_to_process):

        graphs = []

        for i_ev in tqdm(range(i_start, i_stop), disable=not self.verbose):
            if self.input_format == "PFNanoAOD":
                n_jets = min(self.n_store_jets, events.nFatJet[i_ev])
            elif self.input_format == "TreeMaker2":
                if events[f"{self.jet_collection}/{self.jet_collection}.fCoordinates.fPt"][i_ev] is None:
                    continue
                n_jets = min(self.n_store_jets, len(events[f"{self.jet_collection}/{self.jet_collection}.fCoordinates.fPt"][i_ev]))
            for nj in range(n_jets):
                event = events[i_ev]
                if self.input_format == "PFNanoAOD":
                    jet_pt = event[f"{self.jet_collection}_pt"][nj]
                    jet_eta = event[f"{self.jet_collection}_eta"][nj]
                    jet_phi = event[f"{self.jet_collection}_phi"][nj]
                    pf_cands_matching_filter = event[f"{self.jet_collection}PFCands_pFCandsIdx"][event[f"{self.jet_collection}PFCands_jetIdx"] == nj].to_numpy().astype(np.int32)
                    pt = event["PFCands_pt"][pf_cands_matching_filter]
                    eta = event["PFCands_eta"][pf_cands_matching_filter]
                    phi = event["PFCands_phi"][pf_cands_matching_filter]
                    pdgId = event["PFCands_pdgId"][pf_cands_matching_filter]
                    mass = event["PFCands_mass"][pf_cands_matching_filter]
                elif self.input_format == "TreeMaker2":
                    jet_pt = event[f"{self.jet_collection}/{self.jet_collection}.fCoordinates.fPt"][nj]
                    jet_eta = event[f"{self.jet_collection}/{self.jet_collection}.fCoordinates.fEta"][nj]
                    jet_phi = event[f"{self.jet_collection}/{self.jet_collection}.fCoordinates.fPhi"][nj]
                    # In the TreeMaker2 format, we have to count how many constituents are matched to the first nj-1 jets
                    n_constituents_prev = sum([
                        event[f"{self.jet_collection}_constituentsIndexCounts"]
                        for i in range(nj-1)
                              ]) if nj > 0 else 0
                    cands_idx = event[f"{self.jet_collection}_constituentsIndex"][
                        n_constituents_prev:(n_constituents_prev + event[f"{self.jet_collection}_constituentsIndexCounts"][nj])
                        ]
                    pt = event["JetsConstituents/JetsConstituents.fCoordinates.fPt"][cands_idx]
                    eta = event["JetsConstituents/JetsConstituents.fCoordinates.fEta"][cands_idx]
                    phi = event["JetsConstituents/JetsConstituents.fCoordinates.fPhi"][cands_idx]
                    energy = event["JetsConstituents/JetsConstituents.fCoordinates.fE"][cands_idx]
                    mass = vector.array({"pt": pt, "eta": eta, "phi": phi, "E": energy}).mass
                    pdgId = event["JetsConstituents_PdgId"][cands_idx]

                # Order everything by pt and keep the desired number of candidates
                permutation = ak.argsort(pt, ascending=False)
                n_constituents = min(len(permutation), self.n_store_cands) if self.n_store_cands else len(permutation)
                pt = np.array(pt[permutation][:n_constituents], dtype=np.float64)
                eta = np.array(eta[permutation][:n_constituents], dtype=np.float64)
                phi = np.array(phi[permutation][:n_constituents], dtype=np.float64)
                if self.use_relative_angles:
                    eta = eta - jet_eta
                    # Need care to account for circularity
                    phi = (phi - jet_phi + np.pi) % (2*np.pi) - np.pi
                pdgId = np.array(pdgId[permutation][:n_constituents], dtype=np.int64)
                # Clip masses to zero since for some reason sometimes they are slightly negative
                mass = np.clip(np.array(mass[permutation][:n_constituents]), a_min=0., a_max=None, dtype=np.float64)

                # If requested, use random values for features
                if self.use_dummy_values:
                    pt = np.random.random(size=(len(pt),))
                    eta = np.random.random(size=(len(eta),))
                    phi = np.random.random(size=(len(phi),))
                    pdgId = np.random.random(size=(len(pdgId),))
                    mass = np.random.random(size=(len(mass),))

                if not self.use_lund_decomp:
                    pos = [[e, p] for e, p in zip(eta, phi)]

                    # Converting to np.array and subsequently to torch.tensor as suggested in torch docs for performance
                    features = torch.tensor(np.array([
                        pt,
                        eta,
                        phi,
                        pdgId,
                        mass,
                    ]).T, dtype=torch.float)

                    # Calculate edges and edge features
                    matrix_eta = np.repeat(eta, len(eta)).reshape((len(eta), -1))
                    matrix_phi = np.repeat(phi, len(phi)).reshape((len(phi), -1))
                    matrix_pt = np.repeat(pt, len(pt)).reshape((len(pt), -1))
                    matrix_mass = np.repeat(mass, len(mass)).reshape((len(mass), -1))
                    delta_eta = matrix_eta - matrix_eta.T

                    # Calculate delta phi accounting for circularity
                    delta_phi_internal = np.abs(matrix_phi - matrix_phi.T)
                    delta_phi_external = 2*np.pi - np.abs(matrix_phi - matrix_phi.T)
                    delta_phi = np.minimum(delta_phi_internal, delta_phi_external)
                    delta_R = np.sqrt(delta_eta**2 + delta_phi**2)
                    adjacency = (delta_R < self.delta_r_threshold).astype(int)

                    # If requested, substitute actual adjacency matrix with random values (to check for nconstituents dependency)
                    if self.use_dummy_values:
                        adjacency = np.random.binomial(1, 0.5, delta_R.shape)
                        np.fill_diagonal(adjacency, 1)

                    edge_connections = np.where( (adjacency - np.identity(adjacency.shape[0])) == 1)
                    edge_index = torch.tensor([ edge for edge in zip(edge_connections[0], edge_connections[1]) ], dtype=torch.long)

                    if self.save_edge_attributes:
                        # Build pair-wise invariant masses
                        lorentz_vectors = vector.array({"pt": matrix_pt, "eta": matrix_eta, "phi": matrix_phi, "mass": matrix_mass})
                        pair_masses = (lorentz_vectors + lorentz_vectors.T).mass
                        # Only keep connected edges and take log to squeeze the distribution
                        pair_masses = np.clip(np.nan_to_num(np.log(pair_masses[edge_connections])), a_max=1e5, a_min=-1e5)

                        # Build Cambridge-Aachen and kt distances between constituents
                        R = 0.8
                        d_ca = ((delta_R**2) / (R**2))[edge_connections]
                        d_kt = (np.minimum(matrix_pt**2, matrix_pt.T**2) * (delta_R**2) / (R**2))[edge_connections]

                        edge_features = torch.tensor(np.array([
                            pair_masses,
                            d_ca,
                            d_kt,
                        ]).T, dtype=torch.float)

                        # Build the graph
                        graph = Data(
                                x=features,
                                edge_index=edge_index.t().contiguous(),
                                edge_attr=edge_features,
                                num_nodes=n_constituents,
                                num_node_features=int(features.shape[1]),
                                pos=pos,
                                y=torch.Tensor([self.label])
                            )

                    else:
                        # Build the graph
                        graph = Data(
                                x=features,
                                edge_index=edge_index.t().contiguous(),
                                num_nodes=n_constituents,
                                num_node_features=int(features.shape[1]),
                                pos=pos,
                                y=torch.Tensor([self.label])
                            )
                    
                else:
                    # Initialize energy fractions
                    energy_fractions = np.array([
                        1.*(np.abs(pdgId) == 11), # electrons
                        1.*(np.abs(pdgId) == 13), # muons
                        1.*(pdgId == 22), # photons
                        1.*((np.abs(pdgId) != 11) & \
                            (np.abs(pdgId) != 13) & \
                            (np.abs(pdgId) != 22)), # hadrons
                    ])

                    # Get lund decomposition
                    ret = get_lund_decomp(
                        pt,
                        eta,
                        phi,
                        mass,
                        n_lund_vars=self.n_lund_vars,
                        save_4vectors=self.save_4vectors_in_lund_tree,
                        fractions=energy_fractions,
                        save_fractions=self.save_energy_fractions,
                        )

                    feature_matrix, adjacency_matrix = ret

                    # Pass features to torch tensor
                    features = torch.tensor(np.array(feature_matrix).T, dtype=torch.float)
                        
                    # Initialize adjacency matrix
                    adjacency = torch.tensor(np.array(adjacency_matrix))

                    if self.kt_cut is not None:
                        # Convert to tree representation
                        root_node = tensor_to_tree(features, adjacency)
                        # Prune the tree
                        root_node, n_nodes = prune_tree(root_node, 4, self.kt_cut)
                        # Convert back to tensor representation
                        features, adjacency = tree_to_tensor(root_node, (n_nodes, features.shape[1]))
                        # TODO: deal with 1-node or empty graphs after pruning

                    # Build the graph
                    graph = Data(
                            x=features,                                   
                            edge_index=torch.tensor(adjacency, dtype=torch.long),
                            num_nodes=len(features),                            
                            num_node_features=int(features.shape[1]),
                            y=torch.Tensor([self.label])
                        )
                        
                # Add the event weight
                if self.weights == "xsec":
                    if self.input_format == "PFNanoAOD":
                        if not self.max_events_to_process:
                            graph.w = torch.tensor([xsec/gen_events_before_selection], dtype=torch.float)
                        else:
                            graph.w = torch.tensor([xsec/(gen_events_before_selection * events_to_process / total_events)], dtype=torch.float)
                    elif self.input_format == "TreeMaker2":
                        graph.w = torch.tensor(event["Weight"], dtype=torch.float)
                        
                # Add the number of constituents if requested
                if self.save_n_constituents:
                    graph.n_constituents = torch.tensor([n_constituents], dtype=torch.long)

                # Add the event number if requested
                if self.save_event_number:
                    graph.event_number = torch.tensor([i_ev], dtype=torch.long)

                # Add any extra jet-level observables if requested
                for obs in self.extra_obs_to_save_per_jet:
                    graph[obs] = torch.tensor([event[obs][nj]], dtype=torch.float)
                for obs in self.extra_obs_to_save_per_event:
                    graph[obs] = torch.tensor([event[obs]], dtype=torch.float)
                for obs in self.extra_obs_to_compute_per_event:
                    graph[obs.__name__] = torch.tensor([obs(event)], dtype=torch.float)
                        
                graphs.append(graph)

        return graphs

    def get_graphs(self, file_name):

        # Check if xsec weights are requested
        if self.weights is None:
            xsec = 1.
            gen_events_before_selection = None

        if self.input_format == "PFNanoAOD":
            if self.weights == "xsec":
                with uproot.open(file_name) as in_file:
                    if "Metadata" in in_file:
                        if "GenCrossSection" in in_file["Metadata"].keys():
                            xsec = in_file["Metadata"]["GenCrossSection"].array()[0]
                            gen_events_before_selection = in_file["CutFlow"]["Initial"].array()[0]
                        else:
                            raise ValueError("No GenCrossSection found in Metadata")
                    else:
                        raise ValueError("No Metadata tree found in file")

            branches_to_load = [
                "PFCands_pt",
                "PFCands_eta",
                "PFCands_phi",
                "PFCands_pdgId",
                "PFCands_mass",
                "FatJetPFCands_jetIdx",
                "FatJetPFCands_pFCandsIdx",
                "nFatJet",
                #"genWeight",
                "FatJet_eta",
                "FatJet_phi",
                "FatJet_pt",
            ] + self.extra_obs_to_load
            
            for obs in self.extra_obs_to_save_per_event:
                if obs not in branches_to_load: branches_to_load.append(obs)
            for obs in self.extra_obs_to_save_per_jet:
                if obs not in branches_to_load: branches_to_load.append(obs)

            with uproot.open(f"{file_name}:Events") as in_file:
                events = in_file.arrays(branches_to_load, library="ak")

        elif self.input_format == "TreeMaker2":
            branches_to_load = [
                "JetsConstituents/JetsConstituents.fCoordinates.fPt",
                "JetsConstituents/JetsConstituents.fCoordinates.fEta",
                "JetsConstituents/JetsConstituents.fCoordinates.fPhi",
                "JetsConstituents/JetsConstituents.fCoordinates.fE",
                "JetsConstituents_PdgId",
                f"{self.jet_collection}_constituentsIndex",
                f"{self.jet_collection}_constituentsIndexCounts",
                f"{self.jet_collection}/{self.jet_collection}.fCoordinates.fPt",
                f"{self.jet_collection}/{self.jet_collection}.fCoordinates.fEta",
                f"{self.jet_collection}/{self.jet_collection}.fCoordinates.fPhi",
                f"{self.jet_collection}/{self.jet_collection}.fCoordinates.fE",
                "Weight",
            ] + self.extra_obs_to_load

            xsec = None
            gen_events_before_selection = None
       
            for obs in self.extra_obs_to_save_per_event:
                if obs not in branches_to_load: branches_to_load.append(obs)
            for obs in self.extra_obs_to_save_per_jet:
                if obs not in branches_to_load: branches_to_load.append(obs)

            with uproot.open(f"{file_name}:TreeMaker2/PreSelection") as in_file:
                events = in_file.arrays(branches_to_load, library="ak")

            if self.mask:
                events = ak.mask(events, ak.num(events[f"{self.jet_collection}/{self.jet_collection}.fCoordinates.fPt"], axis = 1) > 1)

        total_events = len(events)
        if self.input_format == "PFNanoAOD":
            if gen_events_before_selection is None:
                gen_events_before_selection = total_events
        
        events_to_process = min(self.max_events_to_process, total_events) if self.max_events_to_process else total_events

        graphDataset = self.process_event(
            0,
            events_to_process,
            events,
            xsec,
            gen_events_before_selection,
            total_events,
            events_to_process,
        )

        if self.pre_transform is not None:
            graphDataset = [self.pre_transform(d) for d in graphDataset]

        return graphDataset

    def process(self):
        # Read data into huge `Data` list.
        graphs = []
        for file in self.raw_file_names:
            print(f"Processing {file}")
            graphs += self.get_graphs(file)
        
        data, slices = self.collate(graphs)
        torch.save((data, slices), self.processed_paths[-1])
        
