import os
import numpy as np
import uproot
import torch
import torch_geometric
from torch_geometric.data import InMemoryDataset, Data
import awkward as ak
from tqdm import tqdm
import vector
import logging
from utils.LundTreeUtilities import tensor_to_tree, prune_tree, tree_to_tensor, get_lund_decomp

torch.serialization.add_safe_globals([torch_geometric.data.data.DataEdgeAttr])

class LundGraphProducer(InMemoryDataset):
    """
    Produces graphs from a root file containing jet constituents.
    The graphs are built by reclustering the constituents with the Cambridge-Aachen algorithm.
    Each graph node contains information about a declustering step in the Lund plane.
    1 graph is produced per jet.
    Graphs are stored in a PyTorch Geometric InMemoryDataset and written to disk in a "processed" folder in the dataset folder.

    Args:
        dataset_folder (str): Path to the folder containing the root file(s).
        output_dataset_name (str): Name of the output dataset file (without extension).
        n_store_jets (int): Number of leading jets to store per event.
        max_events_per_file (int): Maximum number of events to process per file. If -1, process all events.
        pre_transform: Optional pre-transform to apply to each graph.
        pre_filter: Optional pre-filter to apply to each graph.
        save_event_number (bool): Whether to save the event number in each graph.
        n_lund_vars (int): Number of Lund variables to store per node (3 or 5).
        four_vectors (bool): Whether to store four-vectors on top of Lund variables.
        energy_fractions (bool): Whether to store energy fractions of different particle types.
        kt_cut (float): Minimum kt value for pruning the Lund tree. If -1, no pruning is applied.
        weights: Not implemented yet.
        extra_obs_to_save_per_jet (list): List of extra jet-level observables to save in each graph.
        extra_obs_to_save_per_event (list): List of extra event-level observables to save in each graph.
        extra_obs_to_compute_per_event (list): List of callables to compute extra event-level observables to save in each graph.
        extra_obs_to_load (list): List of extra observables to load from the root file.
        label (float): Label to assign to each graph.
        input_format (str): Format of the input root file. Currently only "L1Scouting" is supported.
        jet_collection (str): Name of the jet collection in the root file.
    """
    def __init__(
        self,
        dataset_folder:str,
        output_dataset_name:str,
        n_store_jets:int=2,
        max_events_per_file:int=-1,
        pre_transform=None,
        pre_filter=None,
        save_event_number:bool=False,
        n_lund_vars:int=3,
        four_vectors:bool=False,
        energy_fractions:bool=True,
        kt_cut:float=-1,
        weights=None,
        extra_obs_to_save_per_jet=[],
        extra_obs_to_save_per_event=[],
        extra_obs_to_compute_per_event:list[callable]=[],
        extra_obs_to_load=[],
        label:float=0.,
        input_format:str="L1Scouting",
        jet_collection:str="sc8PuppiJets",
    ):

        self.dataset_folder = dataset_folder
        self.output_dataset_name = output_dataset_name
        self.n_store_jets = n_store_jets
        self.max_events_per_file = max_events_per_file
        self.save_event_number = save_event_number
        self.n_lund_vars = n_lund_vars
        self.four_vectors = four_vectors
        self.energy_fractions = energy_fractions
        self.kt_cut = kt_cut
        self.weights = weights
        self.extra_obs_to_save_per_jet = extra_obs_to_save_per_jet
        self.extra_obs_to_save_per_event = extra_obs_to_save_per_event
        self.extra_obs_to_compute_per_event = extra_obs_to_compute_per_event
        self.extra_obs_to_load = extra_obs_to_load
        self.label = label
        self.input_format = input_format
        self.jet_collection = jet_collection

        if self.weights:
            raise NotImplementedError("Weights functionality is not implemented yet.")
        
        if self.input_format not in ["L1Scouting"]:
            raise ValueError(f"Input format {self.input_format} not yet supported.")
        
        if self.n_lund_vars not in [3, 5]:
            raise ValueError("n_lund_vars must be 3 or 5.")

        super().__init__(self.dataset_folder, pre_transform, pre_filter)

        self.data, self.slices = torch.load(self.processed_paths[0], weights_only=False)


    @property
    def raw_file_names(self):
        files = [os.path.abspath(os.path.join(self.dataset_folder, f)) for f in os.listdir(self.dataset_folder) if (f.endswith('.root') and not f.startswith('.'))]
        return files


    @property
    def processed_file_names(self):
        return [self.output_dataset_name + '.pt']


    def process_events(self, events):

        graphs = []

        # Reshape the constituent information so that it's easier to work with
        # Instead of each dau variable being its own array with shape (n_events, n_jets), we want arrays with shape (n_events, n_jets, n_cands)
        dau_vars = ["pt", "eta", "phi", "mass", "pdgId"]

        # Do pt first, since this is the one with meaningful padding values (-1)
        dau_arrays = []
        for i in range(60):
            dau_arrays.append(events[f"{self.jet_collection}_dau{i}_pt"])
        stacked = ak.concatenate([d[..., np.newaxis] for d in dau_arrays], axis=-1)
        # Remove placeholder values (-1) used for padding and save the mask
        mask = stacked != -1
        stacked = stacked[mask]
        events[f"{self.jet_collection}PFCands_pt"] = stacked
        # Remove the original dau arrays to save memory
        for i in range(60):
            del events[f"{self.jet_collection}_dau{i}_pt"]

        for var in dau_vars[1:]: # pt already done
            dau_arrays = []
            for i in range(60):
                dau_arrays.append(events[f"{self.jet_collection}_dau{i}_{var}"])
            stacked = ak.concatenate([d[..., np.newaxis] for d in dau_arrays], axis=-1)
            # Remove placeholder values (-1) used for padding
            stacked = stacked[mask]
            events[f"{self.jet_collection}PFCands_{var}"] = stacked
            # Remove the original dau arrays to save memory
            for i in range(60):
                del events[f"{self.jet_collection}_dau{i}_{var}"]
        

        for i_ev in tqdm(range(len(events) if self.max_events_per_file < 0 else self.max_events_per_file), desc="Processing events"):

            event = events[i_ev]
            for i_jet in range(min(self.n_store_jets, len(events[f"{self.jet_collection}_pt"][i_ev]))):
                
                jet_eta = event[f"{self.jet_collection}_eta"][i_jet]
                jet_phi = event[f"{self.jet_collection}_phi"][i_jet]

                # Move everything to numpy for numba compatibility
                const_pt = np.array(event[f"{self.jet_collection}PFCands_pt"][i_jet], dtype=np.float64)
                const_eta = np.array(event[f"{self.jet_collection}PFCands_eta"][i_jet], dtype=np.float64)
                const_phi = np.array(event[f"{self.jet_collection}PFCands_phi"][i_jet], dtype=np.float64)
                const_mass = np.array(event[f"{self.jet_collection}PFCands_mass"][i_jet], dtype=np.float64)
                const_pdgId = np.array(event[f"{self.jet_collection}PFCands_pdgId"][i_jet], dtype=np.float64)
                
                # Rescale to the jet eta and phi accounting for circularity
                const_eta = const_eta - jet_eta
                const_phi = (const_phi - jet_phi + np.pi) % (2*np.pi) - np.pi

                # Initialize energy fractions
                energy_fractions = np.array([
                    1.*(np.abs(const_pdgId) == 11), # electrons
                    1.*(np.abs(const_pdgId) == 13), # muons
                    1.*(const_pdgId == 22), # photons
                    1.*((np.abs(const_pdgId) != 11) & \
                        (np.abs(const_pdgId) != 13) & \
                        (np.abs(const_pdgId) != 22)), # hadrons
                ])

                # Build the Lund decomposition
                feature_matrix, adjacency_matrix = get_lund_decomp(
                    const_pt,
                    const_eta,
                    const_phi,
                    const_mass,  
                    n_lund_vars=self.n_lund_vars,
                    four_vectors=self.four_vectors,
                    fractions=energy_fractions,
                    save_fractions=self.energy_fractions,
                )

                # Pass features to torch tensor
                features = torch.tensor(np.array(feature_matrix).T, dtype=torch.float)
                        
                # Initialize adjacency matrix
                adjacency = torch.tensor(np.array(adjacency_matrix))

                if self.kt_cut > 0:
                    # Convert to tree representation
                    root_node = tensor_to_tree(features, adjacency)
                    # Prune the tree
                    root_node, n_nodes = prune_tree(root_node, 0, self.kt_cut)
                    # Convert back to tensor representation
                    features, adjacency = tree_to_tensor(root_node, (n_nodes, features.shape[1]))
                    # NOTE: after pruning: the order of the nodes may have changed! Though the
                    # tree structure is preserved.
                    # TODO: deal with 1-node or empty graphs after pruning

                # Build the graph
                graph = Data(
                        x=features,                                   
                        edge_index=torch.tensor(adjacency, dtype=torch.long),
                        num_nodes=len(features),                            
                        num_node_features=int(features.shape[1]),
                        y=torch.Tensor([self.label])
                    )
                
                # TODO: Add functionality to save event weights once I know the scheme
                # Add the event number if requested
                if self.save_event_number:
                    graph.event_number = torch.tensor([i_ev], dtype=torch.long)

                # Add any extra jet-level observables if requested
                for obs in self.extra_obs_to_save_per_jet:
                    graph[obs] = torch.tensor([event[obs][i_jet]], dtype=torch.float)
                for obs in self.extra_obs_to_save_per_event:
                    graph[obs] = torch.tensor([event[obs]], dtype=torch.float)
                for obs in self.extra_obs_to_compute_per_event:
                    graph[obs.__name__] = torch.tensor([obs(event)], dtype=torch.float)
                        
                graphs.append(graph)

        return graphs
    

    def get_graphs(self, file_name):

        #TODO: implement event weights once I know the scheme

        # Define the variables to load
        branches_to_load = [
            f"{self.jet_collection}_pt",
            f"{self.jet_collection}_eta",
            f"{self.jet_collection}_phi",
        ]

        branches_to_load += [f"{self.jet_collection}_dau{i}_pt" for i in range(60)]
        branches_to_load += [f"{self.jet_collection}_dau{i}_eta" for i in range(60)]
        branches_to_load += [f"{self.jet_collection}_dau{i}_phi" for i in range(60)]
        branches_to_load += [f"{self.jet_collection}_dau{i}_mass" for i in range(60)]
        branches_to_load += [f"{self.jet_collection}_dau{i}_pdgId" for i in range(60)]

        for obs in self.extra_obs_to_save_per_event:
            if obs not in branches_to_load: branches_to_load.append(obs)
        for obs in self.extra_obs_to_save_per_jet:
            if obs not in branches_to_load: branches_to_load.append(obs)
        for obs in self.extra_obs_to_load:
            if obs not in branches_to_load: branches_to_load.append(obs)

        # Load the file
        with uproot.open(f"{file_name}:Events") as in_file:
            events = in_file.arrays(branches_to_load, library="ak")

        graphDataset = self.process_events(events)

        if self.pre_transform is not None:
            graphDataset = [self.pre_transform(data) for data in graphDataset]

        return graphDataset
    

    def process(self):
        # Read data into huge `Data` list. FIXME: if this becomes too large, we need
        # to switch to an on-disk dataset instead of InMemoryDataset.
        graphs = []
        for raw_path in self.raw_file_names:
            print(f"Processing file {raw_path}...")
            graphs += self.get_graphs(raw_path)
        
        data, slices = self.collate(graphs)
        torch.save((data, slices), self.processed_paths[-1])